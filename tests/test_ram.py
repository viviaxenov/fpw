from typing import Callable, Union, List, Dict, Generator, Literal
from functools import partial

import numpy as np
import scipy as sp

import torch
from geomloss import SamplesLoss

import matplotlib.pyplot as plt
from time import perf_counter

from fpw import RAMSolver
from fpw.utility import *

S2_dist_fn = SamplesLoss(blur=0.08)

rs = 10
N_particles = 300
dim = 40
n_comp = 10
N_steps = 60
N_steps_warmstart = 0
hist_lens = [1, 2, 3, 5, 10]

# mean = torch.Tensor([(-1.0) ** k for k in range(dim)])
# target = Nonconvex(a=mean)

# ms = torch.randn((n_comp, dim))
# norm_ms = torch.linalg.norm(ms, axis=-1)
# ms_coef = (5.0 + norm_ms) / norm_ms
# ms *= ms_coef[:, None]
#
# mix = torch.distributions.Categorical(
#     torch.ones(
#         n_comp,
#     )
# )
# comp = torch.distributions.Independent(
#     torch.distributions.Normal(ms, torch.full((n_comp, dim), fill_value=0.3)), 1
# )
# gmm = torch.distributions.MixtureSameFamily(mix, comp)
#
# target = gmm

loc = torch.ones(dim)
cov = torch.randn((dim, dim))
cov = torch.mm(cov.t(), cov) + .1 * torch.eye(dim)


target = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

sample_targ = target.rsample((2 * N_particles,))

sample_init = torch.randn((N_particles, dim))

# operator = ULAStep(target, 9.e-4)
operator = MALAStep(target, 1.0)
operator.tune(sample_init)

S2_err = []
sample_ula = sample_init.detach().clone()
for k in range(N_steps):
    if k == N_steps_warmstart:
        sample_init_ram = sample_ula.detach().clone()
    S2_err.append(S2_dist_fn(sample_ula, sample_targ))
    sample_ula = operator(sample_ula)


S2_convs = []
ars_ram = []
for m_history in hist_lens:
    S2_err_ram = []
    ars = []
    solver = RAMSolver(
        operator,
        history_len=m_history,
        relaxation=1.2,
        l_inf_bound_Gamma=.1,
        reg_sinkhorn=0.3,
        sinkhorn_kwargs={"scaling": 0.5},
    )
    solver._initialize_iteration(sample_init_ram.detach().clone())
    for k in range(N_steps - N_steps_warmstart):
        sample_ram = solver._x_prev
        S2_err_ram.append(S2_dist_fn(sample_ram, sample_targ))
        try:
            t = perf_counter()
            solver._step()
            dt_per_iter = perf_counter() - t
        except Exception as e:
            print(f"Test run for {m_history=:-2d} terminated at {k=}")
            print(f"\tError: {e}")
            break
    _, ar = operator.step(sample_ram)
    print(f"{m_history=:2d} {dt_per_iter=:.2e} {ar=:.3f}")
    # print(f"{m_history=:2d} {dt_per_iter=:.2e}")
    S2_convs.append(S2_err_ram)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

ax = axs[0]

s_marker = 5.0
# ax.scatter(*sample_init[:, :2].T, s=s_marker, label="Initial")
ax.scatter(*sample_targ[:, :2].T, s=s_marker, label="Target")
ax.scatter(*sample_ula[:, :2].T, s=s_marker, marker="+", label="ULA approx")
ax.scatter(*sample_ram[:, :2].T, s=s_marker, marker="x", label="RAM approx")

steps_ram = list(range(N_steps_warmstart, N_steps))

ax = axs[1]
ax.plot(S2_err, label="ULA", linewidth=2.0, linestyle="--")

ax.scatter(N_steps_warmstart, S2_err[N_steps_warmstart], 20.0, marker="*", color="g")

for S2_err_ram, m in zip(S2_convs, hist_lens):
    ax.plot(
        steps_ram[: len(S2_err_ram)],
        S2_err_ram,
        label=f"ULA+RAM, ${m=}$",
        linewidth=0.7,
    )
ax.set_yscale("log")
ax.set_xlabel("$k$")
ax.set_ylabel("$S_s(\\mu_k, \\mu^*)$")

for ax in axs:
    ax.grid()
    ax.legend()
fig.tight_layout()
fig.savefig("ram_ula_test.pdf")

fig, ax = plt.subplots(1, 1)

ax.set_title(f"RAM, {m = }")
ax.plot(steps_ram, S2_err_ram, label=f"$W_2(\\rho_k, \\rho^\\infty)$", linewidth=0.7)
ax.plot(steps_ram, solver.W2_between_iterates[:-1], label="$W_2(\\rho_k, \\rho_{k+1})$")
ax.set_yscale("log")
ax.legend(loc="upper right")
ax.grid()

ax = ax.twinx()
ax.plot(steps_ram, solver.norm_rk[:-1], "r--", label="$\|r_k\|_{L^2_{\\rho_k}}$")
ax.plot(steps_ram, solver.norm_Gamma[:-1], "g--", label="$\|\Gamma\|_{l_2}$")
ax.legend(loc="center right")
ax.grid()


plt.show()
