from typing import Callable, Union, List, Dict, Generator, Literal
from types import SimpleNamespace
from functools import partial

import numpy as np

np.random.seed(0)
import scipy as sp

import torch

torch.manual_seed(1)
from geomloss import SamplesLoss
import ot

import matplotlib.pyplot as plt
from time import perf_counter

from fpw import RAMSolver
from fpw.utility import *


S2_dist_fn = SamplesLoss(blur=0.3)
S2_dist_fn = ot.sliced.sliced_wasserstein_distance

N_particles = 300
dim = 500
N_steps = 250
N_steps_warmstart = 0
restart_every = 300
hist_lens = [1, 2, 5, 10]
# hist_lens = [5]


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

loc = torch.ones(dim) * 0.0
ortho = sp.stats.ortho_group.rvs(dim=dim, random_state=10)
sigmas = np.diag(np.linspace(1.0, 100.0, dim, endpoint=True))
cov = torch.Tensor(ortho.T @ sigmas @ ortho)

target = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

sample_targ = target.rsample((2 * N_particles,))

sample_init = torch.randn((N_particles, dim))

operator = MALAStep(target, 1e-2)
operator.tune(sample_init)

S2_err = []
sample_mala = sample_init.detach().clone()
for k in range(N_steps):
    if k == N_steps_warmstart:
        sample_init_ram = sample_mala.detach().clone()
    S2_err.append(S2_dist_fn(sample_mala, sample_targ))
    sample_mala = operator(sample_mala)


def relaxation_gen():
    for k in range(1, 21):
        yield 1.0 + 0.5 / k**0.2
    while True:
        yield 1.0


def relaxation_gen():
    k = 0
    while True:
        yield .95 + 0.45 * 2.0 ** (-k / 8.0)
        k += 1


S2_convs = []
ars_ram = []
for m_history in hist_lens:
    S2_err_ram = []
    ars = []
    solver = RAMSolver(
        operator,
        history_len=m_history,
        # relaxation=1.,
        relaxation=relaxation_gen(),
        l_inf_bound_Gamma=.2*5/m_history,
        reg_sinkhorn=0.3,
        sinkhorn_kwargs={"scaling": 0.5},
    )
    solver._initialize_iteration(sample_init_ram.detach().clone())
    k = 0
    while k < (N_steps - N_steps_warmstart):
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
        if (k + 1) % max(
            restart_every, m_history
        ) == 0 and k + 1 < N_steps - N_steps_warmstart:
            # restart implicitly does 1 Picard step in order to set
            solver.restart(new_relaxation=relaxation_gen())
            sample_ram = solver._x_prev
            S2_err_ram.append(S2_dist_fn(sample_ram, sample_targ))
            k += 1
        k += 1

    _, ar = operator.step(sample_ram)
    print(f"{m_history=:2d} {dt_per_iter=:.2e} {ar=:.3f}")
    # print(f"{m_history=:2d} {dt_per_iter=:.2e}")
    S2_convs.append(S2_err_ram)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

ax = axs[0]

s_marker = 5.0
# ax.scatter(*sample_init[:, :2].T, s=s_marker, label="Initial")
ax.scatter(*sample_targ[:, :2].T, s=s_marker, label="Target")
ax.scatter(*sample_mala[:, :2].T, s=s_marker, marker="+", label="MALA approx")
ax.scatter(*sample_ram[:, :2].T, s=s_marker, marker="x", label="RAM approx")

steps_ram = list(range(N_steps_warmstart, N_steps))

ax = axs[1]
ax.plot(S2_err, label="MALA", linewidth=2.0, linestyle="--")

ax.scatter(N_steps_warmstart, S2_err[N_steps_warmstart], 20.0, marker="*", color="g")


dist_to_self = [
    S2_dist_fn(target.rsample((N_particles,)), target.rsample((N_particles,)))
    for _ in range(N_steps)
]
bottom_line = np.mean(dist_to_self) * 0.5
ax.fill_between(
    list(range(N_steps)),
    dist_to_self,
    bottom_line,
    label="$W^2_2(\\mu_N^1, \\mu_N^2)$",
    color="lightgray",
    alpha=0.2,
)

for S2_err_ram, m in zip(S2_convs, hist_lens):
    ax.plot(
        steps_ram[: len(S2_err_ram)],
        S2_err_ram,
        label=f"MALA+RAM, ${m=}$",
        linewidth=0.7,
    )
ax.set_yscale("log")
ax.set_xlabel("$k$")
ax.set_ylabel("$S_s(\\mu_k, \\mu^*)$")

for ax in axs:
    ax.grid()
    ax.legend()

fig.suptitle(f"Gaussan in $\\mathbb{{R}}^{{{dim}}}$")
fig.tight_layout()
fig.savefig("ram_mala_test.pdf")

plt.show()
exit()

fig, ax = plt.subplots(1, 1)

ax.set_title(f"RAM, {m = }")
ax.plot(steps_ram, S2_err_ram, label=f"$W_2(\\rho_k, \\rho^\\infty)$", linewidth=0.7)
ax.plot(steps_ram, solver.W2_between_iterates[:-1], label="$W_2(\\rho_k, \\rho_{k+1})$")
ax.set_yscale("log")
ax.legend(loc="upper right")
ax.grid()

ax = ax.twinx()
ax.plot(steps_ram, solver.norm_rk[:-1], "r--", label="$\\|r_k\\|_{L^2_{\\rho_k}}$")
ax.plot(steps_ram, solver.norm_Gamma[:-1], "g--", label="$\\|\\Gamma\\|_{l_2}$")
ax.legend(loc="center right")
ax.grid()


plt.show()
