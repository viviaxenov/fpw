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

_S2_dist_fn = SamplesLoss(blur=0.08)
S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(torch.from_numpy(_s1), torch.from_numpy(_s2))

rs = 10
N_particles = 100
dim = 6
N_steps = 2_000
N_steps_warmstart = 0
hist_lens = [1, 2, 3, 5, 10]

mean = np.array([(-1.0) ** k for k in range(dim)])

target = Nonconvex(mean)
sample_targ = target.sample(
    N_particles,
)


sample_init = sp.stats.multivariate_normal(
    cov=np.eye(dim),
).rvs(
    size=N_particles,
    random_state=rs,
)

operator = ULAStep(target, 2e-4)

S2_err = []
sample_ula = sample_init.copy()
for k in range(N_steps):
    if k == N_steps_warmstart:
        sample_init_ram = sample_ula.copy()
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
        l_inf_bound_Gamma=.3,
        reg_sinkhorn=0.3,
        sinkhorn_kwargs={"scaling": 0.5},
    )
    solver._initialize_iteration(sample_init_ram.copy())
    for k in range(N_steps - N_steps_warmstart):
        sample_ram = solver._x_prev
        S2_err_ram.append(S2_dist_fn(sample_ram, sample_targ))
        try:
            t = perf_counter()
            solver._step()
            dt = perf_counter() - t
        except:
            print(f"Test run for {m_history=:-2d} terminated at {k=}")
            break
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
