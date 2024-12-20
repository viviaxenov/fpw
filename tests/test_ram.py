from typing import Callable, Union, List, Dict, Generator, Literal

import numpy as np
import scipy as sp

import torch
from geomloss import SamplesLoss

import matplotlib.pyplot as plt

from fpw import RAMSolver

_S2_dist_fn = SamplesLoss(blur=0.03)
S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(
    torch.from_numpy(_s1), torch.from_numpy(_s2)
)

def get_lpr_and_score_fn_gaussian(m: np.ndarray, sigma: np.ndarray):
    psq = np.linalg.inv(sigma)

    def _lpr(x):
        return (
            -0.5 * np.linalg.norm((psq @ (x - m[np.newaxis, :]).T).T, axis=-1) ** 2
        )

    def _score(x):
        return -(psq.T @ psq @ (x - m[np.newaxis, :]).T).T

    return _lpr, _score

def ula_step(x: np.ndarray, timestep: np.float64, score_fn: Callable):
    noise = sp.stats.norm().rvs(size=x.shape)
    return x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5

def mala_step(
    x: np.ndarray,
    timestep: np.float64,
    log_prob: Callable,
    score_fn: Callable,
):
    noise = sp.stats.norm().rvs(size=x.shape)
    x_prop = x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5

    d_log_prob = log_prob(x_prop) - log_prob(x)
    d_transition = (
        np.linalg.norm((x - x_prop - timestep * score_fn(x_prop)), axis=-1) ** 2
        - 2.0 * timestep * np.linalg.norm(noise, axis=-1) ** 2
    )

    log_alpha = np.minimum(0, d_log_prob - d_transition / (4.0 * timestep))
    u = np.random.rand(x.shape[0])
    is_accepted = np.log(u) <= log_alpha
    acceptance_rate = is_accepted.mean()
    x_new = np.where(is_accepted[:, np.newaxis], x_prop, x)

    return x_new, acceptance_rate

rs = 10
N_particles = 50
dim = 2
N_steps = 50
hist_lens = [1, 2, 5, 10, 15]

if dim == 2:
    sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    m = np.array([10.0, 4.0]) 
else:
    U = sp.stats.ortho_group.rvs(dim, random_state=rs)
    sigma = np.diag([1.0 - 0.9 * 0.8**n for n in range(dim)])
    sigma = U.T @ sigma @ U
    m = sp.stats.uniform.rvs(size=dim, random_state=rs) * 0.0

sample_targ = sp.stats.multivariate_normal(
    mean=m,
    cov=sigma.T @ sigma,
).rvs(size=N_particles)
sample_init = sp.stats.multivariate_normal(
    cov=np.eye(dim),
).rvs(
    size=N_particles,
    random_state=rs,
)

lpr, score = get_lpr_and_score_fn_gaussian(m, sigma)
S2_errs_mala = []
acceptance_rates = []
timesteps = np.linspace(0.01, .11, 51, endpoint=True)

ar_means = []
logS2_rates = []
N_linear = 5

for dt in timesteps:
    sample_mala = sample_init.copy()
    S2_err = []
    ars = []
    for _ in range(N_steps):
        S2_err.append(S2_dist_fn(sample_mala, sample_targ))
        sample_mala, ar = mala_step(sample_mala, dt, lpr, score)
        ars.append(ar)
    S2_errs_mala.append(S2_err)
    acceptance_rates.append(ars)

    ar_means.append(np.mean(ars))
    logS2_rates.append(-np.log(S2_err[N_linear] / S2_err[0]) / N_linear)

idx_opt = logS2_rates.index(max(logS2_rates))
dt_opt = timesteps[idx_opt]
S2_err = S2_errs_mala[idx_opt]

fig, axs = plt.subplots(1, 2)

ax = axs[0]
ax.plot(timesteps, ar_means)
ax.set_xlabel("Timestep")
ax.set_ylabel("Acceptance rate")
ax.axvline(dt_opt, color="g", linestyle="--")

ax1 = ax.twinx()
ax1.plot(timesteps, logS2_rates, "r--")
ax1.set_ylabel("Convergence rate ($\\log S_2$)")

ax = axs[1]
ax.plot(ar_means, logS2_rates)
ax.set_xlabel("Acceptance rate")
ax.set_ylabel("Convergence rate ($\\log S_2$)")

fig.tight_layout()

operator = lambda _x: mala_step(_x, dt_opt, lpr, score)[0]
# operator = lambda _x: ula_step(_x, dt_opt, score)
S2_convs = []
for m_history in hist_lens:
    S2_err_ram = []
    solver = RAMSolver(
        operator,
        history_len=m_history,
        relaxation=0.8,
        reg_sinkhorn=0.01,
        sinkhorn_kwargs={"scaling": 0.3},
    )
    solver._initialize_iteration(sample_init.copy())
    for k in range(N_steps):
        sample_ram = solver._x_prev
        S2_err_ram.append(S2_dist_fn(sample_ram, sample_targ))
        solver._step()
    S2_convs.append(S2_err_ram)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

ax = axs[0]

s_marker = 5.0
ax.scatter(*sample_init[:, :2].T, s=s_marker, label="Initial")
ax.scatter(*sample_targ[:, :2].T, s=s_marker, label="Target")
# ax.scatter(*sample_mala[:, :2].T, s=s_marker, marker="+", label="MALA approx")
ax.scatter(*sample_ram[:, :2].T, s=s_marker, marker="x", label="RAM approx")

ax = axs[1]
ax.plot(S2_err, label="MALA", linewidth=2.0, linestyle="--")
for S2_err_ram, m in zip(S2_convs, hist_lens):
    ax.plot(S2_err_ram, label=f"MALA+RAM, ${m=}$", linewidth=0.7)
ax.set_yscale("log")
ax.set_xlabel("$k$")
ax.set_ylabel("$S_s(\\mu_k, \\mu^*)$")

for ax in axs:
    ax.grid()
    ax.legend()
fig.tight_layout()
fig.savefig("ram_mala_test.pdf")
plt.show()
