from typing import Callable, Union, List, Dict, Generator, Literal
from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import ott

import scipy as sp
import numpy as np

from fpw import RAMSolverJAX


@jax.jit
def S2_dist_fn(x0, x1, reg_sinkhorn=0.03):
    out = ott.tools.sinkhorn_divergence.sinkhorn_divergence(
        ott.geometry.pointcloud.PointCloud,
        x=x0,
        y=x1,
        epsilon=reg_sinkhorn,
        static_b=True,
    )
    return out[0]


def W2_gaussian(x0: jnp.array, x1: jnp.array, device="gpu"):
    """
    Assuming x0, x1 are samples from Gaussian distributions and have the shape
    [N_samples, dimension], compute the (Bures)-Wasserstein distance between them
    """
    m0, m1 = x0.mean(axis=0), x1.mean(axis=0)
    cov0, cov1 = jnp.cov(x0, rowvar=False), jnp.cov(x1, rowvar=False)
    if device == "cpu":
        sqrt_cov0 = jsp.linalg.sqrtm(cov0).real
        K = jsp.linalg.sqrtm(sqrt_cov0 @ cov1 @ sqrt_cov0).real
    else:
        # there is no GPU implementation of matrix sqrt so have to put to CPU
        cov0, cov1 = np.array(cov0), np.array(cov1)
        sqrt_cov0 = sp.linalg.sqrtm(cov0).real
        K = sp.linalg.sqrtm(sqrt_cov0 @ cov1 @ sqrt_cov0).real
        cov0, cov1, K = jnp.array(cov0), jnp.array(cov1), jnp.array(K)

    dsq = jnp.trace(cov0 + cov1 - 2.0 * K) + ((m0 - m1) ** 2).sum()

    return dsq


def get_lpr_and_score_fn_gaussian(m: jnp.ndarray, sigma: jnp.ndarray):
    psq = np.linalg.inv(sigma)
    psq = jnp.array(psq)

    def _lpr(x):
        return -0.5 * jnp.linalg.norm((psq @ (x - m[jnp.newaxis, :]).T).T, axis=-1) ** 2

    def _score(x):
        return -(psq.T @ psq @ (x - m[jnp.newaxis, :]).T).T

    return _lpr, _score


def ula_step(x: jnp.ndarray, timestep: jnp.float64, score_fn: Callable, key):
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=x.shape)
    return x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5, key


@partial(jax.jit, static_argnums=[2, 3])
def mala_step(
    x: jnp.ndarray, timestep: jnp.float64, log_prob: Callable, score_fn: Callable, key
):
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=x.shape)
    x_prop = x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5

    d_log_prob = log_prob(x_prop) - log_prob(x)
    d_transition = (
        jnp.linalg.norm((x - x_prop - timestep * score_fn(x_prop)), axis=-1) ** 2
        - 2.0 * timestep * jnp.linalg.norm(noise, axis=-1) ** 2
    )

    log_alpha = jnp.minimum(0, d_log_prob - d_transition / (4.0 * timestep))
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(x.shape[0],))
    is_accepted = jnp.log(u) <= log_alpha
    acceptance_rate = is_accepted.mean()
    x_new = jnp.where(is_accepted[:, np.newaxis], x_prop, x)

    return x_new, acceptance_rate, key


rs = 10
N_particles = 200
dim = 2
N_steps = 500
hist_lens = [1, 2, 5, 10, 15][:3]

key = jax.random.PRNGKey(rs)

if dim == 2:
    sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    m = np.array([10.0, 4.0]) 
else:
    U = sp.stats.ortho_group.rvs(dim, random_state=rs)
    sigma = np.diag([1.0 - 0.9 * 0.8**n for n in range(dim)])
    sigma = U.T @ sigma @ U
    m = sp.stats.uniform.rvs(size=dim, random_state=rs)

# W2_dist_fn = jax.jit(partial(W2_gaussian, device='cpu'))
W2_dist_fn = S2_dist_fn

key, split = jax.random.split(key)
sample_targ = jax.random.multivariate_normal(
    split, m, sigma.T @ sigma, shape=(N_particles,)
)
key, split = jax.random.split(key)
sample_init = jax.random.normal(split, shape=(N_particles, dim))

print(sample_init.shape, sample_targ.shape)

lpr, score = get_lpr_and_score_fn_gaussian(m, sigma)
S2_errs_mala = []
acceptance_rates = []
timesteps = np.linspace(0.01, .1, 21, endpoint=True)

ar_means = []
logS2_rates = []
N_linear = 5

for dt in timesteps:
    sample_mala = sample_init.copy()
    S2_err = []
    ars = []
    for _ in range(N_steps):
        S2_err.append(W2_dist_fn(sample_mala, sample_targ))
        key, split = jax.random.split(key)
        sample_mala, ar, key = mala_step(sample_mala, dt, lpr, score, split)
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


def operator(_x, keygen):
    split = next(keygen)
    return mala_step(_x, dt_opt, lpr, score, split)[0]


def keygen(key):
    while True:
        key, split = jax.random.split(key)
        yield split


operator = partial(operator, keygen=keygen(key))


S2_convs = []
for m_history in hist_lens:
    S2_err_ram = []
    solver = RAMSolverJAX(
        operator,
        sample_init.copy(),
        history_len=m_history,
        relaxation=1.,
        reg_sinkhorn=0.01,
        sinkhorn_kwargs={"scaling": 0.3},
    )
    for k in range(N_steps):
        sample_ram = solver._x_prev
        S2_err_ram.append(W2_dist_fn(sample_ram, sample_targ))
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
