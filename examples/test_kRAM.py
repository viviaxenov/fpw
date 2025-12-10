import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import jax

jax.config.update("jax_traceback_filtering", "off")

from fpw.kernelRAMSolver import *

from geomloss import SamplesLoss

# import torch
import warnings
from time import perf_counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

_loss_fn = SamplesLoss()

# sinkdiv = lambda _x, _y: _loss_fn(
#     torch.from_numpy(np.array(_x)),
#     torch.from_numpy(np.array(_y)),
# )

sinkdiv = lambda *args: 1.0

# with jax.profiler.trace("/tmp/profile-data"):
N_samples = 2000
dim = 20
N_iter = 3000

sigma_targ = 0.3
m_targ = 0.0
p = 4.0

stepsize_SVGD = 0.5


# Gaussian mixture
# log_density_targ = lambda _x: jnp.log(
#     jnp.exp(-0.5 * ((_x + m_targ) ** 2).sum() / sigma_targ**2)
#     + jnp.exp(-0.5 * ((_x - m_targ) ** 2).sum() / sigma_targ**2)
# )

log_density_targ = lambda _x: -(((jnp.abs(_x - m_targ) ** p).sum()) ** (1.0 / p))


key = jax.random.PRNGKey(5)
k1, k2, k3 = jax.random.split(key, 3)
sample_targ = jnp.concatenate(
    (
        jax.random.normal(k1, (N_samples // 2, dim)) * sigma_targ + m_targ,
        jax.random.normal(k2, (N_samples // 2, dim)) * sigma_targ - m_targ,
    ),
    axis=0,
)
x0 = jax.random.normal(k3, (N_samples, dim))
kern = lambda _x1, _x2: jnp.exp(-((_x1 - _x2) ** 2).sum() / bandwidth**2)
# kern = jax.jit(kern)
bandwidth = bandwidth_median(x0)

oper = getOperatorSteinGradKL(log_density_targ, -stepsize_SVGD)

solver = kernelRAMSolver(
    oper,
    kern,
    relaxation=3.00,
    l2_regularization=8e-3,
    history_len=6,
)

e_init = sinkdiv(x0, sample_targ)
# err_sinkhorn = [[e_init], [e_init]]
x = x0.copy()

# t = perf_counter()
# with jax.profiler.trace("/tmp/profile-data"):
solver, (d_rkhs_kram, d_l2_kram) = solver.iterate(x, max_iter=N_iter)
x_kRAM = solver._x_cur
def stepSVGD(carry, *args):
    x_SVGD = carry
    sg = oper(x_SVGD)
    G = pairwiseScalarProductOfBasisVectors(x_SVGD, x_SVGD, kern)
    v = evalTangent(x_SVGD, sg, x_SVGD, kern)

    x_SVGD += v

    # Sinkhorn distance (== estimation of Wasserstein distance)
    # Need a reference sample ``sample.targ`` which has to be generated e.g. with MCMC
    # err_sinkhorn = sinkdiv(x_SVGD, sample_targ)
    # Estimate the residual in H^d_k norm
    d_rkhs = norm_rkhs(sg, G)
    # Estimate size of the step in l2 norm;
    d_l2 = norm_l2(v)

    return x_SVGD, (d_rkhs, d_l2)
x_SVGD, (d_rkhs, d_l2) = jax.lax.scan(stepSVGD, x0.copy(), length=N_iter)
# dt = perf_counter() - t

# print(f"Execution time {dt:3e}", flush=True)


label_RAM = f"$k$RAM, m={solver._m}"

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(d_rkhs_kram)
ax.plot(d_rkhs, color="r")
ax.set_ylabel(r"$\|r_t\|_{\mathcal{H}^d_k}$")

ax = axs[1]
ax.plot(d_l2_kram, label=label_RAM)
ax.plot(d_l2, label=f"SVGD (Picard), $h$ = {stepsize_SVGD:.2f}", color="r")
ax.set_ylabel(r"$\|\Delta x_t \|_{L_2(\rho_t)}$")

# ax = axs[2]
# ax.plot(err_sinkhorn[0])
# ax.plot(err_sinkhorn[1], color="r")
# ax.set_ylabel(r"$S_{2,\varepsilon}(x_t, x_*)$")

for ax in axs:
    ax.set_yscale("log")
    ax.grid()

axs[-1].legend()

fig.savefig("test_kRAM_convergence.pdf")

fig, axs = plt.subplots(1, 1)
axs.scatter(*x_kRAM[:, :2].T, label=r"$x_\text{ kRAM}$")
axs.scatter(*x_SVGD[:, :2].T, label=r"$x_\text{ SVGD }$")
# axs.scatter(*sample_targ[:, :2].T, label=r"$x_{\infty}$")
axs.legend()

fig.savefig("test_kRAM_scatter.pdf")
