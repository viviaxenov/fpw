from fpw.kernelRAMSolver import *

from geomloss import SamplesLoss
import torch
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

_loss_fn = SamplesLoss()

sinkdiv = lambda _x, _y: _loss_fn(
    torch.from_numpy(np.array(_x)),
    torch.from_numpy(np.array(_y)),
)

sinkdiv = lambda *args: 1.0


N_samples = 100
dim = 3
N_iter = 300

sigma_targ = 0.3
m_targ = 0.0
p = 1.0

stepsize_SVGD = 0.8

bandwidth = 0.5
kern = lambda _x1, _x2: jnp.exp(-((_x1 - _x2) ** 2).sum() / bandwidth**2)
kern = jax.jit(kern)


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

oper = getOperatorSteinGradKL(log_density_targ, -stepsize_SVGD)

solver = kernelRAMSolver(
    oper, x0, kern, relaxation=3.00, l2_regularization=8e-3, history_len=6,
)

e_init = sinkdiv(x0, sample_targ)
err_sinkhorn = [[e_init], [e_init]]
x = x0.copy()
for _ in range(N_iter):
    x = solver._step()
    err_sinkhorn[0].append(sinkdiv(x, sample_targ))

x_kRAM = x.copy()
# baseline
d_rkhs = []
d_l2 = []
x_SVGD = x0.copy()
for _ in range(N_iter):
    sg = oper(x_SVGD)
    G = pairwiseScalarProductOfBasisVectors(x_SVGD, x_SVGD, kern)
    v = evalTangent(x_SVGD, sg, x_SVGD, kern)

    x_SVGD += v

    err_sinkhorn[1].append(sinkdiv(x_SVGD, sample_targ))
    d_rkhs.append(jnp.einsum("ij,ijkl,kl", sg, G, sg) ** 0.5)
    d_l2.append(jnp.linalg.norm(v) / N_samples**0.5)

label_RAM = f"$k$RAM, m={solver._m}"

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(solver._residual_rkhs)
ax.plot(d_rkhs, color="r")
ax.set_ylabel(r"$\|r_t\|_{\mathcal{H}^d_k}$")

ax = axs[1]
ax.plot(solver._dx_l2, label=label_RAM)
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


fig, axs = plt.subplots(1, 1)
axs.scatter(*x_kRAM[:, :2].T, label=r"$x_\text{ kRAM}$")
axs.scatter(*x_SVGD[:, :2].T, label=r"$x_\text{ SVGD }$")
# axs.scatter(*sample_targ[:, :2].T, label=r"$x_{\infty}$")
axs.legend()
axs.grid()

plt.show()
