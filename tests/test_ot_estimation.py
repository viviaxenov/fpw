from typing import Callable, Union, List, Dict, Generator

import numpy as np
import scipy as sp

import ot

import torch
from geomloss import SamplesLoss

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ott

from time import perf_counter

beta = 0.2
_S2_dist_fn = SamplesLoss(blur=beta, scaling=0.8)
S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(torch.from_numpy(_s1), torch.from_numpy(_s2))


def v_torch(_x0: torch.Tensor, _x1: torch.Tensor):
    S2_obj = _S2_dist_fn(_x0, _x1)
    [v] = (torch.autograd.grad(S2_obj, [_x0]),)
    v = v[0]
    v *= -v.shape[0]
    return v


def OT_map_jax_ott(_x0: jnp.array, _x1: jnp.array):
    geom = ott.geometry.pointcloud.PointCloud(_x0, _x1, epsilon=beta)
    ot = ott.solvers.linear.sinkhorn.Sinkhorn()(
        ott.problems.linear.linear_problem.LinearProblem(geom)
    )
    ot_plan = ot.matrix
    x1_barycentric = (ot_plan[:, :, np.newaxis] * _x1[np.newaxis, :, :]).sum(
        axis=1
    ) * _x0.shape[0]

    return x1_barycentric


OT_map_jax_ott = jax.jit(OT_map_jax_ott)


def estimate_OT_map(x0, x1, reg_sinkhorn, **kwargs_sinkhorn):
    ot_plan = ot.bregman.empirical_sinkhorn(x0, x1, reg_sinkhorn, **kwargs_sinkhorn)
    x1_barycentric = (ot_plan[:, :, np.newaxis] * x1[np.newaxis, :, :]).sum(
        axis=1
    ) * x0.shape[0]

    return x1_barycentric


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dim = 100
    rs = 1

    p_nl = 3.0
    c_nl = 0.0

    if dim == 2:
        sigmas_new = np.array([[1.0, 0.4], [0.4, 1.0]])
        m_new = np.array([6, 2.0])
    else:
        U = sp.stats.ortho_group.rvs(dim, random_state=rs)
        sigmas_new = np.diag([1.0 - 0.7 * 0.8**n for n in range(dim)]) 
        sigmas_new = U.T @ sigmas_new @ U
        m_new = sp.stats.uniform.rvs(dim, random_state=rs)

    def ot_map(x0):
        x1 = (
            m_new
            + (sigmas_new @ x0.T).T
            + (c_nl * 0.5 * p_nl * np.linalg.norm(x0, axis=-1) ** (p_nl - 2))[
                :, np.newaxis
            ]
            * x0
        )
        return x1

    Ns = list(range(10, 5100, 200))
    method_names = ["pot", "geomloss", "ott-jax"]
    method_errs = {n: [] for n in method_names}
    dTang = []
    for N_samples in Ns:
        x0 = sp.stats.norm().rvs(size=(N_samples, dim))
        x1 = ot_map(sp.stats.norm().rvs(size=(N_samples * 3 // 4, dim)))

        t = perf_counter()
        x1_pot = estimate_OT_map(x0, x1, reg_sinkhorn=beta, numIterMax=100000)
        dt_pot = perf_counter() - t

        _x0, _x1 = torch.from_numpy(x0), torch.from_numpy(x1)
        _x0.requires_grad = True

        t = perf_counter()
        v = v_torch(_x0, _x1)
        dt_torch = perf_counter() - t

        v = v.numpy()
        x1_torch = x0 + v

        _x0, _x1 = jnp.array(x0), jnp.array(x1)
        t = perf_counter()
        x1_jax = OT_map_jax_ott(_x0, _x1)
        dt_jax = perf_counter() - t
        x1_jax = np.array(x1_jax)

        # W2_error = ot.bregman.empirical_sinkhorn2(x1, x1_barycentric, beta
        x1_opt = ot_map(x0)
        W2_error = S2_dist_fn(x1, x1_pot)
        W2_error_torch = S2_dist_fn(x1, x0 + v)
        tang_error = (
            np.linalg.norm(x1_opt - x1_pot, axis=-1).mean()
            / np.linalg.norm(x1_opt - x0, axis=-1).mean()
        )

        dTang.append(tang_error)

        for name, x1_approx in zip(method_names, [x1_pot, x1_torch, x1_jax]):
            err = S2_dist_fn(x1_approx, x1_opt)
            method_errs[name].append(err)

        print(f"{N_samples=:5d} {dt_pot=:.1e} {dt_torch=:.1e} {dt_jax=:.1e}")

    fig, axs = plt.subplots(1, 2)

    ax = axs[0]

    for i, x in enumerate([x0, x1, x1_pot]):
        print(f"{i}: {x.shape})")

    ax.scatter(*x0[:, :2].T, label="$\\mu_0 = N(0, 1)$")
    ax.scatter(*x1[:, :2].T, label="$\\mu_1 = \\nabla \\varphi_\\sharp\\mu_0$")

    for name, x1_approx in zip(method_names, [x1_pot, x1_torch, x1_jax]):
        ax.scatter(*x1_approx[:, :2].T, label=f"$\\mu_1^\\varepsilon$, {name}")
    ax.quiver(
        *x0[:, :2].T,
        *(x1_pot - x0)[:, :2].T,
        color="tab:grey",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        zorder=-1,
        label="OT barycentric projection",
    )
    ax.legend()

    ax = axs[1]

    for name in method_errs:
        errors = method_errs[name]
        ax.plot(Ns, errors, label=f"$W_2(\\mu_1, \\mu_\\varepsilon)$, {name}")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

    fig.tight_layout()
    plt.show()
