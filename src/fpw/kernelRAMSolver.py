from typing import Callable, Union, List, Dict, Generator, Literal, Generator, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt

from functools import partial
from .utility import _as_generator

@jax.jit
def bandwidth_median(X: jnp.array):
    N, d = X.shape
    X_diffs = X[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]
    idx = jnp.triu_indices(N, k=1)
    X_diffs = X_diffs[*idx, :]
    pairwise_dists = (X_diffs**2).sum(axis=-1)
    h = jnp.median(pairwise_dists)
    h = jnp.sqrt(0.5 * h / jnp.log(d + 1))

    return h


# @jax.jit
def norm_rkhs(x, G):
    rkhs_norm_sq = jnp.einsum("ij,ijkl,kl", x, G, x)
    return jnp.sqrt(jnp.maximum(rkhs_norm_sq, 0.0))


# @jax.jit
def norm_l2(v):
    norm_l2 = jnp.linalg.norm(v, axis=-1).mean()
    return norm_l2


def getOperatorSteinGradKL(log_density_target: Callable, stepsize: jnp.float64):
    grad_log_fn = jax.vmap(jax.grad(log_density_target))

    # @jax.jit
    def steinGradKL(x: jnp.ndarray):
        N_samples, dim = x.shape
        sg = jnp.zeros((N_samples, dim + 1))
        sg = sg.at[:, :-1].set(grad_log_fn(x))

        sg = sg.at[:, -1].set(1.0)

        return -sg / N_samples * stepsize

    return steinGradKL


# @partial(jax.jit, static_argnums=-1)
def evalTangent(
    x_eval: jnp.ndarray, tangent: jnp.ndarray, x_basis: jnp.ndarray, kern: Callable
):
    # TODO: kern vectorized? how to broadcast?
    grad_of_kern = jax.grad(kern, argnums=1)
    vect_kern = jnp.vectorize(kern, signature="(i),(i)->()")
    vect_grad = jnp.vectorize(grad_of_kern, signature="(i),(i)->(i)")

    k = vect_kern(x_eval[:, jnp.newaxis, :], x_basis[jnp.newaxis, :, :])
    gk = vect_grad(x_eval[:, jnp.newaxis, :], x_basis[jnp.newaxis, :, :])

    comp_T0 = (k[:, :, jnp.newaxis] * tangent[jnp.newaxis, :, :-1]).sum(axis=1)
    comp_T1 = (gk * tangent[jnp.newaxis, :, -1, jnp.newaxis]).sum(axis=1)

    return comp_T0 + comp_T1


# to do: reimplement as matvec?
# @partial(jax.jit, static_argnums=-1)
def pairwiseScalarProductOfBasisVectors(x1: jnp.array, x2: jnp.array, kern: Callable):
    N, d = x1.shape
    M, d1 = x2.shape
    assert d1 == d

    grad_of_kern = jax.grad(kern, argnums=1)
    vect_kern = jnp.vectorize(kern, signature="(i),(i)->()")
    vect_grad = jnp.vectorize(grad_of_kern, signature="(i),(i)->(i)")
    jac_of_kern = jax.jacrev(grad_of_kern, argnums=0)

    mixed_deriv_term = jnp.vectorize(
        lambda _x1, _x2: jnp.trace(jac_of_kern(_x1, _x2)), signature="(i),(i)->()"
    )

    res_mat = jnp.zeros((N, d + 1, M, d + 1))

    k = vect_kern(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :])
    for i in range(d):
        res_mat = res_mat.at[:, i, :, i].set(k)

    res_mat = res_mat.at[:, :-1, :, -1].set(
        vect_grad(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :]).swapaxes(1, 2)
    )
    res_mat = res_mat.at[:, -1, :, :-1].set(
        vect_grad(x2[jnp.newaxis, :, :], x1[:, jnp.newaxis, :])
    )
    res_mat = res_mat.at[:, -1, :, -1].set(
        mixed_deriv_term(x1[:, jnp.newaxis, :], x2[jnp.newaxis, :, :])
    )

    return res_mat


# @jax.jit
def vectorTransport(
    x_cur: jnp.ndarray,
    G_cur: jnp.ndarray,
    x_prev: jnp.ndarray,
    tang_prev: jnp.ndarray,
    T_cur: jnp.ndarray,
    reg_proj: jnp.float64 = 1e-6,
):
    N, d = x_cur.shape
    M, d1 = x_prev.shape
    assert d1 == d
    if len(tang_prev.shape) == 2:
        tang_prev = tang_prev[:, :, jnp.newaxis]
    mk = tang_prev.shape[-1]

    # matTransition = pairwiseScalarProductOfBasisVectors(x_cur, x_prev)
    rhs = jnp.einsum("ijkl,klm->ijm", T_cur, tang_prev)

    tang_cur = jnp.stack(
        [
            jsp.sparse.linalg.cg(
                lambda _x: jnp.einsum("ijkl,kl->ij", G_cur, _x) + reg_proj * _x,
                rhs[:, :, i],
            )[0]
            for i in range(mk)
        ],
        axis=-1,
    )

    return tang_cur


# Naming tentative; Riemannian -> Wasserstein? Ottonian? Steinian?
class kernelRAMSolver:
    def __init__(
        self,
        operator: Callable,
        kernel: Callable,
        relaxation: Union[jnp.float64, Generator] = 0.95,
        l2_regularization: Union[jnp.float64, Generator] = 0.0,
        history_len: int = 2,
        metrics: Tuple[Callable] = None,
    ):
        self._operator = operator
        # self._k = 0
        self._m = history_len

        self._kernel = kernel
        self._relaxation = _as_generator(relaxation)
        self._l2_reg = _as_generator(l2_regularization)

        # self._residual_rkhs = []
        self._x_cur, self._x_prev, self._r_prev, self._delta_rs, self._delta_xs = (
            None,
        ) * 5
        self._metrics = metrics
        # self._initialize_iteration(x0)

    # TODO: pytree functional to use @jit with everything
    @jax.jit
    def _initialize_iteration(self, x0: jnp.ndarray):
        N, d = x0.shape
        r0 = self._operator(x0)
        G = pairwiseScalarProductOfBasisVectors(x0, x0, self._kernel)

        v = evalTangent(x0, r0, x0, self._kernel)

        x1 = x0 + v

        self._x_prev = x0.copy()  # x_k-1
        self._x_cur = x1.copy()  # x_k
        self._r_prev = r0.copy()

        self._delta_rs = None
        self._delta_xs = r0.copy()[:, :, jnp.newaxis]
        # self._k = 1
        residual_rkhs = norm_rkhs(r0, G)
        dx_l2 = norm_l2(v)

        metric_vals = (residual_rkhs, dx_l2)

        if self._metrics is not None:
            metric_vals += tuple(m(x1) for m in self._metrics)

        return self, metric_vals

    # TODO jit ?
    @jax.jit
    def _step(
        self,
    ):
        rk = self._operator(self._x_cur)
        # Compute Gram matrix Gk of the current basis
        # And auxilary matrix Tk, needed to compute projections
        Gk = pairwiseScalarProductOfBasisVectors(self._x_cur, self._x_cur, self._kernel)
        Tk = pairwiseScalarProductOfBasisVectors(
            self._x_cur, self._x_prev, self._kernel
        )

        # Transport Delta X and Delta r vectors to the tangent space of the current iter
        self._delta_xs = vectorTransport(
            self._x_cur, Gk, self._x_prev, self._delta_xs[:, :, : self._m], Tk
        )
        delta_r_cur = rk[:, :, jnp.newaxis] - vectorTransport(
            self._x_cur, Gk, self._x_prev, self._r_prev, Tk
        )
        if self._delta_rs is not None:
            self._delta_rs = vectorTransport(
                self._x_cur, Gk, self._x_prev, self._delta_rs[:, :, : self._m - 1], Tk
            )
            self._delta_rs = jnp.concatenate(
                (delta_r_cur, self._delta_rs[:, :, : self._m - 1]),
                axis=-1,
            )
        else:
            self._delta_rs = delta_r_cur

        mk = self._delta_rs.shape[-1]

        R = self._delta_rs
        X = self._delta_xs

        # TODO l_infty CONSTRAINED minimization?
        #       or adaptive regularization
        lam = next(self._l2_reg)
        W_quad = jnp.einsum("ijm,ijkl,kln->mn", R, Gk, R) + lam * jnp.eye(mk)
        rhs = jnp.einsum("ij,ijkl,klm->m", rk, Gk, R)
        # for small matrix < 15 x 15 direct solve should be OK
        Gamma = jnp.linalg.solve(W_quad, rhs)
        Gamma = jnp.atleast_1d(Gamma)

        rk_bar = rk - R @ Gamma
        delta_x_cur = -X @ Gamma + next(self._relaxation) * rk_bar
        self._delta_xs = jnp.concatenate(
            (delta_x_cur[:, :, jnp.newaxis], self._delta_xs[:, :, : self._m]),
            axis=-1,
        )

        self._x_prev = self._x_cur.copy()
        v = evalTangent(
            self._x_cur, delta_x_cur, self._x_cur, self._kernel
        )  # <<Exponential>>

        self._x_cur += v
        self._r_prev = rk
        # self._k += 1

        residual_rkhs = norm_rkhs(rk, Gk)

        dx_l2 = norm_l2(v)
        metric_vals = (residual_rkhs, dx_l2)

        if self._metrics is not None:
            metric_vals += tuple(m(self._x_cur) for m in self._metrics)

        return self, metric_vals

    # @partial(jax.jit, static_argnames=('max_iter'))
    def iterate(
        self,
        x0: jnp.ndarray,
        max_iter: int = 10,
    ):
        solver, metric_vals_orig = self._initialize_iteration(x0)
        metric_vals = [metric_vals_orig]
        for _ in range(solver._m):
            solver, mv = solver._step()
            metric_vals.append(mv)
        metric_vals = jnp.array(metric_vals)

        def body_fn(carry, *args):
            solver = carry
            solver, metric_vals = solver._step()

            return solver, metric_vals

        solver, metric_vals_run = jax.lax.scan(
            body_fn, solver, length=max_iter - solver._m
        )

        metric_vals_run = jnp.array(metric_vals_run)
        metric_vals = jnp.concatenate((metric_vals.T, metric_vals_run), axis=1)

        return solver, metric_vals

    # def restart(
    #     self,
    #     new_history_len=None,
    #     new_relaxation=None,
    # ):
    #     k = self._k
    #     self._initialize_iteration(self._x_cur.copy())
    #     self._k = k

    def _tree_flatten(self):
        children = (
            self._x_cur,
            self._x_prev,
            self._r_prev,
            self._delta_rs,
            self._delta_xs,
        )
        aux_data = {
            "history_len": self._m,
            "operator": self._operator,
            "kernel": self._kernel,
            "relaxation": self._relaxation,
            "l2_regularization": self._l2_reg,
            "metrics": self._metrics,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        sol = cls(**aux_data)
        sol._x_cur, sol._x_prev, sol._r_prev, sol._delta_rs, sol._delta_xs = children

        return sol


jax.tree_util.register_pytree_node(
    kernelRAMSolver, kernelRAMSolver._tree_flatten, kernelRAMSolver._tree_unflatten
)
