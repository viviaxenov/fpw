from typing import Callable, Union, List, Dict, Generator, Literal

import numpy as np

import jax
import jax.numpy as jnp
import ott

# TODO: <<Real>> parallel transport + JAX
# from .pt import parallel_transport


def vector_translation(x0, x1, u0):
    return u0


def get_vector_transport(kind="translation", *args, **kwargs):
    if kind == "translation":
        return vector_translation
    elif kind == "parallel":
        raise NotImplementedError
    else:
        raise RuntimeError(f"Vector transport <<{kind}>> not supported")


# Naming tentative; Riemannian -> Wasserstein? Ottonian?
# TODO: need to pass a bunch of (*args, **kwargs) to both PT and Sinkhorn
class RAMSolverJAX:
    def __init__(
        self,
        operator: Callable,
        x0: jnp.array,
        relaxation: Union[jnp.float64, Generator] = 0.95,
        history_len: int = 2,
        vector_transport_kind: Literal["translation", "parallel"] = "translation",
        vt_args: List = [],
        vt_kwargs: Dict = {},
        reg_sinkhorn: jnp.float64 = 0.1,
        sinkhorn_args: List = [],
        sinkhorn_kwargs: Dict = {},
    ):
        self._operator = operator
        self._k = 0
        self._m = history_len

        if isinstance(relaxation, Generator):
            self._relaxation = relaxation
        elif isinstance(relaxation, float):

            def rgen():
                while True:
                    yield relaxation

            self._relaxation = rgen()
        else:
            raise RuntimeError(f"Type of relaxation ({type(relaxation)}) not supported")

        self._vt = get_vector_transport(vector_transport_kind, *vt_args, **vt_kwargs)
        # self._vt_vmapped = jax.vmap(self._vt)

        self._reg_sinkhorn = reg_sinkhorn
        self._sinkhorn_args = sinkhorn_args
        self._sinkhorn_kwargs = sinkhorn_kwargs
        self._initialize_iteration(x0)

    # TODO: pytree functional to use @jit with everything

    def _operator_and_residual(self, x_cur: jnp.ndarray):
        x0 = x_cur
        x1 = self._operator(x_cur)


        geom = ott.geometry.pointcloud.PointCloud(x0, x1, epsilon=self._reg_sinkhorn)
        ot = ott.solvers.linear.sinkhorn.Sinkhorn()(
            ott.problems.linear.linear_problem.LinearProblem(geom)
        )
        ot_plan = ot.matrix
        x1_barycentric = (ot_plan[:, :, jnp.newaxis] * x1[jnp.newaxis, :, :]).sum(
            axis=1
        ) * x0.shape[0]

        return x1_barycentric, x1_barycentric - x0

    def _initialize_iteration(self, x0: jnp.ndarray):
        N, d = x0.shape
        x1, r0 = self._operator_and_residual(x0)

        self.dim = d
        self.n_particles = N
        self._x_prev = x0.copy()  # x_k-1
        self._x_cur = x1.copy()  # x_k
        self._r_prev = r0.copy()

        self._delta_rs = []
        self._delta_xs = [r0.copy()]
        self._k = 1

    # TODO jit
    def _step(
        self,
    ):
        _, rk = self._operator_and_residual(self._x_cur)
        # Transport Delta X and Delta r vectors to the tangent space of the current iter
        # TODO: use vmap instead
        self._delta_xs = [
            self._vt(self._x_prev, self._x_cur, delta_x)
            for delta_x in self._delta_xs[: self._m]
        ]
        self._delta_rs = [
            self._vt(self._x_prev, self._x_cur, delta_r)
            for delta_r in self._delta_rs[: self._m - 1]
        ]

        delta_r_cur = rk - self._vt(self._x_prev, self._x_cur, self._r_prev)
        self._delta_rs.insert(0, delta_r_cur)

        mk = len(self._delta_rs)
        r = rk.reshape(-1)
        R = jnp.stack(self._delta_rs, axis=-1).reshape(-1, mk)
        X = jnp.stack(self._delta_xs, axis=-1).reshape(-1, mk)

        # TODO: QR solution?
        Gamma = jnp.linalg.lstsq(R, r)[0]
        Gamma = jnp.atleast_1d(Gamma)

        rk_bar = r - R @ Gamma
        delta_x_cur = -X @ Gamma + next(self._relaxation) * rk_bar
        delta_x_cur = delta_x_cur.reshape(self.n_particles, self.dim)
        self._delta_xs.insert(0, delta_x_cur)

        self._x_prev = self._x_cur
        self._x_cur += delta_x_cur  # <<Exponential>>
        self._r_prev = rk
        self._k += 1
        return self._x_cur

    def iterate(self, x0: jnp.ndarray, max_iter: int, residual_conv_tol: jnp.float64):
        if self._k == 0:
            self._initialize_iteration(x0)
        while jnp.linalg.norm(self._r_prev) > residual_conv_tol:
            self._step()
            if self._k >= max_iter:
                break

        return self._x_cur

    def restart(
        self,
        new_history_len=None,
        new_relaxation=None,
    ):
        k = self._k
        self._initialize_iteration(self._x_cur.copy())
        self._k = k
