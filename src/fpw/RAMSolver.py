from typing import Callable, Union, List, Dict, Generator, Literal

import numpy as np
import scipy as sp

import torch
from geomloss import SamplesLoss

from .pt import parallel_transport


def _vector_translation(x0, x1, u0):
    return u0


def _get_vector_transport(kind="translation", *args, **kwargs):
    if kind == "translation":
        return _vector_translation
    elif kind == "parallel":
        eps = kwargs.get("epsilon", 1e-10)
        return lambda *_x: parallel_transport(*_x, epsilon)
    else:
        raise RuntimeError(f"Vector transport <<{kind}>> not implemented")


def _ot_map_geomloss(x0, x1, reg_sinkhorn, *sinkhorn_args, **sinkhorn_kwargs):
    _x0, _x1 = torch.from_numpy(x0), torch.from_numpy(x1)
    _x0.requires_grad = True
    S2_obj = SamplesLoss(loss="sinkhorn", blur=reg_sinkhorn, **sinkhorn_kwargs)(
        _x0, _x1
    )
    [v] = (torch.autograd.grad(S2_obj, [_x0]),)
    v = v[0]
    v *= -v.shape[0]

    v = v.numpy()

    return x0 + v, v

# Naming tentative; Riemannian -> Wasserstein? Ottonian?
# TODO: need to pass a bunch of (*args, **kwargs) to both PT and Sinkhorn
class RAMSolver:
    """Approximate the fixed-point :math:`\\rho^*`  for an operator :math:`F` over the Wasserstein space of probability measures, i.e. 

        .. math::

            F: \\mathcal P^2(\\mathbb{R}^d) \\to \\mathcal P^2(\\mathbb{R}^d)

            \\rho^*: \\rho^* = F(\\rho^*)

        with Riemannian(-like) Anderson Mixing scheme.

    Args:
        operator (Callable): Operator :math:`F`, fixed point of which is in question
        relaxation (Union[np.float64, Generator]): relaxation parameter, used at each iteration; constant of function of the step
        history_len (int): maximal number of previous iterates, used in the method
        vector_transport_kind (Literal["translation", "parallel"]): solver for intermediate vector transport subproblem
        vt_args, vt_kwargs: additional arguments for the vector transport solver
        ot_map_solver (Literal['pot', 'geomloss']): type of OT solver used in intermediate steps
        reg_sinkhorn (float): regularization for Sinkhorn OT solver
        sinkhorn_args, sinkhorn_kwargs: additional arguments for the OT solver

    Attributes:
        dim (int): problem dimension
        n_particles (int): number of particles in the sample approximating the current measure
    """
    def __init__(
        self,
        operator: Callable,
        relaxation: Union[np.float64, Generator] = 0.95,
        history_len: int = 2,
        vector_transport_kind: Literal["translation", "parallel"] = "translation",
        vt_args: List = [],
        vt_kwargs: Dict = {},
        ot_map_solver: Literal['pot', 'geomloss'] = 'geomloss',
        reg_sinkhorn: np.float64 = 0.2,
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

        self._vt = _get_vector_transport(vector_transport_kind, *vt_args, **vt_kwargs)

        if ot_map_solver == 'geomloss':
            self._ot_map = _ot_map_geomloss
        elif ot_map_solver == 'pot':
            self._ot_map = _ot_map_pot
        else:
            raise RuntimeError(f"{ot_map_solver=:} not supported")

        self._reg_sinkhorn = reg_sinkhorn
        self._sinkhorn_args = sinkhorn_args
        self._sinkhorn_kwargs = sinkhorn_kwargs

    def _operator_and_residual(self, x_cur: np.ndarray):
        x0 = x_cur
        x1 = self._operator(x_cur)

        return self._ot_map(x0, x1, self._reg_sinkhorn, *self._sinkhorn_args, **self._sinkhorn_kwargs)

    def _initialize_iteration(self, x0: np.ndarray):
        N, d = x0.shape
        x1, r0 = self._operator_and_residual(x0)

        self.dim = d
        self.n_particles = N
        self._x_prev = x0.copy()  # x_k-1
        self._x_cur = x1.copy()  # x_k
        self._r_prev = r0.copy()

        self._delta_rs = []
        self._delta_xs = [r0.copy()]  # Not sure
        self._k = 1

    def _step(
        self,
    ):
        _, rk = self._operator_and_residual(self._x_cur)
        # Transport Delta X and Delta r vectors to the tangent space of the current iter
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
        R = np.stack(self._delta_rs, axis=-1).reshape(-1, mk)
        X = np.stack(self._delta_xs, axis=-1).reshape(-1, mk)

        # TODO: QR solution?
        Gamma = sp.optimize.lsq_linear(R, r)
        Gamma = np.atleast_1d(Gamma.x)

        rk_bar = r - R @ Gamma
        delta_x_cur = -X @ Gamma + next(self._relaxation) * rk_bar
        delta_x_cur = delta_x_cur.reshape(self.n_particles, self.dim)
        self._delta_xs.insert(0, delta_x_cur)

        self._x_prev = self._x_cur
        self._x_cur += delta_x_cur  # <<Exponential>>
        self._r_prev = rk
        self._k += 1
        return self._x_cur

    def iterate(self, x0: np.ndarray, max_iter: int, residual_conv_tol: np.float64):
        if self._k == 0:
            self._initialize_iteration(x0)
        while np.linalg.norm(self._r_prev) > residual_conv_tol:
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

