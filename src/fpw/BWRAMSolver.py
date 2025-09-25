from typing import Callable, Union, List, Dict, Generator, Literal

import numpy as np
import scipy as sp

import torch

torch.set_default_dtype(torch.float64)
import ot
from geomloss import SamplesLoss
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def check_cov(Cov: np.ndarray):
    assert len(Cov.shape) == 2 and Cov.shape[0] == Cov.shape[1]
    assert np.allclose(Cov, Cov.T)
    assert all(np.linalg.eigvalsh(Cov) > 0.0)


def to_dSigma(U: np.ndarray, Cov: np.ndarray):
    return U @ Cov + Cov @ U


def to_Map(V: np.ndarray, Cov: np.ndarray):
    L_CV = sp.linalg.solve_continuous_lyapunov(Cov, V)
    return L_CV


def rExpGaussian(V: np.ndarray, Cov: np.ndarray, is_map=True):
    check_cov(Cov)
    assert np.allclose(V, V.T)

    if not is_map:
        V = to_Map(V)
    Id_plus_V = np.eye(V.shape[0]) + V

    return Id_plus_V.T @ Cov @ Id_plus_V


def Christoffel(Sigma: np.ndarray, X: np.ndarray, Y: np.ndarray):
    L_CX = sp.linalg.solve_continuous_lyapunov(Sigma, X)
    L_CY = sp.linalg.solve_continuous_lyapunov(Sigma, Y)
    L_CYmulL_CX = L_CY @ L_CX
    Gamma = Sigma @ L_CYmulL_CX + L_CYmulL_CX @ Sigma - L_CX @ Y - L_CY @ X
    return 0.5 * (Gamma + Gamma.T)


def dBW(Cov_0, Cov_1):
    zero = np.zeros(Cov_0.shape[0])
    return ot.gaussian.bures_wasserstein_distance(zero, zero, Cov_0, Cov_1)


def parallel_transport(Sigma_0, Sigma_1, U0, is_map=True):
    dim = Sigma_0.shape[0]
    if is_map:
        U0 = to_dSigma(U0, Sigma_0)

    Id = np.eye(dim)
    T = ot.gaussian.bures_wasserstein_mapping(
        np.zeros(dim), np.zeros(dim), Sigma_0, Sigma_1
    )[0]
    Tdir = T - Id

    def Sigma_dSigma_dt(t):
        T_interp = Id + t * Tdir
        Sigma_t = T_interp @ Sigma_0 @ T_interp
        dSigma_dt = Tdir @ Sigma_0 + Sigma_0 @ Tdir + 2.0 * t * Tdir @ Sigma_0 @ Tdir

        return Sigma_t, dSigma_dt

    def ode_rhs(t, Ucur):
        Sigma_t, dSigma_dt = Sigma_dSigma_dt(t)
        return -Christoffel(Sigma_t, dSigma_dt, Ucur)

    _ode_rhs_flat = lambda _t, _x: ode_rhs(_t, _x.reshape((dim, dim))).reshape(-1)

    ode_result = sp.integrate.solve_ivp(
        _ode_rhs_flat, [0.0, 1.0], U0.reshape(-1), max_step=1e-2
    )
    if not ode_result.success:
        raise RuntimeError(
            "Parallel transport did not converge; Solver status: " + ode_result.message
        )

    U1 = ode_result.y[:, -1].reshape((dim, dim))

    if is_map:
        U1 = to_Map(U1, Sigma_1)

    return U1


def vector_translation(Sigma_0, Sigma_1, U0):
    dim = Sigma_0.shape[0]

    Id = np.eye(dim)
    Tinv = ot.gaussian.bures_wasserstein_mapping(
        np.zeros(dim), np.zeros(dim), Sigma_1, Sigma_0
    )[0]
    U1 = U0 @ Tinv

    return U1


def vt_trival(Sigma_0, Sigma_1, U0):
    return U0


def project_on_tangent(U, Sigma):
    rhs = Sigma @ U + U.T @ Sigma
    return sp.linalg.solve_continuous_lyapunov(Sigma, rhs)


def one_step_approx(Sigma_0, Sigma_1, U0):
    U1 = vector_translation(Sigma_0, Sigma_1, U0)
    return project_on_tangent(U1, Sigma_1)


def _as_generator(r: Union[np.float64, Generator]):
    if isinstance(r, Generator):
        return r
    elif isinstance(r, float):

        def rgen():
            while True:
                yield r

        return rgen()
    else:
        raise RuntimeError(f"Type of relaxation/regularization ({r}) not supported")


def _get_cvxpy_problem(dim: int, mk: int, Gamma_bound: float) -> cp.Problem:
    gamma = cp.Variable((mk,), name="gamma")
    R = cp.Parameter((min(mk, dim**2), mk), name="R")
    r = cp.Parameter((min(mk, dim**2),), name="r")
    Gamma_bound = cp.Constant(Gamma_bound, name="Gamma_bound")

    resid = r - cp.matmul(R, gamma)
    obj = cp.Minimize(cp.sum_squares(resid))
    constrs = [cp.norm_inf(gamma) <= Gamma_bound]
    prob = cp.Problem(obj, constrs)
    return prob


# Naming tentative; Riemannian -> Wasserstein? Ottonian?
# TODO: need to pass a bunch of (*args, **kwargs) to both PT and Sinkhorn
class BWRAMSolver:
    """Approximate the fixed-point :math:`\\rho^*`  for an operator :math:`F` over the Bures-Wasserstein space of probability measures, i.e.

        .. math::

            F: \\mathcal N_{0,d}(\\mathbb{R}^d) \\to \\mathcal N_{0,d}(\\mathbb{R}^d)

            \\rho^*: \\rho^* = F(\\rho^*)

        with Riemannian(-like) Anderson Mixing scheme.

    Args:
        operator (Callable): Operator :math:`F`, fixed point of which is in question
        relaxation (Union[np.float64, Generator]): relaxation parameter, used at each iteration; constant of function of the step
        history_len (int): maximal number of previous iterates, used in the method
        vector_transport_kind (Literal["parallel","translation","one_step" ]): solver for intermediate vector transport subproblem

    Attributes:
        dim (int): problem dimension
        n_particles (int): number of particles in the sample approximating the current measure
    """

    def __init__(
        self,
        operator: Callable,
        relaxation: Union[np.float64, Generator] = 0.95,
        l_inf_bound_Gamma: float = 0.1,
        history_len: int = 2,
        vt_kind: Literal["parallel", "translation", "one-step", "trivial"] = "one-step",
        r_threshold=None,
        restart_every=None,
        **kwargs,
    ):
        self._operator = operator
        self._k = 0
        self._m = history_len
        self._r_threshold = r_threshold
        self._k_restart = restart_every

        self._relaxation = _as_generator(relaxation)
        if l_inf_bound_Gamma < 0.0:
            raise ValueError(
                f"l_inf_bound_Gamma must be positive, got {l_inf_bound_Gamma}"
            )
        self._l_inf_bound_Gamma = l_inf_bound_Gamma

        self.W2_residual = []
        if vt_kind == "parallel":
            self._vt = parallel_transport
        elif vt_kind == "translation":
            self._vt = vector_translation
        elif vt_kind == "one-step":
            self._vt = one_step_approx
        elif vt_kind == "trivial":
            self._vt = vt_trival
        self._k = 0
        self.norm_Gamma = []
        self.norm_rk = []

    def _operator_and_residual(self, x_cur: np.ndarray):
        if hasattr(self._operator, "residual"):
            r = self._operator.residual(x_cur)
            x1 = rExpGaussian(r, x_cur)
        else:
            x0 = x_cur
            x1 = self._operator(x_cur)
            dim = x0.shape[0]

            T = ot.gaussian.bures_wasserstein_mapping(
                np.zeros(dim), np.zeros(dim), x0, x1
            )[0]
            r = T - np.eye(dim)

        return x_cur, r

    def _initialize_iteration(self, x0: np.ndarray):
        check_cov(x0)
        x1, r0 = self._operator_and_residual(x0)

        self.dim = x0.shape[0]
        self._x_prev = x0.copy()  # x_k-1
        self._x_cur = x1.copy()  # x_k
        self._r_prev = r0.copy()  # r_k-1

        self._delta_rs = []
        self._delta_xs = [r0.copy()]
        self.norm_Gamma += [0.0]
        self.norm_rk += [np.sqrt(np.trace(r0 @ self._x_prev @ r0))]
        self._k = 1

    def _check_restart(self, rk):
        norm_rk_cur = np.trace(rk @ self._x_cur @ rk) ** 0.5
        if self._k == self._k_restart:
            return True
        mk = min(self._m, self._k)
        if self._r_threshold is not None and mk > 1:
            if norm_rk_cur > self.norm_rk[-1] * self._r_threshold:
                return True

        return False

    def _step(
        self,
    ):
        _, rk = self._operator_and_residual(self._x_cur)
        norm_rk_cur = np.trace(rk @ self._x_cur @ rk) ** 0.5

        if self._check_restart(rk):
            self.restart()
            return self._x_cur

        self.norm_rk.append(norm_rk_cur)
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

        R = np.stack(self._delta_rs, axis=-1)
        X = np.stack(self._delta_xs, axis=-1)

        mk = min(self._m, self._k)
        Sigma_sq = sp.linalg.sqrtm(self._x_cur)
        R_ = (Sigma_sq @ R).reshape(-1, mk)
        r_ = (Sigma_sq @ rk).reshape(-1)
        Q, R_ = np.linalg.qr(R_)
        r_ = Q.T @ r_

        if self._k <= self._m:
            self._lsq_prob = _get_cvxpy_problem(self.dim, mk, self._l_inf_bound_Gamma)

        self._lsq_prob.param_dict["R"].value = R_
        self._lsq_prob.param_dict["r"].value = r_

        self._lsq_prob.solve()
        Gamma = self._lsq_prob.var_dict["gamma"].value
        Gamma = np.atleast_1d(Gamma)
        self.norm_Gamma.append(np.linalg.norm(Gamma, ord=np.inf))

        rk_bar = rk - R @ Gamma
        delta_x_cur = -X @ Gamma + next(self._relaxation) * rk_bar

        self._delta_xs.insert(0, delta_x_cur)

        self._x_prev = self._x_cur.copy()
        self._x_cur = rExpGaussian(delta_x_cur, self._x_cur)
        assert all(np.linalg.eigvalsh(self._x_cur)) > 0.0
        self._r_prev = rk
        self._k += 1
        return self._x_cur

    def iterate(self, x0: torch.Tensor, max_iter: int, residual_conv_tol: np.float64):
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
        self._initialize_iteration(self._x_cur.copy())
