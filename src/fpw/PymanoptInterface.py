import numpy as np
import scipy as sp
import ot
import torch
torch.set_default_dtype(torch.double)

import warnings
warnings.filterwarnings('error')

import pymanopt


def to_dSigma(U: np.ndarray, Cov: np.ndarray):
    return U @ Cov + Cov @ U


def to_Map(V: np.ndarray, Cov: np.ndarray):
    L_CV = sp.linalg.solve_continuous_lyapunov(Cov, V)


class BuresWassersteinManifold(pymanopt.manifolds.manifold.RiemannianSubmanifold):
    def __init__(self, dimension, pt_type="parallel"):
        name = f"Gaussian distributions in {dimension=} (represented with their covariance matrices)"
        if pt_type == "parallel":
            self.transport = self._parallel_transport
        elif pt_type == "translation":
            self.transport = self._vector_translation
        elif pt_type == "one-step":
            self.transport = self._one_step_approx
        else:
            raise NotImplementedError(f"Parallel transport {pt_type} not implemented")
        super().__init__(name, dimension)

    def inner_product(
        self,
        point: np.ndarray,
        tangent_vector_a: np.ndarray,
        tangent_vector_b: np.ndarray,
    ) -> float:
        return np.trace(tangent_vector_a.T @ point @ tangent_vector_b)

    def projection(self, Sigma, U):
        rhs = Sigma @ U + U.T @ Sigma
        try:
            return sp.linalg.solve_continuous_lyapunov(Sigma, rhs)
        except (ValueError, RuntimeWarning):
            print(Sigma)
            print(U)
            print(rhs)
            raise

    def norm(self, point, tangent_vector):
        nrm = np.trace(tangent_vector.T @ point @ tangent_vector) ** 0.5
        return nrm

    def random_point(self):
        A = np.random.randn(self.dim, self.dim)
        lam_min = np.abs(np.random.randn()) + 1e-3

        return A.T @ A + lam_min * np.eye(self.dim)

    def random_tangent_vector(self, point):
        A = np.random.randn(self.dim, self.dim)
        return A.T + A

    def zero_vector(self, point):
        return np.zeros((self.dim, self.dim))

    def dist(Cov_0, Cov_1):
        zero = np.zeros(Cov_0.shape[0])
        return ot.gaussian.bures_wasserstein_distance(zero, zero, Cov_0, Cov_1)

    def exp(self, Cov, V):
        Id_plus_V = np.eye(V.shape[0]) + V

        return Id_plus_V.T @ Cov @ Id_plus_V

    def retraction(self, Cov, V):
        return self.exp(Cov, V)

    def log(self, Sigma_0, Sigma_1):
        T = ot.gaussian.bures_wasserstein_mapping(
            np.zeros(dim), np.zeros(dim), Sigma_0, Sigma_1
        )[0]
        Tdir = T - Id

        return Tdir

    def _parallel_transport(self, Sigma_0, Sigma_1, U0):
        dim = self.dim
        U0 = to_dSigma(U0, Sigma_0)

        Id = np.eye(dim)
        T = ot.gaussian.bures_wasserstein_mapping(
            np.zeros(dim), np.zeros(dim), Sigma_0, Sigma_1
        )[0]
        Tdir = T - Id

        def Sigma_dSigma_dt(t):
            T_interp = Id + t * Tdir
            Sigma_t = T_interp @ Sigma_0 @ T_interp
            dSigma_dt = (
                Tdir @ Sigma_0 + Sigma_0 @ Tdir + 2.0 * t * Tdir @ Sigma_0 @ Tdir
            )

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
                "Parallel transport did not converge; Solver status: "
                + ode_result.message
            )

        U1 = ode_result.y[:, -1].reshape((dim, dim))

        U1 = to_Map(U1, Sigma_1)

        return U1

    def _vector_translation(self, Sigma_0, Sigma_1, U0):
        dim = self.dim

        Id = np.eye(dim)
        Tinv = ot.gaussian.bures_wasserstein_mapping(
            np.zeros(dim), np.zeros(dim), Sigma_1, Sigma_0
        )[0]
        U1 = U0 @ Tinv

        return U1

    def _one_step_approx(self, Sigma_0, Sigma_1, U0):
        U1 = self._vector_translation(Sigma_0, Sigma_1, U0)
        return self.projection(Sigma_1, U1)
