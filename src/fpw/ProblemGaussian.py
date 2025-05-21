from typing import Callable, Union, List, Dict, Generator, Literal, Tuple
import abc
from docstring_inheritance import GoogleDocstringInheritanceInitMeta

from functools import partial

from time import perf_counter

import numpy as np
import scipy as sp

import torch

torch.set_default_dtype(torch.float64)

import ot
from .PymanoptInterface import *

from num2tex import num2tex


def _sqrtm_torch(M):
    """
    Compute the matrix square root of a positive semi-definite matrix using eigenvalue decomposition.

    Args:
        M: (d, d) tensor, positive semi-definite matrix

    Returns:
        M^{1/2} matrix square root
    """
    # Eigenvalue decomposition
    L, Q = torch.linalg.eigh(M)  # M = Q L Q^T
    # Ensure positive eigenvalues for numerical stability
    # L = torch.clamp(L, min=1e-8)
    # Compute square root of eigenvalues
    sqrt_L = torch.diag_embed(torch.sqrt(L))
    # Reconstruct matrix square root
    return Q @ sqrt_L @ Q.mT


def barycenter_loss_vectorized(Sigmas, weights, Sigma):
    """
    Vectorized implementation of the BW barycenter loss with proper matrix square roots.
    """
    # Normalize weights
    weights = weights / weights.sum()

    # Compute traces of input matrices
    traces_Sigmas = torch.einsum("ijj->i", Sigmas)  # shape (N,)
    trace_Sigma = torch.trace(Sigma)

    # Compute the cross terms
    sqrt_Sigma = _sqrtm_torch(Sigma)
    cross_terms = sqrt_Sigma @ Sigmas @ sqrt_Sigma  # shape (N, d, d)

    # Compute matrix square roots of cross terms
    sqrt_cross_terms = torch.stack(
        [_sqrtm_torch(cross_term) for cross_term in cross_terms]
    )
    trace_cross_terms = torch.einsum("njj->n", sqrt_cross_terms)  # shape (N,)

    # Compute all distances
    distances = trace_Sigma + traces_Sigmas - 2 * trace_cross_terms

    # Weighted sum
    return torch.sum(weights * distances)


def entropic_barycenter_loss_vectorized(Sigmas, weights, gamma, Sigma):
    entr = 0.5 * (
        torch.trace(Sigma) - Sigma.shape[0] - torch.linalg.slogdet(Sigma).logabsdet
    )
    return barycenter_loss_vectorized(Sigmas, weights, Sigma) + gamma * entr


def median_loss_vectorized(Sigmas, weights, eps, scaling, Sigma):
    """
    Vectorized implementation of the BW barycenter loss with proper matrix square roots.
    """
    # Normalize weights
    weights = weights / weights.sum()

    # Compute traces of input matrices
    traces_Sigmas = torch.einsum("ijj->i", Sigmas)  # shape (N,)
    trace_Sigma = torch.trace(Sigma)

    # Compute the cross terms
    sqrt_Sigma = _sqrtm_torch(Sigma)
    cross_terms = sqrt_Sigma @ Sigmas @ sqrt_Sigma  # shape (N, d, d)

    # Compute matrix square roots of cross terms
    sqrt_cross_terms = torch.stack(
        [_sqrtm_torch(cross_term) for cross_term in cross_terms]
    )
    trace_cross_terms = torch.einsum("njj->n", sqrt_cross_terms)  # shape (N,)

    # Compute all distances
    distances_sq = trace_Sigma + traces_Sigmas - 2 * trace_cross_terms

    distances = (eps**2 + distances_sq).sqrt()

    # Weighted sum
    return torch.sum(weights * distances) * scaling


class _Meta(abc.ABC, GoogleDocstringInheritanceInitMeta):
    pass


class Problem(metaclass=_Meta):
    """Deals with fixed-point problems in the Bures-Wasserstein space: find the fixed-point :math:`\\Sigma_*`  for an operator :math:`G` over the Bures-Wasserstein space :math:`\\mathcal{N}_0^d`.

    Consider the space of positive-definite :math:`d\\times d` matrices

    .. math::

        \\mathcal{N}_0^d = \\{\\mathcal{N}(0, \\Sigma), 0 \\prec \\Sigma \\in \\mathbb{R}^{d\\times d}\\}

    endowed with the Wasserstein distance

    .. math::

        W^2_2(\\Sigma_0, \\Sigma_1)  = \\Tr{\\Sigma_0} + \\Tr{\\Sigma_1} - 2\\Tr{\\left(\\Sigma_0^{\\frac{1}{2}}\\Sigma_1 \\Sigma_0^{\\frac{1}{2}}\\right)^{\\frac{1}{2}}}

    Then consider an operator :math:`G: \\mathcal{N}_0^d \\to \\mathcal{N}_0^d` and the following fixed-point for it: find :math:`\\Sigma_*` such that

    .. math::

        \\Sigma_* = G(\\Sigma_*)

    If there is a smooth functional :math:`f: \\mathcal{N}_0^d \\to \\mathbb{R}`, then the problem of finding the critical point of this functional is equaivalent to a fixed-point problem

    .. math::

        \\partial_W f(\\Sigma_*) = 0 \\Longleftrightarrow Exp_{\\Sigma_*}(\\partial_W f(\\Sigma_*)) = \\Sigma_*

    This class wraps the operator :math:`G` and, in case of equivalence to a functional minimization problem, also :math:`f, \\nabla f`
    """

    def __init__(self, dim=None, vt_kind="one-step", **kwargs):
        self._dim = dim
        # TODO: rename pt_type everywhere!
        self.base_manifold = BuresWassersteinManifold(dim, pt_type=vt_kind)

    @property
    def dim(self):
        return self._dim

    @property
    def name(self):
        return self._name

    def __call__(self, Sigma: np.ndarray) -> np.ndarray:
        """Implements the operator :math:`G`, for which we seek the fixed-point

        Args:
           Sigma (np.ndarray): a covariance matrix :math:`\\Sigma \\in \\mathcal{N}_0^d`

        Returns:
            np.ndarray: Value of :math:`G(\\Sigma)`

        """
        return self.base_manifold.retraction(Sigma, self.residual(Sigma))

    def residual(self, Sigma: np.ndarray) -> np.ndarray:
        """Riemannian logarithm of :math:`G(\\Sigma)`.

        Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

        .. math::

            r: \\mathcal{N}_0^d \\ni \\Sigma \\mapsto Tan_{\\Sigma}(\\mathcal{N}_0^d)

            G(\\Sigma) = Exp_{\\Sigma}(-r(\\Sigma))


        Args:
           Sigma (np.ndarray): a covariance matrix :math:`\\Sigma \\in \\mathcal{N}_0^d`

        Returns:
            np.ndarray: a residual. It is an element of the tangent space at :math:`\\Sigma_*`, which is isomorphic to all symmetric matrices in :math:`\\mathbb{R}^{d\\times d}`

        """
        return self.operator_and_residual(Sigma)[1]

    def operator_and_residual(self, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zero = np.zeros(self.dim)
        Id = np.eye(self.dim)
        Sigma_next = self(Sigma)
        T = ot.gaussian.bures_wasserstein_mapping(zero, zero, Sigma, Sigma_next)[0]
        return Sigma_next, T - Id

    # TODO: why is this abstract?
    @abc.abstractmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not (
            cls.residual is not Problem.residual
            or cls.__call__ is not Problem.__call__
            or cls.operator_and_residual is not Problem.operator_and_residual
        ):
            raise TypeError(
                f"{cls.__name__} must overload either `residual` or `__call__`"
            )

    def cost(self, Sigma: np.ndarray) -> float:
        """Cost for the operator, defined by a minimization problem

        Args:
           Sigma (np.ndarray): a covariance matrix :math:`\\Sigma \\in \\mathcal{N}_0^d`

        Returns:
            np.float64: value of the cost function
        """
        pass

    def get_initial_value(self) -> np.ndarray:
        return np.eye(self.dim)

    # TODO: Picard method
    def get_solution_picard(
        self,
        N_steps_max: int = 10_000,
        r_min: np.float64 = 1e-10,
        cost_min: np.float64 = -np.inf,
        **kwargs,
    ):
        kw_reference = locals()
        kw_reference.pop("self")
        kw_reference = kw_reference.pop("kwargs") | kw_reference

        residuals = []
        covs = []
        dts = []

        if self.has_cost:
            costs = []

        cov_init = self.get_initial_value()
        cov_cur = cov_init

        for k in range(N_steps_max):
            cov_prev = cov_cur.copy()

            try:
                t = perf_counter()
                cov_cur, residual = self.operator_and_residual(cov_cur)
                dt = perf_counter() - t
            except RuntimeError:
                break

            # covs.append(cov_prev)
            residuals.append(self.base_manifold.norm(cov_prev, residual))
            dts.append(dt)
            if self.has_cost:
                costs.append(self.cost(cov_prev))
                if costs[-1] <= cost_min:
                    break
            if residuals[-1] <= r_min:
                break

        results = dict()
        self.covs_ref = covs
        if not hasattr(self, "target") and residuals[-1] < r_min:
            self.target = cov_cur
        if self.has_cost:
            self.costs_ref = costs
            results["costs"] = np.array(costs)

        results["residuals"] = np.array(residuals)
        results["dts"] = np.array(dts)
        results["covs"] = covs
        results["cov_final"] = cov_cur
        results = results | kw_reference

        return results


def _get_target(
    dim=2,
    sigma_min=0.5,
    sigma_max=5.0,
    rs=10,
    **kwargs,
):
    target_data = dict(
        dim=dim,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rs=rs,
    )
    ortho = sp.stats.ortho_group.rvs(dim=dim, random_state=rs)
    sigmas = np.diag(np.linspace(sigma_min, sigma_max, dim, endpoint=True))
    cov_target = ortho.T @ sigmas @ ortho

    return cov_target, target_data


def _pprint_target(dim, sigma_min, sigma_max, rs):
    return f"d = {dim},\\ \\sigma_{{min}} = {num2tex(sigma_min)},\\ \\sigma_{{max}} = {num2tex(sigma_max)},\\ r = {rs:d}"


class OUEvolution(Problem):
    # TODO: documentation
    """Operator describing the dynamic of the Ornstein-Uhlenbeck process with invariant distribution :math:`\\Sigma_*`

    Args:
        target (np.ndarray): The invariant distribution
        dt (np.float64): The time step
        **kwargs: args to randomly generate the target
    """

    def __init__(self, target: np.ndarray = None, dt=0.1, **kwargs):
        if target is None:
            target, target_data = _get_target(**kwargs)
        else:
            kwargs["dim"] = target.shape[0]

        super().__init__(**kwargs)
        # TODO: this should be a class method, not instance!
        self.has_cost = False
        self.target_data = target_data

        # self._name = f"OU, {target_data}"
        self._name = f"$OU,\\ {_pprint_target(**target_data)}$"
        self._dt = dt
        self.target = target.copy()
        self.Sigma_inv = np.linalg.inv(target)
        self.tmp1 = sp.linalg.expm(-dt * self.Sigma_inv)
        self.tmp2 = sp.linalg.sqrtm(
            np.eye(self.target.shape[0]) - sp.linalg.expm(-2.0 * dt * self.Sigma_inv)
        )

    @property
    def dt(self):
        return self._dt

    @property
    def name(self):
        return self._name

    def __call__(self, Sigma):
        return self.tmp1 @ Sigma @ self.tmp1 + self.tmp2 @ self.target @ self.tmp2


class WGKL(Problem):
    """Operator describing the minimization of :math:`\\operatorname{KL}(\\cdot|\\Sigma_*)` for target distribution :math:`\\Sigma_*`

    Args:
        target (np.ndarray): The invariant distribution
        scaling (np.float64): Scale the Wasserstein gradient of :math:`\\operatorname{KL}(\\cdot|\\Sigma_*)` by this value
        **kwargs: args to randomly generate the target
    """

    def __init__(self, target: np.ndarray = None, scaling=1.0, **kwargs):
        if target is None:
            target, target_data = _get_target(**kwargs)
        else:
            kwargs["dim"] = target.shape[0]

        super().__init__(**kwargs)
        self.has_cost = True
        self.target_data = target_data
        self.scaling = scaling

        self.Sigma_target = target.copy()
        self.Sigma_target_inv = np.linalg.inv(self.Sigma_target)
        self._name = f"$\\lambda\\partial_W \\operatorname{{KL}},\\ {_pprint_target(**target_data)}$"

    def residual(self, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)

        return -self.scaling * (self.Sigma_target_inv - Sigma_inv)

    # TODO: same implementation for torch and numpy
    def cost(self, Sigma: np.ndarray):
        Sigma_mul = self.Sigma_target_inv @ Sigma
        ld = np.linalg.slogdet(Sigma_mul).logabsdet
        dim = Sigma_mul.shape[-1]
        return 0.5 * self.scaling * (np.trace(Sigma_mul) - ld - dim)

    def get_cost_torch(
        self,
    ):
        def KL(
            Sigma_target_inv: torch.Tensor, scaling: torch.float64, Sigma: torch.Tensor
        ):
            Sigma_mul = torch.matmul(Sigma_target_inv, Sigma)
            ld = torch.linalg.slogdet(Sigma_mul).logabsdet
            dim = Sigma_mul.shape[-1]
            return 0.5 * scaling * (torch.trace(Sigma_mul) - ld - dim)

        return partial(
            KL,
            torch.Tensor(self.Sigma_target_inv),
            self.scaling,
        )


class Barycenter(Problem):
    """Operator describing the Wasserstein barycenter problem

    .. math::

        \\Sigma_* = \\arg\\min_{\\Sigma \\in \\mathcal{N}_0^d} \\sum_{k=1}^{n_\\sigma} w_k W^2_2(\\Sigma, \\Sigma_k)


    Args:
        n_sigmas (int) : Number :math:`n_\\sigma` of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution :math:`\\Sigma \\sim \\mathcal{W}(\\operatorname{I}_d, d)`
        dim (int) : dimension :math:`d` of the distributions
        weights (np.ndarray) : weight vector :math:`w_i > 0`. If not given, set to :math:`w_1 = w_2 = \\dots = w_{n_\\sigma}`
        rs (int) : random seed for the generation of the distribution
        **kwargs: args to randomly generate the target
    """

    def __init__(
        self,
        n_sigmas: int = None,
        dim: int = None,
        weights: np.ndarray = None,
        rs: int = 1,
        **kwargs,
    ):
        super().__init__(dim=dim)
        self.has_cost = True

        if n_sigmas is None:
            n_sigmas = len(weights) if weights is not None else None
        assert n_sigmas > 1

        if weights is None:
            weights = np.ones(n_sigmas)
        else:
            assert all(weights > 0.0)
            weights = np.array(weights)

        self._weights = weights.copy()
        self._weights /= np.sum(self._weights)

        self._sigmas = sp.stats.wishart(df=dim, scale=np.eye(dim), seed=rs).rvs(
            n_sigmas
        )

        # we need this for PyManOpt Riemannian minimization
        self._sigmas_torch = torch.Tensor(self._sigmas)
        self._weights_torch = torch.Tensor(self._weights)

        self._name = f"Barycenter, {dim=}, k={n_sigmas}"

    def _S(self, sqrt_Sigma):
        S = np.sum(
            [
                _lam * sp.linalg.sqrtm(sqrt_Sigma @ _Sig @ sqrt_Sigma)
                for _lam, _Sig in zip(self._weights, self._sigmas)
            ],
            axis=0,
        )
        return S

    def residual(self, Sigma):
        eigvals, eigvecs = sp.linalg.eigh(Sigma)
        sqrt_Sigma = eigvecs @ np.diag(eigvals**0.5) @ eigvecs.T
        sqrt_inv_Sigma = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T

        S = self._S(sqrt_Sigma)

        return -(np.eye(self.dim) - sqrt_inv_Sigma @ S @ sqrt_inv_Sigma)

    def __call__(self, Sigma):
        eigvals, eigvecs = sp.linalg.eigh(Sigma)
        assert all(eigvals > 0.0)
        sqrt_Sigma = eigvecs @ np.diag(eigvals**0.5) @ eigvecs.T
        sqrt_inv_Sigma = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T

        S = self._S(sqrt_Sigma)

        return sqrt_inv_Sigma @ S @ S @ sqrt_inv_Sigma

    def operator_and_residual(self, Sigma):
        eigvals, eigvecs = sp.linalg.eigh(Sigma)
        assert all(eigvals > 0.0)
        sqrt_Sigma = eigvecs @ np.diag(eigvals**0.5) @ eigvecs.T
        sqrt_inv_Sigma = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T

        S = self._S(sqrt_Sigma)

        return sqrt_inv_Sigma @ S @ S @ sqrt_inv_Sigma, -(
            np.eye(self.dim) - sqrt_inv_Sigma @ S @ sqrt_inv_Sigma
        )

    def cost(self, Sigma):
        zero = np.zeros(self.dim)
        dists = np.array(
            [
                ot.gaussian.bures_wasserstein_distance(zero, zero, Sigma, _S) ** 2
                for _S in self._sigmas
            ]
        )
        return np.dot(self._weights, dists)

    def get_cost_torch(self):

        return partial(
            barycenter_loss_vectorized, self._sigmas_torch, self._weights_torch
        )

    @property
    def name(self):
        return self._name

    @property
    def n_sigmas(self):
        return self._n_sigmas

    def get_initial_value(self):
        return self._sigmas[0]


class EntropicBarycenter(Barycenter):
    """Operator describing the entropy-regularized Wasserstein barycenter problem

    .. math::

        \\Sigma_* = \\arg\\min_{\\Sigma \\in \\mathcal{N}_0^d} \\sum_{k=1}^{n_\\sigma} w_k W_2(\\Sigma, \\Sigma_k) + \\gamma\\operatorname{KL}(\\Sigma|\\I_d)


    Args:
        n_sigmas (int) : Number :math:`n_\\sigma` of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution :math:`\\Sigma \\sim \\mathcal{W}(I_d, d)`
        dim (int) : dimension :math:`d` of the distributions
        weights (np.ndarray) : weight vector :math:`w_i > 0`. If not given, set to :math:`w_1 = w_2 = \\dots = w_{n_\\sigma}`
        gamma (np.float64): the regularization parameter :math:`\\gamma`
        rs (int) : random seed for the generation of the distribution
        **kwargs: args to randomly generate the target
    """

    def __init__(
        self,
        n_sigmas: int = None,
        dim: int = None,
        weights: np.ndarray = None,
        rs: int = 1,
        gamma: np.float64 = 1e-2,
        **kwargs,
    ):
        super().__init__(n_sigmas=n_sigmas, dim=dim, weights=weights, rs=rs)
        self.has_cost = True

        assert gamma > 0.0
        self._gamma = gamma
        self._name = f"Entropic Barycenter, {dim=}, k={n_sigmas}, $\\gamma = {num2tex(self._gamma)}$"

    def residual(self, Sigma):
        eigvals, eigvecs = sp.linalg.eigh(Sigma)
        sqrt_Sigma = eigvecs @ np.diag(eigvals**0.5) @ eigvecs.T
        sqrt_inv_Sigma = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T
        inv_Sigma = eigvecs @ np.diag(eigvals ** (-1.0)) @ eigvecs.T

        S = self._S(sqrt_Sigma)

        return (
            sqrt_inv_Sigma @ S @ sqrt_inv_Sigma
            + self._gamma * inv_Sigma
            - (1.0 + self._gamma) * np.eye(self.dim)
        )

    def cost(self, Sigma):
        zero = np.zeros(self.dim)
        dists = np.array(
            [
                (ot.gaussian.bures_wasserstein_distance(zero, zero, Sigma, _S) ** 2)
                for _S in self._sigmas
            ]
        )

        entr = 0.5 * (
            np.trace(Sigma) - Sigma.shape[0] - np.linalg.slogdet(Sigma).logabsdet
        )
        return np.dot(self._weights, dists) + self._gamma * entr

    def get_cost_torch(self):
        return partial(
            entropic_barycenter_loss_vectorized,
            self._sigmas_torch,
            self._weights_torch,
            self._gamma,
        )

    @property
    def name(self):
        return self._name

    @property
    def n_sigmas(self):
        return self._n_sigmas

    def get_initial_value(self):
        return self._sigmas[0]


class Median(Problem):
    """Operator describing the Wasserstein geometric median problem

    .. math::

        \\Sigma_* = \\arg\\min_{\\Sigma \\in \\mathcal{N}_0^d} \\sum_{k=1}^{n_\\sigma} w_k W_2(\\Sigma, \\Sigma_k)

    For the purpose of regularizing the problem, we consider the smoothed version of the problem

    .. math::

        \\Sigma_* = \\arg\\min_{\\Sigma \\in \\mathcal{N}_0^d} \\sum_{k=1}^{n_\\sigma} w_k \\sqrt{W^2_2 + \\varepsilon}(\\Sigma, \\Sigma_k)

    Args:
        n_sigmas (int) : Number :math:`n_\\sigma` of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution :math:`\\Sigma \\sim \\mathcal{W}(\\operatorname{I}_d, d)`
        dim (int) : dimension :math:`d` of the distributions
        weights (np.ndarray) : weight vector :math:`w_i > 0`. If not given, set to :math:`w_1 = w_2 = \\dots = w_{n_\\sigma}`
        rs (int) : random seed for the generation of the distribution
        eps (np.float64): the smoothing parameter :math:`\\varepsilon`
        scaling (np.float64): the stepsize parameter s
        **kwargs: args to randomly generate the target
    """

    def __init__(
        self,
        n_sigmas: int = None,
        dim: int = None,
        weights: np.ndarray = None,
        rs: int = 1,
        eps: np.float64 = 1e-2,
        scaling: np.float64 = 1.0,
        **kwargs,
    ):
        super().__init__(dim=dim)
        self.has_cost = True

        if n_sigmas is None:
            n_sigmas = len(weights) if weights is not None else None
        assert n_sigmas > 1

        if weights is None:
            weights = np.ones(n_sigmas)
        else:
            assert all(weights > 0.0)
            weights = np.array(weights)

        assert eps > 0.0
        self._eps = eps
        assert scaling > 0.0
        self._scaling = scaling

        self._weights = weights.copy()
        self._weights /= np.sum(self._weights)

        self._sigmas = sp.stats.wishart(df=dim, scale=np.eye(dim), seed=rs).rvs(
            n_sigmas
        )

        # we need this for PyManOpt Riemannian minimization
        self._sigmas_torch = torch.Tensor(self._sigmas)
        self._weights_torch = torch.Tensor(self._weights)

        self._name = (
            f"Median, {dim=}, k={n_sigmas}, $\\varepsilon = {num2tex(self._eps)}$"
        )

    def residual(self, Sigma):
        zero = np.zeros(self.dim)
        Id = np.eye(self.dim)
        transports = np.array(
            [
                ot.gaussian.bures_wasserstein_mapping(zero, zero, Sigma, _S)[0]
                for _S in self._sigmas
            ]
        )

        V = transports - Id[np.newaxis, :, :]

        dists_sq = np.einsum("ijk,km,imj->i", V, Sigma, V)
        W2_eps = (self._eps**2 + dists_sq) ** 0.5

        V *= (
            self._weights[:, np.newaxis, np.newaxis] / W2_eps[:, np.newaxis, np.newaxis]
        )

        V_avg = V.sum(axis=0)
        return self._scaling * V_avg

    def cost(self, Sigma):
        zero = np.zeros(self.dim)
        dists = np.array(
            [
                (
                    ot.gaussian.bures_wasserstein_distance(zero, zero, Sigma, _S) ** 2
                    + self._eps**2
                )
                ** 0.5
                for _S in self._sigmas
            ]
        )
        return self._scaling * np.dot(self._weights, dists)

    def get_cost_torch(self):
        return partial(
            median_loss_vectorized,
            self._sigmas_torch,
            self._weights_torch,
            self._eps,
            self._scaling,
        )

    @property
    def name(self):
        return self._name

    @property
    def n_sigmas(self):
        return self._n_sigmas

    def get_initial_value(self):
        return self._sigmas[0] * 1.001
