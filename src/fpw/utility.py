import abc
from docstring_inheritance import GoogleDocstringInheritanceInitMeta
from typing import Union, Tuple
from types import MethodType

import warnings


import numpy as np
import scipy as sp
import torch

import os
import sys
import pickle



class _Meta(abc.ABC, GoogleDocstringInheritanceInitMeta):
    pass


class Distribution(metaclass=_Meta):
    """Convenience class for storing distributions and plotting.

    Args:

    Attributes :
        dim : dimension of the distribution

    .. automethod:: _sample
    """

    N_steps_mcmc_tune = 100
    N_ensemble_mcmc_tune = 20
    N_burn_in_mcmc = 1_000_000
    N_ensemble_mcmc = 100
    N_samples_mcmc = 500
    samples_dir = "./samples"

    def __init__(self, *args, **kwargs):
        self.n_samples_read = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a value, proportional to the pdf of the distribution.

        Args:
            x (torch.Tensor) : array of shape `(N_x, dim)` --- coordinates of N_x points

        Returns:
            torch.Tensor : array of shape `(N_x)` --- value of density (up to normalization constant) at these points
        """
        return torch.exp(self.log_prob(x))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a value, equal to the log pdf of the distribution (up to an additive constant).

        Args:
            x : array of shape `(N_x, dim)` --- coordinates of N_x points

        Returns:
            torch.Tensor : array of shape `(N_x)` --- value of log density (up to an additive constant) at these points
        """
        return torch.log(self.density(x))

    def np_log_prob(self, x: np.ndarray):
        """
        Returns a value, equal to the log pdf of the distribution (up to an additive constant).

        Args:
            x : array of shape `(N_x, dim)` --- coordinates of N_x points

        Returns:
            torch.Tensor : array of shape `(N_x)` --- value of log density (up to an additive constant) at these points
        """
        return self.log_prob(torch.from_numpy(x)).numpy()

    @abc.abstractmethod
    def __getstate__(
        self,
    ):
        if hasattr(self, "mcmc_cov") and hasattr(self, "mcmc_t_ac"):
            return self.mcmc_cov, self.mcmc_t_ac
        return

    @abc.abstractmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not (
            cls.density is not Distribution.density
            or cls.log_prob is not Distribution.log_prob
        ):
            raise TypeError(f"{cls.__name__} must overload either density or log_prob")

    @abc.abstractmethod
    def __setstate__(self, *args):
        if len(args) > 0:
            self.mcmc_cov = args[0]
        if len(args) > 1:
            self.mcmc_t_ac = args[1]

    @abc.abstractmethod
    def __eq__(
        self,
    ):
        pass

    def sample(self, N_samples: Union[int, Tuple[int]], **kwargs):
        """
        Returns the sample from the distribution

        Args:
            N_samples: if `int` --- number of samples; if `tuple(int)` --- the leading shape for the sample, i.e. the output will have shape (N_samples[0], ..., dim)

        Returns:
            torch.Tensor
        """
        if isinstance(N_samples, int):
            N_samples = (N_samples,)
            _N_samples = np.prod(N_samples)
            spl = self._sample(_N_samples, **kwargs)
            return torch.Tensor(spl.reshape(N_samples + (-1,)))

    def _sample(self, N_samples: int, run_mcmc=False) -> torch.Tensor:
        """
        Returns the sample from the distribution

        .. note::
            In child classes, this should implement the actual sampling algorithm. The ``sample`` wrapping method is implemented to have the same call signature as ``torch.distributions.Distribution.sample`` method

            For any ``Distribution``, for which ``log_prob`` is implemented, a basic Metropolis-Hastings MCMC is implemented.
            If one intends to use it, it is advised to override a separate ``np_log_prob`` method for efficiency.


        Args:
            N_samples: number of samples

        Returns:
            torch.Tensor : array of shape `(N_samples, dim)` --- number of samples
        """
        import emcee
        fname_h5, fname_pickle = self._get_sample_file_paths()

        load_ok = False
        if os.path.exists(fname_pickle) and os.path.exists(fname_h5):
            with open(fname_pickle, "rb") as ifile:
                other = pickle.load(ifile)
            load_ok = self == other
            self.mcmc_cov = other.mcmc_cov
            self.mcmc_t_ac = other.mcmc_t_ac

        if load_ok:
            backend = emcee.backends.HDFBackend(
                fname_h5,
            )
            chain = backend.get_chain(flat=True)
        else:
            if not run_mcmc:
                raise RuntimeError(
                    "MCMC sample files not found. To run MCMC sampling, call with run_mcmc=True (may take a lot of time)"
                )
            backend = self._prepare_mcmc_samples()
            chain = backend.get_chain(flat=True)

        if N_samples + self.n_samples_read > chain.shape[0]:
            # do addtitonal MCMC runs
            if self.np_log_prob is Distribution.np_log_prob:
                warnings.warn(
                    "For efficiency of MCMC sampling, {self.__class__.__name__} must override np_log_prob to compute the log density with numpy only (no pytorch)",
                    category=ResourceWarning,
                )

            print("Running additional MCMC")
            n_thin = int(np.ceil(self.mcmc_t_ac))
            N_steps = N_samples + self.n_samples_read - chain.shape[0]
            N_steps = int(np.ceil(N_steps / self.__class__.N_ensemble_mcmc))
            print(self.mcmc_cov, self.mcmc_t_ac)
            sampler = emcee.EnsembleSampler(
                self.__class__.N_ensemble_mcmc,
                self.dim,
                self.np_log_prob,
                moves=[(emcee.moves.GaussianMove(cov=self.mcmc_cov), 1.0)],
                backend=backend,
            )
            sampler.run_mcmc(
                None,
                N_steps,
                progress=True,
                store=True,
                thin_by=n_thin,
            )

            chain = sampler.get_chain().reshape(-1, self.dim)

        self.n_samples_read += N_samples
        return chain[self.n_samples_read - N_samples : self.n_samples_read]

    def _get_sample_file_paths(
        self,
    ):
        fname_h5 = self.name + ".h5"
        fname_pickle = self.name + ".pkl"

        fname_h5 = os.path.join(self.__class__.samples_dir, self.name, fname_h5)
        fname_pickle = os.path.join(self.__class__.samples_dir, self.name, fname_pickle)

        return fname_h5, fname_pickle

    def _prepare_mcmc_samples(
        self,
    ):
        if self.np_log_prob is Distribution.np_log_prob:
            warnings.warn(
                "For efficiency of MCMC sampling, {self.__class__.__name__} must override np_log_prob to compute the log density with numpy only (no pytorch)",
                category=ResourceWarning,
            )
        # tune COV in the MH-MCMC
        n_step_tune = self.__class__.N_steps_mcmc_tune
        n_ensemble_tune = self.__class__.N_ensemble_mcmc_tune

        ar_cur = [1.0]

        def _test_mcmc_ar(cov):
            if cov <= 0.0:
                return 1.0 - 0.23

            init = np.random.randn(self.__class__.N_ensemble_mcmc_tune, self.dim)
            sampler = emcee.EnsembleSampler(
                n_ensemble_tune,
                self.dim,
                self.np_log_prob,
                moves=[(emcee.moves.GaussianMove(cov=cov), 1.0)],
            )
            state = sampler.run_mcmc(init, n_step_tune, progress=False)
            ar = sampler.acceptance_fraction.mean()
            ar_cur[0] = ar

            return ar - 0.23  # rule of thumb optimal AR

        print(
            "Tuning stepsize parameter in MCMC",
            flush=True,
        )
        tune_res = sp.optimize.root_scalar(
            _test_mcmc_ar, bracket=(0.0, 10.0), method="bisect", xtol=0.005
        )
        cov = tune_res.root
        print(f"Stepsize {cov:.2e}; acceptance fraction {ar_cur[0]:.3f}", flush=True)
        self.mcmc_cov = cov

        # Run mcmc to estimate AC time and (hopefully) skip the burn-in phase

        print("Running burn-in phase", flush=True)
        fname_h5, fname_pickle = self._get_sample_file_paths()
        dirname = os.path.dirname(fname_h5)
        os.makedirs(dirname, exist_ok=True)

        init = np.random.randn(self.__class__.N_ensemble_mcmc, self._dim)
        sampler = emcee.EnsembleSampler(
            self.__class__.N_ensemble_mcmc,
            self.dim,
            self.np_log_prob,
            moves=[(emcee.moves.GaussianMove(cov=self.mcmc_cov), 1.0)],
        )
        state = sampler.run_mcmc(
            init,
            self.__class__.N_burn_in_mcmc,
            progress=True,
            store=True,
        )
        ac_times = sampler.get_autocorr_time()
        self.mcmc_t_ac = np.max(ac_times)
        print(f"Autocorrelation time {self.mcmc_t_ac:.2e}", flush=True)

        # Compute a lot of i.i.d. samples and save for the future

        n_thin = int(np.ceil(self.mcmc_t_ac))
        N_steps = int(
            np.ceil(self.__class__.N_samples_mcmc / self.__class__.N_ensemble_mcmc)
        )
        if os.path.exists(fname_h5):
            os.remove(fname_h5)
        sampler = emcee.EnsembleSampler(
            self.__class__.N_ensemble_mcmc,
            self.dim,
            self.np_log_prob,
            moves=[(emcee.moves.GaussianMove(cov=self.mcmc_cov), 1.0)],
            backend=emcee.backends.HDFBackend(fname_h5),
        )
        sampler.run_mcmc(state, N_steps, progress=True, store=True, thin_by=n_thin)

        self.n_samples_read = 0

        with open(fname_pickle, "wb") as ofile:
            pickle.dump(self, ofile)

        return sampler.backend


class Gaussian(Distribution):
    def __init__(
        self,
        mean: torch.Tensor = None,
        cov: torch.Tensor = None,
        dim: int = None,
        m_mag: float = 0.0,
        sigma_min: float = 1,
        sigma_max: float = 1.0,
        rs_params: int = 1,
        rs_sample: int = 1,
        name: str = None,
    ):
        super().__init__(self)
        if mean is not None:
            if cov is not None:
                if cov.shape[0] == mean.shape[0] and cov.shape[1] == mean.shape[0]:
                    self.__setstate__(mean, cov, rs_sample)
                else:
                    raise RuntimeError(
                        f"Dimension mismatch, {mean.shape=}, {cov.shape=}"
                    )
            else:
                self._dim = mean.shape[0]
        elif cov is not None:
            if not np.allclose(cov, cov.T):
                raise RuntimeError("Covariance must be a symmetric matrix")
            self._dim = cov.shape[0]
        elif dim is not None:
            self._dim = dim
        else:
            raise RuntimeError("Can not determine dimension")

        if mean is None:
            self.mean = sp.stats.norm(
                loc=0.0,
                scale=m_mag,
            ).rvs(self._dim, random_state=rs_params)
        else:
            self.mean = mean

        if cov is None:
            assert 0 < sigma_min and sigma_min <= sigma_max
            sigma = np.linspace(sigma_min, sigma_max, self.dim, endpoint=True)
            U = sp.stats.ortho_group(dim=self.dim, seed=rs_params).rvs()
            self.cov = U @ np.diag(sigma**2) @ U.T
            self.sigma = U @ np.diag(sigma) @ U.T
            self.psq = U @ np.diag(sigma ** (-1)) @ U.T
            self.precision = U @ np.diag(sigma ** (-2)) @ U.T
            self._rs_sample = rs_sample
            self._norm_gen = sp.stats.norm()

        self._name = f"normal_d{self._dim}"

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_m = x - self.mean[np.newaxis, :]
        return -0.5 * np.einsum("bi,ij,bj->b", x_m, self.precision, x_m)

    def density(self, x: torch.Tensor) -> torch.Tensor:
        return np.exp(self.log_prob(x))

    def score(self, x):
        return -(self.precision @ (x - self.mean[np.newaxis, :]).T).T

    def _sample(self, N_samples: Union[int, Tuple[int]]):
        x0 = self._norm_gen.rvs(
            size=(N_samples, self.dim), random_state=self._rs_sample
        )

        return (self.sigma @ x0.T).T + self.mean

    def __setstate__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
    ):
        self.m = m.copy()
        self.cov = cov.copy()
        self._rs_sample = rs_sample
        self._norm_gen = sp.stats.norm(random_state=rs_sample)
        self._dim = cov.shape[0]

        if mcmc_cov is not None and mcmc_t_ac is not None:
            self.mcmc_cov = mcmc_cov
            self.mcmc_t_ac = mcmc_t_ac

        svd_dec = np.linalg.svd(cov, hermitian=True, compute_uv=True)
        U, S, Vh = svd_dec.U, svd_dec.S, svd_dec.Vh
        if not np.all(S > 0.0):
            raise RuntimeError
        self.precision = U @ np.diag(S ** (-1)) @ Vh
        self.psq = U @ np.diag(S ** (-0.5)) @ Vh
        self.sigma = U @ np.diag(S ** (0.5)) @ Vh

    def __getstate__(
        self,
    ):
        if hasattr(self, "mcmc_cov") and hasattr(self, "mcmc_t_ac"):
            return self.m, self.cov, self._rs_sample, self.mcmc_cov, self.mcmc_t_ac
        return self.m, self.cov, self._rs_sample

    def __eq__(self, other):
        if not isinstance(other, Gaussian):
            return False
        m0, cov0, _ = self.__getstate__()
        m1, cov1, _ = other.__getstate__()

        if not (m0.shape == m1.shape and cov0.shape == cov1.shape):
            return False

        return np.allclose(m0, m1) and np.allclose(cov0, cov1)


class Nonconvex(Distribution):
    def __init__(
        self,
        a: torch.Tensor = None,
        dim: int = None,
        a_min: float = -1.0,
        a_max: float = 1.0,
        rs_params: int = 1,
    ):
        super().__init__(self)
        if a is None:
            if dim is None:
                raise RuntimeError("Cannot determine dimension")
            a = sp.stats.uniform(loc=a_min, scale=(a_max - a_min)).rvs(
                size=dim, random_state=rs_params
            )
        a = torch.Tensor(a)

        self.__setstate__((a,))

        self._a_np = a.numpy()
        self._name = f"nonconvex_d{self._dim}"

    def __getstate__(
        self,
    ):
        return (self._a,) + super().__getstate__()

    def __eq__(self, other):
        if not isinstance(other, Nonconvex):
            return False
        
        return torch.allclose(self._a, other._a)

    def __setstate__(self, args):
        a = args[0]
        a = a.detach().clone() if isinstance(a, torch.Tensor) else torch.Tensor(a) 
        self._dim = a.shape[0]
        self._a = a
        super().__setstate__(*args[1:])

    def log_prob(self, x: torch.Tensor):
        return -torch.linalg.norm(x - self._a[None, :], ord=0.5, axis=-1)

    def np_log_prob(self, x: np.ndarray):
        return -np.linalg.norm(x - self._a_np[np.newaxis, :], ord=0.5, axis=-1)


class DoubleMoon(Distribution):
    def __init__(
        self,
        a: float,
        dim: int,
    ):
        super().__init__(self)

        self.__setstate__((a, dim))
        self._name = f"doublemoon_d{self._dim}"

    def __getstate__(
        self,
    ):
        return (self._a, self.dim) + super().__getstate__()

    def __eq__(self, other):
        if not isinstance(other, DoubleMoon):
            return False
        return np.allclose(self._a, other._a) and self.dim == other.dim

    def __setstate__(self, args):
        self._a, self._dim = args[:2]
        super().__setstate__(*args[3:])

    def log_prob(self, x: torch.Tensor):
        nx = torch.linalg.norm(x, ord=2, axis=-1)
        x1 = x[..., 0]
        return (-2.0 * (nx - self._a) ** 2) + torch.log(
            torch.exp(-2.0 * (x1 - self._a) ** 2)
            + torch.exp(-2.0 * (x1 + self._a) ** 2)
        )

    def np_log_prob(self, x: np.ndarray):
        nx = np.linalg.norm(x, ord=2, axis=-1)
        x1 = x[..., 0]
        return (-2.0 * (nx - self._a) ** 2) + np.log(
            np.exp(-2.0 * (x1 - self._a) ** 2) + np.exp(-2.0 * (x1 + self._a) ** 2)
        )


class Operator(metaclass=_Meta):
    @property
    def distribution(self) -> str:
        return self._distribution

    @property
    def dim(self) -> int:
        return self._distribution.dim

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        target_distribution: Union[Distribution, torch.distributions.Distribution],
        *args,
        **kwargs,
    ):
        self._distribution = target_distribution

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor):
        pass


class ScoreBasedOperator(Operator):
    def __init__(
        self,
        target_distribution: Union[Distribution, torch.distributions.Distribution],
        *args,
        **kwargs,
    ):
        super().__init__(target_distribution)

        if not hasattr(self._distribution, "score"):

            def score(self, x: torch.Tensor):
                _x = x.detach().clone()
                _x.requires_grad = True
                ld = self.log_prob(_x)
                _score = torch.autograd.grad([ld], [_x], torch.ones(x.shape[0]))[0]
                return _score

            setattr(self._distribution, "score", MethodType(score, self._distribution))


class ULAStep(ScoreBasedOperator):
    def __init__(self, target_distribution, timestep: float):
        super().__init__(target_distribution)
        self._timestep = timestep

    @property
    def name(self):
        return f"ULAStep_target_dist={target_distribution.name}_dt={self._timestep}"

    def step(self, x: torch.Tensor, dt: float = None):
        dt = self._timestep if dt is None else dt
        noise = torch.randn_like(x)
        x_new = x + self._distribution.score(x) * dt + noise * (2.0 * dt) ** 0.5
        return x_new

    def __call__(self, x: torch.Tensor):
        return self.step(x, self._timestep)


class MALAStep(ScoreBasedOperator):
    def __init__(self, target_distribution, timestep: float):
        super().__init__(target_distribution)
        self._timestep = timestep

    @property
    def name(self):
        return f"MALAStep_target_dist={target_distribution.name}_dt={self._timestep}"

    def step(self, x: torch.Tensor, dt: float = None):
        dt = self._timestep if dt is None else dt
        noise = torch.randn_like(x)
        x_prop = x + self._distribution.score(x) * dt + noise * (2.0 * dt) ** 0.5

        d_log_prob = self._distribution.log_prob(x_prop) - self._distribution.log_prob(
            x
        )
        d_transition = (
            torch.linalg.norm(
                (x - x_prop - dt * self._distribution.score(x_prop)),
                axis=-1,
            )
            ** 2
            - 2.0 * dt * torch.linalg.norm(noise, axis=-1) ** 2
        )

        log_alpha = torch.minimum(
            torch.zeros_like(d_log_prob), d_log_prob - d_transition / (4.0 * dt)
        )
        u = torch.rand(x.shape[0])
        is_accepted = torch.log(u) <= log_alpha
        acceptance_rate = is_accepted.to(torch.float).mean()
        x_new = torch.where(is_accepted[:, None], x_prop, x)

        return x_new, acceptance_rate

    def tune(
        self, x_init: torch.Tensor, ar_target: float = 0.574, n_steps_tune: int = 50
    ):
        ar_cur = [
            1.0,
        ]

        def _get_ar_mean(_dt):
            if _dt <= 0.0:
                return 1.0 - ar_target
            ars = []
            x_cur = x_init
            for _ in range(n_steps_tune):
                x_cur, ar = self.step(x_cur, _dt)
                ars.append(ar)
            ar_mean = np.mean(ars)
            ar_cur[0] = ar_mean
            return ar_mean - ar_target

        print(f"Tuning stepsize for MALA steps")
        tune_res = sp.optimize.root_scalar(
            _get_ar_mean,
            bracket=(0.0, 10.0),
            method="bisect",
            xtol=0.00001,
        )

        dt = tune_res.root
        print(
            f"Stepsize {dt:.2e}; acceptance fraction {ar_cur[0]:.3f}; iterations {tune_res.iterations}",
            flush=True,
        )

        self._timestep = dt

    def __call__(self, x: torch.Tensor):
        return self.step(x, self._timestep)[0]


class Constant(Operator):
    def __init__(self, target_distribution: Distribution, n_subsample: int = 10):
        super().__init__(target_distribution)
        self._n_subsample = n_subsample
        self._name = f"const_oper_target_dist={target_distribution.name}"

    def __call__(self, x: torch.Tensor):
        return self._distribution._sample(self._n_subsample)
