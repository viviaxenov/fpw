from typing import Callable, Union, List, Dict, Generator, Literal

import numpy as np
import scipy as sp

import ot

import torch
from geomloss import SamplesLoss

from .pt import parallel_transport


def vector_translation(x0, x1, u0):
    return u0


def get_vector_transport(kind="translation", *args, **kwargs):
    if kind == "translation":
        return vector_translation
    elif kind == "parallel":
        eps = kwargs.get("epsilon", 1e-10)
        return lambda *_x: parallel_transport(*_x, epsilon)
    else:
        raise RuntimeError(f"Vector transport <<{kind}>> not implemented")


def ot_map_geomloss(x0, x1, reg_sinkhorn, *sinkhorn_args, **sinkhorn_kwargs):
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


def ot_map_pot(x0, x1, reg_sinkhorn, *sinkhorn_args, **sinkhorn_kwargs):
    ot_plan = ot.bregman.empirical_sinkhorn(
        x0, x1, self._reg_sinkhorn, *self._sinkhorn_args, **self._sinkhorn_kwargs
    )
    x1_barycentric = (ot_plan[:, :, np.newaxis] * x1[np.newaxis, :, :]).sum(
        axis=1
    ) * x0.shape[0]

    v = x1_barycentric - x0

    return x1_barycentric, v


# Naming tentative; Riemannian -> Wasserstein? Ottonian?
# TODO: need to pass a bunch of (*args, **kwargs) to both PT and Sinkhorn
class RAMSolver:
    def __init__(
        self,
        operator: Callable,
        relaxation: Union[np.float64, Generator] = 0.95,
        history_len: int = 2,
        ot_map_solver: Literal["pot", "geomloss"] = "geomloss",
        vector_transport_kind: Literal["translation", "parallel"] = "translation",
        vt_args: List = [],
        vt_kwargs: Dict = {},
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

        self._vt = get_vector_transport(vector_transport_kind, *vt_args, **vt_kwargs)

        if ot_map_solver == "geomloss":
            self._ot_map = ot_map_geomloss
        elif ot_map_solver == "pot":
            self._ot_map = ot_map_pot
        else:
            raise RuntimeError(f"{ot_map_solver=:} not supported")

        self._reg_sinkhorn = reg_sinkhorn
        self._sinkhorn_args = sinkhorn_args
        self._sinkhorn_kwargs = sinkhorn_kwargs

    def _operator_and_residual(self, x_cur: np.ndarray):
        x0 = x_cur
        x1 = self._operator(x_cur)

        return self._ot_map(
            x0, x1, self._reg_sinkhorn, *self._sinkhorn_args, **self._sinkhorn_kwargs
        )

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _S2_dist_fn = SamplesLoss(blur=0.3)
    S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(
        torch.from_numpy(_s1), torch.from_numpy(_s2)
    )

    def get_lpr_and_score_fn_gaussian(m: np.ndarray, sigma: np.ndarray):
        psq = np.linalg.inv(sigma)

        def _lpr(x):
            return (
                -0.5 * np.linalg.norm((psq @ (x - m[np.newaxis, :]).T).T, axis=-1) ** 2
            )

        def _score(x):
            return -(psq.T @ psq @ (x - m[np.newaxis, :]).T).T

        return _lpr, _score

    def ula_step(x: np.ndarray, timestep: np.float64, score_fn: Callable):
        noise = sp.stats.norm().rvs(size=x.shape)
        return x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5

    def mala_step(
        x: np.ndarray,
        timestep: np.float64,
        log_prob: Callable,
        score_fn: Callable,
    ):
        noise = sp.stats.norm().rvs(size=x.shape)
        x_prop = x + score_fn(x) * timestep + noise * (2.0 * timestep) ** 0.5

        d_log_prob = log_prob(x_prop) - log_prob(x)
        d_transition = (
            np.linalg.norm((x - x_prop - timestep * score_fn(x_prop)), axis=-1) ** 2
            - 2.0 * timestep * np.linalg.norm(noise, axis=-1) ** 2
        )

        log_alpha = np.minimum(0, d_log_prob - d_transition / (4.0 * timestep))
        u = np.random.rand(x.shape[0])
        is_accepted = np.log(u) <= log_alpha
        acceptance_rate = is_accepted.mean()
        x_new = np.where(is_accepted[:, np.newaxis], x_prop, x)

        return x_new, acceptance_rate

    rs = 2
    N_particles = 3000
    dim = 10
    N_steps = 100
    hist_lens = [1, 2, 5, 10, 15]

    if dim == 2:
        sigma = np.array([[1.0, 0.4], [0.4, 1.0]])
        m = np.array([10.0, 4.0])
    else:
        U = sp.stats.ortho_group.rvs(dim, random_state=rs)
        sigma = np.diag([1.0 - 0.8 * 0.8**n for n in range(dim)])
        sigma = U.T @ sigma @ U
        m = sp.stats.uniform.rvs(size=dim, random_state=rs) * 10.0

    sample_targ = sp.stats.multivariate_normal(
        mean=m,
        cov=sigma.T @ sigma,
    ).rvs(size=N_particles)
    sample_init = sp.stats.multivariate_normal(
        cov=np.eye(dim),
    ).rvs(
        size=N_particles,
        random_state=rs,
    )

    lpr, score = get_lpr_and_score_fn_gaussian(m, sigma)
    S2_errs_mala = []
    acceptance_rates = []
    timesteps = np.linspace(0.001, .11, 51, endpoint=True)

    ar_means = []
    logS2_rates = []
    N_linear = 12

    for dt in timesteps:
        sample_mala = sample_init.copy()
        S2_err = []
        ars = []
        for _ in range(N_steps):
            S2_err.append(S2_dist_fn(sample_mala, sample_targ))
            sample_mala, ar = mala_step(sample_mala, dt, lpr, score)
            ars.append(ar)
        S2_errs_mala.append(S2_err)
        acceptance_rates.append(ars)

        ar_means.append(np.mean(ars))
        logS2_rates.append(-np.log(S2_err[N_linear]/S2_err[0])/N_linear)

    idx_opt = logS2_rates.index(max(logS2_rates))
    dt_opt = timesteps[idx_opt]
    S2_err = S2_errs_mala[idx_opt]

    fig, axs = plt.subplots(1, 2)

    ax = axs[0]
    ax.plot(timesteps, ar_means)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Acceptance rate')
    ax.axvline(dt_opt, color='g', linestyle='--')
    
    ax1 = ax.twinx()
    ax1.plot(timesteps, logS2_rates, 'r--')
    ax1.set_ylabel('Convergence rate ($\\log S_2$)')

    ax = axs[1]
    ax.plot(ar_means, logS2_rates)
    ax.set_xlabel('Acceptance rate')
    ax.set_ylabel('Convergence rate ($\\log S_2$)')

    fig.tight_layout()


    operator = lambda _x: mala_step(_x, dt_opt, lpr, score)[0]
    S2_convs = []
    for m_history in hist_lens:
        S2_err_ram = []
        solver = RAMSolver(
            operator,
            history_len=m_history,
            relaxation=0.8,
            reg_sinkhorn=0.03,
            sinkhorn_kwargs={"scaling": 0.6},
        )
        solver._initialize_iteration(sample_init.copy())
        for k in range(N_steps):
            sample_ram = solver._x_prev
            S2_err_ram.append(S2_dist_fn(sample_ram, sample_targ))
            solver._step()
        S2_convs.append(S2_err_ram)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    ax = axs[0]

    s_marker = 5.0
    ax.scatter(*sample_init[:, :2].T, s=s_marker, label="Initial")
    ax.scatter(*sample_targ[:, :2].T, s=s_marker, label="Target")
    # ax.scatter(*sample_mala[:, :2].T, s=s_marker, marker="+", label="MALA approx")
    ax.scatter(*sample_ram[:, :2].T, s=s_marker, marker="x", label="RAM approx")

    ax = axs[1]
    ax.plot(S2_err, label="MALA", linewidth=2.0, linestyle="--")
    for S2_err_ram, m in zip(S2_convs, hist_lens):
        ax.plot(S2_err_ram, label=f"MALA+RAM, ${m=}$", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$S_s(\\mu_k, \\mu^*)$")

    for ax in axs:
        ax.grid()
        ax.legend()
    fig.tight_layout()
    fig.savefig("ram_mala_test.pdf")
    plt.show()
