import numpy as np

np.random.seed(2)
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import colormaps
from time import perf_counter
import ot
import pymanopt

from fpw import BWRAMSolver, dBW
from fpw.BWRAMSolver import OperatorWG, OperatorOU
from fpw.PymanoptInterface import *
from fpw.utility import *

cmap_name = "Greens"


def _run_experiment(
    dim=200,
    N_steps=40,
    N_steps_warmstart=0,
    m_history=5,
    restart_every=40,
    hist_lens=[1, 2, 5, 10],
    vt_kind="one-step",
    dt_ou=1.71e-01,
    sigma_range=[0.1, 5.0],
    rs_cov=10,
    relaxation=1.0,
    l_inf_bound_Gamma=2.0,
    restart_threshold=None,
):
    ortho = sp.stats.ortho_group.rvs(dim=dim, random_state=rs_cov)
    sigmas = np.diag(np.linspace(*sigma_range, dim, endpoint=True))
    cov_target = ortho.T @ sigmas @ ortho

    cov_init = np.eye(dim)

    # operator = get_operator_OU(cov_target, dt_ou)
    scaling = 0.0875
    # scaling = 0.1
    operator = OperatorWG(cov_target, scaling=scaling)
    # operator = OperatorOU(cov_target, 0.17)

    BW_manifold = BuresWassersteinManifold(dim, pt_type=vt_kind)
    Sigma_1 = torch.Tensor(cov_target)

    @pymanopt.function.pytorch(BW_manifold)
    def KL(
        Sigma,
    ):
        Sigma_1_inv_at_Sigma = torch.linalg.solve(Sigma_1, Sigma)
        return (
            0.5
            * scaling
            * (
                torch.trace(Sigma_1_inv_at_Sigma)
                + torch.log(torch.linalg.det(Sigma_1))
                - torch.log(torch.linalg.det(Sigma))
                - dim
            )
        )

    BW_err = []
    KL_err = []
    resid_OU = []
    cov_OU = cov_init.copy()

    for k in range(N_steps):
        if k == N_steps_warmstart:
            cov_init_ram = cov_OU.copy()
        BW_err.append(dBW(cov_OU, cov_target))
        KL_err.append(KL(cov_OU) / scaling)
        cov_prev = cov_OU.copy()
        cov_OU = operator(cov_OU)
        resid_OU.append(dBW(cov_OU, cov_prev))

    BW2_convs = []
    KL_convs = []
    residuals = []
    for m_history in hist_lens:
        BW2_err_ram = []
        KL_err_ram = []
        solver = BWRAMSolver(
            operator,
            history_len=m_history,
            relaxation=relaxation,
            l_inf_bound_Gamma=l_inf_bound_Gamma,
            vt_kind=vt_kind,
            r_threshold=restart_threshold,
        )
        solver._initialize_iteration(cov_init_ram.copy())
        k = 0
        while k < (N_steps - N_steps_warmstart):
            cov_ram = solver._x_prev
            BW2_err_ram.append(dBW(cov_ram, cov_target))
            KL_err_ram.append(KL(cov_ram) / scaling)
            try:
                t = perf_counter()
                solver._step()
                dt_per_iter = perf_counter() - t
            except Exception as e:
                print(f"Test run for {m_history=:-2d} terminated at {k=}")
                print(f"\tError: {e}")
                break
            k += 1
            if (
                k % max(restart_every, m_history) == 0
                and k < N_steps - N_steps_warmstart
            ):
                # restart implicitly does 1 Picard step in order to set x_prev, r_prev
                solver.restart()
                cov_ram = solver._x_prev
                BW2_err_ram.append(dBW(cov_ram, cov_target))
                KL_err_ram.append(KL(cov_ram) / scaling)
                k += 1

        # print(f"{m_history=:2d} {dt_per_iter=:.2e}")
        print(f"{m_history=:-2d} {vt_kind=} {dt_per_iter=:.2e}")
        BW2_convs.append(BW2_err_ram)
        KL_convs.append(KL_err_ram)
        residuals.append(solver.norm_rk)

    problem = pymanopt.Problem(BW_manifold, KL)

    optimizer = pymanopt.optimizers.SteepestDescent(
        log_verbosity=2,
        min_gradient_norm=1e-20,
        max_iterations=N_steps,
        min_step_size=0.0,
    )
    result_RGD = optimizer.run(problem, initial_point=cov_init)
    KL_err_RGD = result_RGD.log["iterations"]["cost"]
    KL_err_RGD = [x / scaling for x in KL_err_RGD]
    residual_RGD = result_RGD.log["iterations"]["gradient_norm"]

    optimizer = pymanopt.optimizers.ConjugateGradient(
        log_verbosity=2,
        min_gradient_norm=1e-20,
        max_iterations=N_steps,
        min_step_size=0.0,
        max_cost_evaluations=N_steps,
    )
    result_RCG = optimizer.run(problem, initial_point=cov_init)
    KL_err_RCG = result_RCG.log["iterations"]["cost"]
    KL_err_RCG = [x / scaling for x in KL_err_RCG]
    residual_RCG = result_RCG.log["iterations"]["gradient_norm"]

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    ax = axs[0]

    s_marker = 5.0

    sample_targ = sp.stats.multivariate_normal(cov=cov_target).rvs(size=300)
    sample_ou = sp.stats.multivariate_normal(cov=cov_OU).rvs(size=300)
    sample_ram = sp.stats.multivariate_normal(cov=cov_ram).rvs(size=300)

    # ax.scatter(*sample_init[:, :2].T, s=s_marker, label="Initial")
    ax.scatter(*sample_targ[:, :2].T, s=s_marker, label="Target")
    ax.scatter(*sample_ou[:, :2].T, s=s_marker, marker="+", label="Picard approx")
    ax.scatter(*sample_ram[:, :2].T, s=s_marker, marker="x", label="RAM approx")

    ax = axs[1]
    ax.plot(KL_err, label=operator.name, linewidth=2.0, linestyle="--")

    ax.scatter(
        N_steps_warmstart, KL_err[N_steps_warmstart], 20.0, marker="*", color="g"
    )

    cmap = colormaps.get_cmap(cmap_name)

    def get_citer():
        k = 0
        while True:
            yield cmap((k + 4) / (len(hist_lens) + 4))
            k = (k + 1) % len(hist_lens)

    citer = get_citer()

    for KL_err_ram, m in zip(KL_convs, hist_lens):
        steps_ram = list(range(N_steps_warmstart, N_steps_warmstart + len(KL_err_ram)))
        ax.plot(
            steps_ram,
            KL_err_ram,
            label=f"{operator.name}+RAM, {m=}",
            linewidth=0.7,
            color=next(citer),
        )

    ax.plot(KL_err_RGD, label="RGD", linewidth=2.0, linestyle="-.")
    ax.plot(KL_err_RCG, label="RCG", linewidth=2.0, linestyle=":", color="r")

    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$KL(\\mu_k, \\mu^*)$")

    ax = axs[2]
    ax.plot(resid_OU, label=operator.name)

    for resid, m in zip(residuals, hist_lens):
        resid = resid[:-1]
        steps_ram = list(range(N_steps_warmstart, N_steps_warmstart + len(resid)))
        ax.plot(
            steps_ram[: len(resid)],
            resid,
            label=f"{operator.name}+RAM, {m=}",
            linewidth=0.7,
            color=next(citer),
        )
    ax.plot(residual_RGD, label="RGD", linewidth=2.0, linestyle="-.")
    ax.plot(residual_RCG, label="RCG", linewidth=2.0, linestyle=":", color="r")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\|r\|_{\Sigma_k}$")

    for ax in axs:
        ax.grid()
        ax.legend()

    fig.suptitle(f"Gaussian in $\\mathbb{{R}}^{{{dim}}}$")
    fig.tight_layout()
    fig.savefig("bwram_ou_test.pdf")

    plt.show()
    fig, ax = plt.subplots(1, 1)

    ax.set_title(f"RAM, {vt_kind = }")
    ax.plot(
        steps_ram, BW2_err_ram, label=f"$W_2(\\rho_k, \\rho^\\infty)$", linewidth=0.7
    )
    # ax.plot(steps_ram, solver.W2_residual[:-1], label="$W_2(\\rho_k, \\rho_{k+1})$")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid()

    ax = ax.twinx()
    ax.plot(steps_ram, solver.norm_rk[:-1], "r--", label="$\\|r_k\\|_{L^2_{\\rho_k}}$")
    ax.plot(steps_ram, solver.norm_Gamma[:-1], "g--", label="$\\|\\Gamma\\|_{l_2}$")
    ax.legend(loc="center right")
    ax.grid()

    plt.show()


if __name__ == "__main__":
    _run_experiment()
