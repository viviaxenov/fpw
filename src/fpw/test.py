#!/usr/bin/env python3


from typing import List, Dict

import numpy as np
import scipy as sp
import ot
import pymanopt

import matplotlib.pyplot as plt
from matplotlib import colormaps
import corner


from time import perf_counter

import inspect
from functools import partial, reduce
from operator import mul, concat, itemgetter
from cycler import cycler

import os, sys
from tqdm import tqdm
from datetime import datetime
import argparse

import json
import pickle


from fpw import BWRAMSolver, dBW
from fpw.BWRAMSolver import OperatorWG, OperatorOU, OperatorBarycenter
from fpw.PymanoptInterface import *
from fpw.utility import *


def _get_operator(
    cov_target: np.ndarray,
    operator_type: Union["dW_KL", "OU", "Barycenter"],
    dt=0.17,
    scaling=0.1,
    **kwargs,
):

    match operator_type:
        case "dW_KL":
            return OperatorWG(cov_target, scaling)
        case "OU":
            return OperatorOU(cov_target, dt)
        case "Barycenter":
            return OperatorBarycenter(
                cov_target,
                **kwargs,
            )
        case _:
            raise RuntimeError("Operator type not recognized")


def _run_test_Picard(
    cov_target: np.ndarray,
    N_steps: int,
    operator_type: Union["dW_KL", "OU"],
    r_min=1e-6,
    KL_min=-1.0,
    dt=0.17,
    scaling=0.1,
    **kwargs,
):
    dim = cov_target.shape[0]
    operator = _get_operator(cov_target, operator_type, **kwargs)

    r_stopping = r_min * scaling if operator_type == "dW_KL" else r_min

    def KL(cov, cov_target):
        Sigma_1_inv_at_Sigma = np.linalg.solve(cov_target, cov)
        return 0.5 * (
            np.trace(Sigma_1_inv_at_Sigma)
            + np.log(np.linalg.det(cov_target))
            - np.log(np.linalg.det(cov))
            - dim
        )

    BW2_err = []
    KL_err = []
    residuals = []
    dts = []

    cov_init = operator._sigmas[0] if operator_type == "Barycenter" else np.eye(dim)
    cov_init = np.eye(dim)
    cov_cur = cov_init

    for k in range(N_steps):
        cov_prev = cov_cur.copy()
        cov_cur = operator(cov_cur)

        BW2_err.append(dBW(cov_prev, cov_target))
        KL_err.append(KL(cov_prev, cov_target))
        residuals.append(dBW(cov_cur, cov_prev))

        if residuals[-1] <= r_stopping:
            break
        if KL_err[-1] <= KL_min:
            break

    match operator_type:
        case "dW_KL":
            results = dict(scaling=operator.scaling)
        case "OU":
            results = dict(dt=operator.dt)
        case "Barycenter":
            results = dict(n_sigmas=operator.n_sigmas)

    results["operator"] = operator.name
    results["N_steps_max"] = N_steps
    results["N_steps_done"] = k
    results["BW2_err"] = np.array(BW2_err)
    results["KL_err"] = np.array(KL_err)
    residuals = np.array(residuals)
    residuals = residuals / scaling if operator_type == "dW_KL" else residuals
    results["residuals"] = residuals
    results["dts"] = np.array(dts)
    results["cov_final"] = cov_cur

    return results


def _run_test_BWRAM(
    cov_target: np.ndarray,
    N_steps: int,
    operator_type: Union["dW_KL", "OU"],
    # dt=0.17,
    # scaling=0.1,
    r_min=1e-6,
    KL_min=None,
    **kwargs,
):
    # save all passed args, including defa

    dim = cov_target.shape[0]
    operator = _get_operator(cov_target, operator_type, **kwargs)
    r_stopping = r_min * scaling if operator_type == "dW_KL" else r_min

    # cov_init = operator._sigmas[0] if operator_type == "Barycenter" else np.eye(dim)
    cov_init = np.eye(dim)

    def KL(cov, cov_target):
        Sigma_1_inv_at_Sigma = np.linalg.solve(cov_target, cov)
        return 0.5 * (
            np.trace(Sigma_1_inv_at_Sigma)
            + np.log(np.linalg.det(cov_target))
            - np.log(np.linalg.det(cov))
            - dim
        )

    BW2_err = []
    KL_err = []
    residuals = []
    dts = []

    valid_params = inspect.signature(BWRAMSolver).parameters.keys()
    kwargs_solver = {k: v for k, v in kwargs.items() if k in valid_params}
    solver = BWRAMSolver(operator, **kwargs_solver)
    solver._initialize_iteration(cov_init)
    for k in range(N_steps):
        cov_ram = solver._x_prev
        residual_ram = solver._r_prev

        BW2_err.append(dBW(cov_ram, cov_target))
        KL_err.append(KL(cov_ram, cov_target))
        residuals.append(np.sqrt(np.trace(residual_ram @ cov_ram @ residual_ram)))

        if residuals[-1] <= r_stopping:
            break
        if KL_min is not None:
            if KL_err[-1] <= KL_min:
                break

        try:
            t = perf_counter()
            solver._step()
            dt = perf_counter() - t
            dts.append(dt)
        except Exception as e:
            print(f"Test run for m = {solver._m:-2d} terminated at {k=}")
            print(f"\tError: {e}")
            break

    results = kwargs_solver.copy()

    match operator_type:
        case "dW_KL":
            results.pop("dt", None)
        case "OU":
            results.pop("scaling", None)

    results["N_steps_max"] = N_steps
    results["N_steps_done"] = k
    results["BW2_err"] = np.array(BW2_err)
    results["KL_err"] = np.array(KL_err)
    residuals = np.array(residuals)
    residuals = residuals / scaling if operator_type == "dW_KL" else residuals
    results["residuals"] = residuals
    results["dts"] = np.array(dts)
    results["cov_final"] = solver._x_cur
    results["operator"] = operator.name

    return results


# these parameters already have defaults in PyManOpt but they dont fit our workflow
_Riemannian_defaults = dict(
    log_verbosity=1,
    verbosity=0,
    min_gradient_norm=0.0,
    min_step_size=0.0,
)

#
_defaults = dict(
    r_min=1e-6,
    KL_min=0.0,
)


def _run_test_Riemannian_minimization(
    cov_target: np.ndarray,
    N_steps: int,
    method: Union["RCG", "RGD"],
    vt_kind="one-step",
    scaling=1.0,
    r_min=1e-6,
    **kwargs,
):

    dim = cov_target.shape[0]
    # cov_init = operator._sigmas[0] if operator_type == "Barycenter" else np.eye(dim)
    cov_init = np.eye(dim)
    Sigma_1 = torch.Tensor(cov_target)

    BW_manifold = BuresWassersteinManifold(dim, pt_type=vt_kind)

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

    problem = pymanopt.Problem(BW_manifold, KL)

    opt_factory = (
        pymanopt.optimizers.SteepestDescent
        if method == "RGD"
        else pymanopt.optimizers.ConjugateGradient
    )
    # Avoiding <<Unexpected key-value argument>> error
    valid_params = inspect.signature(opt_factory).parameters.keys()
    kwargs_solver = {k: v for k, v in kwargs.items() if k in valid_params}
    kwargs_solver = _Riemannian_defaults | kwargs_solver
    kwargs_solver["max_iterations"] = N_steps
    kwargs_solver["min_gradient_norm"] = r_min * scaling
    optimizer = opt_factory(
        **kwargs_solver,
    )
    try:
        opt_result = optimizer.run(problem, initial_point=cov_init)
    except (ValueError, RuntimeWarning):
        print(f"{dim=} {scaling=} {method=} {vt_kind=}")
        print(cov_target)
        raise

    BW2_err = [dBW(cov, cov_target) for cov in opt_result.log["iterations"]["point"]]
    KL_err = np.array(opt_result.log["iterations"]["cost"]) / scaling
    residuals = np.array(opt_result.log["iterations"]["gradient_norm"]) / scaling
    dts = np.array(opt_result.log["iterations"]["time"])
    dts = dts[1:] - dts[:-1]
    results = kwargs_solver.copy()

    results["N_steps_max"] = N_steps
    results["N_steps_done"] = len(BW2_err)
    results["BW2_err"] = np.array(BW2_err)
    results["KL_err"] = KL_err
    results["residuals"] = residuals
    results["dts"] = dts
    results["cov_final"] = opt_result.point

    return results


_method_name_to_fn = {
    "BWRAM": _run_test_BWRAM,
    "RGD": _run_test_Riemannian_minimization,
    "RCG": _run_test_Riemannian_minimization,
    "Picard": _run_test_Picard,
}


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
        rs_cov=rs,
    )
    ortho = sp.stats.ortho_group.rvs(dim=dim, random_state=rs)
    sigmas = np.diag(np.linspace(sigma_min, sigma_max, dim, endpoint=True))
    cov_target = ortho.T @ sigmas @ ortho

    return cov_target, target_data


def _run_experiment(**kwargs):

    args_used = kwargs.copy()
    kwargs = _defaults | kwargs
    cov_target, target_data = _get_target(**kwargs)
    run_method = _method_name_to_fn[kwargs["method"]]

    run_res = run_method(cov_target, **kwargs)
    run_res.update(target_data)
    run_res["method"] = kwargs["method"]

    return run_res, args_used


def _cycler_from_dict(d: dict) -> cycler:
    cs = [
        cycler(**{key: (val if isinstance(val, list) else [val])})
        for key, val in d.items()
    ]
    return reduce(mul, cs)


def _expand_config(config: dict):
    if isinstance(config, list):
        return config
    config = config.copy()
    config.pop("output_dir", None)
    config.pop("plot", None)

    cycler_targets = _cycler_from_dict(config["targets"])
    cyclers_methods = [
        _cycler_from_dict(mtd_dict) * cycler(method=[method])
        for method, mtd_dict in config["methods"].items()
    ]

    cycler_methods = [cycler_targets * c for c in cyclers_methods]

    expanded_conf = reduce(concat, [[x for x in c] for c in cycler_methods])

    # if there are no specific values for these params,  look for them at the higher level of config and update
    # only passing down 'scalar' values

    for param, val in config.items():
        if isinstance(val, (list, dict)):
            continue
        for rec in expanded_conf:
            rec[param] = rec.get(param, val)

    ig = itemgetter("dim", "sigma_min", "sigma_max", "rs")
    expanded_conf.sort(key=ig)

    return expanded_conf


def _group_by_keys(data: List[Dict], *key_names):
    ig = itemgetter(*key_names)
    unique_key_groups = set([ig(rec) for rec in data])

    return [
        (group, [rec for rec in data if ig(rec) == group])
        for group in unique_key_groups
    ]


_methods_colors = dict(Picard="blue", BWRAM="green", RCG="red", RGD="orange")
_methods_palettes = dict(Picard="Blues", BWRAM="Greens", RCG="Reds", RGD="Oranges")
_methods_lineprops = dict(
    Picard=dict(linestyle="--", linewidth=2.5),
    BWRAM=dict(linestyle="-", linewidth=0.7),
    RCG=dict(linestyle=":", linewidth=1.5),
    RGD=dict(linestyle="-.", linewidth=1.5),
)
_max_plot_dim = 5


def _get_legend(rec: Dict):
    match rec["method"]:
        case "Picard":
            return f"{rec['operator']}"
        case "BWRAM":
            return f"{rec['operator']}+BWRAM, m = {rec['history_len']}"
        case "RGD" | "RCG":
            return rec["method"]


def _plot_one_target(target_params: dict, data: List[Dict]):

    fig: plt.Figure
    axs: List[plt.Axes]
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    groups = _group_by_keys(data, "method")
    for method, group in groups:
        cmap = colormaps.get_cmap(_methods_palettes[method])
        for k, rec in enumerate(group):
            legend_text = _get_legend(rec)
            color = cmap((k + 2) / (len(group) + 2))

            axs[0].plot(
                rec["residuals"],
                color=color,
                **_methods_lineprops[method],
                label=legend_text,
            )
            axs[1].plot(
                rec["KL_err"],
                color=color,
                **_methods_lineprops[method],
                label=legend_text,
            )
            axs[2].plot(
                rec["BW2_err"],
                color=color,
                **_methods_lineprops[method],
                label=legend_text,
            )

    axs[0].set_title(r"$\|r_k\|_{\Sigma_k}$")
    axs[1].set_title(r"$KL({\Sigma_k}|\Sigma_\infty)$")
    axs[2].set_title(r"$W^2_2({\Sigma_k}|\Sigma_\infty)$")

    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.grid()
    ax.legend()

    fig.suptitle(
        "$d = {0},\\ \\sigma_{{min}} = {1:.1f},\\ \\sigma_{{max}} = {2:.1f}$".format(
            *target_params
        )
    )

    cov_target, _ = _get_target(*target_params)
    sample_target = sp.stats.multivariate_normal(cov=cov_target).rvs(1000)[
        :, :_max_plot_dim
    ]
    fig_corner = corner.corner(
        sample_target, color="k", hist_kwargs=dict(density=True, label="Reference")
    )

    for method, group in groups:
        res_best = min(group, key=lambda rec: min(rec["residuals"]))
        cov_cur = res_best["cov_final"]
        sample_cur = sp.stats.multivariate_normal(cov=cov_cur, allow_singular=True).rvs(
            1000
        )[:, :_max_plot_dim]
        color = _methods_colors[method]
        fig_corner = corner.corner(
            sample_cur,
            fig=fig_corner,
            color=color,
            hist_kwargs=dict(density=True, label=method),
        )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.7))

    fig_corner.suptitle(
        "Target and approx. distribution "
        + (
            f"(first {_max_plot_dim} parameters)"
            if cov_target.shape[0] > _max_plot_dim
            else ""
        )
    )

    return fig, fig_corner


def _plot_each_target(results: List[Dict], output_dir: str):
    group_by_target = _group_by_keys(results, "dim", "sigma_min", "sigma_max", "rs_cov")
    ig = itemgetter("dim", "sigma_min", "sigma_max", "rs_cov")

    for k, (group_data, group) in enumerate(group_by_target):
        print(f"Plotting target {k+1}")
        print(f"Target data: {group_data}", flush=True)
        fig, fig_corner = _plot_one_target(group_data, group)

        fig.savefig(os.path.join(output_dir, f"convergence_target_{k:02d}.pdf"))
        fig_corner.savefig(os.path.join(output_dir, f"corner_target_{k:02d}.pdf"))

        plt.close("all")


def run_experiments(config: dict):
    default_dirname = f"./tests/{datetime.now():%Y_%b_%d__%X}"

    try:
        dirname = config["output_dir"]
    except (TypeError, KeyError):
        dirname = default_dirname
    try:
        plot = config["plot"]
    except (TypeError, KeyError):
        plot = False

    exp_config = _expand_config(config)

    results, used_args = [], []
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "config.json"), "w") as ofile:
        json.dump(config, ofile, indent=4)

    for k, arguments in enumerate(tqdm(exp_config)):
    # for k, arguments in enumerate(exp_config):
        t = perf_counter()
        res, args = _run_experiment(**arguments)
        dt = t - perf_counter()
        results.append(res)
        used_args.append(args)
        if dt > 1.0 or (k + 1) % 5 == 0 or k + 1 == len(arguments):
            ig = itemgetter("dim", "sigma_min", "sigma_max", "rs_cov")
            results.sort(key=ig)

            with open(os.path.join(dirname, "args.json"), "w") as ofile:
                json.dump(used_args, ofile, indent=2)

            with open(os.path.join(dirname, "res.pkl"), "wb") as ofile:
                pickle.dump(results, ofile)

    if plot:
        dirname_plots = os.path.join(dirname, "plots")
        os.makedirs(dirname_plots, exist_ok=True)
        _plot_each_target(results, dirname_plots)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Config file (.json) with experiments description", type=str
    )
    parser.add_argument("--replot", "-r", action="store_true")

    args = parser.parse_args()

    input_file = args.input_file

    if args.replot:
        if os.path.splitext(input_file)[1] == ".json":
            with open(args.input_file, "r") as ifile:
                config = json.load(ifile)
            try:
                output_dir = config["output_dir"]
            except TypeError:
                output_dir = os.path.dirname(input_file)
        else:
            output_dir = os.path.dirname(input_file)

        res_file = os.path.join(output_dir, "res.pkl")
        print(res_file)
        with open(res_file, "rb") as ifile:
            res = pickle.load(ifile)

        _plot_each_target(res, os.path.join(output_dir, "plots"))
        exit()

    with open(args.input_file, "r") as ifile:
        config = json.load(ifile)

    # config = {
    #     "targets": {"dim": 2, "sigma_min": [1.0, 0.5, 0.1], "sigma_max": 5.0, "rs": 1},
    #     "methods": {
    #         "RGD": {"N_steps": 10, "scaling": 0.1},
    #         "BWRAM": {
    #             "history_len": [1, 2, 5, 10],
    #             "N_steps": 10,
    #             "operator_type": "dW_KL",
    #             "scaling": 0.1,
    #             "vt_kind": "one-step",
    #         },
    #         "Picard": {
    #             "N_steps": 10,
    #             "operator_type": "dW_KL",
    #             "scaling": 0.1,
    #         },
    #     },
    #     # "output_dir": "./many_sigma/",
    #     "plot": True,
    # }

    run_experiments(config)
