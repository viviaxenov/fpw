#!/usr/bin/env python3


from typing import List, Dict, Union

import numpy as np
import scipy as sp
import pymanopt

import matplotlib.pyplot as plt
from matplotlib import colormaps


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
from fpw.BWRAMSolver import BWRAMSolver
from fpw.ProblemGaussian import *
from fpw.PymanoptInterface import *

import num2tex


_max_corner_dim = 5


def _run_test_BWRAM(
    problem: Problem,
    N_steps_max: int,
    N_warmstart: int = 0,
    r_min=1e-10,
    cost_min: np.float64 = -np.inf,
    **kwargs,
):

    residuals = []
    covs = []
    dts = []

    kwargs['vt_kind'] = problem._vt_kind

    if problem.has_cost:
        costs = []

    cov_init = problem.get_initial_value()
    if N_warmstart > 0 and hasattr(problem, "covs_ref"):
        if N_warmstart < len(problem.covs_ref):
            cov_init = problem.covs_ref[N_warmstart]
        else:
            warnings.warn("Incorrect value for warmstart")

    covs = []
    residuals = []
    dts = []

    solver = BWRAMSolver(problem, **kwargs)
    solver._initialize_iteration(cov_init)
    for k in range(N_steps_max):
        cov_ram = solver._x_prev
        residual_ram = solver._r_prev

        # covs.append(cov_ram)
        residuals.append(problem.base_manifold.norm(cov_ram, residual_ram))
        if problem.has_cost:
            costs.append(problem.cost(cov_ram))
            if costs[-1] <= cost_min:
                break

        if residuals[-1] <= r_min:
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

    results = dict()
    results["residuals"] = residuals
    results["dts"] = np.array(dts)
    results["covs"] = covs
    results["cov_final"] = cov_ram
    if problem.has_cost:
        results["costs"] = costs

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


class _LineSearchWithIterCount(pymanopt.optimizers.line_search.AdaptiveLineSearcher):
    def __init__(self, *args, **kwargs):
        self._n_calls_cur_iter = 0
        self.iters = []
        super().__init__(*args, **kwargs)

    def search(self, *args, **kwargs):
        self._n_calls_cur_iter = 0

        self.orig_objective = args[0]

        def _objective_with_count(*_args, **_kwargs):
            self._n_calls_cur_iter += 1
            return self.orig_objective(*_args, **_kwargs)

        args = list(args)
        args[0] = _objective_with_count
        args = tuple(args)

        retvals = super().search(*args, **kwargs)
        self.iters.append(self._n_calls_cur_iter)
        return retvals


def _expand_arr(arr, iters):
    expanded_arr = []
    for val, ntimes in zip(arr, iters):
        expanded_arr += [val]* ntimes

    return expanded_arr


def _run_test_Riemannian_minimization(
    problem: Problem,
    N_steps_max: int,
    method: Union["RCG", "RGD"],
    N_warmstart: int = 0,
    r_min=1e-10,
    cost_min: np.float64 = -np.inf,
    **kwargs,
):
    if not hasattr(problem, "get_cost_torch"):
        raise ValueError("Cannot use Riemannian optimization w.o. toch cost")
    cov_init = problem.get_initial_value()
    if N_warmstart > 0 and hasattr(problem, "covs_ref"):
        if len(problem.covs_ref) > 0:
            if N_warmstart < len(problem.covs_ref):
                cov_init = problem.covs_ref[N_warmstart]
            else:
                warnings.warn("Incorrect value for warmstart")
        else:
            warnings.warn(
                "Reference covariances were not written, warmstart not possible"
            )

    BW_manifold = problem.base_manifold

    cost_torch = pymanopt.function.pytorch(BW_manifold)(problem.get_cost_torch())

    problem.n_cost_calls = 0
    _problem = pymanopt.Problem(BW_manifold, cost_torch)

    opt_factory = (
        pymanopt.optimizers.SteepestDescent
        if method == "RGD"
        else pymanopt.optimizers.ConjugateGradient
    )
    ls = _LineSearchWithIterCount()
    # Avoiding <<Unexpected key-value argument>> error
    valid_params = inspect.signature(opt_factory).parameters.keys()
    kwargs_solver = {k: v for k, v in kwargs.items() if k in valid_params}
    kwargs_solver = _Riemannian_defaults | kwargs_solver
    kwargs_solver["max_iterations"] = N_steps_max
    kwargs_solver["min_gradient_norm"] = r_min
    kwargs_solver["line_searcher"] = ls
    optimizer = opt_factory(
        **kwargs_solver,
    )

    try:
        opt_result = optimizer.run(
            _problem,
            initial_point=cov_init,
            reuse_line_searcher=True,  # to avoid deepcopy of line searcher and be able to use counting
        )
        log = opt_result.log
    except (ValueError, RuntimeWarning) as e:
        print(f"Error encountered in {method}: {e}")
        log = optimizer._log


    covs = log["iterations"]["point"]
    residuals = np.array(log["iterations"]["gradient_norm"])
    costs = log["iterations"]["cost"]

    dts = np.array(log["iterations"]["time"])
    dts = dts[1:] - dts[:-1]

    results = dict()
    # results["covs"] = covs
    iters  = optimizer.line_searcher.iters
    results["covs"] = []
    results["residuals"] = _expand_arr(residuals, iters)
    results["dts"] = dts
    results["cov_final"] = covs[-1]
    results["costs"] = _expand_arr(costs, iters)

    return results


_problem_name_to_fn = {
    "Barycenter": Barycenter,
    "EntBC": EntropicBarycenter,
    "Median": Median,
    "OU": OUEvolution,
    "dW_KL": WGKL,
}
_method_name_to_fn = {
    "BWRAM": _run_test_BWRAM,
    "RGD": _run_test_Riemannian_minimization,
    "RCG": _run_test_Riemannian_minimization,
}
_methods_with_vt = {"BWRAM", "RCG"}


def _run_experiment(**kwargs):
    args_used = kwargs.copy()
    kwargs = _defaults | kwargs
    cov_target, target_data = _get_target(**kwargs)
    run_method = _method_name_to_fn[kwargs["method"]]

    run_res = run_method(cov_target, **kwargs)
    run_res.update(target_data)
    run_res["method"] = kwargs["method"]

    return run_res, args_used


def _parse_dict(d: dict, name: str) -> cycler:
    d = d[name]
    res = []
    for entry_name, entry_dict in d.items():
        cs = [
            cycler(**{key: (val if isinstance(val, list) else [val])})
            for key, val in entry_dict.items()
        ]
        cs.append(cycler(**{name[:-1]: [entry_name]}))
        cycler_all = reduce(mul, cs)

        res += list(cycler_all)

    return res


def _parse_config(config: dict):
    default_dirname = f"./outputs/{datetime.now():%Y_%b_%d__%X}"
    dirname = config.pop("output_dir", default_dirname)
    plot = config.pop("plot", False)
    n_max_plot = config.pop("n_max_plot", 0)

    kws_problems = _parse_dict(config, "problems")
    kws_methods = _parse_dict(config, "methods")

    # if there are no specific values for these params,  look for them at the higher level of config and update
    # only passing down 'scalar' values

    for param, val in config.items():
        if isinstance(val, (list, dict)):
            continue
        for rec in kws_problems:
            rec[param] = rec.get(param, val)
        for rec in kws_methods:
            rec[param] = rec.get(param, val)

    return kws_problems, kws_methods, dirname, plot, n_max_plot


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
    Picard=dict(linestyle="--", linewidth=2.9),
    BWRAM=dict(linestyle="-", linewidth=1.0),
    RCG=dict(linestyle=":", linewidth=1.5),
    RGD=dict(linestyle="-.", linewidth=1.5),
)


def _get_legend(rec: Dict):
    match rec["method"]:
        case "Picard":
            return "Picard"
        case "BWRAM":
            label = f"BWRAM, $m = {rec['history_len']},\\ \\|\\Gamma\\|_{{l_\\infty}} \\leq {num2tex.num2tex(rec['l_inf_bound_Gamma'])}"
            if "r_threshold" in rec:
                label = f"q_{{rest}} = {rec['r_threshold']:.1e}"

            label += "$"
            return label
        case "RGD" | "RCG":
            return rec["method"]


def _get_style(rec: Dict):
    style = _methods_lineprops[rec["method"]]
    label = _get_legend(rec)
    color = _methods_colors[rec["method"]]
    if rec["method"] == "BWRAM":
        m = rec["history_len"]
        palette = _methods_palettes[rec["method"]]
        color = colormaps.get_cmap(palette)((m + 5) / (15 + 5))

    return style | dict(label=label, color=color)


_key_to_ylabel = dict(
    residuals=r"$\|r_k\|_{\Sigma_k}$",
    costs=r"$V(\Sigma_k)$",
    costs_to_0=r"$V(\Sigma_k) - V(\Sigma_*)$",
    dBW=r"$W^2_2(\Sigma_k, \Sigma_*)$",
)


def _plot_figure(
    problem: Problem,
    rp: List[dict],
    dirname_plots: str,
    key: str = "residuals",
    n_max_plot=5,
    **kwargs,
):
    fig: plt.Figure
    ax: plt.Axes

    # Plot residuals
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(problem.name)
    v_min, k_min = np.inf, 0
    dirname_cur = os.path.join(dirname_plots, key)
    os.makedirs(dirname_cur, exist_ok=True)

    rp.sort(key=lambda _r: 0 if _r["method"] != "BWRAM" else len(_r["residuals"]))
    r_to_plot = rp[:n_max_plot]

    def _pretty_sorting(rec: dict):
        match rec["method"]:
            case "Picard":
                return 0
            case "RGD":
                return 1
            case "RCG":
                return 2
            case "BWRAM":
                return 2 + rec["history_len"]

    r_to_plot.sort(key=_pretty_sorting)

    plot_name = os.path.join(dirname_cur, f"{problem.name}.pdf")
    for res in r_to_plot:
        val = res[key]

        k_cur_min = np.argmin(np.array(val))
        v_cur_min = val[k_cur_min]

        if v_cur_min < v_min:
            v_min, k_min = v_cur_min, k_cur_min

        n0 = res.get("N_warmstart", 0)
        iters = range(n0, n0 + len(val))
        ax.plot(iters, val, **_get_style(res))

    ax.set_yscale("log")
    ax.set_xlim(left=0, right=len(rp[0][key]))
    # ax.set_ylim(bottom=v_min)
    ax.set_xlabel("iteration")
    ax.set_ylabel(_key_to_ylabel[key])
    ax.legend(fontsize=14)
    ax.grid()

    fig.savefig(plot_name)
    return fig


def _plot_problem(problem: Problem, rp: List[dict], dirname_plots: str, n_max_plot=0):
    _plot_figure(problem, rp, dirname_plots, "residuals", n_max_plot=n_max_plot)

    if problem.has_cost:
        # Plot convergence of the cost functional
        _plot_figure(problem, rp, dirname_plots, "costs", n_max_plot=n_max_plot)
        pass
        # plot_name = os.path.join(dirname_plots, f"cost_{problem_name}.pdf")
        # fig.savefig(plot_name)

    if hasattr(problem, "target"):
        # for res in rp:
        #     res["dBW"] = [dBW(Cov, problem.target) for Cov in res["covs"]]
        # _plot_figure(problem, rp, dirname_plots, "dBW", n_max_plot=n_max_plot)
        if problem.has_cost:
            cost_ref = problem.cost(problem.target)
            for res in rp:
                res["costs_to_0"] = [c - cost_ref for c in res["costs"]]
            _plot_figure(
                problem, rp, dirname_plots, "costs_to_0", n_max_plot=n_max_plot
            )

    plt.close("all")


def run_experiments(config: dict):

    args_problems, args_methods, dirname, plot, n_max_plot = _parse_config(config)
    os.makedirs(dirname, exist_ok=True)

    problems = []
    results = []
    results_by_problem = []

    with open(os.path.join(dirname, "config.json"), "w") as ofile:
        json.dump(config, ofile, indent=4)

    dirname_chunks = os.path.join(dirname, "chunks")
    os.makedirs(dirname_chunks, exist_ok=True)
    dirname_plots = os.path.join(dirname, "plots")
    os.makedirs(dirname_plots, exist_ok=True)

    n_total_runs = (len(args_methods) + 1) * len(args_problems)
    with tqdm(total=n_total_runs) as pbar:
        for k, kw_prob in enumerate(args_problems):
            output_fname = os.path.join(dirname_chunks, f"p_{k+1}.pkl")
            problem: Problem = _problem_name_to_fn[kw_prob["problem"]](**kw_prob)
            problems.append(problem)
            rp = []

            pbar.set_description(
                f"Computing reference solution for problem {problem.name}"
            )
            res = problem.get_solution_picard(**kw_prob)
            if hasattr(problem, "target"):
                res["dBW"] = [dBW(Cov, problem.target) for Cov in res["covs"]]
                if problem.has_cost:
                    cost_ref = problem.cost(problem.target)
                    res["costs_to_0"] = [c - cost_ref for c in res["costs"]]
            else:
                # If Picard didn't produce a reference solution, don't run other methods either
                print("Picard method didn't converge; can't produce reference solution; Skipping")
                pbar.update(1 + len(args_methods))
                continue

            pbar.update(1)

            res = res | kw_prob
            res["method"] = "Picard"
            results.append(res)
            rp.append(res)

            for kw_method in args_methods:
                if kw_method["method"] in ["RGD", "RCG"] and not problem.has_cost:
                    pbar.update(1)
                    continue
                method = _method_name_to_fn[kw_method["method"]]

                pbar.set_description(
                    f"Computing solution for problem {problem.name} with method {kw_method['method']}"
                )
                try:
                    res = method(problem, **kw_method)
                except (ValueError, RuntimeWarning, RuntimeError, AssertionError) as e:
                    print(e)
                    pbar.update(1)
                    continue
                pbar.update(1)

                res = res | kw_method | kw_prob
                if hasattr(problem, "target"):
                    res["dBW"] = [dBW(Cov, problem.target) for Cov in res["covs"]]
                    if problem.has_cost:
                        cost_ref = problem.cost(problem.target)
                        res["costs_to_0"] = [c - cost_ref for c in res["costs"]]
                rp.append(res)

            with open(output_fname, "wb") as ofile:
                pickle.dump(rp, ofile)

            if plot:
                _plot_problem(problem, rp, dirname_plots, n_max_plot=n_max_plot)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Config file (.json) with experiments description", type=str
    )
    parser.add_argument("--replot", "-r", action="store_true")

    args = parser.parse_args()

    input_file = args.input_file

    if args.replot:
        raise NotImplemented

    with open(args.input_file, "r") as ifile:
        config = json.load(ifile)

    run_experiments(config)
