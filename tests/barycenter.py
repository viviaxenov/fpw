import numpy as np
import scipy as sp
import ot
import pymanopt

import matplotlib.pyplot as plt
from matplotlib import colormaps

from time import perf_counter

import os, sys

from fpw import BWRAMSolver, dBW
from fpw.BWRAMSolver import OperatorBarycenter
from fpw.PymanoptInterface import *
from fpw.utility import *


dim = 2
n_sigmas = 5
n_data = 3
N_steps_max = 200


n_conv = []
n_same_as_pot = 0

for n_run in range(n_data):
    dists = []
    operator = OperatorBarycenter(n_sigmas=n_sigmas, dim=dim, rs=n_run)
    for _k, _sig in enumerate(operator._sigmas):
        operator._sigmas[_k] += 0.2 * np.eye(dim)

    cov_init = np.eye(dim)
    cov_cur = cov_init.copy()
    cov_prev = cov_cur.copy()
    costs = [operator.cost(cov_init)]

    for k in range(N_steps_max):
        cov_cur = operator(cov_cur)
        costs.append(operator.cost(cov_cur))
        dists.append(dBW(cov_cur, cov_prev))
        d_cost = np.abs(costs[-1] - costs[-2])
        if d_cost < 1e-10:
            n_conv.append(k + 1)
            if np.allclose(
                cov_cur,
                ot.gaussian.bures_wasserstein_barycenter(
                    np.zeros((n_sigmas, dim)),
                    operator._sigmas,
                    weights=operator._weights,
                    num_iter=10_000,
                    eps=1e-10,
                )[1],
            ):
                n_same_as_pot += 1
            break
        cov_prev = cov_cur
costs = np.array(costs)
delta_costs = costs[:-1] - costs[-1:]
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.plot(
    delta_costs,
)
ax.set_yscale("log")
ax.set_xlabel(r"$n_{iter}$")
ax.set_ylabel(r"$V(x_{k}) - V(x_{k-1})$")
ax = axs[1]
ax.plot(
    dists,
)
ax.set_yscale("log")
ax.set_xlabel(r"$n_{iter}$")
ax.set_ylabel(r"$W^2_2(x_{k}, x_{k-1})$")

print(f"d={dim} k={n_sigmas}")
print(f"Converged       {len(n_conv):-3d}/{n_data}")
print(f"Agrees with POT {n_same_as_pot:-3d}/{n_data}")
print(f"Avg. # iter.    {np.mean(n_conv):.2f}")

r_vis = n_sigmas

fig, ax = plt.subplots(1, 1)

t = np.linspace(
    0,
    2.0 * np.pi,
    101,
)
X0 = np.stack((np.cos(t), np.sin(t)), axis=0)
X_bc = sp.linalg.sqrtm(cov_cur)[:2, :2] @ X0
ax.plot(*X_bc)

for k, Sigma in enumerate(operator._sigmas):
    phi = k / n_sigmas * (2.0 * np.pi)
    shift = np.stack((r_vis * np.sin(phi), r_vis * np.cos(phi)), axis=0)
    sq_Sigma = sp.linalg.sqrtm(Sigma)
    X_cur = shift[:, np.newaxis] + (sq_Sigma[:2, :2] @ X0)
    ax.plot(*X_cur)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plt.show()
