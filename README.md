<!-- fpw documentation master file, created by
sphinx-quickstart on Wed Oct 30 17:14:49 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Welcome to fpw’s documentation!

Various statistical tasks, including sampling or computing Wasserstein barycenters, can be reformulated as fixed-point problems for operators on probability distributions:

$$
G: \mathcal{P}_2(\mathbb{R}^d) \to \mathbb{R} 
$$

$$
\rho_* : \ G(\rho_* ) = \rho_* 
$$

where $\mathcal{P}_2(\mathbb{R}^d)$ is a space of all probability measures over $\mathbb{R}^d$ with finite second moments, viewed as a metric space with respect to the 2-Wasserstein metric

$$
W^2_2(\rho_1, \rho_2) = \inf_{\pi \in \Pi(\rho_1, \rho_2)} \int \|x - y\|^2_2 d\pi(x, y)
$$

This infinite-dimensional metric space has a structure, similar to a Riemannian manifold.
The goal of this project is to identify interesting fixed-point problems and to provide accelerated iterative solution with Riemannian Anderson Mixing.

## Installation

```bash
pip install --upgrade fpw@git+https://github.com/viviaxenov/fpw
```

### Requirements

* [cvxpy](https://www.cvxpy.org/) for $l_\infty$ regularized minimization
* [torch](https://pytorch.org/)

Optional:

* [emcee](https://emcee.readthedocs.io/en/stable/) for sampling from general distributions
* [pymanopt](https://pymanopt.org/) for comparison with Riemannian minimization methods

# Gaussian case

We currently focus mostly on the Bures-Wasserstein manifold, i.e. the subset of Gaussian measures with zero mean (parametrized with their covariance matrices)

$$
\mathcal{N}_0^d = \{\Sigma: \Sigma^T = \Sigma \succ 0 \}
$$

The Wasserstein distance for Gaussians takes form

$$
W^2_2(\Sigma_0, \Sigma_1)  = \mathrm{Tr}{\Sigma_0} + \mathrm{Tr}{\Sigma_1} - 2\mathrm{Tr}{\left(\Sigma_0^{\frac{1}{2}}\Sigma_1 \Sigma_0^{\frac{1}{2}}\right)^{\frac{1}{2}}}
$$

$\mathcal{N}_0^d$ is a Riemannian manifold with tangent space at $\Sigma$ isomorphic to all symmetric matrices.
The scalar product takes form

$$
\langle U, V \rangle_\Sigma := \frac{1}{2}\mathrm{Tr}(U\Sigma V)
$$

Riemannian Anderson Mixing relies on keeping a set of historical vectors, which is transported to the tangent space of the current iterate with a *vector transport* mapping.
The update direction is then chosen based on a solution of a $l_\infty$ regularized least-squares problem in the tangent space.

## Code example

Here, a solution of the Wasserstein barycenter problem is presented.
[`Barycenter`](fpw.md#fpw.ProblemGaussian.Barycenter) defines the problem, including the relevant fixed-point operator.
We first solve the problem with Picard iteration $\Sigma_{k+1} = G(\Sigma_k)$.
The iteration is run until the fixed-point residual, which is an upper bound for $W_2(\Sigma_{k}, G(\Sigma_{k}))$ , reaches a prescribed tolerance.
Then, the accelerated solution is performed by [`BWRAMSolver`](fpw.md#module-fpw.BWRAMSolver).
The hyperparameters of the method are the number of historical vectors `history_len`, relaxation `relaxation` and the regularization factor in the least squares minimization subproblem `l_inf_bound_Gamma`.

```python
import numpy as np

from fpw import BWRAMSolver, dBW
from fpw.BWRAMSolver import BWRAMSolver
from fpw.ProblemGaussian import *


n_sigmas = 5
dim = 20

N_iter_max = 100
tol = 1e-8

problem_bc = Barycenter(n_sigmas, dim)
cov_init = problem_bc.get_initial_value()

cov_picard = cov_init.copy()

# Reference solution with Picard method
for k in range(N_iter_max):
    cov_next, residual = problem_bc.operator_and_residual(cov_picard)
    r_norm_sq = 0.5 * np.trace(residual @ cov_picard @ residual)
    r_norm = np.sqrt(r_norm_sq)
    if r_norm < tol:
        break
    cov_picard = cov_next
print(k)


# Solution with BWRAM
solver = BWRAMSolver(
    problem_bc,
    relaxation=1.0,
    l_inf_bound_Gamma=1.0,
    history_len=5,
)
cov_bwram = solver.iterate(cov_init, N_iter_max, tol)
```

If [pymanopt](https://pymanopt.org/) is installed, one can use `fpw.PymanoptInterface` to run Riemannian minimization methods and compare

```python
BW_manifold = problem_bc.base_manifold
cost_torch = pymanopt.function.pytorch(BW_manifold)(problem_bc.get_cost_torch())
pymanopt_problem = pymanopt.Problem(BW_manifold, cost_torch)
optimizer = pymanopt.optimizers.SteepestDescent(log_verbosity=1)
opt_result = optimizer.run(pymanopt_problem, initial_point=cov_init)
cov_pymanopt = opt_result.log["iterations"]["point"][-1]
```

## Citation

Currently submitted to NeurIPS

# General case

Solver for the general case can be found in `fpw.RAMSolver`, and the JAX implementation in `fpw.RAMSolverJAX`, with wrappers for operators in `fpw.utility`. This however is still TBD.
