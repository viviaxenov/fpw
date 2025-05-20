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

print("BWRAM coincides with Picard:")
print(np.allclose(cov_picard, cov_bwram))

# Solution with pymanopt
try:
    import pymanopt
    from fpw.PymanoptInterface import *
except ImportError:
    print("Cannot import PyManOpt, can't run test")
    exit()

BW_manifold = problem_bc.base_manifold
cost_torch = pymanopt.function.pytorch(BW_manifold)(problem_bc.get_cost_torch())
pymanopt_problem = pymanopt.Problem(BW_manifold, cost_torch)
optimizer = pymanopt.optimizers.SteepestDescent(log_verbosity=1)
opt_result = optimizer.run(pymanopt_problem, initial_point=cov_init)
cov_pymanopt = opt_result.log["iterations"]["point"][-1]

print("RGD coincides with Picard:")
print(np.allclose(cov_picard, cov_pymanopt))
