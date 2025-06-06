# src

* [fpw package](fpw.md)
  * [fpw.BWRAMSolver module](fpw.md#module-fpw.BWRAMSolver)
    * [`BWRAMSolver`](fpw.md#fpw.BWRAMSolver.BWRAMSolver)
      * [`BWRAMSolver.dim`](fpw.md#fpw.BWRAMSolver.BWRAMSolver.dim)
      * [`BWRAMSolver.n_particles`](fpw.md#fpw.BWRAMSolver.BWRAMSolver.n_particles)
      * [`BWRAMSolver.iterate()`](fpw.md#fpw.BWRAMSolver.BWRAMSolver.iterate)
      * [`BWRAMSolver.restart()`](fpw.md#fpw.BWRAMSolver.BWRAMSolver.restart)
    * [`Christoffel()`](fpw.md#fpw.BWRAMSolver.Christoffel)
    * [`check_cov()`](fpw.md#fpw.BWRAMSolver.check_cov)
    * [`dBW()`](fpw.md#fpw.BWRAMSolver.dBW)
    * [`one_step_approx()`](fpw.md#fpw.BWRAMSolver.one_step_approx)
    * [`parallel_transport()`](fpw.md#fpw.BWRAMSolver.parallel_transport)
    * [`project_on_tangent()`](fpw.md#fpw.BWRAMSolver.project_on_tangent)
    * [`rExpGaussian()`](fpw.md#fpw.BWRAMSolver.rExpGaussian)
    * [`to_Map()`](fpw.md#fpw.BWRAMSolver.to_Map)
    * [`to_dSigma()`](fpw.md#fpw.BWRAMSolver.to_dSigma)
    * [`vector_translation()`](fpw.md#fpw.BWRAMSolver.vector_translation)
  * [fpw.ProblemGaussian module](fpw.md#module-fpw.ProblemGaussian)
    * [`Barycenter`](fpw.md#fpw.ProblemGaussian.Barycenter)
      * [`Barycenter.cost()`](fpw.md#fpw.ProblemGaussian.Barycenter.cost)
      * [`Barycenter.get_cost_torch()`](fpw.md#fpw.ProblemGaussian.Barycenter.get_cost_torch)
      * [`Barycenter.get_initial_value()`](fpw.md#fpw.ProblemGaussian.Barycenter.get_initial_value)
      * [`Barycenter.n_sigmas`](fpw.md#fpw.ProblemGaussian.Barycenter.n_sigmas)
      * [`Barycenter.name`](fpw.md#fpw.ProblemGaussian.Barycenter.name)
      * [`Barycenter.operator_and_residual()`](fpw.md#fpw.ProblemGaussian.Barycenter.operator_and_residual)
      * [`Barycenter.residual()`](fpw.md#fpw.ProblemGaussian.Barycenter.residual)
    * [`EntropicBarycenter`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter)
      * [`EntropicBarycenter.cost()`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.cost)
      * [`EntropicBarycenter.get_cost_torch()`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.get_cost_torch)
      * [`EntropicBarycenter.get_initial_value()`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.get_initial_value)
      * [`EntropicBarycenter.n_sigmas`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.n_sigmas)
      * [`EntropicBarycenter.name`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.name)
      * [`EntropicBarycenter.residual()`](fpw.md#fpw.ProblemGaussian.EntropicBarycenter.residual)
    * [`Median`](fpw.md#fpw.ProblemGaussian.Median)
      * [`Median.cost()`](fpw.md#fpw.ProblemGaussian.Median.cost)
      * [`Median.get_cost_torch()`](fpw.md#fpw.ProblemGaussian.Median.get_cost_torch)
      * [`Median.get_initial_value()`](fpw.md#fpw.ProblemGaussian.Median.get_initial_value)
      * [`Median.n_sigmas`](fpw.md#fpw.ProblemGaussian.Median.n_sigmas)
      * [`Median.name`](fpw.md#fpw.ProblemGaussian.Median.name)
      * [`Median.residual()`](fpw.md#fpw.ProblemGaussian.Median.residual)
    * [`OUEvolution`](fpw.md#fpw.ProblemGaussian.OUEvolution)
      * [`OUEvolution.dt`](fpw.md#fpw.ProblemGaussian.OUEvolution.dt)
      * [`OUEvolution.name`](fpw.md#fpw.ProblemGaussian.OUEvolution.name)
    * [`Problem`](fpw.md#fpw.ProblemGaussian.Problem)
      * [`Problem.cost()`](fpw.md#fpw.ProblemGaussian.Problem.cost)
      * [`Problem.dim`](fpw.md#fpw.ProblemGaussian.Problem.dim)
      * [`Problem.get_initial_value()`](fpw.md#fpw.ProblemGaussian.Problem.get_initial_value)
      * [`Problem.get_solution_picard()`](fpw.md#fpw.ProblemGaussian.Problem.get_solution_picard)
      * [`Problem.name`](fpw.md#fpw.ProblemGaussian.Problem.name)
      * [`Problem.operator_and_residual()`](fpw.md#fpw.ProblemGaussian.Problem.operator_and_residual)
      * [`Problem.residual()`](fpw.md#fpw.ProblemGaussian.Problem.residual)
    * [`WGKL`](fpw.md#fpw.ProblemGaussian.WGKL)
      * [`WGKL.cost()`](fpw.md#fpw.ProblemGaussian.WGKL.cost)
      * [`WGKL.get_cost_torch()`](fpw.md#fpw.ProblemGaussian.WGKL.get_cost_torch)
      * [`WGKL.residual()`](fpw.md#fpw.ProblemGaussian.WGKL.residual)
    * [`barycenter_loss_vectorized()`](fpw.md#fpw.ProblemGaussian.barycenter_loss_vectorized)
    * [`entropic_barycenter_loss_vectorized()`](fpw.md#fpw.ProblemGaussian.entropic_barycenter_loss_vectorized)
    * [`median_loss_vectorized()`](fpw.md#fpw.ProblemGaussian.median_loss_vectorized)
  * [fpw.PymanoptInterface module](fpw.md#module-fpw.PymanoptInterface)
    * [`BuresWassersteinManifold`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold)
      * [`BuresWassersteinManifold.dist()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.dist)
      * [`BuresWassersteinManifold.exp()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.exp)
      * [`BuresWassersteinManifold.inner_product()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.inner_product)
      * [`BuresWassersteinManifold.log()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.log)
      * [`BuresWassersteinManifold.norm()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.norm)
      * [`BuresWassersteinManifold.projection()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.projection)
      * [`BuresWassersteinManifold.random_point()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.random_point)
      * [`BuresWassersteinManifold.random_tangent_vector()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.random_tangent_vector)
      * [`BuresWassersteinManifold.retraction()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.retraction)
      * [`BuresWassersteinManifold.zero_vector()`](fpw.md#fpw.PymanoptInterface.BuresWassersteinManifold.zero_vector)
    * [`to_Map()`](fpw.md#fpw.PymanoptInterface.to_Map)
    * [`to_dSigma()`](fpw.md#fpw.PymanoptInterface.to_dSigma)
  * [fpw.RAMSolver module](fpw.md#module-fpw.RAMSolver)
    * [`RAMSolver`](fpw.md#fpw.RAMSolver.RAMSolver)
      * [`RAMSolver.dim`](fpw.md#fpw.RAMSolver.RAMSolver.dim)
      * [`RAMSolver.n_particles`](fpw.md#fpw.RAMSolver.RAMSolver.n_particles)
      * [`RAMSolver.iterate()`](fpw.md#fpw.RAMSolver.RAMSolver.iterate)
      * [`RAMSolver.restart()`](fpw.md#fpw.RAMSolver.RAMSolver.restart)
  * [fpw.RAMSolverJAX module](fpw.md#module-fpw.RAMSolverJAX)
    * [`RAMSolverJAX`](fpw.md#fpw.RAMSolverJAX.RAMSolverJAX)
      * [`RAMSolverJAX.iterate()`](fpw.md#fpw.RAMSolverJAX.RAMSolverJAX.iterate)
      * [`RAMSolverJAX.restart()`](fpw.md#fpw.RAMSolverJAX.RAMSolverJAX.restart)
    * [`get_vector_transport()`](fpw.md#fpw.RAMSolverJAX.get_vector_transport)
    * [`vector_translation()`](fpw.md#fpw.RAMSolverJAX.vector_translation)
  * [fpw.pt module](fpw.md#module-fpw.pt)
    * [`ddt_hat_fn()`](fpw.md#fpw.pt.ddt_hat_fn)
    * [`dt_hat_fn()`](fpw.md#fpw.pt.dt_hat_fn)
    * [`dudt_matrix()`](fpw.md#fpw.pt.dudt_matrix)
    * [`hat_fn()`](fpw.md#fpw.pt.hat_fn)
    * [`ode_rhs_fn()`](fpw.md#fpw.pt.ode_rhs_fn)
    * [`parallel_transport()`](fpw.md#fpw.pt.parallel_transport)
    * [`u_matrix()`](fpw.md#fpw.pt.u_matrix)
  * [Module contents](fpw.md#module-fpw)
