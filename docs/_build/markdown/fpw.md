# fpw package

## fpw.BWRAMSolver module

### *class* fpw.BWRAMSolver.BWRAMSolver(operator, relaxation=0.95, l_inf_bound_Gamma=0.1, history_len=2, vt_kind='one-step', r_threshold=None, restart_every=None, \*\*kwargs)

Bases: `object`

Approximate the fixed-point $\rho^*$  for an operator $F$ over the Bures-Wasserstein space of probability measures, i.e.

> $$
> F: \mathcal N_{0,d}(\mathbb{R}^d) \to \mathcal N_{0,d}(\mathbb{R}^d)

> \rho^*: \rho^* = F(\rho^*)
> $$

> with Riemannian(-like) Anderson Mixing scheme.
* **Parameters:**
  * **operator** (*Callable*) – Operator $F$, fixed point of which is in question
  * **relaxation** (*Union* *[**np.float64* *,* *Generator* *]*) – relaxation parameter, used at each iteration; constant of function of the step
  * **history_len** (*int*) – maximal number of previous iterates, used in the method
  * **vector_transport_kind** (*Literal* *[* *"parallel"* *,* *"translation"* *,* *"one_step"* *]*) – solver for intermediate vector transport subproblem
  * **l_inf_bound_Gamma** (*float*)
  * **vt_kind** (*Literal* *[* *'parallel'* *,*  *'translation'* *,*  *'one-step'* *]*)

#### dim

problem dimension

* **Type:**
  int

#### n_particles

number of particles in the sample approximating the current measure

* **Type:**
  int

#### iterate(x0, max_iter, residual_conv_tol)

* **Parameters:**
  * **x0** (*torch.Tensor*)
  * **max_iter** (*int*)
  * **residual_conv_tol** (*float64*)

#### restart(new_history_len=None, new_relaxation=None)

### fpw.BWRAMSolver.Christoffel(Sigma, X, Y)

* **Parameters:**
  * **Sigma** (*ndarray*)
  * **X** (*ndarray*)
  * **Y** (*ndarray*)

### fpw.BWRAMSolver.check_cov(Cov)

* **Parameters:**
  **Cov** (*ndarray*)

### fpw.BWRAMSolver.dBW(Cov_0, Cov_1)

### fpw.BWRAMSolver.one_step_approx(Sigma_0, Sigma_1, U0)

### fpw.BWRAMSolver.parallel_transport(Sigma_0, Sigma_1, U0, is_map=True)

### fpw.BWRAMSolver.project_on_tangent(U, Sigma)

### fpw.BWRAMSolver.rExpGaussian(V, Cov, is_map=True)

* **Parameters:**
  * **V** (*ndarray*)
  * **Cov** (*ndarray*)

### fpw.BWRAMSolver.to_Map(V, Cov)

* **Parameters:**
  * **V** (*ndarray*)
  * **Cov** (*ndarray*)

### fpw.BWRAMSolver.to_dSigma(U, Cov)

* **Parameters:**
  * **U** (*ndarray*)
  * **Cov** (*ndarray*)

### fpw.BWRAMSolver.vector_translation(Sigma_0, Sigma_1, U0)

## fpw.ProblemGaussian module

### *class* fpw.ProblemGaussian.Barycenter(n_sigmas=None, dim=None, weights=None, rs=1, \*\*kwargs)

Bases: [`Problem`](#fpw.ProblemGaussian.Problem)

Operator describing the Wasserstein barycenter problem

$$
\Sigma_* = \arg\min_{\Sigma \in \mathcal{N}_0^d} \sum_{k=1}^{n_\sigma} w_k W^2_2(\Sigma, \Sigma_k)
$$

* **Parameters:**
  * **n_sigmas** (*int*) – Number $n_\sigma$ of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution $\Sigma \sim \mathcal{W}(\operatorname{I}_d, d)$
  * **dim** (*int*) – dimension $d$ of the distributions
  * **weights** (*np.ndarray*) – weight vector $w_i > 0$. If not given, set to $w_1 = w_2 = \dots = w_{n_\sigma}$
  * **rs** (*int*) – random seed for the generation of the distribution
  * **\*\*kwargs** – args to randomly generate the target

#### cost(Sigma)

Cost for the operator, defined by a minimization problem

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  value of the cost function
* **Return type:**
  np.float64

#### get_cost_torch()

#### get_initial_value()

#### *property* n_sigmas

#### *property* name

#### operator_and_residual(Sigma)

#### residual(Sigma)

Riemannian logarithm of $G(\Sigma)$.

Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

$$
r: \mathcal{N}_0^d \ni \Sigma \mapsto Tan_{\Sigma}(\mathcal{N}_0^d)

G(\Sigma) = Exp_{\Sigma}(-r(\Sigma))
$$

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  a residual. It is an element of the tangent space at $\Sigma_*$, which is isomorphic to all symmetric matrices in $\mathbb{R}^{d\times d}$
* **Return type:**
  np.ndarray

### *class* fpw.ProblemGaussian.EntropicBarycenter(n_sigmas=None, dim=None, weights=None, rs=1, gamma=1e-2, \*\*kwargs)

Bases: [`Barycenter`](#fpw.ProblemGaussian.Barycenter)

Operator describing the entropy-regularized Wasserstein barycenter problem

$$
\Sigma_* = \arg\min_{\Sigma \in \mathcal{N}_0^d} \sum_{k=1}^{n_\sigma} w_k W_2(\Sigma, \Sigma_k) + \gamma\operatorname{KL}(\Sigma|I_d)
$$

* **Parameters:**
  * **n_sigmas** (*int*) – Number $n_\sigma$ of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution $\Sigma \sim \mathcal{W}(I_d, d)$
  * **dim** (*int*) – dimension $d$ of the distributions
  * **weights** (*np.ndarray*) – weight vector $w_i > 0$. If not given, set to $w_1 = w_2 = \dots = w_{n_\sigma}$
  * **rs** (*int*) – random seed for the generation of the distribution
  * **gamma** (*np.float64*) – the regularization parameter $\gamma$
  * **\*\*kwargs** – args to randomly generate the target

#### cost(Sigma)

Cost for the operator, defined by a minimization problem

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  value of the cost function
* **Return type:**
  np.float64

#### get_cost_torch()

#### get_initial_value()

#### *property* n_sigmas

#### *property* name

#### residual(Sigma)

Riemannian logarithm of $G(\Sigma)$.

Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

$$
r: \mathcal{N}_0^d \ni \Sigma \mapsto Tan_{\Sigma}(\mathcal{N}_0^d)

G(\Sigma) = Exp_{\Sigma}(-r(\Sigma))
$$

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  a residual. It is an element of the tangent space at $\Sigma_*$, which is isomorphic to all symmetric matrices in $\mathbb{R}^{d\times d}$
* **Return type:**
  np.ndarray

### *class* fpw.ProblemGaussian.Median(n_sigmas=None, dim=None, weights=None, rs=1, eps=1e-2, scaling=1.0, \*\*kwargs)

Bases: [`Problem`](#fpw.ProblemGaussian.Problem)

Operator describing the Wasserstein geometric median problem

$$
\Sigma_* = \arg\min_{\Sigma \in \mathcal{N}_0^d} \sum_{k=1}^{n_\sigma} w_k W_2(\Sigma, \Sigma_k)
$$

For the purpose of regularizing the problem, we consider the smoothed version of the problem

$$
\Sigma_* = \arg\min_{\Sigma \in \mathcal{N}_0^d} \sum_{k=1}^{n_\sigma} w_k \sqrt{W^2_2 + \varepsilon}(\Sigma, \Sigma_k)
$$

* **Parameters:**
  * **n_sigmas** (*int*) – Number $n_\sigma$ of distributions. The distributions for the test are taken i.i.d. from the Wishart distribution $\Sigma \sim \mathcal{W}(\operatorname{I}_d, d)$
  * **dim** (*int*) – dimension $d$ of the distributions
  * **weights** (*np.ndarray*) – weight vector $w_i > 0$. If not given, set to $w_1 = w_2 = \dots = w_{n_\sigma}$
  * **rs** (*int*) – random seed for the generation of the distribution
  * **eps** (*np.float64*) – the smoothing parameter $\varepsilon$
  * **scaling** (*np.float64*) – the stepsize parameter s
  * **\*\*kwargs** – args to randomly generate the target

#### cost(Sigma)

Cost for the operator, defined by a minimization problem

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  value of the cost function
* **Return type:**
  np.float64

#### get_cost_torch()

#### get_initial_value()

#### *property* n_sigmas

#### *property* name

#### residual(Sigma)

Riemannian logarithm of $G(\Sigma)$.

Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

$$
r: \mathcal{N}_0^d \ni \Sigma \mapsto Tan_{\Sigma}(\mathcal{N}_0^d)

G(\Sigma) = Exp_{\Sigma}(-r(\Sigma))
$$

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  a residual. It is an element of the tangent space at $\Sigma_*$, which is isomorphic to all symmetric matrices in $\mathbb{R}^{d\times d}$
* **Return type:**
  np.ndarray

### *class* fpw.ProblemGaussian.OUEvolution(target=None, dt=0.1, \*\*kwargs)

Bases: [`Problem`](#fpw.ProblemGaussian.Problem)

Operator describing the dynamic of the Ornstein-Uhlenbeck process with invariant distribution $\Sigma_*$

* **Parameters:**
  * **target** (*np.ndarray*) – The invariant distribution
  * **dt** (*np.float64*) – The time step
  * **\*\*kwargs** – args to randomly generate the target

#### *property* dt

#### *property* name

### *class* fpw.ProblemGaussian.Problem(dim=None, vt_kind='one-step', \*\*kwargs)

Bases: `object`

Deals with fixed-point problems in the Bures-Wasserstein space: find the fixed-point $\Sigma_*$  for an operator $G$ over the Bures-Wasserstein space $\mathcal{N}_0^d$.

Consider the space of positive-definite $d\times d$ matrices

$$
\mathcal{N}_0^d = \{\mathcal{N}(0, \Sigma), 0 \prec \Sigma \in \mathbb{R}^{d\times d}\}
$$

endowed with the Wasserstein distance

$$
W^2_2(\Sigma_0, \Sigma_1)  = \Tr{\Sigma_0} + \Tr{\Sigma_1} - 2\Tr{\left(\Sigma_0^{\frac{1}{2}}\Sigma_1 \Sigma_0^{\frac{1}{2}}\right)^{\frac{1}{2}}}
$$

Then consider an operator $G: \mathcal{N}_0^d \to \mathcal{N}_0^d$ and the following fixed-point for it: find $\Sigma_*$ such that

$$
\Sigma_* = G(\Sigma_*)
$$

If there is a smooth functional $f: \mathcal{N}_0^d \to \mathbb{R}$, then the problem of finding the critical point of this functional is equaivalent to a fixed-point problem

$$
\partial_W f(\Sigma_*) = 0 \Longleftrightarrow Exp_{\Sigma_*}(\partial_W f(\Sigma_*)) = \Sigma_*
$$

This class wraps the operator $G$ and, in case of equivalence to a functional minimization problem, also $f, \nabla f$

#### cost(Sigma)

Cost for the operator, defined by a minimization problem

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  value of the cost function
* **Return type:**
  np.float64

#### *property* dim

#### get_initial_value()

* **Return type:**
  *ndarray*

#### get_solution_picard(N_steps_max=10_000, r_min=1e-10, cost_min=-np.inf, \*\*kwargs)

* **Parameters:**
  * **N_steps_max** (*int*)
  * **r_min** (*float64*)
  * **cost_min** (*float64*)

#### *property* name

#### operator_and_residual(Sigma)

* **Parameters:**
  **Sigma** (*ndarray*)
* **Return type:**
  *Tuple*[*ndarray*, *ndarray*]

#### residual(Sigma)

Riemannian logarithm of $G(\Sigma)$.

Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

$$
r: \mathcal{N}_0^d \ni \Sigma \mapsto Tan_{\Sigma}(\mathcal{N}_0^d)

G(\Sigma) = Exp_{\Sigma}(-r(\Sigma))
$$

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  a residual. It is an element of the tangent space at $\Sigma_*$, which is isomorphic to all symmetric matrices in $\mathbb{R}^{d\times d}$
* **Return type:**
  np.ndarray

### *class* fpw.ProblemGaussian.WGKL(target=None, scaling=1.0, \*\*kwargs)

Bases: [`Problem`](#fpw.ProblemGaussian.Problem)

Operator describing the minimization of $\operatorname{KL}(\cdot|\Sigma_*)$ for target distribution $\Sigma_*$

* **Parameters:**
  * **target** (*np.ndarray*) – The invariant distribution
  * **scaling** (*np.float64*) – Scale the Wasserstein gradient of $\operatorname{KL}(\cdot|\Sigma_*)$ by this value
  * **\*\*kwargs** – args to randomly generate the target

#### cost(Sigma)

Cost for the operator, defined by a minimization problem

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  value of the cost function
* **Return type:**
  np.float64

#### get_cost_torch()

#### residual(Sigma)

Riemannian logarithm of $G(\Sigma)$.

Some fixed point operators can be given in the form of residual, i.e. a mapping to the tangent space

$$
r: \mathcal{N}_0^d \ni \Sigma \mapsto Tan_{\Sigma}(\mathcal{N}_0^d)

G(\Sigma) = Exp_{\Sigma}(-r(\Sigma))
$$

* **Parameters:**
  **Sigma** (*np.ndarray*) – a covariance matrix $\Sigma \in \mathcal{N}_0^d$
* **Returns:**
  a residual. It is an element of the tangent space at $\Sigma_*$, which is isomorphic to all symmetric matrices in $\mathbb{R}^{d\times d}$
* **Return type:**
  np.ndarray

### fpw.ProblemGaussian.barycenter_loss_vectorized(Sigmas, weights, Sigma)

Vectorized implementation of the BW barycenter loss with proper matrix square roots.

### fpw.ProblemGaussian.entropic_barycenter_loss_vectorized(Sigmas, weights, gamma, Sigma)

### fpw.ProblemGaussian.median_loss_vectorized(Sigmas, weights, eps, scaling, Sigma)

Vectorized implementation of the BW barycenter loss with proper matrix square roots.

## fpw.PymanoptInterface module

### *class* fpw.PymanoptInterface.BuresWassersteinManifold(\*args, \*\*kwargs)

Bases: `RiemannianSubmanifold`

#### dist(Cov_1)

#### exp(Cov, V)

#### inner_product(point, tangent_vector_a, tangent_vector_b)

* **Parameters:**
  * **point** (*ndarray*)
  * **tangent_vector_a** (*ndarray*)
  * **tangent_vector_b** (*ndarray*)
* **Return type:**
  float

#### log(Sigma_0, Sigma_1)

#### norm(point, tangent_vector)

#### projection(Sigma, U)

#### random_point()

#### random_tangent_vector(point)

#### retraction(Cov, V)

#### zero_vector(point)

### fpw.PymanoptInterface.to_Map(V, Cov)

* **Parameters:**
  * **V** (*ndarray*)
  * **Cov** (*ndarray*)

### fpw.PymanoptInterface.to_dSigma(U, Cov)

* **Parameters:**
  * **U** (*ndarray*)
  * **Cov** (*ndarray*)

## fpw.RAMSolver module

### *class* fpw.RAMSolver.RAMSolver(operator, relaxation=0.95, l_inf_bound_Gamma=0.1, history_len=2, vector_transport_kind='translation', vt_args=[], vt_kwargs={}, reg_sinkhorn=0.2, sinkhorn_args=[], sinkhorn_kwargs={})

Bases: `object`

Approximate the fixed-point $\rho^*$  for an operator $F$ over the Wasserstein space of probability measures, i.e.

> $$
> F: \mathcal P^2(\mathbb{R}^d) \to \mathcal P^2(\mathbb{R}^d)

> \rho^*: \rho^* = F(\rho^*)
> $$

> with Riemannian(-like) Anderson Mixing scheme.
* **Parameters:**
  * **operator** (*Callable*) – Operator $F$, fixed point of which is in question
  * **relaxation** (*Union* *[**np.float64* *,* *Generator* *]*) – relaxation parameter, used at each iteration; constant of function of the step
  * **history_len** (*int*) – maximal number of previous iterates, used in the method
  * **vector_transport_kind** (*Literal* *[* *"translation"* *,*  *"parallel"* *]*) – solver for intermediate vector transport subproblem
  * **vt_args** (*List*) – additional arguments for the vector transport solver
  * **vt_kwargs** (*Dict*) – additional arguments for the vector transport solver
  * **reg_sinkhorn** (*float*) – regularization for Sinkhorn OT solver
  * **sinkhorn_args** (*List*) – additional arguments for the OT solver
  * **sinkhorn_kwargs** (*Dict*) – additional arguments for the OT solver
  * **l_inf_bound_Gamma** (*float*)

#### dim

problem dimension

* **Type:**
  int

#### n_particles

number of particles in the sample approximating the current measure

* **Type:**
  int

#### iterate(x0, max_iter, residual_conv_tol)

* **Parameters:**
  * **x0** (*torch.Tensor*)
  * **max_iter** (*int*)
  * **residual_conv_tol** (*float64*)

#### restart(new_history_len=None, new_relaxation=None)

## fpw.RAMSolverJAX module

### *class* fpw.RAMSolverJAX.RAMSolverJAX(operator, x0, relaxation=0.95, history_len=2, vector_transport_kind='translation', vt_args=[], vt_kwargs={}, reg_sinkhorn=0.1, sinkhorn_args=[], sinkhorn_kwargs={})

Bases: `object`

* **Parameters:**
  * **operator** (*Callable*)
  * **x0** (*jax.numpy.array*)
  * **relaxation** (*jax.numpy.float64* *|* *Generator*)
  * **history_len** (*int*)
  * **vector_transport_kind** (*Literal* *[* *'translation'* *,*  *'parallel'* *]*)
  * **vt_args** (*List*)
  * **vt_kwargs** (*Dict*)
  * **reg_sinkhorn** (*jax.numpy.float64*)
  * **sinkhorn_args** (*List*)
  * **sinkhorn_kwargs** (*Dict*)

#### iterate(x0, max_iter, residual_conv_tol)

* **Parameters:**
  * **x0** (*jax.numpy.ndarray*)
  * **max_iter** (*int*)
  * **residual_conv_tol** (*jax.numpy.float64*)

#### restart(new_history_len=None, new_relaxation=None)

### fpw.RAMSolverJAX.get_vector_transport(kind='translation', \*args, \*\*kwargs)

### fpw.RAMSolverJAX.vector_translation(x0, x1, u0)

## fpw.pt module

### fpw.pt.ddt_hat_fn(t)

* **Parameters:**
  **t** (*ndarray*)

### fpw.pt.dt_hat_fn(t)

* **Parameters:**
  **t** (*ndarray*)

### fpw.pt.dudt_matrix(xt, epsilon)

Compute the matrix in front of du/dt in the parallel transport equation

* **Parameters:**
  * **xt** (*ndarray*) – position of the particle at time t; shape [N_particles; dim]
  * **epsilon** (*float64*) – characterizes the size of the support of the test functions
* **Returns:**
  shape [N_particles; dim; N_particles; dim]
* **Return type:**
  np.ndarray

### fpw.pt.hat_fn(t)

* **Parameters:**
  **t** (*ndarray*)

### fpw.pt.ode_rhs_fn(x0, x1, epsilon)

* **Parameters:**
  * **x0** (*ndarray*)
  * **x1** (*ndarray*)
  * **epsilon** (*float64*)

### fpw.pt.parallel_transport(x0, x1, u0, epsilon)

### fpw.pt.u_matrix(xt, vt, epsilon)

Compute the matrix in front of du/dt in the parallel transport equation

* **Parameters:**
  * **xt** (*ndarray*) – position of the particles at time t; shape [N_particles; dim]
  * **vt** (*ndarray*) – velocities of the particles at time t; shape [N_particles; dim]
  * **epsilon** (*float64*) – characterizes the size of the support of the test functions
* **Returns:**
  shape [N_particles; dim; N_particles; dim]
* **Return type:**
  np.ndarray

## Module contents
