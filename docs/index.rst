.. fpw documentation master file, created by
   sphinx-quickstart on Wed Oct 30 17:14:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fpw's documentation!
===============================

Various statistical tasks, including sampling or computing Wasserstein barycenters, can be reformulated as fixed-point problems for operators on probability distributions:

.. math::

   G: \mathcal{P}_2(\mathbb{R}^d) \to \mathbb{R} \\
   \text{Find } \rho_* : \ G(\rho_*) = \rho_*


where :math:`\mathcal{P}_2(\mathbb{R}^d)` is a space of all probability measures over :math:`\mathbb{R}^d` with finite second moments, viewed as a metric space with respect to the 2-Wasserstein metric 

.. math::

    W^2_2(\rho_1, \rho_2) = \inf_{\pi \in \Pi(\rho_1, \rho_2)} \int \|x - y\|^2_2 d\pi(x, y)


This infinite-dimensional metric space has a structure, similar to a Riemannian manifold.
The goal of this project is to identify interesting fixed-point problems and to provide accelerated iterative solution with Riemannian Anderson Mixing.


Requirements
^^^^^^^^^^^^^^
* `cvxpy <https://www.cvxpy.org/>`_ for :math:`l_\infty` regularized minimization
* `torch <https://pytorch.org/>`_ 

Optional:

* `emcee <https://emcee.readthedocs.io/en/stable/>`_ for sampling from general distributions
* `pymanopt <https://pymanopt.org/>`_ for comparison with Riemannian minimization methods


Gaussian case
==================

We currently focus mostly on the Bures-Wasserstein manifold, i.e. the subset of Gaussian measures with zero mean (parametrized with their covariance matrices)

.. math::

    \mathcal{N}_0^d = \{\Sigma: \Sigma^T = \Sigma \succ 0 \}


The Wasserstein distance for Gaussians takes form

.. math::

    W^2_2(\Sigma_0, \Sigma_1)  = \Tr{\Sigma_0} + \Tr{\Sigma_1} - 2\Tr{\left(\Sigma_0^{\frac{1}{2}}\Sigma_1 \Sigma_0^{\frac{1}{2}}\right)^{\frac{1}{2}}}

:math:`\mathcal{N}_0^d` is a Riemannian manifold with tangent space at :math:`\Sigma` isomorphic to all symmetric matrices. 
The scalar product takes form

.. math::

    \langle U, V \rangle_\Sigma := \frac{1}{2}\Tr(U\Sigma V)

Riemannian Anderson Mixing relies on keeping a set of historical vectors, which is transported to the tangent space of the current iterate with a *vector transport* mapping. 
The update direction is then chosen based on a solution of a :math:`l_\infty` regularized least-squares problem in the tangent space.

Code example
------------

Here, a solution of the Wasserstein barycenter problem is presented.
:py:class:`Barycenter <fpw.ProblemGaussian.Barycenter>` defines the problem, including the relevant fixed-point operator.
We first solve the problem with Picard iteration :math:`\Sigma_{k+1} = G(\Sigma_k)`.
The iteration is run until the fixed-point residual (which is an upper bound for :math:`W_2(\Sigma_k, G(\Sigma_k))`) reaches a prescribed tolerance.
Then, the accelerated solution is performed by :py:class:`BWRAMSolver <fpw.BWRAMSolver>`. 
The hyperparameters of the method are the number of historical vectors ``history_len``, relaxation ``relaxation`` and the regularization factor in the least squares minimization subproblem ``l_inf_bound_Gamma``.

.. literalinclude:: ../tests/example.py
   :language: python
   :linenos:
   :lines: 1-37

If `pymanopt <https://pymanopt.org/>`_ is installed, one can use :py:mod:`fpw.PymanoptInterface` to run Riemannian minimization methods and compare

.. literalinclude:: ../tests/example.py
   :language: python
   :linenos:
   :lineno-start: 50
   :lines: 50-55


Citation
--------

Currently submitted to NeurIPS

General case
==================

Solver for the general case can be found in :py:mod:`fpw.RAMSolver`, and the JAX implementation in :py:mod:`fpw.RAMSolverJAX`, with wrappers for operators in :py:mod:`fpw.utility`. This however is still TBD.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
