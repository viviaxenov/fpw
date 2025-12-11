import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import jax
from flax import nnx

from jax.typing import ArrayLike
from typing import Tuple, Callable, Union

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

import operator

from time import perf_counter

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from fpw.kernelRAMSolver import *


def _load_from_ucml_repo(name: str):
    import ucimlrepo

    dataset = ucimlrepo.fetch_ucirepo(name=name)
    X = jnp.array(dataset.data.features.to_numpy())
    y = jnp.array(dataset.data.targets.to_numpy())
    return X, y


def _load_local(name: str):
    path = os.path.join("./datasets/", name + ".txt")
    dataset = np.loadtxt(path)
    X, y = dataset[:, :-1], dataset[:, -1]

    return jnp.array(X), jnp.array(y)


def _normalize(
    X: Union[np.ndarray, jnp.ndarray],
    X_mean=None,
    X_std=None,
) -> Tuple[Union[jnp.ndarray, np.ndarray], float, float]:
    if X_mean is None:
        X_mean = X.mean(axis=0)
    if X_std is None:
        X_std = X.std(axis=0)

    X_normalized = (X - X_mean) / X_std**0.5

    return X_normalized, X_mean, X_std


class FullyConnected(nnx.Module):
    def __init__(
        self,
        dim_input: int,
        rngs,
        dim_output: int = 1,
        dim_hidden: int = 50,
        n_hidden: int = 1,
    ):
        self.layers = [
            nnx.Linear(
                dim_input,
                dim_hidden,
                rngs=rngs,
                kernel_init=nnx.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_init=nnx.initializers.constant(
                    0.0,
                ),
            )
        ]
        for _ in range(n_hidden - 1):
            self.layers.append(
                nnx.Linear(
                    dim_hidden,
                    dim_hidden,
                    rngs=rngs,
                    kernel_init=nnx.initializers.variance_scaling(
                        scale=1.0,
                        mode="fan_in",
                        distribution="normal",
                    ),
                    bias_init=nnx.initializers.constant(
                        0.0,
                    ),
                )
            )
        self.layers.append(
            nnx.Linear(
                dim_hidden,
                dim_output,
                rngs=rngs,
                kernel_init=nnx.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_init=nnx.initializers.constant(
                    0.0,
                ),
            )
        )

    def __call__(
        self,
        X: ArrayLike,
    ):
        """Evaluate NN prediction"""
        for layer in self.layers[:-1]:
            X = nnx.relu(layer(X))
        y = self.layers[-1](X)
        return y


class BayesianNetworkRegression:
    def __init__(
        self,
        dataset: str,
        architecture: nnx.Module,
        dataset_src: Literal["local", "remote"] = "local",
        alpha_lambda_0=1.0,
        alpha_gamma_0=1.0,
        beta_lambda_0=0.1,
        beta_gamma_0=0.1,
        random_seed=1,
    ):

        self.rngs = nnx.Rngs(random_seed)

        X, y = (
            _load_local(dataset)
            if dataset_src == "local"
            else _load_from_ucml_repo(dataset)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=random_seed
        )

        # dataset normalization
        self.X, self.X_mean, self.X_std = _normalize(X_train)
        self.y, self.y_mean, self.y_std = _normalize(y_train)

        self.X_test, _, _ = _normalize(X_test, self.X_mean, self.X_std)
        self.y_test, _, _ = _normalize(y_test, self.y_mean, self.y_std)

        self.architecture = architecture
        self.dim = self.X.shape[-1]
        self.nn = self.architecture(self.dim, self.rngs)

        # Parameters of the hyperprior distribution
        # on gamma (inv. covariance of measurement noise)
        self.alpha_gamma_0 = alpha_gamma_0
        self.beta_gamma_0 = beta_gamma_0
        # and lambda (inv. covariance on prior weights)
        self.alpha_lambda_0 = alpha_lambda_0
        self.beta_lambda_0 = beta_lambda_0

        self.rs = random_seed
        self.split = jax.random.key(random_seed)

        gd, par = nnx.split(self.nn)
        self.n_nn_params = sum([np.prod(leaf.shape) for leaf in jax.tree.leaves(par)])
        self.n_total_params = self.n_nn_params + 2

        leaves, tree_structure = jax.tree.flatten(par)
        self._tree_structure = tree_structure
        self._leaf_shapes = [leaf.shape for leaf in leaves]
        leaf_sizes = [leaf.size for leaf in leaves]
        self._split_indices = np.cumsum(leaf_sizes)[:-1]
        self._dtype = leaves[0].dtype if leaves else jnp.float32

    def flatten_params(self, params):
        leaves = jax.tree.leaves(params)
        flat_arrays = [jnp.ravel(leaf) for leaf in leaves]
        return jnp.concatenate(flat_arrays)

    def unflatten_params(self, params_flat):
        log_gamma, log_lambda = params_flat[:2]
        params_nn = params_flat[2:]
        split_arrays = jnp.split(params_nn, self._split_indices)
        reconstructed_leaves = [
            arr.reshape(shape) for arr, shape in zip(split_arrays, self._leaf_shapes)
        ]
        return (
            log_gamma,
            log_lambda,
            jax.tree.unflatten(self._tree_structure, reconstructed_leaves),
        )

    def prediction(
        self,
        X: jnp.ndarray,
        nn_params=None,
    ):
        graphdef, nn_params_cur = nnx.split(self.nn)
        if nn_params is None:
            nn_params = nn_params_cur
        nn_cur = nnx.merge(graphdef, nn_params)

        def eval_nn(X_):
            return nn_cur(X_)

        return nnx.vmap(eval_nn)(X)

    def log_posterior(
        self, log_gamma: jnp.float64, log_lambda: jnp.float64, nn_params=None
    ) -> jnp.float64:
        """Evaluates the posterior distribution

        Params:
            log_gamma: log of covariance of the measurement noise
            log_lambda: log of coviariance of the prior on the weights
            nn_params: weights

        Returns:
            log_posterior for the set of parameters (gamma, lambda, nn_params)
        """
        if nn_params is None:
            gd, nn_params_cur = nnx.split(self.nn)
            nn_params = nn_params_cur

        gamma = jnp.exp(log_gamma)
        lambda_ = jnp.exp(log_lambda)

        y_pred = self.prediction(self.X, nn_params)

        log_likelihood = -0.5 * (
            ((self.y - y_pred) ** 2).sum() * gamma - self.y.shape[0] * log_gamma
        )
        log_prior_weights = -0.5 * (
            jax.tree.reduce_associative(
                operator.add,
                jax.tree.map(
                    lambda _x: jnp.sum(_x**2), nn_params
                ),  # since leaves are arrays, have to sum within the arrays first
            )
            * lambda_
            - self.n_nn_params * log_lambda
        )
        log_prior_lambda = jsp.stats.gamma.logpdf(
            lambda_, self.alpha_lambda_0, scale=1.0 / self.beta_lambda_0
        )
        log_prior_gamma = jsp.stats.gamma.logpdf(
            gamma, self.alpha_gamma_0, scale=1.0 / self.beta_gamma_0
        )

        # jax.debug.print(
        #     "lp_gamma = {0:2e}, lp_lambda = {1:2e}, lp_weights = {2:.2e} ll = {3:.2e}",
        #     log_prior_gamma,
        #     log_prior_lambda,
        #     log_prior_weights,
        #     log_likelihood,
        # )

        return log_likelihood + log_prior_weights + log_prior_gamma + log_prior_lambda

    def test_rmse(
        self, log_gamma: jnp.float64, log_lambda: jnp.float64, nn_params=None
    ):
        y_pred = self.prediction(self.X_test, nn_params)

        return jnp.sqrt(((y_pred - self.y_test) ** 2).mean())

    # TODO: how to manage random number generators???
    def sample_prior(
        self,
    ):
        key1, key2, split = jax.random.split(self.split, num=3)
        gamma = jax.random.gamma(key1, self.alpha_gamma_0) / self.beta_gamma_0
        lambda_ = jax.random.gamma(key2, self.alpha_lambda_0) / self.beta_lambda_0

        gd, nn_params_base = nnx.split(self.nn)
        td_params = jax.tree.structure(nn_params_base)

        new_leaves = []
        for leaf in jax.tree.leaves(nn_params_base):
            key, split = jax.random.split(split)
            new_leaves.append(
                jax.random.normal(key=key, shape=leaf.shape) / lambda_**0.5
            )

        new_params = jax.tree.unflatten(td_params, new_leaves)

        self.split = split
        return jnp.log(gamma), jnp.log(lambda_), new_params

    def sample_prior_flat_params(self, n_samples: int = 1):
        def one_sample(split, sample):
            key1, key2, key3, split = jax.random.split(split, num=4)

            gamma = jax.random.gamma(key1, self.alpha_gamma_0) / self.beta_gamma_0
            lambda_ = jax.random.gamma(key2, self.alpha_lambda_0) / self.beta_lambda_0

            # jax.debug.print("lambda = {0:.2e}, gamma = {1:.2e}", lambda_, gamma)
            w = jax.random.normal(key=key3, shape=(self.n_nn_params,)) / jnp.sqrt(700)
            return split, jnp.concatenate([jnp.log(jnp.array([gamma, lambda_])), w])

        split, sample = jax.lax.scan(one_sample, self.split, length=n_samples)
        self.split = split
        return sample

    def init_samples(self, n_samples: int = 1):
        def one_sample(rngs, *args):
            gamma = rngs.gamma(self.alpha_gamma_0) / self.beta_gamma_0
            lambda_ = rngs.gamma(self.alpha_lambda_0) / self.beta_lambda_0

            # jax.debug.print("lambda = {0:.2e}, gamma = {1:.2e}", lambda_, gamma)
            # as in Viktor's implementation, the weights have initial covariance \sqrt{d_input + 1} and biases are set to zero
            nn_cur = self.architecture(self.dim, rngs)
            _, state = nnx.split(nn_cur)

            return self.flatten_params((jnp.log(gamma), jnp.log(lambda_), state))

        arr_rngs = self.rngs.fork(split=n_samples)

        sample = nnx.vmap(one_sample)(arr_rngs)

        return sample


if __name__ == "__main__":

    bayes_nn = BayesianNetworkRegression(
        # "Concrete Compressive Strength",
        "wine",
        FullyConnected,
    )
    params_init = bayes_nn.sample_prior()

    params_flat = bayes_nn.flatten_params(params_init)
    print(params_flat.shape)
    params_reconstructed = bayes_nn.unflatten_params(params_flat)

    def compare_trees(a, b):
        return all(
            jnp.allclose(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
        )

    def log_posterior_flat(params_flat):
        return bayes_nn.log_posterior(*bayes_nn.unflatten_params(params_flat))

    @nnx.vmap
    def test_rmse_flat(params_flat):
        return bayes_nn.test_rmse(*bayes_nn.unflatten_params(params_flat))

    lp_flat_vmap = jax.vmap(log_posterior_flat)
    lp_metric = nnx.jit(lambda _X: lp_flat_vmap(_X).max())
    rmse_metric = nnx.jit(lambda _X: test_rmse_flat(_X).min())

    params_flat = bayes_nn.sample_prior_flat_params(10)

    def sample_langevin(
        log_posterior: Callable,
        sample_init: jnp.ndarray,
        tau: jnp.float_ = 0.01,
        N_iter=100,
        rs=1,
    ):
        key = jax.random.key(rs)
        value_and_grad_jitted = nnx.jit(nnx.vmap(nnx.value_and_grad(log_posterior)))
        vals = []
        rmses = []
        sample_cur = sample_init.copy()
        for _ in range(N_iter):
            v, g = value_and_grad_jitted(sample_cur)
            vals.append(v.mean())
            rmses.append(test_rmse_flat(sample_cur).mean())

            key, split = jax.random.split(key)
            noise = jax.random.normal(split, shape=sample_cur.shape)

            sample_cur = sample_cur + tau * g + (2.0 * tau) ** 0.5 * noise

        return sample_cur, vals, rmses

    def sample_SVGD(
        log_posterior: Callable,
        kern: Callable,
        sample_init: jnp.ndarray,
        tau: jnp.float_ = 0.01,
        N_iter=100,
    ):
        x_SVGD = sample_init.copy()
        oper = getOperatorSteinGradKL(log_posterior, -tau)

        def step_SVGD(carry, *args):
            x_SVGD = carry
            sg = oper(x_SVGD)
            G = pairwiseScalarProductOfBasisVectors(x_SVGD, x_SVGD, kern)
            v = evalTangent(x_SVGD, sg, x_SVGD, kern)

            x_SVGD += v

            d_rkhs = norm_rkhs(sg, G)
            d_l2 = norm_l2(v)
            lp_cur = lp_metric(x_SVGD)
            rmse_cur = rmse_metric(x_SVGD)

            return x_SVGD, (d_rkhs, d_l2, lp_cur, rmse_cur)

        x_SVGD, metrics = jax.lax.scan(step_SVGD, x_SVGD, length=N_iter)

        return x_SVGD, metrics

    def sample_kRAM(
        log_posterior: Callable,
        kern: Callable,
        sample_init: jnp.ndarray,
        tau: jnp.float_ = 0.01,
        N_iter=100,
    ):
        def gen_regularization():
            k = 0
            while True:
                yield 1e0 + 1e3 ** (-k)
                k += 1

        x0 = sample_init.copy()
        oper = getOperatorSteinGradKL(log_posterior, -tau)

        solver = kernelRAMSolver(
            oper,
            kern,
            relaxation=1.00,
            l2_regularization=1e0,
            history_len=8,
            metrics=(lp_metric, rmse_metric),
        )

        t = perf_counter()
        solver, metric_vals = solver.iterate(x0, N_iter)

        dt = perf_counter() - t
        print(f"Total time {dt = :.2e}; avg. per iter {dt / N_iter:.2e}")

        return solver._x_cur, metric_vals

    #  sample_init = bayes_nn.sample_prior_flat_params(10)
    sample_init = bayes_nn.init_samples(10)
    bandwidth = bandwidth_median(sample_init)
    print(f"{bandwidth=:.2e}", flush=True)
    kern = lambda _x1, _x2: jnp.exp(-0.5 * ((_x1 - _x2) ** 2).sum() / bandwidth**2)
    sample_init, _ = sample_SVGD(
        log_posterior_flat, kern, sample_init, tau=1e-7, N_iter=10
    )

    N_iter = 2000
    print("Starting SVGD", flush=True)
    _, (r_rkhs_SVGD, dx_l2_SVGD, vals_SVGD, rmses_SVGD) = sample_SVGD(
        log_posterior_flat, kern, sample_init, tau=5e-6, N_iter=N_iter
    )
    print("             ... Done", flush=True)
    print("Starting kRAM", flush=True)
    with jax.profiler.trace("/tmp/profile-data"):
        _, (r_rkhs_kRAM, dx_l2_kRAM, vals_kRAM, rmses_kRAM) = sample_kRAM(
            log_posterior_flat, kern, sample_init, tau=1e-5, N_iter=N_iter
        )
    print("             ... Done", flush=True)

    # vals_floor = np.min(np.minimum(vals_SVGD, vals_kRAM))
    # vals_SVGD -= vals_floor
    # vals_kRAM -= vals_floor

    fig, axs = plt.subplots(2, 2, sharex=True)

    ax = axs[0, 0]
    ax.set_title(r"Residual, $\| \cdot \|_{RKHS}$")
    ax.plot(r_rkhs_SVGD, "b-", label="SVGD")
    ax.plot(r_rkhs_kRAM, "r-", label="kRAM")
    ax.set_yscale("log")

    ax = axs[1, 0]
    ax.set_title(r"Size of step, $\| \cdot \|_{L_2}$")
    ax.plot(dx_l2_SVGD, "b-", label="SVGD")
    ax.plot(dx_l2_kRAM, "r-", label="kRAM")
    ax.set_yscale("log")

    ax = axs[0, 1]
    ax.set_title("$\\log \\rho_\\infty$")
    ax.plot(vals_SVGD, "b-", label="SVGD")
    ax.plot(vals_kRAM, "r-", label="kRAM")

    ax = axs[1, 1]
    ax.set_title("Test RMSE (mean over ensemble)")
    ax.plot(rmses_SVGD, "b-", label="SVGD")
    ax.plot(rmses_kRAM, "r-", label="kRAM")
    ax.set_yscale("log")
    ax.legend()

    for ax in axs.ravel():
        ax.grid()

    fig.tight_layout()
    fig.savefig("bayes_nn.pdf")

    plt.show()
