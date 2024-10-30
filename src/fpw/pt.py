import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.stats import norm

import sys

# todo: control epsilon so that the matrix is bounded

# test fns;
# todo: should be variable in the PT method


def hat_fn(t: np.ndarray):
    return np.where(np.abs(t) < 1.0, np.exp(-1.0 / (1.0 - t**2)), 0.0)


def dt_hat_fn(t: np.ndarray):
    return -hat_fn(t) * 2.0 * t / (1.0 - t**2) ** 2


def ddt_hat_fn(t: np.ndarray):
    return hat_fn(t) * (
        (2.0 * t / (1.0 - t**2) ** 2) ** 2
        - 8.0 * t**2 / (1.0 - t**2) ** 3
        - 2.0 / (1.0 - t**2) ** 2
    )


# PT == solve ODE
# dudt_matrix @ {d/dt u} + u_matrix@u = 0
# d/dt u = - dudt_matrix^-1 @ u_matrix @ u


def dudt_matrix(xt: np.ndarray, epsilon: np.float64):
    """Compute the matrix in front of du/dt in the parallel transport equation

    Args:
        xt: position of the particle at time t; shape [N_particles; dim]
        epsilon: characterizes the size of the support of the test functions

    Returns:
        np.ndarray: shape [N_particles; dim; N_particles; dim]
    """
    assert epsilon > 0.0
    Xi = xt[:, np.newaxis, :]
    Xj = xt[np.newaxis, :, :]
    ksi = (Xi - Xj) / epsilon
    mat = hat_fn(ksi)

    return mat


def u_matrix(xt: np.ndarray, vt: np.ndarray, epsilon: np.float64):
    """Compute the matrix in front of du/dt in the parallel transport equation

    Args:
        xt: position of the particles at time t; shape [N_particles; dim]
        vt: velocities of the particles at time t; shape [N_particles; dim]
        epsilon: characterizes the size of the support of the test functions

    Returns:
        np.ndarray: shape [N_particles; dim; N_particles; dim]
    """
    assert epsilon > 0.0
    Xi = xt[:, np.newaxis, :]
    Xj = xt[np.newaxis, :, :]
    Vj = vt[np.newaxis, :, :]
    ksi = (Xi - Xj) / epsilon

    return dt_hat_fn(ksi) / epsilon * Vj


def ode_rhs_fn(x0: np.ndarray, x1: np.ndarray, epsilon: np.float64):
    N_particles, dim = x0.shape

    def _v(_t, _ut):
        u = _ut.reshape(N_particles, dim)
        xt = (1.0 - _t) * x0 + _t * x1
        # todo: compute jointly
        A = dudt_matrix(xt, epsilon)
        B = u_matrix(xt, v, epsilon)
        Bu = np.einsum("ijk,jk->ik", B, u)
        try:
            AinvBu = np.array(
                [sp.linalg.solve(A[:, :, k], Bu[:, k]) for k in range(dim)]
            )

        except np.linalg.LinAlgError:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.suptitle(f"{_t=:.2e}")

            axs[0].matshow(A[:, :, 0])
            axs[1].matshow(A[:, :, 1])
            plt.show()

            raise

        return -AinvBu.ravel()

    return _v


def parallel_transport(x0, x1, u0, epsilon):
    N_particles, dim = x0.shape
    assert x1.shape == (N_particles, dim)
    assert u0.shape == (N_particles, dim)
    assert epsilon > 0.0

    ode_rhs = ode_rhs_fn(x0, x1, epsilon)
    u_init = np.ravel(u0)

    ode_res = solve_ivp(
        ode_rhs,
        [
            0.0,
            1.0,
        ],
        u_init,
        rtol=1e-2,
        atol=1e-3,
    )

    return ode_res.y[:, -1].reshape(N_particles, dim)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 1001, endpoint=True)

    plt.plot(x, hat_fn(x), label="$\\varphi$")
    plt.plot(x, dt_hat_fn(x), label="$\\varphi^'$")
    # plt.plot(x, ddt_hat_fn(x), label="$\\varphi^{''}$")

    plt.title("Test functions and their derivatives")
    plt.grid()
    plt.legend()

    N_particles = 1000
    dim = 2
    epsilon = 1e-10
    x0 = norm.rvs(size=(N_particles, dim), random_state=1)

    m_new = np.array([2.0, 2.0])
    sigmas_new = (
        np.array(
            [
                [1.0, -0.3],
                [-0.3, 1.0],
            ]
        )
        / 1000.0
    )

    # Gaussian to Gaussian transport
    x1 = m_new + (sigmas_new @ x0.T).T

    # some other transport that is a gradient of a convex fn (thus optimal due to Brenier theorem)
    # c = np.ones(dim)
    # x1 += -np.exp(-c @ x0.T)[:, np.newaxis] * c[np.newaxis, :]

    x1 = x0 + np.sign(x0)

    v = x1 - x0  # optimal transport, McCann's interpolation, geodesic goes from 0 to 1

    pname = sys.argv[1] if len(sys.argv) > 1 else "const"
    if pname == "cn-test":
        epsilons = [10 ** (-p) for p in range(10)]
        xt = (x0 + x1) / 2.0
        mats = [dudt_matrix(xt, eps) for eps in epsilons]
        fig, ax = plt.subplots(1, 1)
        for k in range(dim):
            cns = [np.linalg.cond(mat[:, :, k]) for mat in mats]
            ax.plot(epsilons, cns, label=f"{k=}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()
        ax.set_xlabel("$r_c$")
        ax.set_ylabel("$\\mu(A_{\\frac{du}{dt}})$")
        plt.show()
        exit()
    if pname == "const":
        # Constant field
        angle = 3.0 * np.pi / 4.0
        direct = np.array([np.cos(angle), np.sin(angle)])
        u0 = np.zeros_like(v)
        u0[:, :] = direct[np.newaxis, :]
        ptitle = "Constant"
    elif pname == "tangent":
        # Tangent field
        u0 = v.copy()
        ptitle = "Tangent"
    elif pname == "random":
        # Random field
        u0 = norm.rvs(size=v.shape)
        ptitle = "Random"
    u1 = parallel_transport(x0, x1, u0, epsilon)

    fig, axs = plt.subplots(1, 1)
    axs.scatter(*x0.T, label="$\\mu_0 = N(0, 1)$")
    axs.scatter(*x1.T, label="$\\mu_1 = N(\\mu, \\Sigma)$")
    vf_scale = 50
    axs.quiver(
        *x0.T,
        *u0.T,
        color="tab:blue",
        width=0.001,
        scale=vf_scale,
        angles="xy",
    )
    axs.quiver(
        *x1.T,
        *u1.T,
        color="tab:orange",
        width=0.001,
        scale=vf_scale,
        angles="xy",
    )
    axs.quiver(
        *x0.T,
        *(x1 - x0).T,
        color="tab:grey",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        zorder=-1,
        label="OT coupling",
    )
    axs.legend()
    fig.suptitle(ptitle)
    fig.tight_layout()
    fig.savefig(f"{pname}.pdf")
    plt.show()
