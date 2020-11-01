import numpy as np
import pylab as plt
from parameters import *
from utilities import create_random_dirac


def scaling_update(u, p, q, c, lam, rho):
    """ One scaling iteration, starting from u """
    temp = np.expand_dims(u, 1) - c + np.expand_dims(lam * np.log(p), 1)
    m = - np.max(temp, axis=0)
    v = (m - lam * np.log(np.sum(np.exp((np.expand_dims(m, axis=0) + temp) / lam), axis=0))) * rho / (lam + rho)
    temp = np.expand_dims(v, 1) - np.transpose(c) + np.expand_dims(lam * np.log(q), 1)
    m = - np.max(temp, axis=0)
    u_new = (m - lam * np.log(np.sum(np.exp((np.expand_dims(m, axis=0) + temp) / lam), axis=0))) * rho / (lam + rho)
    err = np.sum(np.abs(u - u_new) * p)  # L1 norm of the gradient of the dual at (u,v)

    return u_new, v, err


def scaling(c, p, q, lam=1.0, rho=1.0, tol=1e-4):
    """The iterative scaling algorithm
    C: cost matrix
    p,q : vector of weights for the two marginals
    """
    m, n = np.size(p), np.size(q)
    u, v = np.zeros(m), np.zeros(n)
    errs, err = np.empty(0), 1.0

    while err > tol:
        u, v, err = scaling_update(u, p, q, c, lam, rho)
        errs = np.concatenate((errs, np.array([err])))

    gamma = np.exp((np.expand_dims(u, 1)
                    + np.expand_dims(v, 0) - c) / lam) * np.expand_dims(p, 1) * np.expand_dims(q, 0)
    return u, v, gamma, errs


def create_cost(xs, ys, cost_type=COST_TYPE, scale_notes=SCALE_NOTES):
    c = np.empty((np.size(xs), np.size(ys)))
    if cost_type == "conic":
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                c[i, j] = -2 * np.log(np.cos(np.pi * np.min((np.abs(x - y) / scale_notes,
                                                             1 / (2 * scale_notes)))))
    elif cost_type == "square":
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                c[i, j] = ((x - y) / scale_notes) ** 2
    else:
        raise ValueError("Cost type not understood.")

    return c


if __name__ == '__main__':
    _n, _m = 5, 6
    _xs, _ys, _p, _q = create_random_dirac(_n, _m)

    plt.figure(1)
    plt.stem(_xs, _p, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k")
    plt.stem(_ys, _q, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k")
    plt.legend()

    _c = create_cost(_xs, _ys)

    _u, _v, _gamma, _errs = scaling(_c, _p, _q, lam=1e-2, rho=1, tol=1e-5)

    plt.figure(2, figsize=[12, 3])
    plt.subplot(131)
    plt.semilogy(_errs, "k")
    plt.xlabel("niter")
    plt.ylabel("$\\Vert \\nabla G(u,v)\\Vert_{distances^1(\\mu)}$")
    plt.title("Convergence")

    plt.subplot(132)
    plt.plot(_xs, _u)
    plt.plot(_ys, _v, "C3")
    plt.title("Dual potentials $(u,v)$")

    plt.subplot(133)
    plt.pcolor(_xs, _ys, np.transpose(_gamma))
    plt.title("Calibration measure $\\gamma$")

    plt.show()
