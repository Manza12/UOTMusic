import numpy as np
import pylab as plt

# plt.rcParams.update({
#     "text.usetex": True
# })


def scaling_update(u, p, q, c, lam, rho):
    """ One scaling iteration, starting from u """
    temp = np.expand_dims(u, 1) - c + lam * np.log(rho)
    m = - np.max(temp, axis=0)
    v = (m - lam * np.log(np.sum(np.exp((m + temp) / lam), axis=0))) * rho / (lam + rho)
    temp = np.expand_dims(v, 1) - np.transpose(c) + np.expand_dims(lam * np.log(q), 1)
    m = - np.max(temp, axis=0)
    u_new = (m - lam * np.log(np.sum(np.exp((m + temp) / lam), axis=0))) * rho / (lam + rho)
    err = np.sum(np.abs(u - u_new) * p)  # L1 norm of the gradient of the dual at (u,v)

    return u_new, v, err


def scaling(c, p, q, lam=1.0, rho=1.0, tol=1e-4):
    """The iterative scaling algorithm
    C: cost matrix
    p,q : vector of weights for the two marginals
    """
    m, n = np.size(p), np.size(q)
    u, v = np.zeros(m), np.zeros(n)
    errs, err = np.empty(0), 1.0 * np.ones(1)

    while err > tol:
        u, v, err = scaling_update(u, p, q, c, lam, rho)
        err = err * np.ones(1)
        errs = np.concatenate((errs, np.array(err)))

    gamma = np.exp((np.expand_dims(u, 1) + np.expand_dims(v, 0) - c) / lam) * np.expand_dims(p, 1) * np.expand_dims(q, 0)
    return u, v, gamma, errs


if __name__ == '__main__':
    _n, _m = 5, 6
    xs, ys = np.sort(np.random.uniform(0, 1, _n)), np.sort(np.random.uniform(0, 1, _m))  # random positions
    _p, _q = np.random.uniform(0, 1, _n), np.random.uniform(0, 1, _m)  # random weights

    plt.figure(1)
    plt.stem(xs, _p, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k")
    plt.stem(ys, _q, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k")
    plt.legend()

    _c = np.empty((np.size(xs), np.size(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            _c[i, j] = -2 * np.log(np.cos(np.pi * np.min((np.abs(x - y) / np.pi, 1 / 2))))  # conic cost

    # C = [(x - y)**2 for x in xs for y in ys]  # square cost
    _u, _v, _gamma, _errs = scaling(_c, _p, _q, lam=1e-2, rho=1, tol=1e-5)

    plt.figure(2, figsize=[12, 3])
    plt.subplot(131)
    plt.semilogy(_errs, "k")
    plt.xlabel("niter")
    plt.ylabel("\\Nabla G(u,v)\\Vert_{L^1(\\mu)}")
    plt.title("Convergence")

    plt.subplot(132)
    plt.plot(xs, _u)
    plt.plot(ys, _v, "C3")
    plt.title("Dual potentials $(u,v)$")

    plt.subplot(133)
    plt.pcolor(xs, ys, np.transpose(_gamma), shading='auto')
    plt.title("Calibration measure $\\gamma$")

    plt.show()
