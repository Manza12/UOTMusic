import pylab as plt

from plot import plot_gamma
from uot import scaling, create_cost
from utilities import create_random_dirac
from parameters import *


def conic_interp(a0, a1, distances, t):
    a = (1 - t) ** 2 * a0 + t ** 2 * a1 + 2 * t * (1 - t) * np.sqrt(a0 * a1) * np.cos(distances)
    b = np.arctan2((1 - t) * np.sqrt(a0) + t * np.cos(distances) * np.sqrt(a1), t * np.sin(distances) * np.sqrt(a1))

    return a, b


def create_distances(xs, ys):
    dist = np.empty((np.size(xs), np.size(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            dist[i, j] = np.abs(x - y)

    return dist


def conic_interp_measures(xs, ys, p, q, ts, lam=LAMBDA, tol=TOL, rho=RHO, thr=THR, straight=STRAIGHT):
    distances = create_distances(xs, ys)  # pairwise distances
    cost = create_cost(xs, ys)  # conic cost

    u, v, gamma, errs = scaling(cost, p, q, lam=lam, rho=rho, tol=tol)

    gamma_a = gamma * np.expand_dims(np.exp(u), 1)
    gamma_b = gamma * np.expand_dims(np.exp(v), 0)

    mask = (gamma_a + gamma_b) > thr  # mask for the (non-negligible) travelling Dirac masses
    a = np.zeros((np.sum(mask), np.size(ts)))
    x = np.zeros((np.sum(mask), np.size(ts)))

    k = 0
    for i in range(np.size(p)):
        for j in range(np.size(q)):
            if mask[i, j]:
                if straight:
                    term_1 = (1 - ts) ** 2 * gamma_a[i, j]
                    term_2 = ts ** 2 * gamma_b[i, j]
                    term_3 = 2 * ts * (1 - ts) * np.sqrt(gamma_a[i, j] * gamma_b[i, j])
                    a[k, :] = term_1 + term_2 + term_3  # * cos(distances[i,j])
                else:
                    # this one is the true geodesic, but it's a bit weird because the mass is not k monotonous
                    a[k, :] = (1 - ts) ** 2 * gamma_a[i, j] + ts ** 2 * gamma_b[i, j] \
                              + 2 * ts * (1 - ts) * np.sqrt(gamma_a[i, j] * gamma_b[i, j]) * np.cos(distances[i, j])

                bs = np.arctan2(ts * np.sin(distances[i, j]) * np.sqrt(gamma_b[i, j]),
                                (1. - ts) * np.sqrt(gamma_a[i, j]) + ts * np.cos(distances[i, j])
                                * np.sqrt(gamma_b[i, j])) / distances[i, j]
                x[k, :] = xs[i] + bs * (ys[j] - xs[i])
                k = k + 1

    for i in range(np.size(p)):
        if np.abs(np.sum(gamma_a[i, :]) / p[i] - 1) > 1 / 2:  # mass disappears
            a = np.concatenate((a, (1. - np.transpose(ts)) ** 2 * p[i]), axis=0)
            x = np.concatenate((x, np.transpose(xs[i] * np.ones((1, np.size(ts))))), axis=0)
    for j in range(np.size(q)):
        if np.abs(np.sum(gamma_b[:, j]) / q[j] - 1) > 1 / 2:  # mass appears
            a = np.concatenate((a, np.expand_dims(ts ** 2, 0) * q[j]), axis=0)
            x = np.concatenate((x, ys[j] * np.ones((1, np.size(ts)))), axis=0)

    return a, x


def log_interp(a, b, t):
    assert (np.all(0 <= t) and np.all(t <= 1))
    assert (a > 0 and b > 0)
    return np.exp(np.log(a) * t + np.log(b) * (1 - t))


def music_interp_measures(xs, ys, p, q, ts, lam=LAMBDA, tol=TOL, rho=RHO, thr=THR):
    cost = create_cost(xs, ys)

    u, v, gamma, errs = scaling(cost, p, q, lam=lam, rho=rho, tol=tol)

    gamma_a = gamma * np.expand_dims(np.exp(u), 1)
    gamma_b = gamma * np.expand_dims(np.exp(v), 0)

    mask = (gamma_a + gamma_b) > thr  # mask for the (non-negligible) travelling Dirac masses
    a = np.zeros((np.sum(mask), np.size(ts)))
    x = np.zeros((np.sum(mask), np.size(ts)))

    k = 0
    for i in range(np.size(p)):
        for j in range(np.size(q)):
            if mask[i, j]:
                a[k, :] = log_interp(gamma_a[i, j], gamma_b[i, j], ts)
                x[k, :] = xs[i] + ts * (ys[j] - xs[i])
                k = k + 1

    for i in range(np.size(p)):
        if np.sum(gamma_a[i, :]) / p[i] > 1:  # mass disappears
            a = np.concatenate((a, (1. - np.expand_dims(ts, 0)) * p[i]), axis=0)
            x = np.concatenate((x, xs[i] * np.ones((1, np.size(ts)))), axis=0)
    for j in range(np.size(q)):
        if np.sum(gamma_b[:, j]) / q[j] > 1:  # mass appears
            a = np.concatenate((a, np.expand_dims(ts, 0) * q[j]), axis=0)
            x = np.concatenate((x, ys[j] * np.ones((1, np.size(ts)))), axis=0)

    return a, x


if __name__ == '__main__':
    n_figures = 11
    _ts = np.linspace(0, 1, n_figures)

    _n, _m = 2, 3
    _xs, _ys, _p, _q = create_random_dirac(_n, _m)

    _c = create_cost(_xs, _ys)

    _u, _v, _gamma, _errs = scaling(_c, _p, _q, lam=LAMBDA, rho=RHO, tol=TOL)

    # Plot transport plan
    plot_gamma(_gamma, log_scale=True)

    _gamma_a = _gamma * np.expand_dims(np.exp(_u), 1)
    _gamma_b = _gamma * np.expand_dims(np.exp(_v), 0)

    # checks that its first marginal is indeed μ (up to numerical error)
    first_marginal = np.sum(_gamma_a, axis=1)
    first_marginal_error = np.sum(np.abs(first_marginal / _p - 1))
    print("First marginal error:", first_marginal_error)

    # checks that its second marginal is indeed ν (up to optimization error)
    second_marginal = np.sum(_gamma_b, axis=0)
    second_marginal_error = np.sum(np.abs(second_marginal / _q - 1))
    print("Second marginal error:", second_marginal_error)

    A, X = music_interp_measures(_xs, _ys, _p, _q, _ts, lam=1e-2, tol=1e-14, thr=1e-14)

    if CREATE_FIGURES:
        for _t in range(np.size(A, 1)):
            fig = plt.figure(figsize=[6, 4])
            plt.stem(_xs, _p, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k",
                     use_line_collection=True)
            plt.stem(_ys, _q, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k",
                     use_line_collection=True)
            plt.stem(X[:, _t], A[:, _t], linefmt="k", markerfmt="ko", label="\\mu_t", basefmt="k",
                     use_line_collection=True)
            plt.plot([0, 1], [0, 0], "k", lw=4)
            plt.savefig(path.join(FIGURES_PATH, "interp_" + str(_t) + ".png"))

            fig.canvas.manager.window.wm_geometry('+500+150')

        plt.show()
