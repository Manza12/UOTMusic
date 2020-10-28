import numpy as np
import pylab as plt
import os
import time

from uot import scaling, create_cost
from utilities import create_random_dirac
from parameters import *


def conic_interp(a0, a1, L, t):
    a = (1 - t)**2 * a0 + t**2 * a1 + 2 * t * (1 - t) * np.sqrt(a0 * a1) * np.cos(L)
    b = np.arctan2((1-t) * np.sqrt(a0) + t * np.cos(L) * np.sqrt(a1), t * np.sin(L) * np.sqrt(a1)) #np.arctan((1-t) * np.sqrt(a0) + t * np.cos(L) * np.sqrt(a1), t * np.sin(L) * np.sqrt(a1)) ici il faut arctan de dimension 2

    return a, b



def create_distances(xs, ys):
    dist = np.empty((np.size(xs), np.size(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            dist[i, j] = np.abs(x - y)

    return dist


def conic_interp_measures(xs, ys, p, q, ts, lam=1e-2, tol=1e-6, thr=1e-2, straight=True):
    L = create_distances(xs, ys)  # pairwise distances
    C = create_cost(xs, ys)  # conic cost

    u, v, gamma, errs = scaling(C, p, q, lam=lam, rho=1.0, tol=tol)

    gamma_a = _gamma * np.expand_dims(np.exp(u), 1)
    gamma_b = _gamma * np.expand_dims(np.exp(v), 0)

    M = (gamma_a + gamma_b) > thr  # mask for the (non-negligible) travelling Dirac masses
    A = np.zeros((np.sum(M), np.size(ts)))
    X = np.zeros((np.sum(M), np.size(ts)))

    k = 0
    for i in range(np.size(p)):
        for j in range(np.size(q)):
            if M[i, j]:
                if straight:
                    term_1 = (1 - ts)**2 * gamma_a[i, j]
                    term_2 = ts**2 * gamma_b[i, j]
                    term_3 = 2 * ts * (1 - ts) * np.sqrt(gamma_a[i, j] * gamma_b[i, j])
                    A[k, :] = term_1 + term_2 + term_3  # * cos(L[i,j])
                else:
                    # this one is the true geodesic, but it's a bit weird because the mass isn't monotonous
                    A[k, :] = (1 - ts)**2 * gamma_a[i, j] + ts**2 * gamma_b[i, j] + 2 * ts * (1 - ts) * np.sqrt(gamma_a[i, j] * gamma_b[i, j]) * np.cos(L[i, j])

                bs = np.arctan2(ts * np.sin(L[i, j]) * np.sqrt(gamma_b[i, j]), (1. - ts) * np.sqrt(gamma_a[i, j]) + ts * np.cos(L[i, j]) * np.sqrt(gamma_b[i, j])) / L[i, j] #ici arctan2 et non arctan
                X[k, :] = xs[i] + bs * (ys[j] - xs[i])
                k = k + 1

    for i in range(np.size(p)):
        if np.abs(np.sum(gamma_a[i, :]) / p[i] - 1) > 1 / 2:  # mass disappears
            A = np.concatenate((A, (1. - np.transpose(ts))**2 * p[i]), axis=0)
            X = np.concatenate((X, np.transpose(xs[i] * np.ones( (1,np.size(ts)) ))), axis=0)#np.ones(np.size(ts)))), axis=0)
    for j in range(np.size(q)):
        if np.abs(np.sum(gamma_b[:, j]) / q[j] - 1) > 1 / 2:  # mass appears
            A = np.concatenate((A, np.expand_dims(ts**2, 0) * q[j]), axis=0)
            X = np.concatenate((X, ys[j] * np.ones((1, np.size(ts)))), axis=0)

    return A, X


if __name__ == '__main__':
    n_figures = 11
    _ts = np.linspace(0, 1, n_figures)

    _n, _m = 1, 2
    _xs, _ys, _p, _q = create_random_dirac(_n, _m)

    _c = create_cost(_xs, _ys)

    _u, _v, _gamma, _errs = scaling(_c, _p, _q, lam=1e-2, rho=1, tol=1e-14)

    _gamma_a = _gamma * np.expand_dims(np.exp(_u), 1)
    _gamma_b = _gamma * np.expand_dims(np.exp(_v), 0)

    first_marginal = np.sum(np.abs(np.sum(_gamma_a, axis=1) / _p - 1))  # checks that its first marginal is indeed μ (up to numerical error)
    print(first_marginal)

    second_marginal = np.sum(np.abs(np.sum(_gamma_b, axis=0) / _q - 1))  # checks that its second marginal is indeed ν (up to optimization error)
    print(second_marginal)
    # ToDo: second marginal optimization error is too big

    A, X = conic_interp_measures(_xs, _ys, _p, _q, _ts, lam=1e-2, tol=1e-14, thr=1e-14)

    if create_figures:
        for t in range(np.size(A, 1)):
            plt.figure(figsize=[4, 4])
            plt.stem(_xs, _p, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k")
            plt.stem(_ys, _q, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k")
            plt.stem(X[:, t], A[:, t], linefmt="k", markerfmt="ko", label="\\mu_t", basefmt="k")
            plt.plot([0, 1], [0, 0], "k", lw=4)
            plt.savefig(path.join(figures_path, "interp_" + str(t) + ".png"))

        # time.sleep(1)
        # os.system('ffmpeg -framerate 4 -i interp_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p animation.mp4')

        plt.show()
