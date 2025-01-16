import numpy as np
import scipy.sparse as sp


def spnabla(M, N):
    dx = spnabla_x(M, N)
    dy = spnabla_y(M, N)
    dxy = sp.vstack((dx, dy))
    return dxy


def spnabla_x(M, N):
    a = np.append(np.ones(N-1), 0)
    dx = sp.diags([np.tile(-a, M), np.tile(a, M)], [0, 1], (M*N, M*N))
    return dx.tocsr()


def spnabla_y(M, N):
    b = np.append(np.tile(np.ones(N), M-1), np.zeros(N))
    dy = sp.diags([-b, b], [0, N], (M*N, M*N))
    return dy.tocsr()


def spnabla_hp(M, N) -> sp.csr_matrix:
    dx = spnabla_x_hp(M, N)
    dy = spnabla_y_hp(M, N)
    dxy = sp.vstack((dx, dy))
    return dxy


def spnabla_x_hp(M, N):
    a = np.append(np.ones(N-1) * 0.5, 0)
    vals = np.append(np.tile(a, M-1), np.zeros(N))
    dx = sp.diags([-vals, vals, -vals, vals], [0, 1, N, N+1], (M*N, M*N))
    return dx.tocsr()


def spnabla_y_hp(M, N):
    a = np.append(np.ones(N-1) * 0.5, 0)
    vals = np.append(np.tile(a, M-1), np.zeros(N))
    dy = sp.diags([-vals, -vals, vals, vals], [0, 1, N, N+1], (M*N, M*N))
    return dy.tocsr()
