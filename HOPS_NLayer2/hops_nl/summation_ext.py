"""Series summation that also covers M = 0 (HOPS in eps only, used by the exact-dispersion mode).

hops.summation.sum_series sums the joint (eps, delta) series along rays (K = min(N, M)), which degenerates for
M = 0; here M = 0 is summed as a 1D series in eps (Taylor or [N/2, N/2] Pade).
"""
import numpy as np

from . import _paths  # noqa: F401
from hops.summation import sum_series as _sum_series_2d, _pade_batch, polyval_asc


def sum_series(kind, c, Eps, delta, N, M, taylor_full_order=False):
    """as hops.summation.sum_series; c[..., n, m]"""
    if M > 0:
        return _sum_series_2d(kind, c, Eps, delta, N, M, taylor_full_order)
    c1 = np.asarray(c)[..., :N + 1, 0]
    Eps = np.asarray(Eps, dtype=float)
    shape = np.broadcast_shapes(c1.shape[:-1], Eps.shape, np.shape(delta))
    c1 = np.broadcast_to(c1, shape + (N + 1,))
    rho = np.broadcast_to(Eps, shape)
    if kind == 1:
        return polyval_asc(c1, rho)
    return _pade_batch(c1, rho, N // 2)


def energies_eps(res, stack, alpha, Eps, N, kind=2):
    """(ee, ru, rl) of length len(Eps) for an M = 0 solve at the single frequency of res (hops energy convention:
    propagating orders of the top layer for R, of the substrate for T; ee = 1 - R - T)."""
    Nx = res['ubar'].shape[0]
    p = np.fft.fftfreq(Nx, 1.0 / Nx)
    ap = alpha + p
    gu, gw = res['gp'][0], res['gp'][-1]
    ku2, kw2 = res['kbar'][0] ** 2, res['kbar'][-1] ** 2
    pu = np.real(ap ** 2) < np.real(ku2)
    pw = np.real(ap ** 2) < np.real(kw2)
    Eps = np.atleast_1d(np.asarray(Eps, float))
    uh = np.fft.fft(res['ubar'][:, 0, :], axis=0) / Nx                    # (Nx, N+1)
    wh = np.fft.fft(res['wbar'][:, 0, :], axis=0) / Nx
    U = sum_series(kind, uh[:, None, :, None], Eps[None, :], 0.0, N, 0)      # (Nx, nE)
    W = sum_series(kind, wh[:, None, :, None], Eps[None, :], 0.0, N, 0)
    R = np.sum(np.real(gu[pu] / gu[0])[:, None] * np.abs(U[pu]) ** 2, axis=0)
    T = np.sum(np.real(res['tau2'] * gw[pw] / gu[0])[:, None] * np.abs(W[pw]) ** 2, axis=0)
    return 1 - R - T, R, T
