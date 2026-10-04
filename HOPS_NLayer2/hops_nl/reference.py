"""Independent references for the n-layer HOPS/AWE solver.

tmm(...)        exact reflectance / transmittance of the FLAT stack (eps = 0) by the transfer-matrix method
                (scalar TE: u and u_z continuous; TM: u and u_z / n^2 continuous -- the tau^2 convention of hops)
pointwise(...)  HOPS solved at ONE frequency (delta = 0, M = 0 -> no frequency expansion) with Pade in eps:
                the converged reference used for the summation-error analysis
"""
import warnings

import numpy as np

from . import _paths  # noqa: F401
from hops import csqrt
from hops.summation import padesum


def tmm(n, d, omega, alpha=0.0, mode="TM", layers=False):
    """(R, T) of the flat stack n[0] | n[1] (d[0]) | ... | n[-1] at frequency omega, Bloch number alpha."""
    n = np.asarray(n, complex)
    L = n.size
    k = n * omega
    g = csqrt(k ** 2 - alpha ** 2)
    g = np.where(np.imag(g) < 0, -g, g)
    w = np.ones(L) if mode == 'TE' else 1.0 / n ** 2
    # unknowns: per layer (A_l, B_l) with u_l = A e^{i g (z - h)} + B e^{-i g (z - h)} referenced to the layer's
    # bottom interface (top layer: interface 1); top layer A_0 = r (up), B_0 = 1 (incident, down);
    # bottom layer A_{L-1} = 0 (only the downgoing wave), B_{L-1} = t
    h = [0.0]
    for t in d:
        h.append(h[-1] - t)
    nunk = 2 * L
    A = np.zeros((nunk, nunk), complex)
    rhs = np.zeros(nunk, complex)
    row = 0
    # reference levels: layer 0 -> h[0]; layer l (1..L-2) -> h[l] (its bottom); layer L-1 -> h[L-2]
    ref = [h[0]] + [h[l] for l in range(1, L - 1)] + [h[L - 2]]
    for j in range(1, L):
        z = h[j - 1]
        for l, sgn in ((j - 1, 1.0), (j, -1.0)):
            e_p, e_m = np.exp(1j * g[l] * (z - ref[l])), np.exp(-1j * g[l] * (z - ref[l]))
            A[row, 2 * l] += sgn * e_p
            A[row, 2 * l + 1] += sgn * e_m
            A[row + 1, 2 * l] += sgn * w[l] * 1j * g[l] * e_p
            A[row + 1, 2 * l + 1] += sgn * w[l] * (-1j * g[l]) * e_m
        row += 2
    A[row, 1] = 1.0; rhs[row] = 1.0; row += 1          # B_0 = 1 (incident amplitude at z = 0)
    A[row, 2 * (L - 1)] = 1.0; row += 1                 # A_{L-1} = 0
    c = np.linalg.solve(A, rhs)
    r, t = c[0], c[2 * (L - 1) + 1]
    R = abs(r) ** 2
    T = np.real(w[-1] * g[-1] / (w[0] * g[0])) * abs(t) ** 2
    if layers:
        # z-flux F = Im(conj(u) w u_z) at both faces of every interface; A_l = F(top of l) - F(bottom of l)
        def F(l, z):
            e_p, e_m = np.exp(1j * g[l] * (z - ref[l])), np.exp(-1j * g[l] * (z - ref[l]))
            u = c[2 * l] * e_p + c[2 * l + 1] * e_m
            uz = 1j * g[l] * (c[2 * l] * e_p - c[2 * l + 1] * e_m)
            return np.imag(np.conj(u) * w[l] * uz)
        Finc = np.imag(np.conj(1.0) * w[0] * (-1j * g[0]))                # incident wave alone (downward)
        Fi = [(F(j - 1, h[j - 1]), F(j, h[j - 1])) for j in range(1, L)]   # (above, below) interface j
        Al = np.zeros(L)
        for l in range(1, L - 1):
            Al[l] = (Fi[l - 1][1] - Fi[l][0]) / Finc
        Al[L - 1] = Fi[L - 2][1] / Finc - (T if np.all(np.abs(np.imag(n[-1])) < 1e-12) else 0.0)
        return float(R), float(T), Al
    return float(R), float(T)


def pade_eps(c, Eps):
    """[N/2, N/2] Pade sum of a 1D series c_0..c_N at the points Eps"""
    N = len(c) - 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.array([padesum(np.asarray(c), float(e), N // 2)[0] for e in np.atleast_1d(Eps)]).ravel()


def pointwise(stack, omega, alpha, Eps, N=24):
    """(R, T, D) arrays over Eps from HOPS at the single frequency omega (no delta expansion), Pade in eps."""
    from .solver import multilayer_solve
    res = multilayer_solve(stack, omega, alpha, N, 0)
    Nx = res['ubar'].shape[0]
    from hops import setup_2d
    _, _, alphap, _, _, _ = setup_2d(Nx, 2 * np.pi, alpha, 1.0)
    gu, gw = res['gp'][0], res['gp'][-1]
    ku2, kw2 = res['kbar'][0] ** 2, res['kbar'][-1] ** 2
    pu = np.real(alphap ** 2) < np.real(ku2)
    pw = np.real(alphap ** 2) < np.real(kw2)
    Eps = np.atleast_1d(np.asarray(Eps, float))
    R = np.zeros(Eps.size); T = np.zeros(Eps.size)
    uh = np.fft.fft(res['ubar'][:, 0, :], axis=0) / Nx
    wh = np.fft.fft(res['wbar'][:, 0, :], axis=0) / Nx
    for p in np.nonzero(pu)[0]:
        R += np.real(gu[p] / gu[0]) * np.abs(pade_eps(uh[p], Eps)) ** 2
    for p in np.nonzero(pw)[0]:
        T += np.real(res['tau2'] * gw[p] / gu[0]) * np.abs(pade_eps(wh[p], Eps)) ** 2
    return R, T, 1 - R - T


def check_map(res, info, cols=(0.0, -0.8, 0.8), N=24):
    """true error of a refl_map_nlayer result: pointwise HOPS (no frequency expansion, Pade in eps, N = 24) on the
    columns delta = c * delta_max of every window; returns dict(max, median, per_window) of |R_AWE - R_true|."""
    from .solver import Stack
    errs, per = [], {}
    for r in res:
        e_w = []
        ab = r.get('alpha_bar', info['alpha'])
        for c in cols:
            j = int(np.argmin(np.abs(r['delta'] - c * r['dmax'])))
            ns = r['n_cols'][j] if 'n_cols' in r else r['n_layers']
            st = Stack(list(ns), info['thick'], info['f'], info['fx'], info['a'], info['b'], info['Nz'], info['mode'])
            aj = r['alpha_cols'][j] if 'alpha_cols' in r else ab * (1 + r['delta'][j])
            Rt = pointwise(st, float(r['omega'][j]), aj, r['Eps'], N=N)[0]
            e_w.append(np.abs(np.real(r['ru'][:, j]) - Rt))
        e_w = np.concatenate(e_w)
        per[r['key']] = float(e_w.max())
        errs.append(e_w)
    e = np.concatenate(errs)
    return dict(max=float(e.max()), median=float(np.median(e)), per_window=per)
