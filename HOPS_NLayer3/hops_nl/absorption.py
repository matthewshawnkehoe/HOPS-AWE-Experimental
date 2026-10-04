"""Absorption in EACH layer of a corrugated multilayer from the HOPS/AWE volume series.

The volume coefficients u_{l,n,m}(x, s) (multilayer_solve(..., keep_volume=True)) are Pade-summed at (eps, delta);
the field lives on the TFE grid of each layer,
    Z(x, s) = z_b + eps f_b + s (d + eps (f_t - f_b)),   s in [0, 1] (Chebyshev nodes),
so the physical gradient is  u_z = u_s / J,  u_x = u_x' - Z_x u_s / J  with J = dZ/ds, and every layer integral is
a Clenshaw-Curtis (s) x trapezoid (x) quadrature of the exact curved layer.  Normalised by the incident power
(unit-amplitude plane wave in layer 0):

    TE (u = E_y):   A_l = k0^2 Im(n_l^2) / (2 pi gamma_0) * int |u|^2 dA
    TM (u = H_y):   A_l = n_0^2 Im(n_l^2) / (|n_l^2|^2 2 pi gamma_0) * int |grad(e^{i alpha x} u)|^2 dA

Check: sum_l A_l = 1 - R - T (the absorptance returned by energies) -- see tests/test_nlayer.py.
"""
import warnings

import numpy as np

from . import _paths  # noqa: F401
from .summation_ext import sum_series


def _cc_weights(Nz):
    """Clenshaw-Curtis weights on t_j = cos(pi j / Nz), j = 0..Nz (integral over [-1, 1])"""
    th = np.pi * np.arange(Nz + 1) / Nz
    w = np.zeros(Nz + 1)
    v = np.ones(Nz - 1)
    if Nz % 2 == 0:
        w[0] = w[Nz] = 1.0 / (Nz ** 2 - 1)
        for k in range(1, Nz // 2):
            v -= 2 * np.cos(2 * k * th[1:-1]) / (4 * k ** 2 - 1)
        v -= np.cos(Nz * th[1:-1]) / (Nz ** 2 - 1)
    else:
        w[0] = w[Nz] = 1.0 / Nz ** 2
        for k in range(1, (Nz - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * th[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2 * v / Nz
    return w


def _cheb_D(Nz):
    """Chebyshev differentiation matrix d/dt on t_j = cos(pi j / Nz) (Trefethen)"""
    t = np.cos(np.pi * np.arange(Nz + 1) / Nz)
    c = np.hstack([2, np.ones(Nz - 1), 2]) * (-1) ** np.arange(Nz + 1)
    T = np.tile(t, (Nz + 1, 1)).T
    dT = T - T.T
    D = np.outer(c, 1 / c) / (dT + np.eye(Nz + 1))
    return D - np.diag(D.sum(axis=1)), t


def _dx(u):
    Nx = u.shape[0]
    k = np.fft.fftfreq(Nx, 1.0 / Nx)
    k[Nx // 2] = 0 if Nx % 2 == 0 else k[Nx // 2]
    return np.fft.ifft(1j * k[:, None] * np.fft.fft(u, axis=0), axis=0)


def column_view(r, info, delta):
    """(window, delta, M) to sum the volume series at delta: for an exact-dispersion window ('vol_cols') the column
    nearest to delta is returned as a one-frequency window (M = 0, delta = 0)."""
    if 'vol_cols' not in r:
        return r, delta, info['M']
    j = int(np.argmin(np.abs(r['delta'] - delta)))
    rv = dict(vol=r['vol_cols'][j], n_layers=r['n_cols'][j], omega_bar=r['omega'][j],
              alpha_bar=(r['alpha_cols'][j] if 'alpha_cols' in r else r.get('alpha_bar', info['alpha']) * (1 + r['delta'][j])))
    return rv, 0.0, 0


def layer_absorption(r, info, eps, delta, kind=2):
    """per-layer absorptance A_l (array of length L) at one (eps, delta) of a refl_map_nlayer window r computed
    with keep_fields=True (r['vol'] present)."""
    r, delta, M = column_view(r, info, delta)
    L, N = info['L'], info['N']
    thick, a, b = info['thick'], info['a'], info['b']
    h = [0.0]
    for t in thick:
        h.append(h[-1] - t)
    s1 = 1 + delta
    om = r['omega_bar'] * s1
    alpha = r.get('alpha_bar', info['alpha']) * s1
    n = np.asarray(r['n_layers'], complex)
    k0u = np.real(n[0]) * om
    g0 = np.sqrt(complex(k0u ** 2 - alpha ** 2))
    f = [np.asarray(v, float) for v in info['f']]
    fx = [np.asarray(v, float) for v in info['fx']]
    Z0 = np.zeros_like(f[0])
    A = np.zeros(L)
    for l in range(L):
        eps_l = n[l] ** 2
        if abs(np.imag(eps_l)) < 1e-14:
            continue
        v = r['vol'][l]                                                  # (Nx, Nz+1, M+1, N+1)
        Nx, Nz1 = v.shape[:2]
        Nz = Nz1 - 1
        c = np.transpose(v.reshape(Nx * Nz1, M + 1, N + 1), (0, 2, 1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u = sum_series(kind, c, np.array([[eps]]), np.array([[delta]]), N, M).reshape(Nx, Nz1)
        if l == 0:
            zb, zt, fb, ft, fbx, ftx = h[0], h[0] + a, f[0], Z0, fx[0], Z0
        elif l == L - 1:
            zb, zt, fb, ft, fbx, ftx = h[-1] - b, h[-1], Z0, f[-1], Z0, fx[-1]
        else:
            zb, zt, fb, ft, fbx, ftx = h[l], h[l - 1], f[l], f[l - 1], fx[l], fx[l - 1]
        D, t = _cheb_D(Nz)
        s = (t + 1) / 2
        J = ((zt - zb) + eps * (ft - fb))[:, None] * np.ones((1, Nz1))          # dZ/ds
        Zx = eps * fbx[:, None] + s[None, :] * eps * (ftx - fbx)[:, None]
        if l == 0:                                                       # total field = scattered + incident
            Z = (zb + eps * fb)[:, None] + s[None, :] * ((zt - zb) + eps * (ft - fb))[:, None]
            u = u + np.exp(-1j * g0 * Z)
        us = 2 * (u @ D.T)                                               # d/ds = 2 d/dt
        uxp = _dx(u)
        uz = us / J
        ux = uxp - Zx * uz + 1j * alpha * u                              # Bloch: e^{i alpha x} (u_x + i alpha u)
        w = _cc_weights(Nz) / 2                                          # ds weights on [0, 1]
        dA = (2 * np.pi / Nx) * J * w[None, :]
        if info['mode'] == 'TE':
            I = np.sum(np.abs(u) ** 2 * dA)
        else:
            I = np.sum((np.abs(ux) ** 2 + np.abs(uz) ** 2) * dA)
        if l == L - 1:
            # analytic tail of the absorbing substrate below the artificial boundary z = h_{L-1} - b:
            # u = sum_p W_p e^{i alpha_p x - i gamma_p (z - z_b)},  int_{-inf}^{z_b} |.|^2 = |W_p|^2 / (2 Im gamma_p)
            W = np.fft.fft(u[:, -1]) / Nx
            p = np.fft.fftfreq(Nx, 1.0 / Nx)
            ap = alpha + p
            gp = np.sqrt((n[l] * om) ** 2 - ap ** 2 + 0j)
            gp = np.where(np.imag(gp) < 0, -gp, gp)
            wt = 1.0 if info['mode'] == 'TE' else (np.abs(ap) ** 2 + np.abs(gp) ** 2)
            I += 2 * np.pi * np.sum(wt * np.abs(W) ** 2 / (2 * np.imag(gp)))
        if info['mode'] == 'TE':
            A[l] = om ** 2 * np.imag(eps_l) * I / (2 * np.pi * np.real(g0))
        else:
            A[l] = np.real(n[0]) ** 2 * np.imag(eps_l) / abs(eps_l) ** 2 * I / (2 * np.pi * np.real(g0))
    return A


def absorption_spectrum(res, info, eps, kind=2, step=1):
    """per-layer absorptance along the whole frequency axis at fixed eps: returns (lam, omega, A[L, n])"""
    lam, om, AA = [], [], []
    for r in sorted(res, key=lambda r: r['omega_bar']):
        for j in range(0, len(r['delta']), step):
            AA.append(layer_absorption(r, info, eps, r['delta'][j], kind))
            lam.append(r['lam'][j]); om.append(r['omega'][j])
    o = np.argsort(om)
    return np.asarray(lam)[o], np.asarray(om)[o], np.asarray(AA)[o].T
