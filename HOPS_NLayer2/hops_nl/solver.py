"""HOPS/AWE for a stack of L >= 2 layers separated by L-1 periodic, independently corrugated interfaces (2D).

Geometry (period d = 2 pi in x, layers numbered from the top, l = 0 .. L-1):

    layer 0   (index n_0)          superstrate, artificial boundary z = h_1 + a      (incident wave from above)
    -------   interface 1          z = h_1 + eps f_1(x),   h_1 = 0
    layer 1   (n_1, thickness d_1)
    -------   interface 2          z = h_2 + eps f_2(x),   h_2 = -d_1
      ...
    -------   interface L-1        z = h_{L-1} + eps f_{L-1}(x)
    layer L-1 (n_{L-1})            substrate, artificial boundary z = h_{L-1} - b

Every layer gets its own Transformed Field Expansion: between its (possibly corrugated) top boundary
z_t + g_t and bottom boundary z_b + g_b the map

    z = z_b + g_b + s (d + g_t - g_b),   s = (z' - z_b)/d in [0, 1],   d = z_t - z_b,

flattens the layer to z' in [z_b, z_t].  Writing lambda = 1 + eps mu(x), mu = (f_t - f_b)/d (thickness ratio)
and q(x, s) = f_b' + s (f_t' - f_b') (slope of the coordinate lines), the Helmholtz equation for the
Bloch-reduced field u~ = e^{-i alpha x} u, multiplied by lambda^2, is EXACTLY quadratic in eps:

    E[u] = sum_{k<=2} eps^k { P_k u + 2 i alpha B_k u + (k^2 - alpha^2) C_k u } = 0,
      P_0 = d_xx + d_z'z'
      P_1 = d_x(2 mu u_x - q u_z) - mu' u_x - d_z(q u_x)
      P_2 = d_x(mu^2 u_x - mu q u_z) - mu'(mu u_x - q u_z) + d_z(-mu q u_x + q^2 u_z)
      B_0 = u_x,  B_1 = 2 mu u_x - q u_z,  B_2 = mu^2 u_x - mu q u_z
      C_0 = u,    C_1 = 2 mu u,            C_2 = mu^2 u
(top/bottom layer with a flat artificial boundary: this is exactly the 2D two-layer TFE of hops/, coefficient by
coefficient: A1_xx = 2 mu, A1_xz = -q, B1_x = -mu', S1 = 2 mu, ...).  The frequency enters through
alpha = alpha_bar s, k^2 - alpha^2 = gamma_bar^2 s^2, s = 1 + delta (the joint (eps, delta) HOPS/AWE expansion).

Interface j (layer a = j-1 above, b = j below), at z' = h_j for both layers:
    u_a - u_b                              = zeta_j          (zeta_1 = -u_inc; 0 for j > 1)
    lambda_b N~_a - tau_j^2 lambda_a N~_b   = lambda_a lambda_b psi_j        (psi_1 = -d_N u_inc)
with N~ = lambda d_N u = (1 + g_x^2) u_z' - lambda g_x u_x - i alpha lambda g_x u (cubic in eps), and
tau_j^2 = (n_{j-1}/n_j)^2 (TM) or 1 (TE).  Transparent conditions at z = a (outgoing up) and z = -b (outgoing
down) as in 2D: u_z' = lambda T(omega) u, with T the delta-expansion of i gamma_p (T_dno).

Solver (coupled HOPS/AWE recursion, the multi-layer version of hops/coupled.py):
  * the flat (eps = 0, delta = 0) problem couples ALL layers through the interface rows; per Fourier mode p it is
    one block Chebyshev-collocation matrix of size sum_l (Nz_l + 1).  It is well posed whenever the physical flat
    stack is (no Dirichlet-eigenvalue breakdown of interior layers, unlike a per-layer DNO elimination);
    it is inverted ONCE per frequency window (batched over p);
  * order (n, m): the right-hand side collects all lower orders (interior TFE terms, cubic interface terms,
    transparent-boundary terms) -> one FFT, one batched mat-vec per mode, one inverse FFT for all layers;
  * cost: (N+1)(M+1) such solves, independent of the number of layers except through the matrix size.
"""
import time

import numpy as np

from . import _paths  # noqa: F401  (puts HOPS_Python on sys.path)
from hops import cheb, T_dno, setup_2d, csqrt, setup_zeta_psi_n_m
from hops.energy import energy_defect

_fft = lambda u: np.fft.fft(u, axis=0)
_ifft = lambda u: np.fft.ifft(u, axis=0)
_INV_CACHE = {}


class Stack:
    """Geometry + materials of one frequency window.

    n      : refractive indices of the L layers, top to bottom (complex allowed)
    d      : thicknesses of the L-2 interior layers (nondimensional, period = 2 pi)
    f, fx  : lists of the L-1 interface profiles f_j(x) and slopes on the x grid (zeros = flat interface)
    a, b   : distances of the artificial boundaries above interface 1 / below interface L-1
    Nz     : Chebyshev order per layer (int or list of L ints)
    mode   : 'TM' (tau_j^2 = (n_{j-1}/n_j)^2) or 'TE' (tau_j^2 = 1)
    """

    def __init__(self, n, d, f, fx, a=1.0, b=1.0, Nz=24, mode='TM'):
        self.n = [complex(v) for v in n]
        self.L = len(self.n)
        assert self.L >= 2 and len(d) == self.L - 2 and len(f) == self.L - 1 == len(fx)
        self.d = [float(v) for v in d]
        self.f = [np.asarray(v, float) for v in f]
        self.fx = [np.asarray(v, float) for v in fx]
        self.a, self.b, self.mode = float(a), float(b), mode
        self.Nz = [int(Nz)] * self.L if np.isscalar(Nz) else [int(v) for v in Nz]
        h = [0.0]
        for t in self.d:
            h.append(h[-1] - t)
        self.h = h                                   # flat interface levels h_1 .. h_{L-1}
        self.tau2 = [1.0 + 0j if mode == 'TE' else (self.n[j - 1] / self.n[j]) ** 2 for j in range(1, self.L)]
        self.tau2_total = 1.0 + 0j if mode == 'TE' else (self.n[0] / self.n[-1]) ** 2

    def bounds(self, l):
        """(z_b, z_t, f_b, f_t, fx_b, fx_t) of layer l in transformed coordinates."""
        Z = np.zeros_like(self.f[0])
        if l == 0:
            return self.h[0], self.h[0] + self.a, self.f[0], Z, self.fx[0], Z
        if l == self.L - 1:
            return self.h[-1] - self.b, self.h[-1], Z, self.f[-1], Z, self.fx[-1]
        return self.h[l], self.h[l - 1], self.f[l], self.f[l - 1], self.fx[l], self.fx[l - 1]

    def max_eps(self):
        """largest eps for which no two interfaces (or an interface and an artificial boundary) touch"""
        gaps = []
        for l in range(self.L):
            zb, zt, fb, ft, _, _ = self.bounds(l)
            m = np.max(fb - ft)
            if m > 0:
                gaps.append((zt - zb) / m)
        return min(gaps) if gaps else np.inf


class _LayerOps:
    """Discrete TFE operators of one layer (same discretisation as hops/coupled.py)."""

    def __init__(self, stack, l, p, Nx, dense_x):
        zb, zt, fb, ft, fxb, fxt = stack.bounds(l)
        self.Nz = Nz = stack.Nz[l]
        self.dthk = zt - zb
        Dz, t = cheb(Nz)
        self.s = (t + 1) / 2                                     # ell = 0: s = 1 (top), ell = Nz: s = 0
        self.D = (2.0 / self.dthk) * Dz                          # d/dz'
        self.D2 = self.D @ self.D
        self.mu = ((ft - fb) / self.dthk)[:, None]               # (Nx, 1)
        self.mux = ((fxt - fxb) / self.dthk)[:, None]
        self.q = fxb[:, None] + self.s[None, :] * (fxt - fxb)[:, None]       # (Nx, Nz+1)
        if dense_x:
            Dx = _ifft(1j * p[:, None] * _fft(np.eye(Nx)))
            self.dx = lambda u: Dx @ u
        else:
            ip = (1j * p)[:, None]
            self.dx = lambda u: _ifft(ip * _fft(u))

    def dz(self, u):
        return u @ self.D.T

    def terms(self, u, k, ux=None, uz=None):
        """(P_k u, B_k u, C_k u) for k = 0, 1, 2 (u: (Nx, Nz+1) nodal values; ux, uz: its derivatives if known)."""
        ux = self.dx(u) if ux is None else ux
        uz = self.dz(u) if uz is None else uz
        mu, mux, q = self.mu, self.mux, self.q
        if k == 0:
            return None, ux, u
        if k == 1:
            P = self.dx(2 * mu * ux - q * uz) - mux * ux - self.dz(q * ux)
            return P, 2 * mu * ux - q * uz, 2 * mu * u
        P = self.dx(mu ** 2 * ux - mu * q * uz) - mux * (mu * ux - q * uz) + self.dz(-mu * q * ux + q ** 2 * uz)
        return P, mu ** 2 * ux - mu * q * uz, mu ** 2 * u


def _flat_inverse(stack, ops, gp, kbar2, alphap, tau2):
    """Inverse of the flat multi-layer collocation operator, one (S x S) block per Fourier mode."""
    L = stack.L
    sizes = [o.Nz + 1 for o in ops]
    off = np.concatenate([[0], np.cumsum(sizes)])
    S = off[-1]
    Nx = alphap.size
    key = (S, Nx, np.concatenate([np.ravel(g) for g in gp] + [alphap, kbar2, np.asarray(tau2)]).astype(complex).tobytes(),
           tuple(o.dthk for o in ops), tuple(sizes))
    if key in _INV_CACHE:
        return _INV_CACHE[key], off
    A = np.zeros((Nx, S, S), dtype=complex)
    for l, o in enumerate(ops):
        i0, Nz = off[l], o.Nz
        blk = o.D2[None, :, :] + (kbar2[l] - alphap ** 2)[:, None, None] * np.eye(Nz + 1)[None]
        A[:, i0 + 1:i0 + Nz, i0:i0 + Nz + 1] = blk[:, 1:Nz, :]
        # node 0: top TBC (l = 0) or the flux condition of interface l (l >= 1)
        if l == 0:
            A[:, i0, i0:i0 + Nz + 1] = o.D[0][None, :]
            A[:, i0, i0] -= 1j * gp[0]
        else:
            ip_, op_ = off[l - 1], ops[l - 1]
            A[:, i0, ip_:ip_ + op_.Nz + 1] = op_.D[-1][None, :]
            A[:, i0, i0:i0 + Nz + 1] -= tau2[l - 1] * o.D[0][None, :]
        # node Nz: bottom TBC (l = L-1) or continuity at interface l+1
        if l == L - 1:
            A[:, i0 + Nz, i0:i0 + Nz + 1] = o.D[-1][None, :]
            A[:, i0 + Nz, i0 + Nz] += 1j * gp[-1]
        else:
            A[:, i0 + Nz, i0 + Nz] = 1.0
            A[:, i0 + Nz, off[l + 1]] = -1.0
    Ainv = np.linalg.inv(A)
    if len(_INV_CACHE) > 8:
        _INV_CACHE.clear()
    _INV_CACHE[key] = Ainv
    return Ainv, off


def multilayer_solve(stack, omega_bar, alpha_bar, N, M, zeta=None, psi=None, keep_volume=False, dense_x=None,
                     verbose=False):
    """Joint (eps, delta) HOPS/AWE solve of the stack at frequency omega = omega_bar (1 + delta).

    zeta, psi : dict {j: (Nx, M+1, N+1)} of interface jumps u_{j-1} - u_j and d_N u_{j-1} - tau_j^2 d_N u_j
                (default: plane-wave incidence on interface 1, as setup_zeta_psi_n_m)
    Returns dict with
        ubar (Nx, M+1, N+1)  trace of the top-layer (scattered) field at z = a      (as two_layer_solve)
        wbar (Nx, M+1, N+1)  trace of the substrate field at z = h_{L-1} - b
        V    list of L-1 arrays (Nx, M+1, N+1): field values at interface j (from above)
        vol  (keep_volume) list of L arrays (Nx, Nz_l+1, M+1, N+1): nodal fields u_{l, n, m}
        gamma_bar, gp, tau2, ...
    """
    L, Nx = stack.L, stack.f[0].size
    xx, pp, alphap, _, _, _ = setup_2d(Nx, 2 * np.pi, alpha_bar, 1.0)
    kbar = np.array([nl * omega_bar for nl in stack.n])
    kbar2 = kbar ** 2
    gbar = np.array([csqrt(k2 - alpha_bar ** 2) for k2 in kbar2])
    gp = [setup_2d(Nx, 2 * np.pi, alpha_bar, gb)[3] for gb in gbar]
    if dense_x is None:
        dense_x = Nx <= 64
    ops = [_LayerOps(stack, l, pp, Nx, dense_x) for l in range(L)]
    t0 = time.time()
    Ainv, off = _flat_inverse(stack, ops, [gp[0], gp[-1]], kbar2, alphap, stack.tau2)
    t_inv = time.time() - t0
    T_top = T_dno(alpha_bar, alphap, gbar[0], gp[0], kbar2[0], Nx, M)
    T_bot = T_dno(alpha_bar, alphap, gbar[-1], gp[-1], kbar2[-1], Nx, M)
    tm = lambda T, k, v: _ifft(T[:, k] * _fft(v))
    if zeta is None:
        f1, fx1 = stack.f[0], stack.fx[0]
        z1, p1 = setup_zeta_psi_n_m(xx, pp, alpha_bar, gbar[0], f1, fx1, Nx, N, M)
        zeta, psi = {1: z1}, {1: p1}
    nlev = N + 1 if keep_volume else 4
    V = [np.zeros((nlev, M + 1, Nx, o.Nz + 1), dtype=complex) for o in ops]
    VX = [np.zeros((4, M + 1, Nx, o.Nz + 1), dtype=complex) for o in ops]     # cached d/dx, d/dz' of stored fields
    VZ = [np.zeros((4, M + 1, Nx, o.Nz + 1), dtype=complex) for o in ops]
    get = lambda l, n, m: V[l][n % nlev if not keep_volume else n, m]
    getd = lambda l, n, m: (VX[l][n % 4, m], VZ[l][n % 4, m])
    ubar = np.zeros((Nx, M + 1, N + 1), dtype=complex)
    wbar = np.zeros_like(ubar)
    Vif = [np.zeros_like(ubar) for _ in range(L - 1)]
    ab = alpha_bar
    g2 = [kbar2[l] - ab ** 2 for l in range(L)]
    cleared = -1
    S = off[-1]
    for n in range(N + 1):
        if not keep_volume and cleared != n:
            for l in range(L):
                V[l][n % nlev] = 0.0
                VX[l][n % 4] = 0.0
                VZ[l][n % 4] = 0.0
            cleared = n
        for m in range(M + 1):
            rhs = np.zeros((Nx, S), dtype=complex)
            # ---- interior rows: -sum over lower orders of the TFE operator
            for l, o in enumerate(ops):
                F = np.zeros((Nx, o.Nz + 1), dtype=complex)
                for k in range(3):
                    if n - k < 0:
                        continue
                    for j in range(3):
                        if (k, j) == (0, 0) or m - j < 0:
                            continue
                        u = get(l, n - k, m - j)
                        if not np.any(u):
                            continue
                        P, B, C = o.terms(u, k, *getd(l, n - k, m - j))
                        if j == 0 and P is not None:
                            F -= P
                        if j <= 1 and ab != 0:
                            F -= 2j * ab * B
                        F -= g2[l] * (1.0, 2.0, 1.0)[j] * C
                rhs[:, off[l] + 1:off[l] + o.Nz] = F[:, 1:o.Nz]
            # ---- top transparent condition (layer 0, node 0)
            o0 = ops[0]
            J = np.zeros(Nx, dtype=complex)
            for r in range(m):
                J += tm(T_top, m - r, get(0, n, r)[:, 0])
            if n >= 1:
                acc = np.zeros(Nx, dtype=complex)
                for r in range(m + 1):
                    acc += tm(T_top, m - r, get(0, n - 1, r)[:, 0])
                J += o0.mu[:, 0] * acc
            rhs[:, off[0]] = J
            # ---- bottom transparent condition (layer L-1, node Nz)
            oL = ops[-1]
            J = np.zeros(Nx, dtype=complex)
            for r in range(m):
                J -= tm(T_bot, m - r, get(L - 1, n, r)[:, -1])
            if n >= 1:
                acc = np.zeros(Nx, dtype=complex)
                for r in range(m + 1):
                    acc += tm(T_bot, m - r, get(L - 1, n - 1, r)[:, -1])
                J -= oL.mu[:, 0] * acc
            rhs[:, off[L - 1] + oL.Nz] = J
            # ---- interfaces
            for j in range(1, L):
                a_, b_ = j - 1, j
                oa, ob = ops[a_], ops[b_]
                rhs[:, off[a_] + oa.Nz] = zeta[j][:, m, n] if j in zeta else 0.0
                fx = stack.fx[j - 1]
                mua, mub = oa.mu[:, 0], ob.mu[:, 0]
                t2 = stack.tau2[j - 1]
                R = np.zeros(Nx, dtype=complex)
                if j in psi:                     # lambda_a lambda_b psi_j
                    R += psi[j][:, m, n]
                    if n >= 1:
                        R += (mua + mub) * psi[j][:, m, n - 1]
                    if n >= 2:
                        R += mua * mub * psi[j][:, m, n - 2]
                for k in range(4):
                    if n - k < 0:
                        continue
                    for jj in range(2):
                        if (k, jj) == (0, 0) or m - jj < 0:
                            continue
                        ua = get(a_, n - k, m - jj)
                        ub = get(b_, n - k, m - jj)
                        if not (np.any(ua) or np.any(ub)):
                            continue
                        R -= (_iface_term(oa, ua, -1, mub, fx, ab, k, jj, getd(a_, n - k, m - jj))
                              - t2 * _iface_term(ob, ub, 0, mua, fx, ab, k, jj, getd(b_, n - k, m - jj)))
                rhs[:, off[b_]] = R
            # ---- one solve for every layer
            sol = _ifft(np.matmul(Ainv, _fft(rhs)[:, :, None])[:, :, 0])
            for l, o in enumerate(ops):
                ul = sol[:, off[l]:off[l] + o.Nz + 1]
                VX[l][n % 4, m] = o.dx(ul)
                VZ[l][n % 4, m] = o.dz(ul)
                if keep_volume:
                    V[l][n, m] = ul
                else:
                    V[l][n % nlev, m] = ul
            ubar[:, m, n] = get(0, n, m)[:, 0]
            wbar[:, m, n] = get(L - 1, n, m)[:, -1]
            for j in range(1, L):
                Vif[j - 1][:, m, n] = get(j - 1, n, m)[:, -1]
        if verbose:
            print(f'  multilayer_solve: n = {n}/{N} ({time.time() - t0:.1f} s)', flush=True)
    out = dict(ubar=ubar, wbar=wbar, V=Vif, gamma_bar=gbar, gp=gp, tau2=stack.tau2_total, kbar=kbar,
               t_inverse=t_inv, t_solve=time.time() - t0, size=int(S))
    if keep_volume:
        out['vol'] = [np.transpose(v, (2, 3, 1, 0)) for v in V]       # (Nx, Nz+1, M+1, N+1)
    return out


def _iface_term(o, u, node, mu_other, fx, ab, k, jj, d=None):
    """coefficient eps^k delta^jj of lambda_other * N~(u) at the interface node of layer o
    (N~ = (1 + g_x^2) u_z' - lambda g_x u_x - i alpha lambda g_x u,  lambda = 1 + eps mu, alpha = ab (1 + delta))."""
    uu = u[:, node]
    if d is None:
        ux, uz = o.dx(u)[:, node], u @ o.D[node]
    else:
        ux, uz = d[0][:, node], d[1][:, node]
    mu = o.mu[:, 0]

    def n_(kk, j_):                      # coefficient eps^kk delta^j_ of N~
        if kk == 0:
            return uz if j_ == 0 else 0.0
        if kk == 1:
            v = -1j * ab * fx * uu
            return (-fx * ux + v) if j_ == 0 else v
        if kk == 2:
            v = -1j * ab * mu * fx * uu
            return (fx ** 2 * uz - mu * fx * ux + v) if j_ == 0 else v
        return 0.0
    # lambda_other N~ = N~ + eps mu_other N~
    return n_(k, jj) + (mu_other * n_(k - 1, jj) if k >= 1 else 0.0)


def energies(res, stack, alpha_bar, Eps, delta, N, M, SumType=2, taylor_full_order=False):
    """(ee, ru, rl) maps (len(Eps), len(delta)) from the top and bottom traces (hops.energy_defect)."""
    Nx = res['ubar'].shape[0]
    ub = np.transpose(res['ubar'], (2, 1, 0))
    wb = np.transpose(res['wbar'], (2, 1, 0))
    Eps = np.atleast_1d(Eps); delta = np.atleast_1d(delta)
    return energy_defect(res['tau2'], ub, wb, 2 * np.pi, alpha_bar, res['gamma_bar'][0], res['gamma_bar'][-1],
                         Eps, delta, Nx, N, M, Eps.size, delta.size, SumType, taylor_full_order=taylor_full_order)
