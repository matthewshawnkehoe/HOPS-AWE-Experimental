"""Method of Manufactured Solutions for the n-layer HOPS/AWE solver.

Each layer l carries an exact (Bloch-reduced) Helmholtz solution with one Fourier mode r:
    top    layer 0     u_0 = A_0 e^{i r x} e^{+i gamma_0 (z - h_1)}                 (outgoing up)
    interior layer l   u_l = e^{i r x} [A_l e^{+i gamma_l (z - z_b)} + B_l e^{-i gamma_l (z - z_t)}]
    bottom layer L-1   u_  = B e^{i r x} e^{-i gamma (z - h_{L-1})}                   (outgoing down)
with gamma_l(delta) = sqrt(k_l^2 s^2 - (alpha_bar s + r)^2), s = 1 + delta (analytic continuation from s = 1).
They satisfy the Helmholtz equations and the transparent conditions exactly, so feeding the solver the interface
jumps of these fields
    zeta_j = u_{j-1} - u_j,    psi_j = d_N u_{j-1} - tau_j^2 d_N u_j      at z = h_j + eps f_j(x)
must return the exact interface traces V_j, the top trace ubar (z = a) and the bottom trace wbar (z = h_{L-1} - b),
order by order in (eps, delta).  The exact (eps, delta) Taylor coefficients of every quantity are computed by 2D
Cauchy integrals (FFT on the circles |eps| = rho_e, |delta| = rho_d) -- valid for any profile and any L.
"""
import numpy as np

from . import _paths  # noqa: F401
from hops import setup_2d


def _gamma(k2bar, ab, r, s):
    """analytic continuation of sqrt(k^2 s^2 - (ab s + r)^2) from s = 1 (outgoing branch there)"""
    g0 = np.lib.scimath.sqrt(k2bar - (ab + r) ** 2)
    if np.imag(g0) < 0:
        g0 = -g0
    return g0 * np.sqrt((k2bar * s ** 2 - (ab * s + r) ** 2) / g0 ** 2)


def manufactured(stack, omega_bar, alpha_bar, N, M, r=1, amps=None, rho_e=0.5, rho_d=0.1, K=64):
    """(zeta, psi, exact) for the stack: zeta/psi dicts {j: (Nx, M+1, N+1)}, exact dict with V (list), ubar, wbar."""
    L, Nx = stack.L, stack.f[0].size
    xx, _, _, _, _, _ = setup_2d(Nx, 2 * np.pi, alpha_bar, 1.0)
    rng = np.random.default_rng(7)
    if amps is None:
        amps = [(1 + rng.random() + 1j * rng.random(), 1 + rng.random() - 1j * rng.random()) for _ in range(L)]
    # delta circle: stay inside the disc of analyticity (branch points k_l s = +-(alpha s + r) of every layer)
    bps = []
    for nl in stack.n:
        k = nl * omega_bar
        for sg in (1, -1):
            den = sg * k - alpha_bar
            if abs(den) > 1e-12:
                bps.append(abs(r / den - 1))
    rho_d = min(rho_d, 0.4 * min(bps))
    th = 2 * np.pi * np.arange(K) / K
    E = rho_e * np.exp(1j * th)[:, None]                    # (K, 1)   eps on its circle
    Dl = rho_d * np.exp(1j * th)[None, :]                   # (1, K)   delta on its circle
    s = 1 + Dl
    k2 = [(nl * omega_bar) ** 2 for nl in stack.n]
    gam = [_gamma(k2[l], alpha_bar, r, s) for l in range(L)]
    alpha = alpha_bar * s
    ex = np.exp(1j * r * xx)[:, None, None]

    def field(l, z):
        """u_l and u_l,z at height z (array (Nx, K, K) or broadcastable), Bloch-reduced"""
        A, B = amps[l]
        g = gam[l][None]
        if l == 0:
            e = np.exp(1j * g * (z - stack.h[0]))
            return ex * A * e, ex * A * 1j * g * e
        if l == L - 1:
            e = np.exp(-1j * g * (z - stack.h[-1]))
            return ex * B * e, ex * B * (-1j * g) * e
        zb, zt = stack.h[l], stack.h[l - 1]
        ep, em = np.exp(1j * g * (z - zb)), np.exp(-1j * g * (z - zt))
        return ex * (A * ep + B * em), ex * 1j * g * (A * ep - B * em)

    def coeffs(F):
        """(Nx, K, K) samples -> (Nx, M+1, N+1) Taylor coefficients [:, m, n]"""
        C = np.fft.fft(np.fft.fft(F, axis=1), axis=2) / K ** 2
        n = np.arange(N + 1)
        m = np.arange(M + 1)
        C = C[:, :N + 1, :M + 1] / (rho_e ** n[None, :, None] * rho_d ** m[None, None, :])
        return np.transpose(C, (0, 2, 1))

    zeta, psi, V = {}, {}, []
    for j in range(1, L):
        g = stack.h[j - 1] + E[None] * stack.f[j - 1][:, None, None]      # (Nx, K, 1)
        gx = E[None] * stack.fx[j - 1][:, None, None]
        ua, uza = field(j - 1, g)
        ub, uzb = field(j, g)
        Na = uza - gx * (1j * r * ua + 1j * alpha[None] * ua)
        Nb = uzb - gx * (1j * r * ub + 1j * alpha[None] * ub)
        zeta[j] = coeffs(ua - ub)
        psi[j] = coeffs(Na - stack.tau2[j - 1] * Nb)
        V.append(coeffs(ua))
    ztop = stack.h[0] + stack.a
    zbot = stack.h[-1] - stack.b
    ubar = coeffs(field(0, ztop + 0 * E[None])[0] * np.ones((1, K, 1)))
    wbar = coeffs(field(L - 1, zbot + 0 * E[None])[0] * np.ones((1, K, 1)))

    def exact(eps, delta):
        """exact V_j (list), ubar, wbar on the x grid at real (eps, delta)"""
        s_ = 1 + delta
        gl = [_gamma(k2[l], alpha_bar, r, s_) for l in range(L)]
        e1 = np.exp(1j * r * xx)

        def fld(l, z):
            A, B = amps[l]
            g = gl[l]
            if l == 0:
                return e1 * A * np.exp(1j * g * (z - stack.h[0]))
            if l == L - 1:
                return e1 * B * np.exp(-1j * g * (z - stack.h[-1]))
            return e1 * (A * np.exp(1j * g * (z - stack.h[l])) + B * np.exp(-1j * g * (z - stack.h[l - 1])))
        Vx = [fld(j - 1, stack.h[j - 1] + eps * stack.f[j - 1]) for j in range(1, L)]
        out = dict(V=Vx, ubar=fld(0, stack.h[0] + stack.a + 0 * xx), wbar=fld(L - 1, stack.h[-1] - stack.b + 0 * xx))
        # volume fields at the TFE collocation nodes of every layer (physical z of each node)
        from hops import cheb
        vol = []
        for l in range(L):
            zb, zt, fb, ft, _, _ = stack.bounds(l)
            _, t = cheb(stack.Nz[l])
            sn = (t + 1) / 2
            Z = zb + eps * fb[:, None] + sn[None, :] * ((zt - zb) + eps * (ft - fb))[:, None]
            A, B = amps[l]
            g = gl[l]
            e1 = np.exp(1j * r * xx)[:, None]
            if l == 0:
                vol.append(e1 * A * np.exp(1j * g * (Z - stack.h[0])))
            elif l == L - 1:
                vol.append(e1 * B * np.exp(-1j * g * (Z - stack.h[-1])))
            else:
                vol.append(e1 * (A * np.exp(1j * g * (Z - stack.h[l])) + B * np.exp(-1j * g * (Z - stack.h[l - 1]))))
        out['vol'] = vol
        return out
    return zeta, psi, dict(V=V, ubar=ubar, wbar=wbar, rho_d=rho_d, rho_e=rho_e, exact=exact, amps=amps)


def coefficient_errors(res, exact):
    """max_{n,m} |c - c_exact| / max |c_exact| for V_j (each interface), ubar, wbar"""
    rel = lambda A, B: float(np.max(np.abs(A - B)) / np.max(np.abs(B)))
    out = {f'V{j + 1}': rel(res['V'][j], exact['V'][j]) for j in range(len(exact['V']))}
    out.update(ubar=rel(res['ubar'], exact['ubar']), wbar=rel(res['wbar'], exact['wbar']))
    return out
