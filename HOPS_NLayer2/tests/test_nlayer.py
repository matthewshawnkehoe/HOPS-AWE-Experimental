"""Validation of the n-layer HOPS/AWE solver (python -m pytest tests -q, or python test_mms_error_nlayer.py)."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
from hops_nl import Stack, multilayer_solve, energies          # noqa: E402
from hops_nl.reference import tmm, pointwise                   # noqa: E402
from hops_nl.mms import manufactured, coefficient_errors       # noqa: E402
from hops import two_layer_solve_coupled, setup_2d, setup_zeta_psi_n_m, cheb, csqrt   # noqa: E402


def _grid(Nx=32):
    xx = 2 * np.pi * np.arange(Nx) / Nx
    return xx, np.cos(xx), -np.sin(xx)


def test_two_layers_equal_2d_solver():
    """L = 2: identical (to round-off) to hops.two_layer_solve_coupled, normal and oblique incidence"""
    xx, f, fx = _grid()
    N = M = 10
    for nw, alpha in ((1.1, 0.0), (1.5, 0.13), (0.4 + 2j, 0.05)):
        st = Stack([1.0, nw], [], [f], [fx], 1.0, 1.0, 32)
        r = multilayer_solve(st, 1.5, alpha, N, M)
        gu, gw = csqrt(1.5 ** 2 - alpha ** 2), csqrt((nw * 1.5) ** 2 - alpha ** 2)
        _, pp, ap, gup, _, _ = setup_2d(32, 2 * np.pi, alpha, gu)
        _, _, _, gwp, _, _ = setup_2d(32, 2 * np.pi, alpha, gw)
        z, p = setup_zeta_psi_n_m(xx, pp, alpha, gu, f, fx, 32, N, M)
        Dz, _ = cheb(32)
        U, W, ub, wb = two_layer_solve_coupled((1 / nw) ** 2, z, p, gup, gwp, N, 32, f, fx, pp, alpha, gu, gw, Dz,
                                               1.0, 1.0, 32, M, np.eye(33), ap)
        for A, B in ((r['V'][0], U), (r['ubar'], ub), (r['wbar'], wb)):
            assert np.max(np.abs(A - B)) / np.max(np.abs(B)) < 1e-10


def test_flat_stacks_vs_transfer_matrix():
    """eps = 0: R(omega) over a whole delta window equals the exact transfer-matrix result (3, 4, 6 layers, TE/TM)"""
    xx, f, fx = _grid()
    N = M = 12
    for n, d, mode, alpha in (([1.0, 1.5, 1.0, 2.0], [0.8, 1.3], 'TM', 0.0),
                              ([1.0, 2.3, 1.45, 2.3, 1.45, 1.52], [0.5, 0.7, 0.5, 0.7], 'TE', 0.1),
                              ([1.0, 0.2 + 3j, 1.5], [0.15], 'TM', 0.0)):
        L = len(n)
        st = Stack(n, d, [f] * (L - 1), [fx] * (L - 1), 1.0, 1.0, 24, mode)
        r = multilayer_solve(st, 1.3, alpha, N, M)
        deltas = np.linspace(-0.12, 0.12, 7)
        ee, ru, rl = energies(r, st, alpha, [0.0], deltas, N, M, 2)
        for i, dl in enumerate(deltas):
            R, T = tmm(n, d, 1.3 * (1 + dl), alpha * (1 + dl), mode)
            assert abs(np.real(ru[0, i]) - R) < 1e-11


def test_fictitious_interface_invariance():
    """splitting the substrate by a flat interface between identical media changes nothing"""
    xx, f, fx = _grid()
    Z = 0 * xx
    R2 = pointwise(Stack([1.0, 1.5], [], [f], [fx], 1.0, 1.5, 32), 1.3, 0.0, [0.1, 0.2], N=20)[0]
    R3 = pointwise(Stack([1.0, 1.5, 1.5], [0.7], [f, Z], [fx, Z], 1.0, 0.8, 32), 1.3, 0.0, [0.1, 0.2], N=20)[0]
    assert np.max(np.abs(R2 - R3)) < 1e-11


def test_energy_conservation_lossless_four_layers():
    """lossless 4-layer stack with three different corrugations: R + T = 1"""
    xx, f, fx = _grid()
    st = Stack([1.0, 1.5, 2.0, 1.2], [0.8, 1.0], [f, np.cos(xx + 1.0), np.cos(2 * xx)],
               [fx, -np.sin(xx + 1.0), -2 * np.sin(2 * xx)], 1.0, 1.0, 24)
    R, T, D = pointwise(st, 1.3, 0.0, [0.05, 0.1, 0.2], N=24)
    assert np.max(np.abs(D)) < 1e-11


def test_manufactured_solution_all_orders():
    """MMS: interface traces, artificial-boundary traces recovered order by order (L = 3, 4, 6).
    Individual high-order coefficients carry round-off amplified by the Chebyshev operator (grows with Nz, ~1e-8 at
    (n, m) = (4, 2) for L = 6); the SUMMED fields are accurate to ~1e-12 (test_single_eps_delta_volume_fields)."""
    xx, f, fx = _grid()
    pr = [(f, fx), (np.cos(2 * xx + 0.5) / 2, -np.sin(2 * xx + 0.5)), (0.8 * np.sin(xx), 0.8 * np.cos(xx))]
    for L in (3, 4, 6):
        n = [1.0, 1.5, 2.0 + 0.1j, 1.2, 1.8, 1.05][:L - 1] + [1.05]
        st = Stack(n, [0.9, 1.1, 0.8, 1.0][:L - 2], [pr[j % 3][0] for j in range(L - 1)],
                   [pr[j % 3][1] for j in range(L - 1)], 1.0, 1.0, 32)
        z, p, ex = manufactured(st, 1.3, 0.0, 4, 2)
        res = multilayer_solve(st, 1.3, 0.0, 4, 2, zeta=z, psi=p)
        assert max(coefficient_errors(res, ex).values()) < 1e-7


def test_single_eps_delta_volume_fields():
    """test_single_eps_delta_nlayer RunNumber 1 and 3: every trace and every layer's volume field to 1e-9"""
    import test_single_eps_delta_nlayer as T
    for rn in (1, 3):
        out = T.run(rn, make_plots=False, verbose=False)
        N, M = out['cfg']['N'], out['cfg']['M']
        assert np.max(out['nm'][:, N, M]) < 1e-9


def test_mms_oblique_incidence():
    """alpha = 0.15, L = 5: the Bloch terms (2 i alpha in the TFE operator, i alpha g_x in the flux) are right"""
    import mms_error_nlayer as mms
    out = mms.run(dict(L=5, alpha=0.15, N=8, M=8, eps_max=0.05, delta_max=0.02, n_eps=5, n_delta=5), verbose=False)
    assert max(np.max(v) for (k, _), v in out['err'].items() if k == 'pade') < 1e-8


def test_refl_map_two_layers_equals_refl_map_py():
    """the full n-layer pipeline with L = 2 reproduces refl_map.py (silver, cos 4x) on a reduced grid"""
    import refl_map_nlayer as R
    import refl_map as rm
    res, _ = R.run('silver_L2_ref', qq=(1, 2), N_Eps=5, N_delta=5, workers=1, verbose=False)
    ref, _ = rm.run('silver', qq=(1, 2), N_Eps=5, N_delta=5, workers=1, verbose=False)
    for a, b in zip(res, ref):
        assert np.max(np.abs(np.real(a['ru']) - np.real(b['ru']))) < 1e-9
        assert np.max(np.abs(np.real(a['ee']) - np.real(b['ee']))) < 1e-9


def test_layer_absorption():
    """per-layer absorptance from the volume series: equals TMM layer by layer (flat, TE and TM) and sums to
    1 - R - T on the corrugated stack (gold film on glass, metal-insulator-metal absorber)"""
    import refl_map_nlayer as R
    from hops_nl.absorption import layer_absorption
    for sc in ('mim_absorber', 'gold_film_glass'):
        for mode in ('TM', 'TE'):
            res, info = R.run(sc, qq=(0,), N_Eps=3, N_delta=3, workers=1, verbose=False, keep_fields=True, mode=mode,
                             omega_range=(0.5, 0.6))
            r = res[0]
            A0 = layer_absorption(r, info, 0.0, r['delta'][1])
            _, _, At = tmm(r['n_layers'], info['thick'], r['omega'][1], 0.0, mode, layers=True)
            assert np.max(np.abs(A0 - At)) < 1e-8
            for ie in (1, 2):
                A = layer_absorption(r, info, r['Eps'][ie], r['delta'][2])
                lossy_sub = abs(np.imag(r['n_layers'][-1])) > 0
                target = 1 - np.real(r['ru'][ie, 2]) if lossy_sub else np.real(r['ee'][ie, 2])
                assert abs(A.sum() - target) < 1e-6


def test_exact_dispersion_fixed_angle():
    """exact-dispersion mode (one eps-only HOPS solve per frequency with n_l(lambda)) at a FIXED incidence angle:
    the flat (eps = 0) row equals the transfer matrix with the dispersive indices (Kretschmann: BK7 | Au | water,
    68 deg), the corrugated rows agree with the converged pointwise reference, and the per-layer absorption sums
    to 1 - R"""
    import refl_map_nlayer as R
    from hops_nl.reference import check_map
    from hops_nl.absorption import layer_absorption
    res, info = R.run('kretschmann_au', N_Eps=3, N_delta=4, workers=1, verbose=False, keep_fields=True,
                      omega_range=(0.6 / 0.75, 0.6 / 0.70))
    for r in res:
        for j, lam in enumerate(r['lam'] * 0.6 / (2 * np.pi)):
            ns = [complex(R._index(k, lam)) for k in ('BK7', 'Au', 'water')]
            a = np.real(ns[0]) * (0.6 / lam) * np.sin(np.radians(68.0))
            assert abs(np.real(r['ru'][0, j]) - tmm(ns, [0.05 * 2 * np.pi / 0.6], 0.6 / lam, a, 'TM')[0]) < 1e-7
    assert check_map(res, info)['max'] < 1e-6
    r = res[0]
    A = layer_absorption(r, info, r['Eps'][-1], r['delta'][1])
    assert abs(A.sum() - (1 - np.real(r['ru'][-1, 1]))) < 1e-6
