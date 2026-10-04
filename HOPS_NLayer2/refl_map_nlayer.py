"""refl_map_nlayer.py -- reflectivity map R(eps, omega) and energy defect D of a 2D grating stack with L >= 2 layers
(the n-layer extension of refl_map.py / refl_map.m), computed with the joint (eps, delta) HOPS/AWE expansion:
one multi-layer solve per frequency window, then Taylor/Pade summation at every (eps, omega).

    python refl_map_nlayer.py                                  # SCENARIO below
    python refl_map_nlayer.py --scenario bragg_L6
    python refl_map_nlayer.py --list-scenarios
    python refl_map_nlayer.py --layers 1 1.5 Ag 1.0 --thick 0.6 0.25 --profiles cosx cosx flat --q 1 2

Figures go to figures/refl_map/L<L>/ (refl_map_<scenario>_{R,D}.png, plus a stack sketch in the R figure).
Stack keys: layers (L materials top -> bottom: numbers or hops.materials keys), thick (L-2 interior thicknesses,
nondimensional: period = 2 pi) or thick_um (micrometres, with period), profiles (L-1 interface profiles: refl_map
profile names, 'flat', or 'expr:<numpy expression in x>'), amps (L-1 amplitude factors: g_j = eps * amp_j * f_j).
"""
import argparse
import os
import sys
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hops_nl import Stack, multilayer_solve, energies       # noqa: E402
from hops_nl.summation_ext import energies_eps              # noqa: E402
import refl_map as rm                                       # noqa: E402  (HOPS_Python, via hops_nl._paths)
from hops import materials as mat                           # noqa: E402
from hops.plotting import safe_log10, matlab_contourf, shared_colorbar   # noqa: E402

SCENARIO = 'film_dielectric'
SHOW = True
WORKERS = max(1, (os.cpu_count() or 2))
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'refl_map')

_P = dict(eps_max=0.2, alpha=0.0, M=12, Nx=32, Nz=24, a=1.0, b=1.0, mode='TM', taylor=False, q=(1, 2, 3, 4, 5, 6),
          period=None, windows='joint', sigma=0.99, max_delta=0.1, amps=None, relative=False)


def _s(desc, **kw):
    d = dict(_P)
    d.update(kw)
    d['desc'] = desc
    return d


# quarter-wave Bragg stacks (n_H = 2.3, n_L = 1.45) designed for omega_0 = 1.5 (centre of band q = 1)
_W0, _NH, _NL, _NS = 1.5, 2.3, 1.45, 1.52
_tH, _tL = np.pi / (2 * _NH * _W0), np.pi / (2 * _NL * _W0)


def _bragg(L, conformal):
    inner = [_NH if i % 2 == 0 else _NL for i in range(L - 2)]
    return dict(layers=[1.0] + inner + [_NS], thick=[_tH if v == _NH else _tL for v in inner],
                profiles=['cosx'] * (L - 1) if conformal else ['cosx'] + ['flat'] * (L - 2))


def _graded(L):
    ns = list(np.round(np.linspace(1.1, 1.5, L - 2), 3)) if L > 2 else []
    return dict(layers=[1.0] + ns + [1.6], thick=[0.6] * (L - 2), profiles=['cosx'] * (L - 1))


SCENARIOS = {
    # ------------------------------------------------------------------ L = 3
    'film_dielectric': _s('L=3: vacuum / n=1.3 film (d=1) / n=1.1, conformal cos x (3-layer analogue of paper Fig. 9), Pade',
                          layers=[1.0, 1.3, 1.1], thick=[1.0], profiles=['cosx', 'cosx']),
    'film_dielectric_paper': _s('L=3: as film_dielectric with the paper settings (MATLAB half-order Taylor, paper windows)',
                                layers=[1.0, 1.3, 1.1], thick=[1.0], profiles=['cosx', 'cosx'], taylor=True, M=14,
                                windows='paper', max_delta=None, relative=True),
    'film_dielectric_top': _s('L=3: as film_dielectric, only the top interface corrugated', layers=[1.0, 1.3, 1.1],
                              thick=[1.0], profiles=['cosx', 'flat']),
    'film_dielectric_anti': _s('L=3: as film_dielectric, bottom interface in anti-phase (-cos x)', layers=[1.0, 1.3, 1.1],
                               thick=[1.0], profiles=['cosx', 'expr:-cos(x)']),
    'silver_film': _s('L=3: vacuum / silver film 0.05+2.275i (d=0.3) / vacuum (insulator-metal-insulator: coupled SPPs), '
                      'conformal cos x, Pade', layers=[1.0, 0.05 + 2.275j, 1.0], thick=[0.3], profiles=['cosx', 'cosx']),
    'silver_film_thick': _s('L=3: as silver_film with d = 1.5 (thick film: the two SPPs decouple)',
                            layers=[1.0, 0.05 + 2.275j, 1.0], thick=[1.5], profiles=['cosx', 'cosx']),
    'gold_film_glass': _s('L=3: vacuum / Au film 40 nm / BK7, period 0.6 um, conformal cos x (SPR on a film)',
                          layers=['air', 'Au', 'BK7'], thick_um=[0.04], period=0.6, profiles=['cosx', 'cosx'],
                          q=(0, 1, 2), max_delta=0.05, omega_range=(0.4, 2.0)),
    'gmr_filter': _s('L=3: vacuum / TiO2 120 nm / fused silica, period 0.4 um, top grating: guided-mode resonance filter',
                     layers=['air', 'TiO2', 'fused_silica'], thick_um=[0.12], period=0.4, profiles=['cosx', 'flat'],
                     q=(0,), windows='joint', max_delta=0.03, eps_max=0.1, omega_range=(0.33, 0.999)),
    'ar_coating': _s('L=3: vacuum / MgF2 quarter-wave (100 nm) / BK7, period 0.5 um: anti-reflection coated grating',
                     layers=['air', 'MgF2', 'BK7'], thick_um=[0.1], period=0.5, profiles=['cosx', 'cosx'],
                     q=(0, 1), windows='joint', max_delta=0.05, omega_range=(0.4, 1.6)),
    # ------------------------------------------------------------------ L = 4
    'mim_absorber': _s('L=4: vacuum / Ag 25 nm / SiO2 60 nm / Ag, period 0.5 um, top grating: metal-insulator-metal gap plasmons',
                       layers=['air', 'Ag', 'SiO2', 'Ag'], thick_um=[0.025, 0.06], period=0.5,
                       profiles=['cosx', 'flat', 'flat'], q=(0, 1), max_delta=0.05, eps_max=0.12,
                       omega_range=(0.3, 1.5)),
    'bragg_L4': _s('L=4: vacuum / H / L / glass (one quarter-wave pair, n_H 2.3, n_L 1.45), top grating', **_bragg(4, False),
                   q=(1, 2), windows='joint'),
    'dielectric_L4': _s('L=4: vacuum / 1.3 / 1.2 / 1.1, conformal cos x, Taylor', layers=[1.0, 1.3, 1.2, 1.1],
                        thick=[1.0, 1.0], profiles=['cosx'] * 3),
    # ------------------------------------------------------------------ L = 5 .. 8
    'bragg_L5': _s('L=5: vacuum / H / L / H / glass, top grating', **_bragg(5, False), q=(1, 2), windows='joint'),
    'bragg_L6': _s('L=6: vacuum / (H L)x2 / glass, top grating', **_bragg(6, False), q=(1, 2), windows='joint'),
    'bragg_L7': _s('L=7: vacuum / H L H L H / glass, top grating', **_bragg(7, False), q=(1, 2), windows='joint'),
    'bragg_L8': _s('L=8: vacuum / (H L)x3 / glass, top grating (stop band at omega ~ 1.5)', **_bragg(8, False),
                   q=(1, 2), windows='joint'),
    'bragg_L8_conformal': _s('L=8: as bragg_L8 with all 7 interfaces corrugated (conformal)', **_bragg(8, True),
                             q=(1, 2), windows='joint'),
    'graded_L5': _s('L=5: graded index 1.1 -> 1.5 (3 layers of d = 0.6) over n = 1.6, conformal', **_graded(5)),
    'graded_L8': _s('L=8: graded index 1.1 -> 1.5 (6 layers) over n = 1.6, conformal', **_graded(8)),
    'hmm_L8': _s('L=8: hyperbolic metamaterial (Ag 0.05+2.275i d=0.15 / n=2.3 d=0.3) x3 on glass, conformal cos x',
                 layers=[1.0] + [0.05 + 2.275j, 2.3] * 3 + [1.5], thick=[0.15, 0.3] * 3,
                 profiles=['cosx'] * 7, q=(1, 2, 3)),
    'hmm_L8_top': _s('L=8: as hmm_L8 but only the top interface corrugated (eps <= 0.1 < d_Ag = 0.15)',
                     layers=[1.0] + [0.05 + 2.275j, 2.3] * 3 + [1.5], thick=[0.15, 0.3] * 3,
                     profiles=['cosx'] + ['flat'] * 6, q=(1, 2, 3), eps_max=0.1),
}


# ======================================================================================================================
# Materials gallery for L = 3 ... 9 (refractiveindex.info pages bundled with HOPS_Python + hops_nl.materials_ext):
# the multilayer analogues of the 2-layer material scenarios (SPPs -> coupled / long-range SPPs, SPR sensing ->
# Kretschmann + adhesion + biolayer, ENZ / phase change / phonon polaritons -> thin films on mirrors, absorbers,
# Tamm plasmons, cavities, mirrors).  Thicknesses in um, period in um.
# ======================================================================================================================
def _qw(key, lam0):
    """quarter-wave thickness (um) of material key at lam0 (um)"""
    return lam0 / (4 * np.real(complex(mat.refractive_index(key, lam0, warn=False))))


def _lam(lo, hi, P):
    """omega_range covering lambda in [lo, hi] um for period P"""
    return (P / hi, P / lo)


_TH, _TL = _qw('TiO2', 0.7), _qw('SiO2', 0.7)


def _cavity(p, lam0=0.7):
    """microcavity: air | (H L)^p H | L (half wave) | H (L H)^p | BK7  ->  L = 4p + 5 layers"""
    mir = ['TiO2', 'SiO2'] * p + ['TiO2']
    tm = [_TH, _TL] * p + [_TH]
    return dict(layers=['air'] + mir + ['SiO2'] + mir[::-1] + ['BK7'], thick_um=tm + [2 * _qw('SiO2', lam0)] + tm[::-1],
                profiles=['cosx'] + ['flat'] * (4 * p + 3))


_GAAS, _ALAS = _qw('GaAs', 0.98), 0.98 / (4 * 2.95)
_GE, _ZNS = _qw('Ge_IR', 4.0), _qw('ZnS', 4.0)
_G = dict(windows='joint', max_delta=0.05, q=(0, 1, 2), M=12, dispersion='exact')
SCENARIOS.update({
    # ---------------------------------------------------------------- L = 3: plasmonic films
    'lrspp_ag10': _s('L=3: silica / Ag 10 nm / silica (symmetric IMI): long-range SPP, conformal cos x, P 0.5 um',
                     layers=['fused_silica', 'Ag', 'fused_silica'], thick_um=[0.010], period=0.5,
                     profiles=['cosx', 'cosx'], omega_range=_lam(0.5, 1.0, 0.5), **_G),
    'lrspp_ag20': _s('L=3: silica / Ag 20 nm / silica: long-range + short-range SPP', layers=['fused_silica', 'Ag', 'fused_silica'],
                     thick_um=[0.020], period=0.5, profiles=['cosx', 'cosx'], omega_range=_lam(0.5, 1.0, 0.5), **_G),
    'lrspp_ag40': _s('L=3: silica / Ag 40 nm / silica: the two SPPs decouple', layers=['fused_silica', 'Ag', 'fused_silica'],
                     thick_um=[0.040], period=0.5, profiles=['cosx', 'cosx'], omega_range=_lam(0.5, 1.0, 0.5), **_G),
    'imi_asym_ag20': _s('L=3: air / Ag 20 nm / silica (asymmetric: no long-range SPP)', layers=['air', 'Ag', 'fused_silica'],
                        thick_um=[0.020], period=0.5, profiles=['cosx', 'cosx'], omega_range=_lam(0.5, 1.0, 0.5), **_G),
    'kretschmann_au': _s('L=3: BK7 prism / Au 50 nm / water, TM at 68 deg: Kretschmann SPR + grating, P 0.6 um',
                         layers=['BK7', 'Au', 'water'], thick_um=[0.050], period=0.6, profiles=['cosx', 'cosx'],
                         theta=68.0, eps_max=0.15, omega_range=_lam(0.55, 0.86, 0.6), **_G),
    # ---------------------------------------------------------------- L = 3: ENZ, phase change, phonons, absorbers
    'berreman_ito': _s('L=3: air / ITO 30 nm / silica, TM at 60 deg: Berreman ENZ absorption at 1.26 um, P 1 um',
                       layers=['air', 'ITO', 'fused_silica'], thick_um=[0.030], period=1.0, profiles=['cosx', 'cosx'],
                       theta=60.0, omega_range=_lam(1.0, 1.65, 1.0), **_G),
    'ge_on_au': _s('L=3: air / Ge 10 nm / Au: ultrathin lossy film on a metal (Kats 2013), near-perfect absorption, P 0.5 um',
                   layers=['air', 'Ge', 'Au'], thick_um=[0.010], period=0.5, profiles=['cosx', 'cosx'],
                   omega_range=_lam(0.45, 0.95, 0.5), **_G),
    'vo2_cold_sapphire': _s('L=3: air / VO2 150 nm (25 C, insulating) / sapphire, mid-IR, P 8 um',
                            layers=['air', 'VO2_cold', 'sapphire_IR'], thick_um=[0.15], period=8.0,
                            profiles=['cosx', 'cosx'], omega_range=_lam(8.0, 14.0, 8.0), **_G),
    'vo2_hot_sapphire': _s('L=3: air / VO2 150 nm (100 C, metallic) / sapphire, mid-IR, P 8 um',
                           layers=['air', 'VO2_hot', 'sapphire_IR'], thick_um=[0.15], period=8.0,
                           profiles=['cosx', 'cosx'], omega_range=_lam(8.0, 14.0, 8.0), **_G),
    'vo2_f30_sapphire': _s('L=3: air / VO2 150 nm in the transition (30 % metallic, Bruggeman) / sapphire, P 8 um',
                           layers=['air', 'VO2_f30', 'sapphire_IR'], thick_um=[0.15], period=8.0,
                           profiles=['cosx', 'cosx'], omega_range=_lam(8.0, 14.0, 8.0), **_G),
    # ---------------------------------------------------------------- L = 4
    'kretschmann_cr': _s('L=4: BK7 / Cr 2 nm adhesion / Au 48 nm / water, 68 deg: adhesion-layer damping of the SPR',
                         layers=['BK7', 'Cr', 'Au', 'water'], thick_um=[0.002, 0.048], period=0.6, profiles=['cosx'] * 3,
                         theta=68.0, eps_max=0.15, omega_range=_lam(0.55, 0.86, 0.6), **_G),
    'kretschmann_ti': _s('L=4: BK7 / Ti 2 nm adhesion / Au 48 nm / water, 68 deg',
                         layers=['BK7', 'Ti', 'Au', 'water'], thick_um=[0.002, 0.048], period=0.6, profiles=['cosx'] * 3,
                         theta=68.0, eps_max=0.15, omega_range=_lam(0.55, 0.86, 0.6), **_G),
    'biosensor_L4': _s('L=4: BK7 / Au 50 nm / 10 nm protein layer (n 1.45) / water, 68 deg: SPR shift by binding',
                       layers=['BK7', 'Au', 1.45, 'water'], thick_um=[0.050, 0.010], period=0.6, profiles=['cosx'] * 3,
                       theta=68.0, eps_max=0.15, omega_range=_lam(0.55, 0.86, 0.6), **_G),
    'salisbury': _s('L=4: air / Cr 6 nm / SiO2 100 nm / Al: Salisbury-screen absorber, P 0.4 um',
                    layers=['air', 'Cr', 'SiO2', 'Al'], thick_um=[0.006, 0.10], period=0.4, profiles=['cosx'] * 3,
                    omega_range=_lam(0.4, 0.95, 0.4), **_G),
    'perovskite_cell': _s('L=4: air / ITO 80 nm / MAPbI3 300 nm / Ag: thin perovskite cell with a grating, P 0.5 um',
                          layers=['air', 'ITO', 'MAPbI3', 'Ag'], thick_um=[0.08, 0.30], period=0.5, profiles=['cosx'] * 3,
                          omega_range=_lam(0.45, 0.95, 0.5), Nz=32, **_G),
    'si_cell': _s('L=4: air / Si3N4 70 nm / Si 300 nm / Ag: thin c-Si cell, grating light trapping near the band edge, P 0.5 um',
                  layers=['air', 'Si3N4', 'Si', 'Ag'], thick_um=[0.07, 0.30], period=0.5, profiles=['cosx'] * 3,
                  omega_range=_lam(0.5, 1.1, 0.5), Nz=32, **_G),
    # ---------------------------------------------------------------- L = 5, 6
    'mim_filter': _s('L=5: air / Ag 30 nm / SiO2 120 nm / Ag 30 nm / BK7: Fabry-Perot colour filter, P 0.5 um',
                     layers=['air', 'Ag', 'SiO2', 'Ag', 'BK7'], thick_um=[0.03, 0.12, 0.03], period=0.5,
                     profiles=['cosx'] * 4, omega_range=_lam(0.45, 0.95, 0.5), **_G),
    'hmm_au_tio2_L6': _s('L=6: air / (Au 20 nm / TiO2 40 nm) x 2 / BK7, conformal, P 0.5 um',
                         layers=['air', 'Au', 'TiO2', 'Au', 'TiO2', 'BK7'], thick_um=[0.02, 0.04] * 2, period=0.5,
                         profiles=['cosx'] * 5, omega_range=_lam(0.5, 0.95, 0.5), **_G),
    'ir_mirror_L6': _s('L=6: air / (Ge / ZnS quarter-wave at 4 um) x 2 / CaF2: high-contrast IR mirror, top grating, P 3 um',
                       layers=['air', 'Ge_IR', 'ZnS', 'Ge_IR', 'ZnS', 'CaF2'], thick_um=[_GE, _ZNS] * 2, period=3.0,
                       profiles=['cosx'] + ['flat'] * 4, omega_range=_lam(2.6, 7.0, 3.0), **_G),
    # ---------------------------------------------------------------- L = 8, 9
    'tamm_ag': _s('L=8: air / Ag 30 nm / TiO2 (0.6 x quarter) / SiO2 / TiO2 / SiO2 / TiO2 / BK7: Tamm plasmon, P 0.65 um',
                  layers=['air', 'Ag', 'TiO2', 'SiO2', 'TiO2', 'SiO2', 'TiO2', 'BK7'],
                  thick_um=[0.03, 0.6 * _TH, _TL, _TH, _TL, _TH], period=0.65, profiles=['cosx', 'cosx'] + ['flat'] * 5,
                  omega_range=_lam(0.55, 0.95, 0.65), **_G),
    'tamm_au': _s('L=8: as tamm_ag with Au 30 nm', layers=['air', 'Au', 'TiO2', 'SiO2', 'TiO2', 'SiO2', 'TiO2', 'BK7'],
                  thick_um=[0.03, 0.6 * _TH, _TL, _TH, _TL, _TH], period=0.65, profiles=['cosx', 'cosx'] + ['flat'] * 5,
                  omega_range=_lam(0.55, 0.95, 0.65), **_G),
    'hmm_au_tio2_L8': _s('L=8: air / (Au 20 nm / TiO2 40 nm) x 3 / BK7, conformal, P 0.5 um',
                         layers=['air'] + ['Au', 'TiO2'] * 3 + ['BK7'], thick_um=[0.02, 0.04] * 3, period=0.5,
                         profiles=['cosx'] * 7, omega_range=_lam(0.5, 0.95, 0.5), **_G),
    'vcsel_dbr_L8': _s('L=8: air / (GaAs / AlAs (n 2.95) quarter-wave at 0.98 um) x 3 / GaAs: VCSEL-type mirror, P 0.5 um',
                       layers=['air'] + ['GaAs', 2.95] * 3 + ['GaAs'], thick_um=[_GAAS, _ALAS] * 3, period=0.5,
                       profiles=['cosx'] + ['flat'] * 6, omega_range=_lam(0.88, 1.15, 0.5), **_G),
    'ir_mirror_L8': _s('L=8: air / (Ge / ZnS) x 3 / CaF2: high-contrast IR mirror (stop band 3-5.5 um), P 3 um',
                       layers=['air'] + ['Ge_IR', 'ZnS'] * 3 + ['CaF2'], thick_um=[_GE, _ZNS] * 3, period=3.0,
                       profiles=['cosx'] + ['flat'] * 6, omega_range=_lam(2.6, 7.0, 3.0), **_G),
    'microcavity_L9': _s('L=9: air / H L H / SiO2 half-wave cavity / H L H / BK7 (TiO2/SiO2 at 0.7 um), top grating, P 0.65 um',
                         **_cavity(1), period=0.65, omega_range=_lam(0.55, 0.95, 0.65), **_G),
})

SCENARIOS.update({
    'silver_L2_ref': _s('L=2 reference: the paper silver map (vacuum / silver, cos 4x) through the n-layer solver',
                        layers=[1.0, 0.05 + 2.275j], thick=[], profiles=['cos4x'], M=15, Nz=32, windows='paper', max_delta=None,
                        relative=True),
})


def profile(name, xx):
    if name in ('flat', 'none', '0'):
        return 0 * xx, 0 * xx
    return rm.profile_fn(name, xx)


def _index(spec, lam_um, db=None):
    lit = mat.parse_index(spec) if not isinstance(spec, (int, float, complex)) else complex(spec)
    if lit is not None:
        return complex(lit)
    return complex(mat.refractive_index(spec, lam_um, db=db))


def describe(sc):
    """short text of the stack: 'air | TiO2 0.12um | ... | BK7'"""
    lay = sc['layers']
    th = sc.get('thick_um') or sc.get('thick') or []
    u = ' um' if sc.get('thick_um') else ''
    fmt = lambda v: (f'{complex(v).real:.3g}' + (f'{complex(v).imag:+.3g}i' if complex(v).imag else '')) \
        if isinstance(v, (int, float, complex)) else str(v)
    mid = [f'{fmt(lay[i])} ({th[i - 1]:.3g}{u})' for i in range(1, len(lay) - 1)]
    return ' | '.join([fmt(lay[0])] + mid + [fmt(lay[-1])])


def _window(win, c):
    """one frequency window: build the stack with the window's indices, solve, sum (module level: worker).

    c['dispersion'] == 'window' (default): ONE joint (eps, delta) HOPS/AWE solve, refractive indices frozen at the
        window centre (as refl_map.m).
    c['dispersion'] == 'exact': every frequency column omega_j of the window is solved separately (M = 0: HOPS in
        eps only, Pade in eps) with the indices n_l(lambda_j) -- exact material dispersion, ~5-10x the cost."""
    key, q, omega_bar, dmax = win
    t0 = time.time()
    alpha_bar = c.get('alpha_w', {}).get(key, c['alpha'])
    delta = np.array([0.0]) if c['N_delta'] == 1 else np.linspace(-dmax, dmax, c['N_delta'])
    omega = omega_bar * (1 + delta)
    st_ = 1 if c['taylor'] else 2
    tfo = c.get('taylor_full_order', False)
    if c.get('dispersion', 'window') == 'exact':
        ne = len(c['Eps'])
        ee, ru, rl = (np.zeros((ne, delta.size)) for _ in range(3))
        ru_f = np.zeros((ne, delta.size))
        vols, ncols, acols, t_s = [], [], [], 0.0
        for j, om in enumerate(omega):
            lam_um = c['period'] / om if c.get('period') else None
            ns = [_index(v, lam_um) for v in c['layers']]
            st = Stack(ns, c['thick'], c['f'], c['fx'], c['a'], c['b'], c['Nz'], c['mode'])
            aj = alpha_bar * (1 + delta[j])
            if c.get('theta') is not None:           # fixed angle with the dispersive top-layer index at lambda_j
                aj = np.real(ns[0]) * om * np.sin(np.radians(c['theta']))
            res = multilayer_solve(st, om, aj, c['N'], 0, keep_volume=c.get('keep_fields', False))
            t_s += res['t_solve']
            ee[:, j], ru[:, j], rl[:, j] = energies_eps(res, st, aj, c['Eps'], c['N'], st_)
            ru_f[:, j] = ru[0, j]
            ncols.append(ns)
            acols.append(aj)
            if c.get('keep_fields'):
                vols.append(res['vol'])
        out = dict(key=key, q=q, omega_bar=omega_bar, delta=delta, omega=omega, lam=2 * np.pi / omega, Eps=c['Eps'],
                   ee=ee, ru=ru, rl=rl, ru_flat=ru_f, RR=ru / ru_f if c['relative'] else ru,
                   n_layers=np.array(ncols[len(ncols) // 2]), n_cols=np.array(ncols), alpha_cols=np.array(acols), n_u=ncols[0][0], n_w=ncols[0][-1],
                   t_solve=t_s, size=res['size'], dmax=dmax, alpha_bar=alpha_bar, dispersion='exact')
        if c.get('keep_fields'):
            out['vol_cols'] = vols
        if c['verbose']:
            print(f'{key}: omega in [{omega.min():.3f}, {omega.max():.3f}], exact dispersion, {delta.size} columns: '
                  f'{time.time() - t0:.1f} s', flush=True)
        return out
    ns = c['n_w'][key]
    st = Stack(ns, c['thick'], c['f'], c['fx'], c['a'], c['b'], c['Nz'], c['mode'])
    res = multilayer_solve(st, omega_bar, alpha_bar, c['N'], c['M'], keep_volume=c.get('keep_fields', False))
    t1 = time.time()
    ee, ru, rl = energies(res, st, alpha_bar, c['Eps'], delta, c['N'], c['M'], st_, taylor_full_order=tfo)
    # flat reference R_flat(omega): the eps = 0 row of the same (eps, delta) summation (a multilayer's flat
    # reflectance oscillates with omega -- Fabry-Perot -- so the delta = 0 value alone is not enough)
    ee_f, ru_f, rl_f = energies(res, st, alpha_bar, [0.0], delta, c['N'], c['M'], st_, taylor_full_order=tfo)
    ru_f = np.broadcast_to(ru_f, ru.shape)
    if c['verbose']:
        print(f'{key}: omega in [{omega.min():.3f}, {omega.max():.3f}], L = {st.L}, matrix {res["size"]}: '
              f'solve {res["t_solve"]:.1f} s (inverse {res["t_inverse"]:.2f} s), energy {time.time() - t1:.1f} s', flush=True)
    out = dict(key=key, q=q, omega_bar=omega_bar, delta=delta, omega=omega, lam=2 * np.pi / omega, Eps=c['Eps'],
               ee=ee, ru=ru, rl=rl, ru_flat=ru_f, RR=ru / ru_f if c['relative'] else ru, n_layers=np.array(ns),
               n_u=ns[0], n_w=ns[-1], ubar_n_m=res['ubar'], wbar_n_m=res['wbar'], t_solve=res['t_solve'],
               size=res['size'], dmax=dmax, alpha_bar=alpha_bar, dispersion='window')
    if c.get('keep_fields'):
        out.update(vol=res['vol'], gamma_bar=res['gamma_bar'], tau2=res['tau2'], V=res['V'])
    return out


def _angle_windows(lo, hi, n_top, n_bot, sin_t, sigma, max_delta):
    """windows on [lo, hi] for a FIXED incidence angle (alpha = n_top omega sin theta): split at the Rayleigh
    frequencies of the top layer, omega = p / (n_top (1 -/+ sin theta)), and of a real substrate,
    omega = p / (n_bot -/+ n_top sin theta); then |delta| <= max_delta pieces."""
    cuts = set()
    for p in range(1, 60):
        for sg in (1, -1):
            cuts.add(p / (n_top * (1 + sg * sin_t)) if n_top * (1 + sg * sin_t) > 0 else np.inf)
            if n_bot is not None and n_bot - sg * n_top * sin_t > 0:
                cuts.add(p / (n_bot - sg * n_top * sin_t))
    cuts = sorted(w for w in cuts if lo * (1 + 1e-6) < w < hi * (1 - 1e-6))
    edges = [lo] + cuts + [hi]
    out = []
    md = (max_delta or 0.1) / sigma
    j = 0
    for a_, b_ in zip(edges[:-1], edges[1:]):
        k = max(1, int(np.ceil(np.log(b_ / a_) / np.log((1 + md) / (1 - md)))))
        pieces = [a_] + list(a_ * (b_ / a_) ** (np.arange(1, k + 1) / k))
        for c_, e_ in zip(pieces[:-1], pieces[1:]):
            ob = 0.5 * (c_ + e_)
            out.append((f't{j}', 0, ob, sigma * (e_ - c_) / (2 * ob)))
            j += 1
    return out


def _split_top_rayleigh(wins, n_top_q, alpha, sigma, max_delta):
    """oblique incidence: the Rayleigh frequencies of the TOP layer, n_top omega = |alpha - p| (p >= 1), lie INSIDE
    the paper's bands [(alpha + q), (alpha + q + 1)] / n_top; split every window that contains one (the delta series
    has a branch point there)."""
    from refl_map import _wood_anomalies
    out = []
    for key, q, ob, dm in wins:
        lo, hi = ob * (1 - dm / sigma), ob * (1 + dm / sigma)
        nt = np.real(n_top_q[q])
        cuts = [w for w in _wood_anomalies(nt, alpha, np.array([lo, hi])) if lo * (1 + 1e-6) < w < hi * (1 - 1e-6)]
        if not cuts:
            out.append((key, q, ob, dm))
            continue
        edges = [lo] + cuts + [hi]
        for j, (a_, b_) in enumerate(zip(edges[:-1], edges[1:])):
            if (b_ - a_) < 0.02 * (hi - lo):
                continue
            o = 0.5 * (a_ + b_)
            out.append((f'{key}r{j}', q, o, sigma * (b_ - a_) / (2 * o)))
    return out


def run(scenario=SCENARIO, qq=None, N_Eps=None, N_delta=None, verbose=True, workers=None, **over):
    """Compute the n-layer reflectivity map (keyword overrides of any scenario key)."""
    sc = dict(SCENARIOS[scenario]) if scenario in SCENARIOS else dict(_P, desc='custom stack')
    sc.update({k: v for k, v in over.items() if v is not None})
    qq = tuple(qq) if qq is not None else tuple(sc['q'])
    N_Eps = N_Eps or sc.get('n_eps', 100)
    N_delta = N_delta or sc.get('n_delta', 100)
    layers, L = sc['layers'], len(sc['layers'])
    M = sc['M']
    N = sc.get('N') or M
    period = sc.get('period')
    named = [v for v in layers if not isinstance(v, (int, float, complex)) and mat.parse_index(v) is None]
    if named and period is None:
        period = mat.MATERIALS.get(named[-1], {}).get('period', 1.0)
    if sc.get('thick_um') is not None:
        thick = [t * 2 * np.pi / period for t in sc['thick_um']]
    else:
        thick = list(sc.get('thick') or [])
    sigma, alpha = sc['sigma'], sc['alpha']
    lam_ref = lambda om: (period / om) if period else None
    # band indices of the top / bottom layers (the windows depend on their Rayleigh frequencies)
    from refl_map import _windows, _band_window
    n_top_q, n_bot_q = {}, {}
    for q in qq:
        ob, _ = _band_window(q, 1.0, alpha, sigma, 'rayleigh')
        n_top_q[q] = _index(layers[0], lam_ref(ob))
        ob, _ = _band_window(q, np.real(n_top_q[q]), alpha, sigma, 'rayleigh')
        n_bot_q[q] = _index(layers[-1], lam_ref(ob))
    wins = _windows(qq, n_top_q, n_bot_q, alpha, sigma, 'rayleigh', sc['windows'], sc.get('min_window', 0.02),
                    verbose, sc.get('max_delta'))
    if alpha != 0 and sc['windows'] == 'joint':
        wins = _split_top_rayleigh(wins, n_top_q, alpha, sigma, sc.get('max_delta'))
    alpha_w = {}
    if sc.get('theta') is not None:              # fixed incidence angle: alpha_bar = n_top omega_bar sin(theta)
        sin_t = np.sin(np.radians(sc['theta']))
        o_lo, o_hi = sc['omega_range']
        lam_mid = lam_ref(0.5 * (o_lo + o_hi))
        nt = np.real(_index(layers[0], lam_mid))
        nb = _index(layers[-1], lam_mid)
        nb = np.real(nb) if abs(np.imag(nb)) < 1e-3 * abs(nb) else None
        wins = _angle_windows(o_lo, o_hi, nt, nb, sin_t, sigma, sc.get('max_delta'))
        alpha_w = {w[0]: np.real(_index(layers[0], lam_ref(w[2]))) * w[2] * sin_t for w in wins}
    if sc.get('omega_range'):
        o_lo, o_hi = sc['omega_range']
        wins = [w for w in wins if w[2] * (1 + w[3]) >= o_lo and w[2] * (1 - w[3]) <= o_hi]
    Nx = sc['Nx']
    xx = (2 * np.pi / Nx) * np.arange(Nx)
    amps = sc.get('amps') or [1.0] * (L - 1)
    fs = [profile(p, xx) for p in sc['profiles']]
    f = [amps[j] * fs[j][0] for j in range(L - 1)]
    fx = [amps[j] * fs[j][1] for j in range(L - 1)]
    st0 = Stack([1.0] * L, thick, f, fx, sc['a'], sc['b'], sc['Nz'])
    if sc['eps_max'] >= st0.max_eps():
        raise ValueError(f"eps_max = {sc['eps_max']} lets two interfaces touch: the TFE map of a layer is singular at "
                         f"eps* = d / max(f_b - f_t) = {st0.max_eps():.3g} (use conformal profiles or a smaller eps_max)")
    common = dict(n_w={w[0]: [_index(v, lam_ref(w[2])) for v in layers] for w in wins}, thick=thick, f=f, fx=fx,
                  a=sc['a'], b=sc['b'], Nz=sc['Nz'], mode=sc['mode'], alpha=alpha, N=N, M=M, N_delta=N_delta,
                  Eps=np.linspace(0, sc['eps_max'], N_Eps), taylor=sc['taylor'], verbose=verbose,
                  relative=sc.get('relative', False), keep_fields=sc.get('keep_fields', False),
                  taylor_full_order=sc.get('taylor_full_order', False), alpha_w=alpha_w,
                  dispersion=sc.get('dispersion', 'window'), layers=layers, period=period, theta=sc.get('theta'))
    if verbose:
        print(f"{scenario}: L = {L} layers, {len(wins)} windows; {describe(sc)}", flush=True)
    workers = min(workers or WORKERS, len(wins))
    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_window, wins, [common] * len(wins)))
    else:
        results = [_window(w, common) for w in wins]
    lossless = all(np.all(np.abs(np.imag(r['n_layers'])) < 1e-6) for r in results)
    info = dict(scenario=scenario, L=L, N=N, M=M, Nx=Nx, Nz=sc['Nz'], layers=layers, thick=thick, profiles=sc['profiles'],
                amps=amps, period=period, eps_max=sc['eps_max'], alpha=alpha, mode=sc['mode'], Taylor=sc['taylor'],
                lossless=lossless, relative=common['relative'], desc=sc.get('desc', ''), stack=describe(sc),
                a=sc['a'], b=sc['b'], f=f, fx=fx, t_total=time.time() - t0, theta=sc.get('theta'),
                dispersion=sc.get('dispersion', 'window'))
    return results, info


def sketch(ax, info, eps=None):
    """small drawing of the stack (one period, interfaces at eps = 0.6 * eps_max)"""
    L, thick = info['L'], info['thick']
    xx = np.linspace(0, 2 * np.pi, 200)
    eps = 0.6 * info['eps_max'] if eps is None else eps
    h = [0.0]
    for t in thick:
        h.append(h[-1] - t)
    top, bot = h[0] + max(1.0, 0.6 * sum(thick) / max(1, L - 2) if thick else 1.0), h[-1] - 1.0
    levels = [np.full_like(xx, top)]
    for j, p in enumerate(info['profiles']):
        fj = profile(p, xx)[0] * info['amps'][j]
        levels.append(h[j] + eps * fj)
    levels.append(np.full_like(xx, bot))
    cmap = plt.get_cmap('Pastel1')
    for l in range(L):
        v = info['layers'][l]
        nre = np.real(_index(v, 0.6)) if not isinstance(v, (int, float, complex)) else np.real(v)
        col = '0.95' if l == 0 else cmap(l % 9)
        ax.fill_between(xx, levels[l + 1], levels[l], color=col, lw=0)
        lbl = (f'{complex(v).real:.3g}' + (f'{complex(v).imag:+.2g}i' if complex(v).imag else '')) \
            if isinstance(v, (int, float, complex)) else str(v)
        ax.text(2 * np.pi * 1.02, 0.5 * (levels[l + 1].mean() + levels[l].mean()), lbl, fontsize=6, va='center')
    for lv in levels[1:-1]:
        ax.plot(xx, lv, 'k-', lw=0.6)
    ax.set_xlim(0, 2 * np.pi * 1.35)
    ax.set_ylim(bot, top)
    ax.axis('off')


def plot(results, info, outdir=None, tag=None):
    """MATLAB-like rendering as refl_map.py (one contourf per window, shared 'hot' scale), plus a stack sketch."""
    from matplotlib.colors import Normalize
    outdir = outdir or os.path.join(OUTDIR, f"L{info['L']}")
    os.makedirs(outdir, exist_ok=True)
    tag = tag or info['scenario']
    period = info.get('period')
    xfun = (lambda r: r['lam'] * period / (2 * np.pi)) if period else (lambda r: r['lam'])
    xlabel = (r'$\lambda$ ($\mu$m)' + f'   (period {period:g} $\\mu$m)') if period else r'$\lambda$'
    sub = f"  L = {info['L']}, {info['mode']}, {'Taylor' if info.get('Taylor') else 'Pade'}"
    figs = []
    for num, (title, getZ) in enumerate([('$D$', lambda r: safe_log10(r['ee'])),
                                         ('$R/R_{flat}$' if info['relative'] else '$R$', lambda r: np.real(r['RR']))], start=1):
        Zs = [getZ(r) for r in results]
        allz = np.concatenate([z[np.isfinite(z)].ravel() for z in Zs])
        vmin, vmax = allz.min(), allz.max()
        if num == 2 and np.mean(allz > 1.0 + 1e-9) < 1e-3 and vmax > 1.0:
            vmax = 1.0
            Zs = [np.minimum(z, 1.0) for z in Zs]
        norm = Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(num=num, figsize=(9.6, 5), clear=True)
        for r, Z in zip(results, Zs):
            matlab_contourf(ax, xfun(r), r['Eps'], Z, 'hot', norm,
                            step=1.0 if (num == 1 and vmax - vmin >= 4) else None)
        shared_colorbar(fig, ax, 'hot', norm)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'$\varepsilon$', fontsize=15)
        note = '' if (num == 2 or info.get('lossless', True)) else '  (absorbing: D = absorptance)'
        ax.set_title(title + sub + note + '\n' + info['stack'][:110], fontsize=9)
        fig.tight_layout(rect=(0, 0, 0.83, 1))
        sketch(fig.add_axes((0.835, 0.18, 0.15, 0.62)), info)
        fname = os.path.join(outdir, f'refl_map_{tag}_{"D" if num == 1 else "R"}.png')
        fig.savefig(fname, dpi=130)
        figs.append(fname)
    return figs


def save(results, info, tag, outdir=None):
    outdir = outdir or os.path.join(OUTDIR, f"L{info['L']}")
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f'refl_map_{tag}.npz')
    np.savez_compressed(fn, period=np.nan if info['period'] is None else info['period'], L=info['L'],
                        **{f'{r["key"]}_{k}': r[k] for r in results for k in ('lam', 'omega', 'Eps', 'ee', 'ru', 'rl', 'RR')})
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default=SCENARIO)
    ap.add_argument('--list-scenarios', action='store_true')
    g = ap.add_argument_group('stack / numerics (override the scenario)')
    g.add_argument('--layers', nargs='*', help='L materials top -> bottom (numbers like 1.5 or 0.05+2.275i, or keys)')
    g.add_argument('--thick', type=float, nargs='*', help='L-2 interior thicknesses (period = 2 pi)')
    g.add_argument('--thick-um', type=float, nargs='*', help='L-2 interior thicknesses in micrometres (needs --period)')
    g.add_argument('--profiles', nargs='*', help="L-1 interface profiles (cosx, cos4x, flat, 'expr:...')")
    g.add_argument('--amps', type=float, nargs='*')
    g.add_argument('--period', type=float)
    g.add_argument('--mode', choices=['TE', 'TM'])
    g.add_argument('--alpha', type=float)
    g.add_argument('--eps-max', type=float)
    g.add_argument('--M', type=int)
    g.add_argument('--N', type=int)
    g.add_argument('--Nx', type=int)
    g.add_argument('--Nz', type=int)
    g.add_argument('--summation', choices=['taylor', 'pade'])
    g.add_argument('--windows', choices=['paper', 'joint'])
    g.add_argument('--max-delta', type=float)
    g.add_argument('--q', type=int, nargs='*')
    g.add_argument('--neps', type=int)
    g.add_argument('--ndelta', type=int)
    g.add_argument('--absolute', action='store_true', help='plot R instead of R/R_flat')
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--tag')
    ap.add_argument('--no-show', action='store_true')
    a = ap.parse_args(argv)
    if a.list_scenarios:
        w = max(map(len, SCENARIOS))
        for k, v in SCENARIOS.items():
            print(f'{k:{w}s}  L={len(v["layers"])}  {v["desc"]}')
        return
    if a.no_show or not SHOW:
        matplotlib.use('Agg')
    parse = lambda v: complex(mat.parse_index(v)) if mat.parse_index(v) is not None else v
    over = dict(layers=[parse(v) for v in a.layers] if a.layers else None, thick=a.thick, thick_um=a.thick_um,
                profiles=a.profiles, amps=a.amps, period=a.period, mode=a.mode, alpha=a.alpha, eps_max=a.eps_max,
                M=a.M, N=a.N, Nx=a.Nx, Nz=a.Nz, windows=a.windows, max_delta=a.max_delta,
                taylor=None if a.summation is None else a.summation == 'taylor',
                relative=False if a.absolute else None)
    if a.layers and a.thick is None and a.thick_um is None:
        over['thick'] = [1.0] * (len(a.layers) - 2)
    if a.layers and a.profiles is None:
        over['profiles'] = ['cosx'] * (len(a.layers) - 1)
    t0 = time.time()
    res, info = run(a.scenario, a.q, a.neps, a.ndelta, workers=a.workers, **over)
    tag = a.tag or (a.scenario if not a.layers else f'custom_L{info["L"]}')
    for fn in plot(res, info, tag=tag):
        print('saved', fn)
    print('saved', save(res, info, tag))
    print(f'total {time.time() - t0:.1f} s')
    if SHOW and not a.no_show:
        plt.show()


if __name__ == '__main__':
    main()
