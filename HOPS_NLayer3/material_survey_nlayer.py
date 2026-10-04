"""material_survey_nlayer.py -- material survey for multilayer gratings (the n-layer analogue of material_survey.py).

Part A  thin films (L = 3):  vacuum | 50 nm film of every library material | fused silica, conformal cos x grating,
        period 0.4 um, lambda in [0.4, 1.0] um (omega in [0.4, 1.0], band q = 0), 21 x 21 per window, Pade.
        The same material as a BULK substrate (L = 2) is run with identical settings: what does making the
        material a thin film do to the reflectivity, the grating resonances and the absorption?
Part B  Bragg mirrors (L = 4, 6, 8): vacuum | (H L)^P | BK7 with P = 1, 2, 3 quarter-wave pairs (design 0.6 um)
        for every pair (H, L) of library dielectrics; top interface corrugated (cos x, period 0.4 um).
        Stop-band width and peak reflectance of the flat stack are compared with the textbook formulas
            delta omega / omega_0 = (4 / pi) arcsin((n_H - n_L) / (n_H + n_L)),
            R_P = ((n_L^{2P} - n_s n_H^{2P}) / (n_L^{2P} + n_s n_H^{2P}))^2   (quarter-wave stack, n_0 = 1),
        and the grating's effect (dips of R / R_flat inside the stop band: guided modes of the mirror) is recorded.
Part C  10 nm absorbing film (14 semiconductors / lossy metals) on 4 mirror metals (L = 3): peak absorption,
        share absorbed in the film, reflected colour (flat and corrugated).
Part D  Tamm plasmons: 15 metals (30 nm) on a TiO2/SiO2 mirror (L = 8): dip, absorption in the metal, transmission.
Part E  metal-insulator-metal absorbers (L = 4): 4 metals x 6 spacer dielectrics.
All parts use EXACT material dispersion (every frequency with its own n(lambda); set HOPS_NL_DISPERSION=window
for the HOPS/AWE expansion with indices frozen per window, as refl_map.m).
Writes results/material_survey_*.{csv,md}, figures/material_survey/*.png.

    python material_survey_nlayer.py                         # all parts
    python material_survey_nlayer.py --part films --keys Ag Au Al TiN Si
    python material_survey_nlayer.py --part absorbers tamm mim
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                 # noqa: E402
from hops import materials as mat           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures', 'material_survey')
OUT = os.path.join(HERE, 'results', 'material_survey')
PERIOD, FILM_UM, LAM0 = 0.4, 0.05, 0.6
COMMON = dict(period=PERIOD, q=(0,), omega_range=(0.4, 0.999), max_delta=0.1, M=10, Nx=32, Nz=24, eps_max=0.2,
              windows='joint', alpha=0.0, a=1.0, b=1.0, mode='TM', taylor=False, sigma=0.99,
              dispersion=os.environ.get('HOPS_NL_DISPERSION', 'exact'))
HIGHS = ('TiO2', 'Si3N4', 'HfO2', 'Ta2O5', 'ZnO', 'GaN', 'AlN', 'diamond', 'ZnS', 'LiNbO3')
LOWS = ('SiO2', 'MgF2', 'sapphire')


def _run(layers, thick_um, profiles, n=21):
    sc = dict(R._P); sc.update(COMMON); sc.update(layers=list(layers), thick_um=list(thick_um), profiles=list(profiles),
                                                  desc='survey')
    key = '_survey'
    R.SCENARIOS[key] = sc
    return R.run(key, N_Eps=n, N_delta=n, workers=1, verbose=False)


def _summ(res, info):
    cat = lambda k: np.concatenate([np.real(r[k]).ravel() for r in res])
    Rr, D, Rf = cat('ru'), cat('ee'), cat('ru_flat')
    RR = Rr / np.where(Rf > 1e-12, Rf, np.nan)
    return dict(R_min=float(np.nanmin(Rr)), R_max=float(np.nanmax(Rr)), R_mean=float(np.nanmean(Rr)),
                RR_min=float(np.nanpercentile(RR, 1)), RR_max=float(np.nanpercentile(RR, 99)),
                A_mean=float(np.nanmean(D)), A_max=float(np.nanmax(D)), lossless=bool(info['lossless']),
                nonphys=float(np.mean((Rr < -1e-3) | (Rr > 1.001) | (D < -1e-3))),
                Rflat_min=float(np.min(Rf)), Rflat_max=float(np.max(Rf)))


def film_one(key):
    fn = os.path.join(OUT, f'film_{key}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    try:
        rf, inf = _run(['air', key, 'fused_silica'], [FILM_UM], ['cosx', 'cosx'])
        rb, inb = _run(['air', key], [], ['cosx'])
    except Exception as e:                     # noqa: BLE001
        return dict(key=key, error=repr(e))
    n = np.array([complex(r['n_layers'][1]) for r in rf])
    s = dict(key=key, category=mat.MATERIALS[key].get('category', ''), n_re=float(n.real.mean()), n_im=float(n.imag.mean()),
             epsr=float(np.mean((n ** 2).real)), **{f'film_{k}': v for k, v in _summ(rf, inf).items()},
             **{f'bulk_{k}': v for k, v in _summ(rb, inb).items()}, time=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(s, open(fn, 'w'))
    np.savez_compressed(os.path.join(OUT, f'film_{key}.npz'), **{f'{t}{k}_{q}': np.real(r[q]) for t, rr in (('f', rf), ('b', rb))
                                                                 for k, r in enumerate(rr) for q in ('lam', 'Eps', 'ru', 'ru_flat', 'ee')})
    print(f"{key:12s} n {s['n_re']:.2f}+{s['n_im']:.2f}i  film: R {s['film_R_min']:.2f}-{s['film_R_max']:.2f} A {s['film_A_mean']:.2f}"
          f"  bulk: R {s['bulk_R_min']:.2f}-{s['bulk_R_max']:.2f} A {s['bulk_A_mean']:.2f}  ({s['time']:.0f} s)", flush=True)
    return s


def bragg_one(args):
    hi, lo, P = args
    fn = os.path.join(OUT, f'bragg_{hi}_{lo}_{P}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    nH, nL = np.real(mat.refractive_index(hi, LAM0)), np.real(mat.refractive_index(lo, LAM0))
    tH, tL = LAM0 / (4 * nH), LAM0 / (4 * nL)
    layers = ['air'] + [hi, lo] * P + ['BK7']
    npz = os.path.join(OUT, f'bragg_{hi}_{lo}_{P}.npz')
    if os.path.exists(npz):                    # re-analysis of a stored run (no new HOPS solve)
        z = np.load(npz)
        om, Rf, R15, lossless = z['omega'], z['Rflat'], z['R15'], True
    else:
        try:
            res, info = _run(layers, [tH, tL] * P, ['cosx'] + ['flat'] * (2 * P))
        except Exception as e:                     # noqa: BLE001
            return dict(hi=hi, lo=lo, P=P, error=repr(e))
        om = np.concatenate([r['omega'] for r in res]); o = np.argsort(om)
        Rf = np.concatenate([np.real(r['ru_flat'][0]) for r in res])[o]; om = om[o]
        ie = int(np.argmin(np.abs(res[0]['Eps'] - 0.15)))
        R15 = np.concatenate([np.real(r['ru'][ie]) for r in res])[o]
        lossless = bool(info['lossless'])
    w0 = PERIOD / LAM0
    # stop band = the region around omega_0 where R_flat stays above the half level between R_flat(omega_0) and
    # the bare-substrate reflectance (FWHM of the Bragg lobe; robust to the small steps of R_flat at window joints,
    # which come from freezing the dispersive indices per window)
    ns_ = np.real(mat.refractive_index('BK7', LAM0)); R_sub = ((ns_ - 1) / (ns_ + 1)) ** 2
    i0 = int(np.argmin(np.abs(om - w0))); half = 0.5 * (Rf[i0] + R_sub); i1 = i0; i2 = i0
    while i1 > 0 and Rf[i1 - 1] > half:
        i1 -= 1
    while i2 < len(Rf) - 1 and Rf[i2 + 1] > half:
        i2 += 1
    width = (om[i2] - om[i1]) / w0
    R0 = float(np.interp(w0, om, Rf))                  # R_flat at the design frequency
    ns = np.real(mat.refractive_index('BK7', LAM0))
    # quarter-wave stack (H next to the incidence medium n_0 = 1), normal incidence, design wavelength
    R_th = ((nL ** (2 * P) - ns * nH ** (2 * P)) / (nL ** (2 * P) + ns * nH ** (2 * P))) ** 2
    width_th = (4 / np.pi) * np.arcsin((nH - nL) / (nH + nL))
    in_band = (om >= om[i1]) & (om <= om[i2])
    dip = float(np.min(R15[in_band] / Rf[in_band])) if in_band.any() else np.nan
    s = dict(hi=hi, lo=lo, P=P, L=len(layers), nH=float(nH), nL=float(nL), contrast=float((nH - nL) / (nH + nL)),
             Rflat_peak=R0, R_theory=float(R_th), width=float(width), width_inf_theory=float(width_th),
             R15_min_in_band_ratio=dip, lossless=lossless, time=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(s, open(fn, 'w'))
    if not os.path.exists(npz):
        np.savez_compressed(npz, omega=om, Rflat=Rf, R15=R15)
    print(f"{hi:7s}/{lo:8s} P={P}: R_flat(omega_0) {s['Rflat_peak']:.3f} (theory {s['R_theory']:.3f}), width {width:.3f} "
          f"(P->inf theory {width_th:.3f}), min R/R_flat in band at eps .15 {dip:.3f}  ({s['time']:.0f} s)", flush=True)
    return s


def films(keys, procs):
    if procs > 1:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            rows = list(ex.map(film_one, keys))
    else:
        rows = [film_one(k) for k in keys]
    rows = [r for r in rows if 'error' not in r]
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, 'results', 'material_survey_films.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    A = lambda k: np.array([r[k] for r in rows], float)
    dRR = A('film_RR_min') - A('bulk_RR_min')
    L = [f'{len(rows)} materials as a {FILM_UM * 1000:.0f} nm film on fused silica (L = 3) vs the bulk material (L = 2); '
         f'conformal cos x, period {PERIOD} um, lambda 0.4 - 1.0 um, eps <= 0.2.', '',
         '| material | n (mean) | film R range | bulk R range | film A mean / max | bulk A mean / max | film min R/R_flat | bulk min R/R_flat |',
         '|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['film_A_mean']):
        L.append(f"| {r['key']} | {r['n_re']:.2f}+{r['n_im']:.2f}i | {r['film_R_min']:.2f} - {r['film_R_max']:.2f} | "
                 f"{r['bulk_R_min']:.2f} - {r['bulk_R_max']:.2f} | {r['film_A_mean']:.2f} / {r['film_A_max']:.2f} | "
                 f"{r['bulk_A_mean']:.2f} / {r['bulk_A_max']:.2f} | {r['film_RR_min']:.3f} | {r['bulk_RR_min']:.3f} |")
    rho = sps.spearmanr(A('n_im'), A('film_A_mean') - A('bulk_A_mean'))
    L += ['', f"Film minus bulk absorption vs Im n: Spearman rho = {rho.correlation:.2f} (p = {rho.pvalue:.0e}).",
          f"Materials whose grating resonance is DEEPER as a film (min R/R_flat lower by > 0.02): "
          f"{', '.join(r['key'] for r in rows if r['film_RR_min'] < r['bulk_RR_min'] - 0.02) or 'none'}.",
          f"... SHALLOWER as a film: {', '.join(r['key'] for r in rows if r['film_RR_min'] > r['bulk_RR_min'] + 0.02) or 'none'}."]
    open(os.path.join(HERE, 'results', 'material_survey_films.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[-3:]))
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (x, y, xl, yl, t) in zip(axs, (
            ('bulk_A_mean', 'film_A_mean', 'bulk mean absorptance', 'film mean absorptance', 'absorption: film vs bulk'),
            ('bulk_R_mean', 'film_R_mean', 'bulk mean R', 'film mean R', 'reflectance: film vs bulk'),
            ('bulk_RR_min', 'film_RR_min', 'bulk min R/R_flat (grating resonance depth)', 'film min R/R_flat',
             'grating resonances: film vs bulk'))):
        ax.scatter(A(x), A(y), c=np.log10(A('n_im') + 1e-3), cmap='viridis', s=35)
        for r in rows:
            ax.annotate(r['key'], (r[x], r[y]), fontsize=6)
        lim = [min(A(x).min(), A(y).min()), max(A(x).max(), A(y).max())]
        ax.plot(lim, lim, 'k:', lw=1); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'films_vs_bulk.png'), dpi=100)
    plt.close(fig)
    gallery([r['key'] for r in sorted(rows, key=lambda r: r['film_RR_min'])[:8]])


def gallery(keys):
    fig, axs = plt.subplots(len(keys), 2, figsize=(11, 2.6 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        d = np.load(os.path.join(OUT, f'film_{key}.npz'))
        for j, t in enumerate(('b', 'f')):
            ax = axs[i, j]
            ks = sorted({k.split('_')[0] for k in d.files if k.startswith(t)})
            for kk in ks:
                ax.pcolormesh(d[kk + '_lam'] * PERIOD / (2 * np.pi), d[kk + '_Eps'], d[kk + '_ru'] / d[kk + '_ru_flat'],
                              cmap='hot', vmin=0.5, vmax=1.05, shading='auto')
            ax.set_title(f"{key}: {'bulk (L = 2)' if t == 'b' else f'{FILM_UM * 1000:.0f} nm film on silica (L = 3)'}: R/R_flat", fontsize=9)
            ax.set_xlabel(r'$\lambda$ ($\mu$m)'); ax.set_ylabel(r'$\varepsilon$')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'films_gallery.png'), dpi=90)
    plt.close(fig)


def bragg(procs):
    combos = [(h, l, P) for h in HIGHS for l in LOWS for P in (1, 2, 3)]
    if procs > 1:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            rows = list(ex.map(bragg_one, combos))
    else:
        rows = [bragg_one(c) for c in combos]
    rows = [r for r in rows if 'error' not in r]
    with open(os.path.join(HERE, 'results', 'material_survey_bragg.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    A = lambda k, rs=rows: np.array([r[k] for r in rs], float)
    L = [f'{len(rows)} Bragg mirrors: vacuum | (H L)^P | BK7, quarter-wave at {LAM0} um, top grating cos x, period {PERIOD} um.', '',
         '| H / L | n_H / n_L | P (L layers) | R_flat at omega_0 (HOPS) / theory | stop-band width (HOPS, FWHM of the lobe at omega_0 above the bare-substrate R) / P->inf theory | '
         'min R/R_flat in the band at eps = 0.15 |', '|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (-r['contrast'], r['P'])):
        L.append(f"| {r['hi']} / {r['lo']} | {r['nH']:.2f} / {r['nL']:.2f} | {r['P']} ({r['L']}) | {r['Rflat_peak']:.4f} / "
                 f"{r['R_theory']:.4f} | {r['width']:.3f} / {r['width_inf_theory']:.3f} | {r['R15_min_in_band_ratio']:.3f} |")
    for P in (1, 2, 3):
        rs = [r for r in rows if r['P'] == P]
        e = np.abs(A('Rflat_peak', rs) - A('R_theory', rs))
        rho = sps.spearmanr(A('contrast', rs), A('width', rs)).correlation
        L.append(f"\nP = {P}: |peak R - theory| max {e.max():.1e}; Spearman(contrast, width) = {rho:.2f}; "
                 f"median min R/R_flat in band {np.nanmedian(A('R15_min_in_band_ratio', rs)):.3f}")
    L += ['', f"Dispersion mode: {COMMON['dispersion']} ('exact': every frequency with its own n(lambda); 'window': indices frozen at "
          "each window centre, which left a ~1e-2 residual against the formula).  "
          'The stop-band width of a finite stack (P <= 3) differs from the P -> inf formula (it is only reached as P grows), and for P = 1 the lobe is '
          'so broad that it fills most of the 0.4 - 1.0 um range.']
    open(os.path.join(HERE, 'results', 'material_survey_bragg.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[-3:]))
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
    for P, mk in zip((1, 2, 3), 'osD'):
        rs = [r for r in rows if r['P'] == P]
        axs[0].plot(A('contrast', rs), A('Rflat_peak', rs), mk, label=f'P = {P} (L = {2 * P + 2}) HOPS')
        axs[0].plot(A('contrast', rs), A('R_theory', rs), 'k+', ms=6)
        axs[1].plot(A('contrast', rs), A('width', rs), mk, label=f'P = {P}')
        axs[2].plot(A('contrast', rs), A('R15_min_in_band_ratio', rs), mk, label=f'P = {P}')
    cc = np.linspace(0, A('contrast').max() * 1.05, 50)
    axs[1].plot(cc, (4 / np.pi) * np.arcsin(cc), 'k--', label=r'P$\to\infty$ theory')
    for ax, t in zip(axs, ('peak reflectance of the flat mirror (+ = textbook formula)', 'stop-band width / omega_0',
                           'deepest grating dip R/R_flat inside the stop band (eps = 0.15)')):
        ax.set_xlabel('index contrast (n_H - n_L)/(n_H + n_L)'); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bragg_survey.png'), dpi=100)
    plt.close(fig)


# ======================================================================================================================
# Part C: ultrathin absorber on a mirror (L = 3), Part D: Tamm plasmons for every metal (L = 8),
# Part E: metal-insulator-metal absorbers, metal x spacer (L = 4).   Exact material dispersion.
# ======================================================================================================================
ABSORBERS = ('Ge', 'Si', 'GaAs', 'InP', 'CdTe', 'MoS2', 'MAPbI3', 'Cr', 'Ti', 'W', 'Fe', 'Ni', 'Pt', 'TiN')
MIRRORS = ('Au', 'Ag', 'Al', 'Cu')
TAMM_METALS = ('Ag', 'Au', 'Cu', 'Al', 'AuAg50', 'Na', 'K', 'Mg', 'TiN', 'W', 'Cr', 'Ni', 'Pt', 'Pd', 'Ti')
MIM_METALS = ('Ag', 'Au', 'Al', 'Cu')
MIM_SPACERS = ('MgF2', 'SiO2', 'sapphire', 'HfO2', 'Si3N4', 'TiO2')


def _run_x(layers, thick_um, profiles, period, lam_lo, lam_hi, n_eps=3, n_delta=16, eps_max=0.2):
    sc = dict(R._P); sc.update(COMMON)
    sc.update(layers=list(layers), thick_um=list(thick_um), profiles=list(profiles), period=period,
              omega_range=(period / lam_hi, period / lam_lo), q=(0, 1, 2), max_delta=0.05, eps_max=eps_max, desc='survey')
    key = '_survey_x'
    R.SCENARIOS[key] = sc
    res, info = R.run(key, N_Eps=n_eps, N_delta=n_delta, workers=1, verbose=False)
    sc_ = period / (2 * np.pi)
    lam = np.concatenate([r['lam'] for r in res]) * sc_
    Rr = np.concatenate([np.real(r['ru']) for r in res], axis=1)
    o = np.argsort(lam)
    o = o[(lam[o] >= lam_lo * 0.999) & (lam[o] <= lam_hi * 1.001)]
    return lam[o], Rr[:, o], res[0]['Eps']


def absorber_one(args):
    ab, mi = args
    fn = os.path.join(OUT, f'abs_{ab}_{mi}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    from deep_dives_nlayer import reflected_srgb
    from hops_nl.reference import tmm
    t0 = time.time()
    try:
        lam, Rr, Eps = _run_x(['air', ab, mi], [0.010], ['cosx', 'cosx'], 0.4, 0.40, 0.85)
    except Exception as e:                          # noqa: BLE001
        return dict(absorber=ab, mirror=mi, error=repr(e))
    A0, A2 = 1 - Rr[0], 1 - Rr[-1]
    i = int(np.argmax(A0))
    li = lam[i]
    _, _, Al = tmm([1.0, R._index(ab, li), R._index(mi, li)], [0.010 * 2 * np.pi / 0.4], 0.4 / li, 0, 'TM', layers=True)
    vis = (lam >= 0.40) & (lam <= 0.70)
    s = dict(absorber=ab, mirror=mi, A_peak=float(A0[i]), lam_peak=float(li), film_share=float(Al[1] / max(A0[i], 1e-12)),
             A_vis_mean=float(np.mean(A0[vis])), A_peak_eps0p2=float(np.max(A2)), lam_peak_eps0p2=float(lam[np.argmax(A2)]),
             rgb_flat=[float(v) for v in reflected_srgb(lam, Rr[0])], rgb_eps0p2=[float(v) for v in reflected_srgb(lam, Rr[-1])],
             time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"{ab:7s} on {mi:3s}: peak A {s['A_peak']:.3f} at {li:.3f} um ({100 * s['film_share']:.0f} % in the film), "
          f"eps 0.2: {s['A_peak_eps0p2']:.3f}  ({s['time']:.0f} s)", flush=True)
    return s


def tamm_one(metal):
    fn = os.path.join(OUT, f'tamm_{metal}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    from hops_nl.reference import tmm
    t0 = time.time()
    sc = R.SCENARIOS['tamm_ag']
    layers = ['air', metal] + sc['layers'][2:]
    try:
        lam, Rr, Eps = _run_x(layers, sc['thick_um'], sc['profiles'], 0.65, 0.60, 0.80, eps_max=0.1)
    except Exception as e:                          # noqa: BLE001
        return dict(metal=metal, error=repr(e))
    band = (lam > 0.62) & (lam < 0.78)
    i = int(np.argmin(np.where(band, Rr[0], 9)))
    j = int(np.argmin(np.where(band, Rr[-1], 9)))
    li = lam[i]
    Rt, Tt, Al = tmm([R._index(k, li) for k in layers], [t * 2 * np.pi / 0.65 for t in sc['thick_um']], 0.65 / li, 0, 'TM',
                     layers=True)
    s = dict(metal=metal, lam_tamm=float(li), R_dip=float(Rr[0, i]), A_metal=float(Al[1]), T_at_dip=float(Tt),
             lam_dip_eps0p1=float(lam[j]), R_dip_eps0p1=float(Rr[-1, j]), time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"Tamm {metal:7s}: dip {li:.4f} um R {s['R_dip']:.3f} (A_metal {s['A_metal']:.3f}, T {Tt:.3f}); eps 0.1: "
          f"{s['lam_dip_eps0p1']:.4f} R {s['R_dip_eps0p1']:.3f}  ({s['time']:.0f} s)", flush=True)
    return s


def mim_one(args):
    me, sp = args
    fn = os.path.join(OUT, f'mim_{me}_{sp}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    try:
        lam, Rr, Eps = _run_x(['air', me, sp, me], [0.025, 0.06], ['cosx', 'flat', 'flat'], 0.5, 0.40, 0.95, eps_max=0.12)
    except Exception as e:                          # noqa: BLE001
        return dict(metal=me, spacer=sp, error=repr(e))
    A = 1 - Rr
    i = int(np.argmax(A[-1]))
    s = dict(metal=me, spacer=sp, n_spacer=float(np.real(R._index(sp, 0.7))), A_max_eps0p12=float(A[-1, i]),
             lam_eps0p12=float(lam[i]), A_max_flat=float(A[0].max()), lam_flat=float(lam[np.argmax(A[0])]), time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"MIM {me:3s} / {sp:8s}: eps 0.12 peak A {s['A_max_eps0p12']:.3f} at {s['lam_eps0p12']:.3f} um; flat "
          f"{s['A_max_flat']:.3f} at {s['lam_flat']:.3f}  ({s['time']:.0f} s)", flush=True)
    return s


def _pmap(fn, items, procs):
    if procs > 1:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            return list(ex.map(fn, items))
    return [fn(i) for i in items]


def absorbers(procs):
    rows = [r for r in _pmap(absorber_one, [(a, m) for a in ABSORBERS for m in MIRRORS], procs) if 'error' not in r]
    L = ['10 nm absorbing film on a metal mirror (L = 3): air | film | mirror, conformal cos x, period 0.4 um, '
         'lambda 0.40 - 0.85 um, exact dispersion.', '',
         '| film / mirror | peak absorptance flat (lambda um) | share absorbed in the film | mean A 0.40-0.70 um | peak A at eps = 0.2 (lambda um) |',
         '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['A_peak']):
        L.append(f"| {r['absorber']} / {r['mirror']} | {r['A_peak']:.3f} ({r['lam_peak']:.3f}) | {100 * r['film_share']:.0f} % | "
                 f"{r['A_vis_mean']:.2f} | {r['A_peak_eps0p2']:.3f} ({r['lam_peak_eps0p2']:.3f}) |")
    open(os.path.join(HERE, 'results', 'material_survey_absorbers.md'), 'w').write('\n'.join(L) + '\n')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for k, (key, t) in enumerate((('rgb_flat', 'reflected colour, flat'), ('rgb_eps0p2', 'reflected colour, grating eps = 0.2'))):
        img = np.ones((len(ABSORBERS), len(MIRRORS), 3))
        for r in rows:
            img[ABSORBERS.index(r['absorber']), MIRRORS.index(r['mirror'])] = r[key]
        axs[k].imshow(img, aspect='auto')
        axs[k].set_xticks(range(len(MIRRORS))); axs[k].set_xticklabels(MIRRORS)
        axs[k].set_yticks(range(len(ABSORBERS))); axs[k].set_yticklabels([f'{a} 10 nm' for a in ABSORBERS], fontsize=8)
        axs[k].set_title(t)
    Z = np.full((len(ABSORBERS), len(MIRRORS)), np.nan)
    for r in rows:
        Z[ABSORBERS.index(r['absorber']), MIRRORS.index(r['mirror'])] = r['A_peak']
    im = axs[2].imshow(Z, aspect='auto', cmap='inferno', vmin=0, vmax=1)
    for r in rows:
        axs[2].text(MIRRORS.index(r['mirror']), ABSORBERS.index(r['absorber']), f"{r['A_peak']:.2f}\n{1e3 * r['lam_peak']:.0f}nm",
                    ha='center', va='center', fontsize=6, color='w' if r['A_peak'] < 0.6 else 'k')
    axs[2].set_xticks(range(len(MIRRORS))); axs[2].set_xticklabels(MIRRORS); axs[2].set_yticks([])
    axs[2].set_title('peak absorptance (flat) and its wavelength'); fig.colorbar(im, ax=axs[2])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'absorbers_on_mirrors.png'), dpi=110); plt.close(fig)
    print('\n'.join(L[:12]))


def tamms(procs):
    rows = [r for r in _pmap(tamm_one, TAMM_METALS, procs) if 'error' not in r]
    L = ['Tamm plasmons: air | metal 30 nm | TiO2 (0.6 quarter) / SiO2 / TiO2 / SiO2 / TiO2 | BK7 (L = 8), period 0.65 um, '
         'top two interfaces corrugated, exact dispersion.', '',
         '| metal | Tamm dip (um) | R at dip (flat) | absorbed in the metal | transmitted | dip at eps = 0.1 (um) / R |', '|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: r['R_dip']):
        L.append(f"| {r['metal']} | {r['lam_tamm']:.4f} | {r['R_dip']:.3f} | {r['A_metal']:.3f} | {r['T_at_dip']:.3f} | "
                 f"{r['lam_dip_eps0p1']:.4f} / {r['R_dip_eps0p1']:.3f} |")
    open(os.path.join(HERE, 'results', 'material_survey_tamm.md'), 'w').write('\n'.join(L) + '\n')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(rows))
    rs = sorted(rows, key=lambda r: r['R_dip'])
    ax.bar(x - 0.2, [r['A_metal'] for r in rs], 0.4, label='absorbed in the metal')
    ax.bar(x + 0.2, [r['T_at_dip'] for r in rs], 0.4, label='transmitted')
    ax.plot(x, [r['R_dip'] for r in rs], 'ko', label='R at the Tamm dip')
    ax.set_xticks(x); ax.set_xticklabels([r['metal'] for r in rs]); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Tamm plasmon resonance for every metal (flat stack, at the dip)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'tamm_metals.png'), dpi=110); plt.close(fig)
    print('\n'.join(L))


def mims(procs):
    rows = [r for r in _pmap(mim_one, [(m, s) for m in MIM_METALS for s in MIM_SPACERS], procs) if 'error' not in r]
    L = ['Metal-insulator-metal absorbers: air | metal 25 nm | spacer 60 nm | metal (L = 4), top grating, period 0.5 um, '
         'lambda 0.40 - 0.95 um, exact dispersion.', '',
         '| metal / spacer | n spacer (0.7 um) | peak A at eps = 0.12 (lambda um) | peak A flat (lambda um) |', '|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['A_max_eps0p12']):
        L.append(f"| {r['metal']} / {r['spacer']} | {r['n_spacer']:.2f} | {r['A_max_eps0p12']:.3f} ({r['lam_eps0p12']:.3f}) | "
                 f"{r['A_max_flat']:.3f} ({r['lam_flat']:.3f}) |")
    open(os.path.join(HERE, 'results', 'material_survey_mim.md'), 'w').write('\n'.join(L) + '\n')
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, (key, t) in enumerate((('A_max_eps0p12', 'peak absorptance (eps = 0.12)'), ('lam_eps0p12', 'its wavelength (um)'))):
        Z = np.full((len(MIM_METALS), len(MIM_SPACERS)), np.nan)
        for r in rows:
            Z[MIM_METALS.index(r['metal']), MIM_SPACERS.index(r['spacer'])] = r[key]
        im = axs[k].imshow(Z, cmap='inferno' if k == 0 else 'viridis', aspect='auto')
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                axs[k].text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', fontsize=8, color='w')
        axs[k].set_xticks(range(len(MIM_SPACERS))); axs[k].set_xticklabels(MIM_SPACERS)
        axs[k].set_yticks(range(len(MIM_METALS))); axs[k].set_yticklabels(MIM_METALS); axs[k].set_title(t)
        fig.colorbar(im, ax=axs[k])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'mim_metal_spacer.png'), dpi=110); plt.close(fig)
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', nargs='*', default=['films', 'bragg', 'absorbers', 'tamm', 'mim'],
                    choices=['films', 'bragg', 'both', 'absorbers', 'tamm', 'mim'])
    ap.add_argument('--keys', nargs='*')
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    if 'both' in a.part:
        a.part += ['films', 'bragg']
    if 'films' in a.part:
        films(a.keys or [k for k, v in mat.MATERIALS.items() if v.get('category') != 'superstrate'
                         and not k.startswith('VO2_f') and k != 'hBN_ip'], a.procs)
    if 'bragg' in a.part:
        bragg(a.procs)
    if 'absorbers' in a.part:
        absorbers(a.procs)
    if 'tamm' in a.part:
        tamms(a.procs)
    if 'mim' in a.part:
        mims(a.procs)
