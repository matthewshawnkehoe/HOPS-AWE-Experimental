"""explore_materials_nlayer.py -- the multilayer MATERIALS GALLERY (L = 3 ... 9): reflectivity map, energy defect /
absorptance, true error and the absorption of EVERY layer for the scenarios of refl_map_nlayer.GALLERY.

For each scenario:
  figures/explore/L<L>/refl_map_<name>_{R,D}.png        maps (as refl_map_nlayer.py, stack sketch on the right)
  figures/explore/L<L>/<name>_spectra.png                R, T and the per-layer absorptance A_l(lambda) at
                                                         eps = 0, eps_max / 2, eps_max (hops_nl.absorption)
  results/explore/<name>.json                            statistics, true error, resonances, per-layer maxima
and results/explore_summary.md (one table over the gallery).

    python explore_materials_nlayer.py                       # whole gallery (about 25 min on 2 cores)
    python explore_materials_nlayer.py --only tamm_ag ge_on_au --n 40
"""
import argparse
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                         # noqa: E402
from hops_nl.absorption import layer_absorption     # noqa: E402
from hops_nl.reference import check_map             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures', 'explore')
RES = os.path.join(HERE, 'results', 'explore')

GALLERY = ['lrspp_ag10', 'lrspp_ag20', 'lrspp_ag40', 'imi_asym_ag20', 'kretschmann_au', 'berreman_ito', 'ge_on_au',
           'vo2_cold_sapphire', 'vo2_f30_sapphire', 'vo2_hot_sapphire', 'kretschmann_cr', 'kretschmann_ti',
           'biosensor_L4', 'salisbury', 'perovskite_cell', 'si_cell', 'mim_filter', 'hmm_au_tio2_L6', 'ir_mirror_L6',
           'tamm_ag', 'tamm_au', 'hmm_au_tio2_L8', 'vcsel_dbr_L8', 'ir_mirror_L8', 'microcavity_L9']
# database-wide gallery (refractiveindex.info pages found by screen_rii_database.py)
GALLERY_DB = ['gst_colour_a', 'gst_colour_c', 'gst_mir_a', 'gst_mir_c', 'sqib_polariton', 'lc_cavity_o', 'lc_cavity_e',
              'vo2_window_T20', 'vo2_window_T80', 'berreman_sio2_ir', 'vdw_mirror_TE', 'ws2_polariton_L15',
              'polymer_dbr_L22', 'polymer_dbr_L32']


def _label(v):
    return (f'{complex(v).real:.3g}' + (f'{complex(v).imag:+.2g}i' if complex(v).imag else '')) \
        if isinstance(v, (int, float, complex)) else str(v)


def spectra(res, info, eps_list, step=2):
    """lam (um), R, T(lossless substrate), A_l for each eps in eps_list"""
    sc = info['period'] / (2 * np.pi)
    out = {}
    for e in eps_list:
        rows = []
        for r in sorted(res, key=lambda r: r['omega_bar']):
            ie = int(np.argmin(np.abs(r['Eps'] - e)))
            for j in range(0, len(r['delta']), step):
                A = layer_absorption(r, info, r['Eps'][ie], r['delta'][j])
                rows.append(np.r_[r['lam'][j] * sc, np.real(r['ru'][ie, j]), np.real(r['rl'][ie, j]), A])
        rows = np.array(rows)
        rows = rows[np.argsort(rows[:, 0])]
        out[float(e)] = rows
    return out


def plot_spectra(sp, info, fn):
    L = info['L']
    fig, axs = plt.subplots(1, len(sp), figsize=(5.2 * len(sp), 4.2), sharey=True)
    cols = plt.get_cmap('tab10')
    for ax, (e, rows) in zip(np.atleast_1d(axs), sp.items()):
        lam = rows[:, 0]
        ax.plot(lam, rows[:, 1], 'k-', lw=1.6, label='R')
        A = rows[:, 3:]
        bottom = np.zeros_like(lam)
        for l in range(L):
            if np.max(A[:, l]) < 1e-4:
                continue
            lab = f'A {l}: {_label(info["layers"][l])}' + (' (transmitted)' if l == L - 1 else '')
            ax.fill_between(lam, bottom, bottom + A[:, l], color=cols(l % 10), alpha=0.55, lw=0, label=lab)
            bottom = bottom + A[:, l]
        T = 1 - rows[:, 1] - A.sum(axis=1)               # transmitted into a transparent substrate
        if np.max(T) > 1e-3:
            ax.fill_between(lam, bottom, bottom + T, color='0.75', alpha=0.6, lw=0, label='T (into the substrate)')
        ax.set_title(f'eps = {e:.3g}', fontsize=10)
        ax.set_xlabel(r'$\lambda$ ($\mu$m)')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel('R, absorptance per layer (stacked)')
    np.atleast_1d(axs)[-1].legend(fontsize=7, loc='best')
    fig.suptitle(f"{info['scenario']}: {info['stack']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fn, dpi=120)
    plt.close(fig)


def resonances(rows, k=3):
    """the k deepest local minima of R(lambda): (lambda, R, FWHM-ish width)"""
    lam, Rr = rows[:, 0], rows[:, 1]
    mins = [i for i in range(1, len(Rr) - 1) if Rr[i] < Rr[i - 1] and Rr[i] <= Rr[i + 1]]
    mins = sorted(mins, key=lambda i: Rr[i])[:k]
    out = []
    for i in mins:
        base = np.max(Rr[max(0, i - 15):i + 16])
        half = 0.5 * (base + Rr[i])
        a = i
        while a > 0 and Rr[a] < half:
            a -= 1
        b = i
        while b < len(Rr) - 1 and Rr[b] < half:
            b += 1
        out.append(dict(lam=float(lam[i]), R=float(Rr[i]), width=float(lam[b] - lam[a])))
    return sorted(out, key=lambda d: d['lam'])


def explore(name, n=60, workers=2):
    t0 = time.time()
    res, info = R.run(name, N_Eps=n, N_delta=n, workers=workers, verbose=False, keep_fields=True)
    outdir = os.path.join(FIG, f"L{info['L']}")
    R.plot(res, info, outdir=outdir)
    plt.close('all')
    em = info['eps_max']
    sp = spectra(res, info, [0.0, em / 2, em])
    plot_spectra(sp, info, os.path.join(outdir, f'{name}_spectra.png'))
    chk = check_map(res, info, cols=(0.0, -0.5, 0.5, -0.95, 0.95))
    disp = None
    if info.get('dispersion') == 'exact':
        # the same map with the HOPS/AWE frequency expansion and indices frozen at each window centre
        t1 = time.time()
        res_w, _ = R.run(name, N_Eps=n, N_delta=n, workers=workers, verbose=False, dispersion='window')
        dR = np.concatenate([np.abs(np.real(a['ru']) - np.real(b['ru'])).ravel() for a, b in zip(res_w, res)])
        disp = dict(max=float(dR.max()), median=float(np.median(dR)), p95=float(np.percentile(dR, 95)),
                    t_awe=time.time() - t1)
    cat = lambda k: np.concatenate([np.real(r[k]).ravel() for r in res])
    Rr, D = cat('ru'), cat('ee')
    s = dict(scenario=name, L=info['L'], stack=info['stack'], desc=info['desc'], windows=len(res),
             lossless=info['lossless'], R_min=float(Rr.min()), R_max=float(Rr.max()),
             frac_nonphysical=float(np.mean((Rr < -1e-3) | (Rr > 1.001) | (D < -1e-3))),
             err_max=chk['max'], err_median=chk['median'], wall=time.time() - t0, size=int(res[0]['size']),
             dispersion=info.get('dispersion'), awe_frozen_vs_exact=disp, theta=info.get('theta'))
    if info['lossless']:
        s.update(D_max=float(np.log10(np.max(np.abs(D)) + 1e-300)), D_median=float(np.log10(np.median(np.abs(D)) + 1e-300)))
    for e, rows in sp.items():
        A = rows[:, 3:]
        key = f'eps{e:.3g}'
        s[key] = dict(resonances=resonances(rows),
                      A_layer_max=[float(v) for v in A.max(axis=0)],
                      A_layer_mean=[float(v) for v in A.mean(axis=0)],
                      lam_A_max=[float(rows[np.argmax(A[:, l]), 0]) for l in range(A.shape[1])],
                      sumA_minus_1mR=float(np.max(np.abs(A.sum(axis=1) - (1 - rows[:, 1]))))
                      if not info['lossless'] and abs(np.imag(res[0]['n_layers'][-1])) > 0 else None)
    os.makedirs(RES, exist_ok=True)
    json.dump(s, open(os.path.join(RES, f'{name}.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(RES, f'{name}_spectra.npz'), **{f'eps{e:.3g}': rows for e, rows in sp.items()})
    e0 = s['eps0']
    print(f"{name:20s} L={s['L']} win {s['windows']:2d}  R {s['R_min']:.3f}-{s['R_max']:.3f}  true err "
          f"{s['err_max']:.1e}/{s['err_median']:.1e}  resonances(eps=0) "
          + ', '.join(f"{d['lam']:.3f}um R={d['R']:.2f}" for d in e0['resonances'])
          + (f"  |R_AWE(frozen n) - R| max {disp['max']:.1e} median {disp['median']:.1e}" if disp else '')
          + f"  ({s['wall']:.0f} s)", flush=True)
    return s


def table(rows):
    L = ['| scenario | L | stack | R range | true error max / median | AWE with frozen indices: \\|dR\\| max / median | energy defect log10\\|D\\| max / median | '
         'deepest R dips at eps = 0 (lambda um: R) | deepest R dips at eps_max | max absorptance per layer at eps_max |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for s in sorted(rows, key=lambda r: (r['L'], r['scenario'])):
        em = [k for k in s if k.startswith('eps') and k != 'eps0'][-1]
        dd = f"{s['D_max']:.1f} / {s['D_median']:.1f}" if s['lossless'] else '(absorbing)'
        z = s.get('awe_frozen_vs_exact')
        fz = f"{z['max']:.1e} / {z['median']:.1e}" if z else '-'
        rz = lambda k: ', '.join(f"{d['lam']:.3f}: {d['R']:.2f}" for d in s[k]['resonances'])
        am = ', '.join(f'{v:.2f}' for v in s[em]['A_layer_max'])
        L.append(f"| {s['scenario']} | {s['L']} | {s['stack'].replace('|', '/')[:80]} | {s['R_min']:.3f} - {s['R_max']:.3f} | "
                 f"{s['err_max']:.1e} / {s['err_median']:.1e} | {fz} | {dd} | {rz('eps0')} | {rz(em)} | {am} |")
    open(os.path.join(HERE, 'results', 'explore_summary.md'), 'w').write('\n'.join(L) + '\n')
    return L


def replot(name):
    """redraw <name>_spectra.png from results/explore/<name>_spectra.npz (no recomputation)"""
    sc = R.SCENARIOS[name]
    s = json.load(open(os.path.join(RES, f'{name}.json')))
    d = np.load(os.path.join(RES, f'{name}_spectra.npz'))
    sp = {float(k[3:]): d[k] for k in d.files}
    info = dict(L=s['L'], layers=sc['layers'], scenario=name, stack=s['stack'], lossless=s['lossless'])
    plot_spectra(sp, info, os.path.join(FIG, f"L{s['L']}", f'{name}_spectra.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--db', action='store_true', help='the database-wide gallery (GALLERY_DB)')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--table-only', action='store_true')
    ap.add_argument('--replot', action='store_true', help='redraw the spectra figures from the saved npz files')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    if a.replot:
        for name in (a.only or GALLERY):
            if os.path.exists(os.path.join(RES, f'{name}.json')):
                replot(name)
    elif not a.table_only:
        for name in (a.only or (GALLERY_DB if a.db else GALLERY)):
            try:
                explore(name, a.n, a.workers)
            except Exception as e:                       # noqa: BLE001
                print(f'{name}: FAILED {e!r}', flush=True)
    rows = [json.load(open(os.path.join(RES, f))) for f in sorted(os.listdir(RES)) if f.endswith('.json')]
    print('\n'.join(table(rows)))
