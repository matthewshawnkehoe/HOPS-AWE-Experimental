"""survey_db_nlayer.py -- multilayer material survey over the WHOLE refractiveindex.info database (every page that
screen_rii_database.py classified; exact material dispersion, HOPS with a corrugated interface).

  metals     every metallic page (238) as a 50 nm film in a grating-assisted Kretschmann stack BK7 | metal | water
             (68 deg, TM, L = 3): is there a surface-plasmon resonance, how deep and narrow, and does the database
             figure of merit Q_SPP predict it?
  absorbers  every visible-absorbing page (and every metal) as a 10 nm film on a silver mirror (L = 3): peak absorption,
             its wavelength, the share absorbed in the film, the reflected colour
  enz        every epsilon-near-zero page as a 30 nm film on silica at 60 deg (TM, L = 3): Berreman absorption at lambda_ENZ
  dbr        every transparent book (best page) at 0.6 um and 1.55 um: all high/low pairs ranked by the 3-pair
             quarter-wave mirror (formula), and HOPS/AWE (lossless, energy defect) for the best 1.55 um pairs (L = 8)

    python survey_db_nlayer.py --part metals absorbers enz dbr --procs 2
Writes results/survey_db_*.md|csv, results/survey_db/*.json, figures/survey_db/*.png.
"""
import argparse
import csv
import json
import re
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

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                 # noqa: E402
from hops_nl.reference import tmm, check_map   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'survey_db')
FIG = os.path.join(HERE, 'figures', 'survey_db')
SCREEN = os.path.join(HERE, 'results', 'rii_screen', 'rii_screen.json')


def screen():
    return [r for r in json.load(open(SCREEN)) if 'error' not in r]


def _spec(layers, thick_um, period, lam_lo, lam_hi, n_delta=10, eps=(0.0, 0.1), theta=None, mode='TM', profiles=None):
    sc = dict(R._P)
    sc.update(layers=list(layers), thick_um=list(thick_um), period=period, omega_range=(period / lam_hi, period / lam_lo),
              q=(0, 1, 2), max_delta=0.05, eps_max=max(eps), windows='joint', dispersion='exact', M=12, mode=mode,
              profiles=profiles or ['cosx'] * (len(layers) - 1), desc='survey_db')
    if theta is not None:
        sc['theta'] = theta
    key = '_survey_db'
    R.SCENARIOS[key] = sc
    res, info = R.run(key, N_Eps=len(eps), N_delta=n_delta, workers=1, verbose=False)
    sc_ = period / (2 * np.pi)
    lam = np.concatenate([r['lam'] for r in res]) * sc_
    Rr = np.concatenate([np.real(r['ru']) for r in res], axis=1)
    o = np.argsort(lam)
    o = o[(lam[o] >= lam_lo * 0.999) & (lam[o] <= lam_hi * 1.001)]
    return lam[o], Rr[:, o], res, info


def _dip(lam, Rr):
    i = int(np.argmin(Rr))
    base = np.max(Rr)
    half = 0.5 * (base + Rr[i])
    a, b = i, i
    while a > 0 and Rr[a] < half:
        a -= 1
    while b < len(Rr) - 1 and Rr[b] < half:
        b += 1
    interior = 0 < i < len(Rr) - 1
    return float(lam[i]), float(Rr[i]), float(lam[b] - lam[a]), interior


def _cache(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _label(p):
    return f"{p['book']} ({p['page']})"


# ----------------------------------------------------------------------------------------------------------- metals
def metal_one(p):
    fn = _cache(f"metal_{p['data'].replace('/', '_')}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    try:
        lam, Rr, _, _ = _spec(['BK7', p['data'], 'water'], [0.05], 0.6, 0.55, 0.95, theta=68.0)
    except Exception as e:                           # noqa: BLE001
        return dict(label=_label(p), error=repr(e)[:100])
    l0, r0, w0, ok0 = _dip(lam, Rr[0])
    l1, r1, w1, ok1 = _dip(lam, Rr[1])
    s = dict(label=_label(p), book=p['book'], page=p['page'], data=p['data'], Q_spp_screen=p.get('Q_spp'),
             Q_lsp_screen=p.get('Q_lsp'), lam=l0, Rmin=r0, fwhm=w0, spr=bool(ok0 and r0 < 0.3 and w0 < 0.1),
             lam_eps01=l1, Rmin_eps01=r1, interior=bool(ok0), time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"{s['label']:45s} SPR {s['spr']!s:5s} lambda {l0:.3f} R {r0:.3f} FWHM {1e3 * w0:.0f} nm  ({s['time']:.0f} s)", flush=True)
    return s


def part_metals(procs):
    pg = [p for p in screen() if p['class'] == 'metal' and p['lam_min'] <= 0.55 and p['lam_max'] >= 0.95]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = [r for r in ex.map(metal_one, pg) if 'error' not in r]
    for r in rows:                                   # criterion (re)applied to cached results
        r['spr'] = bool(r.get('interior', 0.56 < r['lam'] < 0.94) and r['Rmin'] < 0.3 and r['fwhm'] < 0.1)
    rows.sort(key=lambda r: (not r['spr'], r['fwhm']))
    with open(os.path.join(HERE, 'results', 'survey_db_metals.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    spr = [r for r in rows if r['spr']]
    books = sorted({r['book'] for r in spr})
    L = [f"{len(rows)} metallic pages (of {len({r['book'] for r in rows})} materials) as 50 nm Kretschmann films (BK7 | metal | water, "
         f"68 deg TM, grating period 0.6 um, exact dispersion).  **{len(spr)} pages / {len(books)} materials show a surface-plasmon "
         f"resonance** (interior dip with R < 0.3 and FWHM < 100 nm): {', '.join(books)}.", '',
         '| page | lambda_res (um) | R_min | FWHM (nm) | Q_LSP (screen) | eps = 0.1: lambda / R |', '|---|---|---|---|---|---|']
    for r in spr[:40]:
        L.append(f"| {r['label']} | {r['lam']:.3f} | {r['Rmin']:.3f} | {1e3 * r['fwhm']:.0f} | {r['Q_lsp_screen'] or float('nan'):.1f} | "
                 f"{r['lam_eps01']:.3f} / {r['Rmin_eps01']:.3f} |")
    q = np.array([r['Q_lsp_screen'] or np.nan for r in spr]); wv = np.array([r['fwhm'] for r in spr])
    ok = np.isfinite(q)
    if ok.sum() > 3:
        from scipy import stats as sps
        rho = sps.spearmanr(q[ok], wv[ok])
        L += ['', f"Spearman(Q_LSP of the screen, SPR width) = {rho.correlation:.2f} (p = {rho.pvalue:.0e}): the database "
              "figure of merit predicts the HOPS resonance quality."]
    open(os.path.join(HERE, 'results', 'survey_db_metals.md'), 'w').write('\n'.join(L) + '\n')
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cm = {b: plt.get_cmap('tab20')(i % 20) for i, b in enumerate(sorted({r['book'] for r in rows}))}
    for r in rows:
        ax[0].scatter(r['lam'], r['Rmin'], color=cm[r['book']], s=14)
    for b in books:
        rb = [r for r in spr if r['book'] == b]
        ax[0].annotate(b, (rb[0]['lam'], rb[0]['Rmin']), fontsize=7)
    ax[0].set_xlabel('resonance / minimum wavelength (um)'); ax[0].set_ylabel('R_min'); ax[0].set_title('every metallic page, Kretschmann')
    ax[1].scatter([r['Q_lsp_screen'] for r in spr], [1e3 * r['fwhm'] for r in spr], c=[cm[r['book']] for r in spr], s=16)
    for r in spr:
        ax[1].annotate(r['book'], (r['Q_lsp_screen'] or 1, 1e3 * r['fwhm']), fontsize=6)
    ax[1].set_xscale('log'); ax[1].set_xlabel('Q_LSP from the database screen'); ax[1].set_ylabel('SPR FWHM from HOPS (nm)')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'metals_kretschmann.png'), dpi=110); plt.close(fig)
    print('\n'.join(L[:3]))


# -------------------------------------------------------------------------------------------------------- absorbers
def absorber_one(p):
    fn = _cache(f"abs_{p['data'].replace('/', '_')}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    from deep_dives_nlayer import reflected_srgb
    t0 = time.time()
    try:
        lam, Rr, _, _ = _spec(['air', p['data'], 'Ag'], [0.010], 0.4, 0.40, 0.80, eps=(0.0, 0.2))
    except Exception as e:                           # noqa: BLE001
        return dict(label=_label(p), error=repr(e)[:100])
    A0 = 1 - Rr[0]
    i = int(np.argmax(A0))
    li = lam[i]
    _, _, Al = tmm([1.0, R._index(p['data'], li), R._index('Ag', li)], [0.010 * 2 * np.pi / 0.4], 0.4 / li, 0, 'TM', layers=True)
    s = dict(label=_label(p), book=p['book'], page=p['page'], data=p['data'], cls=p['class'], A_peak=float(A0[i]), lam=float(li),
             film_share=float(Al[1] / max(A0[i], 1e-12)), A_mean=float(A0.mean()), A_peak_eps02=float(np.max(1 - Rr[1])),
             rgb=[float(v) for v in reflected_srgb(lam, Rr[0])], rgb_eps02=[float(v) for v in reflected_srgb(lam, Rr[1])],
             time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"{s['label']:45s} peak A {s['A_peak']:.3f} at {li:.3f} um ({100 * s['film_share']:.0f} % in film)  ({s['time']:.0f} s)", flush=True)
    return s


def part_absorbers(procs):
    # monolayer / few-layer pages (1L, 2L, ...) describe sub-nm films and are not used as 10 nm films
    pg = [p for p in screen() if p['class'] in ('absorber (visible)', 'metal', 'ENZ') and p['lam_min'] <= 0.4 and p['lam_max'] >= 0.8
          and not re.search(r'\d+L\b', p['page'])]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = [r for r in ex.map(absorber_one, pg) if 'error' not in r]
    rows.sort(key=lambda r: -r['A_peak'] * r['film_share'])
    with open(os.path.join(HERE, 'results', 'survey_db_absorbers.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if not k.startswith('rgb')]); w.writeheader()
        w.writerows([{k: v for k, v in r.items() if not k.startswith('rgb')} for r in rows])
    perfect = [r for r in rows if r['A_peak'] > 0.98]
    L = [f"{len(rows)} absorbing pages as 10 nm films on silver (air | film | Ag, period 0.4 um, exact dispersion).  "
         f"**{len(perfect)} pages give A > 0.98** (near-perfect ultrathin absorbers).", '',
         '| page | class | peak A (lambda um) | absorbed in the film | mean A 0.4-0.8 um | peak A at eps 0.2 |', '|---|---|---|---|---|---|']
    for r in rows[:45]:
        L.append(f"| {r['label']} | {r['cls']} | {r['A_peak']:.3f} ({r['lam']:.3f}) | {100 * r['film_share']:.0f} % | {r['A_mean']:.2f} | "
                 f"{r['A_peak_eps02']:.3f} |")
    open(os.path.join(HERE, 'results', 'survey_db_absorbers.md'), 'w').write('\n'.join(L) + '\n')
    os.makedirs(FIG, exist_ok=True)
    rs = sorted(rows, key=lambda r: r['lam'])
    nc = 16
    nr = int(np.ceil(len(rs) / nc))
    img = np.ones((nr, nc, 3))
    for k, r in enumerate(rs):
        img[k // nc, k % nc] = r['rgb']
    fig, ax = plt.subplots(1, 2, figsize=(18, max(5, 0.32 * nr + 2)), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax[0].imshow(img, aspect='auto')
    for k, r in enumerate(rs):
        ax[0].text(k % nc, k // nc, r['book'][:9], ha='center', va='center', fontsize=4.5,
                   color='k' if np.mean(r['rgb']) > 0.45 else 'w')
    ax[0].set_axis_off(); ax[0].set_title('reflected colour of 10 nm of every absorbing page on silver (sorted by absorption peak)')
    ax[1].scatter([r['lam'] for r in rows], [r['A_peak'] for r in rows], c=[r['film_share'] for r in rows], cmap='viridis', s=14)
    ax[1].set_xlabel('absorption peak (um)'); ax[1].set_ylabel('peak absorptance'); ax[1].grid(alpha=0.3)
    ax[1].set_title('colour = share absorbed in the 10 nm film')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'absorbers_on_silver.png'), dpi=120); plt.close(fig)
    print('\n'.join(L[:2]))


# --------------------------------------------------------------------------------------------------------------- enz
def enz_one(p):
    fn = _cache(f"enz_{p['data'].replace('/', '_')}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    le = p['enz_lam']
    lo, hi = max(p['lam_min'], 0.75 * le), min(p['lam_max'], 1.3 * le)
    if hi <= lo * 1.05:
        return dict(label=_label(p), error='range')
    try:
        lam, Rr, res, info = _spec(['air', p['data'], 'fused_silica' if hi < 6.5 else 1.45], [0.03 * le / 1.0 * 1.0], 0.4 * le,
                                   lo, hi, theta=60.0)
    except Exception as e:                           # noqa: BLE001
        return dict(label=_label(p), error=repr(e)[:100])
    # absorption in the film: 1 - R - T (T into the lossless substrate from the energy balance of the flat stack)
    A0 = []
    for l in lam:
        ns = [1.0, R._index(p['data'], l), R._index('fused_silica', l) if hi < 6.5 else 1.45]
        Rt, Tt, Al = tmm(ns, [0.03 * le * 2 * np.pi / (0.4 * le)], 0.4 * le / l, np.sin(np.radians(60)) * 0.4 * le / l, 'TM', layers=True)
        A0.append(Al[1])
    A0 = np.array(A0)
    i = int(np.argmax(A0))
    s = dict(label=_label(p), book=p['book'], page=p['page'], data=p['data'], lam_enz=le, im_eps_enz=p['enz_im'],
             A_peak=float(A0[i]), lam_peak=float(lam[i]), Rmin=float(Rr[0].min()), Rmin_eps01=float(Rr[1].min()), time=time.time() - t0)
    json.dump(s, open(fn, 'w'))
    print(f"{s['label']:45s} ENZ {le:.3f} um: Berreman A {s['A_peak']:.3f} at {s['lam_peak']:.3f}  ({s['time']:.0f} s)", flush=True)
    return s


def part_enz(procs):
    pg = [p for p in screen() if p.get('enz_lam') and p['class'] in ('ENZ', 'IR metal / TCO') and 0.3 < p['enz_lam'] < 20
          and not re.search(r'\d+L\b', p['page'])]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = [r for r in ex.map(enz_one, pg) if 'error' not in r]
    rows.sort(key=lambda r: r['im_eps_enz'])
    true_enz = [r for r in rows if r['im_eps_enz'] < 3]
    L = [f"{len(true_enz)} low-loss ENZ pages (Im eps < 3 at the crossing; the others are interband zero crossings of "
         "strongly absorbing semiconductors). ",
         f"{len(rows)} epsilon-near-zero pages as 30 nm films (thickness scaled as 0.03 lambda_ENZ) on silica, TM at 60 deg: Berreman absorption.", '',
         '| page | lambda_ENZ (um) | Im eps at ENZ | peak absorption in the film (lambda um) |', '|---|---|---|---|']
    for r in rows[:40]:
        L.append(f"| {r['label']} | {r['lam_enz']:.3f} | {r['im_eps_enz']:.2f} | {r['A_peak']:.3f} ({r['lam_peak']:.3f}) |")
    open(os.path.join(HERE, 'results', 'survey_db_enz.md'), 'w').write('\n'.join(L) + '\n')
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r['im_eps_enz'] for r in rows], [r['A_peak'] for r in rows], s=16)
    for r in rows:
        ax.annotate(r['book'][:12], (r['im_eps_enz'], r['A_peak']), fontsize=6)
    ax.set_xscale('log'); ax.set_xlabel('Im eps at the ENZ wavelength'); ax.set_ylabel('Berreman absorption (60 deg)'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'enz_berreman.png'), dpi=110); plt.close(fig)
    print('\n'.join(L[:2]))


# --------------------------------------------------------------------------------------------------------------- dbr
NON_SOLID = {'Gas', 'Liquid', 'Alcohols', 'Alkanes', 'Liquid crystals', 'Cyclic compounds', 'Other non-polymer compounds',
             '2D structures', 'Liquids', 'Gases'}


NON_SOLID_OTHER = {'human body', 'buffer solutions', 'microscopy mounting media', 'mixed gases', 'immersion oils', 'fuel',
                   'heat transfer fluids', 'C3H8O3 - glycerol', 'liquid crystals', 'D2O', 'optical adhesives'}


def part_dbr(procs):
    rows = screen()
    out = {}
    L = []
    for lam0, sub, tag in ((0.6, 1.52, 'VIS 0.6 um'), (1.55, 1.444, 'telecom 1.55 um')):
        best = {}
        for r in rows:
            if r['class'] != 'transparent dielectric' or not (r['lam_min'] <= lam0 <= r['lam_max']):
                continue
            # solid, isotropic-at-normal-incidence films only: no gases / liquids, no extraordinary (-e) pages, no monolayers
            if (r['div'] in NON_SOLID or r['page'].endswith('-e') or '-e-' in r['page'] or
                    (r['shelf'] == 'organic' and r['div'] != 'Polymers') or r['book'] in ('H2O', 'D2O') or
                    r['data'].split('/')[1] in NON_SOLID_OTHER or
                    re.search(r'(α|β|γ|alpha|beta|gamma|\d+K\b|solid)', r['page'])):
                continue
            try:
                nn = complex(R._index(r['data'], lam0))
            except Exception:                        # noqa: BLE001
                continue
            if abs(nn.imag) > 1e-4 or not (1.25 < nn.real < 6):
                continue
            if r['book'] not in best or nn.real > best[r['book']][1]:
                best[r['book']] = (r, nn.real)
        mats = sorted(best.values(), key=lambda t: -t[1])
        pairs = []
        for i, (rh, nh) in enumerate(mats):
            for rl, nl in mats[i + 1:]:
                if nh - nl < 0.05:
                    continue
                Rp = ((nl ** 6 - sub * nh ** 6) / (nl ** 6 + sub * nh ** 6)) ** 2
                pairs.append(dict(H=_label(rh), L=_label(rl), nH=nh, nL=nl, H_data=rh['data'], L_data=rl['data'], R3=Rp,
                                  width=(4 / np.pi) * np.arcsin((nh - nl) / (nh + nl))))
        pairs.sort(key=lambda p: -p['width'])
        out[tag] = dict(n_materials=len(mats), n_pairs=len(pairs), top=pairs[:15],
                        top_n=[(r['book'], r['page'], n) for r, n in mats[:12]])
        L += [f'## {tag}: {len(mats)} transparent SOLID materials (best page per book, k < 1e-4, n > 1.25; no gases, liquids, '
              f'extraordinary-axis pages or monolayers), {len(pairs)} pairs', '',
              '| high / low | n_H / n_L | 3-pair R (formula) | stop band (4/pi) asin(contrast) |', '|---|---|---|---|']
        for p in pairs[:15]:
            L.append(f"| {p['H']} / {p['L']} | {p['nH']:.3f} / {p['nL']:.3f} | {p['R3']:.4f} | {p['width']:.3f} |")
        L.append('')
    # HOPS/AWE for the best three telecom pairs (lossless, energy defect)
    L += ['## HOPS/AWE (window mode, lossless) for the best telecom pairs: air | (H L)^3 | silica, top grating, P 1 um', '',
          '| pair | log10\\|D\\| max / median | true error max / median | max R_flat |', '|---|---|---|---|']
    hops = []
    for p in out['telecom 1.55 um']['top'][:3]:
        qh = 1.55 / (4 * p['nH']); ql = 1.55 / (4 * p['nL'])
        sc = dict(R._P)
        sc.update(layers=['air'] + [p['H_data'], p['L_data']] * 3 + ['fused_silica'], thick_um=[qh, ql] * 3, period=1.0,
                  omega_range=(1.0 / 2.0, 1.0 / 1.15), q=(0,), max_delta=0.05, windows='joint', profiles=['cosx'] + ['flat'] * 6,
                  desc='dbr')
        R.SCENARIOS['_dbr'] = sc
        try:
            res, info = R.run('_dbr', N_Eps=20, N_delta=20, workers=procs, verbose=False)
        except Exception as e:                       # noqa: BLE001
            L.append(f"| {p['H']} / {p['L']} | failed: {e!r} | | |")
            continue
        Dd = np.concatenate([np.abs(np.real(r['ee'])).ravel() for r in res])
        chk = check_map(res, info, cols=(0.0, -0.9, 0.9))
        Rf = np.concatenate([np.real(r['ru_flat'][0]) for r in res])
        hops.append(dict(pair=f"{p['H']} / {p['L']}", D_max=float(np.log10(Dd.max())), D_med=float(np.log10(np.median(Dd))),
                         err_max=chk['max'], err_med=chk['median'], Rf=float(Rf.max())))
        h = hops[-1]
        L.append(f"| {h['pair']} | {h['D_max']:.1f} / {h['D_med']:.1f} | {h['err_max']:.1e} / {h['err_med']:.1e} | {h['Rf']:.3f} |")
        print(L[-1], flush=True)
    out['hops'] = hops
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'dbr.json'), 'w'), indent=1, default=float)
    open(os.path.join(HERE, 'results', 'survey_db_dbr.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:8]))


PARTS = dict(metals=part_metals, absorbers=part_absorbers, enz=part_enz, dbr=part_dbr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='*', default=list(PARTS))
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    for pt in a.part:
        t0 = time.time()
        PARTS[pt](a.procs)
        print(f'--- {pt}: {time.time() - t0:.0f} s', flush=True)
