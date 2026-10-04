"""screen_rii_database.py -- go through EVERY page of the refractiveindex.info database (shelves main, other, organic,
glass: 1704 pages of n,k data) and classify each material by the optical regime that matters for multilayer
gratings, to choose the materials of the n-layer scenarios.

The database is CC0 (M. N. Polyanskiy, https://github.com/polyanskiy/refractiveindex.info-database).  Get it with
    python fetch_rii_database.py            (downloads catalog-nk.yml and the 1704 YAML pages, ~50 MB)
or  git clone https://github.com/polyanskiy/refractiveindex.info-database   and set RII_DB=<clone>/database

For every page, n + ik is sampled on a log grid inside its own wavelength range (clipped to 0.2 - 30 um) and:
  plasmonic       Re eps < -2 somewhere:  SPP quality Q_SPP = (Re eps)^2 / Im eps, LSP quality Q_LSP = -Re eps / Im eps
                  (best values and their wavelengths; Blaber et al. 2009 figures of merit)
  ENZ             Re eps changes sign: crossing wavelength(s) and Im eps there
  phonon pol.     Re eps < 0 for lambda > 3 um (Reststrahlen band) with its best Q
  transparent     k < 1e-3: highest n (visible 0.4-0.8 um, NIR 0.8-2 um, MIR 2-20 um)
  absorber        n ~ k ~ 1-5 in the visible (lossy films for ultrathin absorbers)
  phase change    books/pages with two states (amorphous / crystalline, cold / hot, insulating / metallic)
Writes results/rii_screen/rii_screen.csv, results/rii_screen.md, figures/rii_screen/*.png.
"""
import csv
import json
import os
import re
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hops_nl import _paths  # noqa: F401,E402
from hops.materials import RIIMaterial     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'rii_screen')
FIG = os.path.join(HERE, 'figures', 'rii_screen')


def db_root():
    for d in (os.environ.get('RII_DB'), os.path.join(HERE, '..', 'rii'), os.path.join(HERE, 'rii_database')):
        if d and os.path.exists(os.path.join(d, 'catalog-nk.yml')):
            return os.path.abspath(d)
    raise FileNotFoundError('refractiveindex.info database not found: run python fetch_rii_database.py or set RII_DB')


def pages(root):
    import yaml
    c = yaml.safe_load(open(os.path.join(root, 'catalog-nk.yml'), encoding='utf-8'))
    out = []

    def walk(node, shelf=None, book=None, bookname=None, div=None):
        for it in node:
            if 'DIVIDER' in it:
                div = it['DIVIDER']
                continue
            if 'SHELF' in it:
                walk(it['content'], it['SHELF'])
            elif 'BOOK' in it:
                walk(it['content'], shelf, it['BOOK'], it.get('name'), div)
            elif 'PAGE' in it and it.get('data'):
                out.append(dict(shelf=shelf, book=book, bookname=re.sub('<[^>]+>', '', bookname or book or ''),
                                page=it['PAGE'], name=it.get('name', ''), data=it['data'], div=div or ''))
    walk(c)
    return [p for p in out if p['shelf'] in ('main', 'other', 'organic', 'glass')]


PCM = re.compile(r'(amorph|cryst|-a\b|-c\b|cold|hot|insulat|metall|\d+C\b|annealed|as-dep)', re.I)


def screen_page(p, root):
    path = os.path.join(root, 'data', p['data'])
    try:
        m = RIIMaterial(path)
    except Exception as e:                           # noqa: BLE001
        return dict(p, error=repr(e)[:80])
    lo, hi = max(m.range[0], 0.2), min(m.range[1], 30.0)
    if not hi > lo:
        return dict(p, error='range outside 0.2-30 um')
    lam = np.geomspace(lo, hi, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nk = np.asarray(m(lam, warn=False), complex)
    if not np.all(np.isfinite(nk)):
        return dict(p, error='non-finite')
    has_k = m.k_part is not None
    e = nk ** 2
    r = dict(p, lam_min=lo, lam_max=hi, has_k=has_k)
    # plasmonic figures of merit
    pl = (np.real(e) < -2) & (lam >= 0.35) & (lam <= 1.0)          # plasmonic quality in the visible / near IR
    vis_metal = bool(np.any((np.real(e) < -2) & (lam >= 0.7) & (lam <= 2.0)))
    r = dict(r, vis_metal=vis_metal)
    if has_k and pl.any():
        qs = np.real(e) ** 2 / np.maximum(np.imag(e), 1e-12)
        ql = -np.real(e) / np.maximum(np.imag(e), 1e-12)
        i = int(np.argmax(np.where(pl, qs, -1)))
        j = int(np.argmax(np.where(pl, ql, -1)))
        r.update(plasmonic=True, Q_spp=float(qs[i]), lam_Qspp=float(lam[i]), Q_lsp=float(ql[j]), lam_Qlsp=float(lam[j]),
                 lam_metal_onset=float(lam[np.argmax(pl)]))
    else:
        r.update(plasmonic=False)
    # epsilon-near-zero crossings
    s = np.sign(np.real(e))
    cr = np.nonzero(np.diff(s) != 0)[0]
    if has_k and cr.size:
        r.update(enz_lam=float(lam[cr[0]]), enz_im=float(np.imag(e[cr[0]])), n_enz=int(cr.size))
    # phonon polariton (mid-IR Reststrahlen)
    ph = (np.real(e) < 0) & (lam > 3)
    # a Reststrahlen band is bounded: the crystal is a dielectric (Re eps > 0) at shorter wavelengths
    if ph.any():
        first = int(np.argmax(ph))
        if not np.any(np.real(e[:first]) > 0) or np.any(np.real(e[(lam > 0.7) & (lam < 2.0)]) < 0):
            ph[:] = False
    if has_k and ph.any():
        qs = np.real(e) ** 2 / np.maximum(np.imag(e), 1e-12)
        i = int(np.argmax(np.where(ph, qs, -1)))
        r.update(phonon=True, Q_phonon=float(qs[i]), lam_phonon=float(lam[i]),
                 rest_band=(float(lam[ph].min()), float(lam[ph].max())))
    else:
        r.update(phonon=False)
    # transparent high-index windows
    kk = np.imag(nk) if has_k else np.zeros(lam.size)
    for tag, a, b in (('vis', 0.4, 0.8), ('nir', 0.8, 2.0), ('mir', 2.0, 20.0)):
        w = (lam >= a) & (lam <= b) & (kk < 1e-3)
        r[f'n_{tag}'] = float(np.max(np.real(nk[w]))) if w.any() else np.nan
    # visible absorber
    v = (lam >= 0.4) & (lam <= 0.8)
    if has_k and v.any():
        r.update(n_600=float(np.real(m(0.6, warn=False))) if lo <= 0.6 <= hi else np.nan,
                 k_600=float(np.imag(m(0.6, warn=False))) if lo <= 0.6 <= hi else np.nan)
    r['pcm_hint'] = bool(PCM.search(p['page'] + ' ' + p['name']))
    return r


def classify(r):
    if 'error' in r:
        return 'error'
    if r.get('vis_metal'):
        return 'metal'
    if r.get('phonon'):
        return 'phonon polariton' if r['rest_band'][0] > 3.5 else 'IR metal / TCO'
    if r.get('enz_lam') and 0.3 < r['enz_lam'] < 30:
        return 'ENZ'
    if r.get('has_k') and np.isfinite(r.get('k_600', np.nan)) and r['k_600'] > 0.3:
        return 'absorber (visible)'
    if any(np.isfinite(r.get(f'n_{t}', np.nan)) for t in ('vis', 'nir', 'mir')):
        return 'transparent dielectric'
    return 'other'


def main():
    root = db_root()
    ps = pages(root)
    rows = []
    for k, p in enumerate(ps):
        r = screen_page(p, root)
        r['class'] = classify(r)
        rows.append(r)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, 'rii_screen.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT, 'rii_screen.json'), 'w'), default=str)
    report(rows)


def _best_per_book(rows, key, cond=lambda r: True, n=20, reverse=True):
    best = {}
    for r in rows:
        if 'error' in r or not cond(r) or not np.isfinite(r.get(key, np.nan)):
            continue
        b = r['book']
        if b not in best or (r[key] > best[b][key]) == reverse:
            best[b] = r
    return sorted(best.values(), key=lambda r: r[key], reverse=reverse)[:n]


def report(rows):
    from collections import Counter
    ok = [r for r in rows if 'error' not in r]
    cnt = Counter(r['class'] for r in rows)
    L = ['# refractiveindex.info database screen (screen_rii_database.py)', '',
         f'{len(rows)} pages ({len({r["book"] for r in rows})} books) of the shelves main / other / organic / glass; '
         f'{len(ok)} parsed and sampled on 0.2 - 30 um.', '',
         '| class | pages |', '|---|---|'] + [f'| {k} | {v} |' for k, v in cnt.most_common()]
    fmt = lambda r: f"{r['bookname'][:40]} ({r['page']})"
    L += ['', '## Best plasmonic metals in 0.35 - 1.0 um (ranked by Q_LSP = -Re eps / Im eps; Q_SPP = (Re eps)^2 / Im eps; best page per book)', '',
          '| material | Q_SPP (lambda um) | Q_LSP (lambda um) | metallic from (um) |', '|---|---|---|---|']
    for r in _best_per_book(ok, 'Q_lsp', lambda r: r.get('plasmonic') and r['class'] == 'metal', 25):
        L.append(f"| {fmt(r)} | {r['Q_spp']:.3g} ({r['lam_Qspp']:.2f}) | {r['Q_lsp']:.3g} ({r['lam_Qlsp']:.2f}) | {r['lam_metal_onset']:.3f} |")
    L += ['', '## Epsilon-near-zero materials (first crossing in 0.2 - 30 um, low Im eps first)', '',
          '| material | lambda_ENZ (um) | Im eps there |', '|---|---|---|']
    enz = sorted([r for r in ok if r.get('enz_lam') and r['class'] in ('ENZ', 'IR metal / TCO') and r['enz_lam'] > 0.3],
                 key=lambda r: r['enz_im'])
    seen = set()
    for r in enz:
        if r['book'] in seen:
            continue
        seen.add(r['book'])
        L.append(f"| {fmt(r)} | {r['enz_lam']:.3f} | {r['enz_im']:.3g} |")
        if len(seen) >= 25:
            break
    L += ['', '## Phonon-polariton crystals (Re eps < 0 for lambda > 3 um)', '',
          '| material | Reststrahlen band (um) | best Q (lambda um) |', '|---|---|---|']
    for r in _best_per_book(ok, 'Q_phonon', lambda r: r.get('phonon'), 25):
        L.append(f"| {fmt(r)} | {r['rest_band'][0]:.2f} - {r['rest_band'][1]:.2f} | {r['Q_phonon']:.3g} ({r['lam_phonon']:.2f}) |")
    for tag, t in (('vis', 'visible 0.4-0.8 um'), ('nir', 'near IR 0.8-2 um'), ('mir', 'mid IR 2-20 um')):
        L += ['', f'## Highest-index transparent materials, {t} (k < 1e-3)', '', '| material | max n |', '|---|---|']
        for r in _best_per_book(ok, f'n_{tag}', n=15):
            L.append(f"| {fmt(r)} | {r[f'n_{tag}']:.3f} |")
    L += ['', '## Phase-change / switchable pairs (pages whose names mark two states)', '']
    books = {}
    for r in ok:
        if r['pcm_hint']:
            books.setdefault(r['book'], []).append(r['page'])
    for b, pg in sorted(books.items()):
        if len(pg) >= 2:
            L.append(f"* **{b}**: {', '.join(pg[:8])}{' ...' if len(pg) > 8 else ''}")
    open(os.path.join(HERE, 'results', 'rii_screen.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    # figures: n-k map at 0.6 um, plasmonic quality factors, phonon bands
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    cols = {'metal': 'C3', 'IR metal / TCO': 'C1', 'ENZ': 'C6', 'absorber (visible)': 'C5', 'transparent dielectric': 'C0',
            'phonon polariton': 'C2', 'other': '0.6'}
    for c, col in cols.items():
        rr = [r for r in ok if r['class'] == c and np.isfinite(r.get('k_600', np.nan))]
        ax[0].scatter([r['n_600'] for r in rr], [max(r['k_600'], 1e-4) for r in rr], s=9, color=col, label=f'{c} ({len(rr)})')
    ax[0].set_yscale('log'); ax[0].set_xlabel('n (0.6 um)'); ax[0].set_ylabel('k (0.6 um)'); ax[0].legend(fontsize=7)
    ax[0].set_title('every page with n,k at 0.6 um')
    best = _best_per_book(ok, 'Q_lsp', lambda r: r.get('plasmonic') and r['class'] == 'metal', 25)
    ax[1].barh(range(len(best)), [r['Q_lsp'] for r in best], color='C3')
    ax[1].set_yticks(range(len(best))); ax[1].set_yticklabels([f"{r['book']} ({r['lam_Qlsp']:.2f} um)" for r in best], fontsize=7)
    ax[1].set_xscale('log'); ax[1].invert_yaxis(); ax[1].set_xlabel('Q_LSP = -Re eps / Im eps (best in 0.35 - 1.0 um)')
    ax[1].set_title('plasmonic quality, best page per material')
    ph = _best_per_book(ok, 'Q_phonon', lambda r: r.get('phonon'), 25)
    for i, r in enumerate(ph):
        ax[2].plot(r['rest_band'], [i, i], '-', lw=5, color='C2')
    ax[2].set_yticks(range(len(ph))); ax[2].set_yticklabels([r['book'] for r in ph], fontsize=7); ax[2].invert_yaxis()
    ax[2].set_xscale('log'); ax[2].set_xlabel('wavelength (um)'); ax[2].set_title('Reststrahlen bands (Re eps < 0, lambda > 3 um)')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'rii_screen.png'), dpi=110); plt.close(fig)
    print('\n'.join(L[:14]))


if __name__ == '__main__':
    main()
