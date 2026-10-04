"""run_all_maps.py -- every refl_map_nlayer.py scenario (figures/refl_map/L<L>/) + summary table results/maps_summary.md.

    python run_all_maps.py                  # all scenarios, 100 x 100 per window (as refl_map.py)
    python run_all_maps.py --only bragg_L8 --n 60
"""
import argparse
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R          # noqa: E402
from hops_nl.reference import check_map   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'maps')


def stats(res, info):
    cat = lambda k: np.concatenate([np.real(r[k]).ravel() for r in res])
    Rr, D, Rf = cat('ru'), cat('ee'), cat('ru_flat')
    s = dict(scenario=info['scenario'], L=info['L'], stack=info['stack'], windows=len(res), lossless=info['lossless'],
             summation='Taylor' if info['Taylor'] else 'Pade', R_min=float(Rr.min()), R_max=float(Rr.max()),
             RR_min=float(np.min(Rr / Rf)), RR_max=float(np.max(Rr / Rf)),
             frac_nonphysical=float(np.mean((Rr < -1e-6) | (Rr > 1 + 1e-6))),
             t_solve=float(sum(r['t_solve'] for r in res)), size=int(res[0]['size']), wall=info['t_total'])
    if info['lossless']:
        s.update(D_max=float(np.log10(np.max(np.abs(D)) + 1e-300)), D_median=float(np.log10(np.median(np.abs(D)) + 1e-300)))
    else:
        s.update(A_min=float(D.min()), A_max=float(D.max()), A_median=float(np.median(D)))
    return s


def table(rows):
    L = ['| scenario | L | stack | windows | summation | R range | R/R_flat range | energy defect max / median log10\\|D\\| '
         '(lossless) or absorptance range | true error of R: max / median | matrix size | solve time |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for s in sorted(rows, key=lambda r: (r['L'], r['scenario'])):
        dd = (f"{s['D_max']:.1f} / {s['D_median']:.1f}" if s['lossless'] else f"A = {s['A_min']:.2f} - {s['A_max']:.2f}")
        L.append(f"| {s['scenario']} | {s['L']} | {s['stack'][:70].replace('|', '/')} | {s['windows']} | {s['summation']} | "
                 f"{s['R_min']:.3f} - {s['R_max']:.3f} | {s['RR_min']:.3f} - {s['RR_max']:.3f} | {dd} | {s['err_max']:.1e} / {s['err_median']:.1e} | {s['size']} | "
                 f"{s['t_solve']:.0f} s |")
    open(os.path.join(HERE, 'results', 'maps_summary.md'), 'w').write('\n'.join(L) + '\n')
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    os.makedirs(RES, exist_ok=True)
    for name in (a.only or list(R.SCENARIOS)):
        t0 = time.time()
        res, info = R.run(name, N_Eps=a.n, N_delta=a.n, workers=a.workers, verbose=False)
        R.plot(res, info)
        R.save(res, info, name)
        s = stats(res, info)
        chk = check_map(res, info)
        s.update(err_max=chk['max'], err_median=chk['median'])
        json.dump(s, open(os.path.join(RES, f'{name}.json'), 'w'), indent=1)
        print(f"{name:22s} L={s['L']} windows {s['windows']:3d} R {s['R_min']:.3f}-{s['R_max']:.3f} " +
              (f"log10|D| max {s['D_max']:.1f} median {s['D_median']:.1f}" if s['lossless'] else
               f"A {s['A_min']:.2f}-{s['A_max']:.2f}") + f"  true err {s['err_max']:.1e}/{s['err_median']:.1e}  ({time.time() - t0:.0f} s)", flush=True)
    rows = [json.load(open(os.path.join(RES, f))) for f in sorted(os.listdir(RES)) if f.endswith('.json')]
    print('\n'.join(table(rows)))
