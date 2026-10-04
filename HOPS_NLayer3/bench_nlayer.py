"""bench_nlayer.py -- clean single-process cost of one HOPS/AWE window vs the number of layers L
(Bragg-type stack, top interface cos x, Nz = 24 per layer, N = M = 12).  Writes results/bench.md.

    python bench_nlayer.py
"""
import os
import time
import warnings

import numpy as np

from hops_nl import Stack, multilayer_solve
from hops_nl import solver as S

HERE = os.path.dirname(os.path.abspath(__file__))


def stack(L, Nx, Nz=24):
    x = 2 * np.pi * np.arange(Nx) / Nx
    n = [1.0] + [2.3 if i % 2 == 0 else 1.45 for i in range(L - 2)] + [1.52]
    f = [np.cos(x)] + [0 * x] * (L - 2)
    fx = [-np.sin(x)] + [0 * x] * (L - 2)
    return Stack(n, [0.45 if i % 2 == 0 else 0.72 for i in range(L - 2)], f, fx, 1.0, 1.0, Nz, 'TM')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    rows = ['| L | Nx | matrix size | inverse (s) | full solve (s) | per extra layer (s) |', '|---|---|---|---|---|---|']
    prev = None
    for Nx in (32, 64):
        for L in range(2, 9):
            st = stack(L, Nx)
            ts = []
            for rep in range(3):
                S._INV_CACHE.clear()
                t0 = time.perf_counter()
                r = multilayer_solve(st, 1.5, 0.0, 12, 12)
                ts.append((time.perf_counter() - t0, r['t_inverse']))
            t, ti = min(ts)
            inc = '' if (prev is None or L == 2) else f'{t - prev:.3f}'
            rows.append(f'| {L} | {Nx} | {r["size"]} | {ti:.3f} | {t:.3f} | {inc} |')
            print(rows[-1], flush=True)
            prev = t
    open(os.path.join(HERE, 'results', 'bench.md'), 'w').write('\n'.join(rows) + '\n')
