"""test_single_eps_delta_nlayer.py -- the n-layer analogue of test_single_eps_delta.py.

Checks every output of the n-layer HOPS/AWE pipeline at a single (eps, delta) for all truncation orders
0 <= n <= N, 0 <= m <= M, against the manufactured exact solution (hops_nl/mms.py):
    V_j      interface traces (j = 1 .. L-1)
    ubar     top trace (z = a),  wbar  bottom trace (z = -b)
    u_l      the VOLUME field of every layer l at all TFE collocation nodes (physical z of each node)
Figures (figures/test_single_eps_delta/):
    single_L<L>_taylor_nm.png   log10 relative error of the truncated Taylor sum over (n, m), every quantity
    single_L<L>_orders.png      error vs N = M: Taylor and Pade, every quantity
RunNumber 1: L = 3, N = M = 8,  eps = delta = 1e-2   (fast)
RunNumber 2: L = 5, N = M = 10, eps = 0.1, delta = 0.05
RunNumber 3: L = 8, N = M = 12, eps = 0.1, delta = 0.05
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hops_nl import multilayer_solve             # noqa: E402
from hops_nl.mms import manufactured             # noqa: E402
from hops.summation import fcn_sum_fast          # noqa: E402
import mms_error_nlayer as mms                   # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'test_single_eps_delta')
SHOW = True
RUNS = {1: dict(L=3, N=8, M=8, Nx=16, Nz=16, eps=1e-2, delta=1e-2),
        2: dict(L=5, N=10, M=10, Nx=32, Nz=24, eps=0.1, delta=0.05),
        3: dict(L=8, N=12, M=12, Nx=32, Nz=24, eps=0.1, delta=0.05)}


def run(RunNumber=1, make_plots=True, verbose=True):
    c = dict(mms.BASE); c.update(RUNS[RunNumber])
    st = mms.build(c)
    N, M, e, d = c['N'], c['M'], c['eps'], c['delta']
    zeta, psi, ex = manufactured(st, c['omega_bar'], c['alpha'], N, M)
    res = multilayer_solve(st, c['omega_bar'], c['alpha'], N, M, zeta=zeta, psi=psi, keep_volume=True)
    exact = ex['exact'](e, d)
    names = [f'V{j}' for j in range(1, st.L)] + ['ubar', 'wbar'] + [f'u{l}' for l in range(st.L)]
    series = res['V'] + [res['ubar'], res['wbar']]
    truth = exact['V'] + [exact['ubar'], exact['wbar']]
    for l in range(st.L):                                  # volume (Nx, Nz+1, M+1, N+1) -> flatten nodes
        v = res['vol'][l]
        series.append(v.reshape(-1, M + 1, N + 1))
        truth.append(exact['vol'][l].ravel())
    nm = np.zeros((len(names), N + 1, M + 1))
    orders = {k: np.zeros((len(names), min(N, M) + 1)) for k in ('taylor', 'pade')}
    pw = (e ** np.arange(N + 1))[None, None, :] * (d ** np.arange(M + 1))[None, :, None]
    for q, (S, u) in enumerate(zip(series, truth)):
        T = S * pw                                       # (P, M+1, N+1) terms eps^n delta^m
        cum = np.cumsum(np.cumsum(T, axis=1), axis=2)    # truncated double sums
        scale = np.max(np.abs(u))
        for nn in range(N + 1):
            for mm_ in range(M + 1):
                nm[q, nn, mm_] = np.max(np.abs(cum[:, mm_, nn] - u)) / scale
        for k in range(min(N, M) + 1):
            orders['taylor'][q, k] = nm[q, k, k]
            if k >= 2:
                v = fcn_sum_fast(2, np.transpose(S[:, :k + 1, :k + 1], (2, 1, 0)), np.array([[e]]), np.array([[d]]),
                                 S.shape[0], k, k)
                orders['pade'][q, k] = np.max(np.abs(np.ravel(v) - u)) / scale
            else:
                orders['pade'][q, k] = np.nan
    if verbose:
        print(f"RunNumber {RunNumber}: L = {st.L}, N = M = {N}, eps = {e:g}, delta = {d:g}")
        for q, n_ in enumerate(names):
            print(f"  {n_:5s} Taylor (N, M) {nm[q, N, M]:.1e}   Pade {orders['pade'][q, -1]:.1e}")
    out = dict(names=names, nm=nm, orders=orders, cfg=c, L=st.L)
    if make_plots:
        plot(out, RunNumber)
    return out


def plot(out, RunNumber, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    names, nm, L = out['names'], out['nm'], out['L']
    k = len(names)
    ncol = min(6, k)
    nrow = int(np.ceil(k / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), squeeze=False)
    for ax, q in zip(axs.ravel(), range(k)):
        m = ax.imshow(np.log10(nm[q] + 1e-17), origin='lower', cmap='viridis', vmin=-16, vmax=0, aspect='auto')
        ax.set_title(names[q], fontsize=9); ax.set_xlabel('M (delta order)'); ax.set_ylabel('N (eps order)')
        fig.colorbar(m, ax=ax)
    for ax in axs.ravel()[k:]:
        ax.axis('off')
    c = out['cfg']
    fig.suptitle(f"L = {L}: log10 relative error of the truncated Taylor sums at eps = {c['eps']:g}, delta = {c['delta']:g} "
                 f"(V_j interface traces, ubar/wbar artificial-boundary traces, u_l volume fields)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'single_L{L}_taylor_nm.png'), dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, kind in zip(axs, ('taylor', 'pade')):
        K = np.arange(out['orders'][kind].shape[1])
        for q in range(k):
            ax.semilogy(K, out['orders'][kind][q] + 1e-17, '-' if names[q].startswith(('V', 'u')) else '--', label=names[q])
        ax.set_xlabel('N = M'); ax.set_ylabel('relative error'); ax.set_title(f'L = {L}, {kind}')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'single_L{L}_orders.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=int, nargs='*', default=[1, 2, 3])
    ap.add_argument('--no-show', action='store_true')
    a = ap.parse_args()
    if a.no_show or not SHOW:
        matplotlib.use('Agg')
    for rn in a.run:
        run(rn)
    if SHOW and not a.no_show:
        plt.show()
