"""mms_error_nlayer.py -- Method of Manufactured Solutions for the n-layer HOPS/AWE solver (analogue of mms_error.py).

Every layer carries an exact Helmholtz solution (outgoing in the top and bottom layers, two-way in the interior,
see hops_nl/mms.py).  Their interface jumps are fed to multilayer_solve; the recovered interface traces V_j
(j = 1 .. L-1), the top trace ubar (z = a) and the bottom trace wbar (z = -b) are summed (Taylor / Pade) on an
(eps, delta) grid and compared with the EXACT values: relative error max_x |S - u_exact| / max_x |u_exact|.

Presets (figures/mms_error/):
  L3_fig2   L = 3, N = M = 4,  eps, delta <= 1e-2, Taylor   (3-layer analogue of paper Fig. 2)
  L3        L = 3, N = M = 12, eps <= 0.2, |delta| <= 0.1, Taylor and Pade
  L4, L5, L6, L8                 as L3 with 4, 5, 6, 8 layers (lossy interior layers, three different profiles)
  L8_oblique                     L = 8 with oblique incidence alpha = 0.15
  convergence                    max error over the grid vs N = M for L = 3 ... 8 (one figure)

    python mms_error_nlayer.py --preset L5
    python mms_error_nlayer.py --preset convergence
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hops_nl import Stack, multilayer_solve      # noqa: E402
from hops_nl.mms import manufactured             # noqa: E402
from hops.summation import fcn_sum_fast          # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'mms_error')
SHOW = True
_N_ALL = [1.0, 1.5, 2.0 + 0.1j, 1.2, 1.8 + 0.05j, 1.3, 2.2, 1.05]
_D_ALL = [0.9, 1.1, 0.8, 1.0, 0.9, 1.2]


def _profiles(xx):
    return [(np.cos(xx), -np.sin(xx)), (np.cos(2 * xx + 0.5) / 2, -np.sin(2 * xx + 0.5)),
            (0.8 * np.sin(xx), 0.8 * np.cos(xx))]


BASE = dict(L=3, N=12, M=12, Nx=32, Nz=32, eps_max=0.2, delta_max=0.1, omega_bar=1.3, alpha=0.0, mode='TM', r=1,
            n_eps=21, n_delta=21, sums=('taylor', 'pade'))
PRESETS = {
    'L3_fig2': dict(L=3, N=4, M=4, eps_max=1e-2, delta_max=1e-2, sums=('taylor',)),
    'L3': dict(L=3), 'L4': dict(L=4), 'L5': dict(L=5), 'L6': dict(L=6), 'L8': dict(L=8),
    'L8_oblique': dict(L=8, alpha=0.15),
}


def build(c):
    L, Nx = c['L'], c['Nx']
    xx = 2 * np.pi * np.arange(Nx) / Nx
    pr = _profiles(xx)
    f = [pr[j % 3][0] for j in range(L - 1)]
    fx = [pr[j % 3][1] for j in range(L - 1)]
    return Stack(_N_ALL[:L - 1] + [_N_ALL[-1]], _D_ALL[:L - 2], f, fx, 1.0, 1.0, c['Nz'], c['mode'])


def run(cfg, verbose=True):
    c = dict(BASE); c.update(cfg)
    st = build(c)
    N, M = c['N'], c['M']
    zeta, psi, ex = manufactured(st, c['omega_bar'], c['alpha'], N, M, r=c['r'])
    t0 = time.time()
    res = multilayer_solve(st, c['omega_bar'], c['alpha'], N, M, zeta=zeta, psi=psi)
    t_solve = time.time() - t0
    Eps = np.linspace(0, c['eps_max'], c['n_eps'])
    delta = np.linspace(-c['delta_max'], c['delta_max'], c['n_delta'])
    names = [f'V{j}' for j in range(1, st.L)] + ['ubar', 'wbar']
    series = res['V'] + [res['ubar'], res['wbar']]
    Nx = st.f[0].size
    exact = [[ex['exact'](e, d) for d in delta] for e in Eps]
    err = {}
    for kind in c['sums']:
        stype = 1 if kind == 'taylor' else 2
        for name, S in zip(names, series):
            vals = fcn_sum_fast(stype, np.transpose(S, (2, 1, 0)), Eps[:, None], delta[None, :], Nx, N, M,
                                taylor_full_order=True)          # (n_eps, n_delta, Nx)
            E = np.zeros((Eps.size, delta.size))
            for i in range(Eps.size):
                for j in range(delta.size):
                    u = exact[i][j]
                    ue = u['V'][int(name[1:]) - 1] if name.startswith('V') else u[name]
                    E[i, j] = np.max(np.abs(vals[i, j] - ue)) / np.max(np.abs(ue))
            err[(kind, name)] = E
    if verbose:
        print(f"L = {st.L}, N = M = {N}, eps <= {c['eps_max']:g}, |delta| <= {c['delta_max']:g}, alpha = {c['alpha']:g}: "
              f"solve {t_solve:.2f} s (matrix {res['size']})")
        for kind in c['sums']:
            print(f'  {kind:6s} ' + '  '.join(f"{n} {np.max(err[(kind, n)]):.1e}" for n in names))
    return dict(err=err, names=names, Eps=Eps, delta=delta, cfg=c, L=st.L, t_solve=t_solve)


def plot(out, tag, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    c, names = out['cfg'], out['names']
    files = []
    for kind in c['sums']:
        k = len(names)
        ncol = min(5, k)
        nrow = int(np.ceil(k / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
        for ax, name in zip(axs.ravel(), names):
            Z = np.log10(out['err'][(kind, name)] + 1e-17)
            m = ax.pcolormesh(out['delta'], out['Eps'], Z, cmap='hot', shading='auto', vmin=-16, vmax=max(-6, Z.max()))
            fig.colorbar(m, ax=ax)
            ax.set_title(f'{name}: max {np.max(out["err"][(kind, name)]):.1e}', fontsize=9)
            ax.set_xlabel(r'$\delta$'); ax.set_ylabel(r'$\varepsilon$')
        for ax in axs.ravel()[k:]:
            ax.axis('off')
        fig.suptitle(f"MMS, L = {out['L']} layers, N = M = {c['N']}, {kind}: log10 relative error of the interface "
                     f"traces V_j, ubar, wbar" + (f", alpha = {c['alpha']:g}" if c['alpha'] else ''), fontsize=10)
        fig.tight_layout()
        fn = os.path.join(outdir, f'mms_{tag}_{kind}.png')
        fig.savefig(fn, dpi=100)
        plt.close(fig)
        files.append(fn)
    return files


def convergence(outdir=OUTDIR, Ls=(2, 3, 4, 5, 6, 8), NMs=(2, 4, 6, 8, 10, 12, 14)):
    """max relative error over the (eps, delta) grid vs N = M, for L = 2 ... 8 (Taylor and Pade)"""
    os.makedirs(outdir, exist_ok=True)
    rows = {}
    for L in Ls:
        for nm in NMs:
            o = run(dict(L=L, N=nm, M=nm, n_eps=9, n_delta=9), verbose=False)
            for kind in ('taylor', 'pade'):
                rows[(L, kind, nm)] = max(np.max(v) for (k, _), v in o['err'].items() if k == kind)
            print(f'L = {L}, N = M = {nm}: taylor {rows[(L, "taylor", nm)]:.1e}, pade {rows[(L, "pade", nm)]:.1e}', flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, kind in zip(axs, ('taylor', 'pade')):
        for L in Ls:
            ax.semilogy(NMs, [rows[(L, kind, nm)] for nm in NMs], 'o-', label=f'L = {L}')
        ax.set_xlabel('N = M'); ax.set_ylabel('max relative error (all traces, eps <= 0.2, |delta| <= 0.1)')
        ax.set_title(f'MMS convergence, {kind}'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fn = os.path.join(outdir, 'mms_convergence.png')
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    np.savez(os.path.join(outdir, 'mms_convergence.npz'), Ls=Ls, NMs=NMs,
             taylor=np.array([[rows[(L, 'taylor', nm)] for nm in NMs] for L in Ls]),
             pade=np.array([[rows[(L, 'pade', nm)] for nm in NMs] for L in Ls]))
    return fn


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default='L3', choices=list(PRESETS) + ['convergence', 'all'])
    ap.add_argument('--nm', type=int)
    ap.add_argument('--no-show', action='store_true')
    a = ap.parse_args()
    if a.no_show or not SHOW:
        matplotlib.use('Agg')
    names = list(PRESETS) if a.preset == 'all' else [a.preset]
    for p in names:
        if p == 'convergence':
            print('saved', convergence())
            continue
        cfg = dict(PRESETS[p])
        if a.nm:
            cfg.update(N=a.nm, M=a.nm)
        out = run(cfg)
        for fn in plot(out, p):
            print('saved', fn)
    if a.preset == 'all':
        print('saved', convergence())
    if SHOW and not a.no_show:
        plt.show()
