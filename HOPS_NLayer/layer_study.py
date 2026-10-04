"""layer_study.py -- what happens to the reflectivity map, the energy defect, the HOPS/AWE accuracy and the cost as the
number of layers grows (L = 2 ... 8), for three families of stacks (vacuum on top, corrugated top interface or
conformal corrugation, f = cos x):

  bragg      quarter-wave Bragg stacks n_H = 2.3 / n_L = 1.45 on glass (1.52), designed for omega_0 = 1.5:
             L = 2 (bare glass), 3 (H), 4 (HL), 5 (HLH), 6 (HLHL), 7, 8 (HLHLHL)
  bragg_conf the same with ALL interfaces corrugated (conformal)
  graded     graded index 1.1 -> 1.5 in L-2 layers of d = 0.6 over n = 1.6, conformal
  hmm        metal / dielectric multilayer: (Ag 0.05+2.275i, d = 0.15 | n = 2.3, d = 0.3) repeated, on glass, conformal
For every (family, L): the band q = 1 map (omega in [1, 2], eps <= 0.2; joint windows, |delta| <= 0.1, Pade),
  R_flat(omega) (the flat stack), R(omega) at eps = 0.15, the energy defect (lossless) or absorptance,
  the TRUE error of the AWE map against pointwise HOPS, the cost (matrix size, solve time per window), and the
  radii of convergence of the delta- and eps-series of the specular amplitude (root test on the HOPS coefficients):
  they shrink as guided / coupled modes of the stack accumulate near the real frequency axis.
Writes figures/layer_study/*.png and results/layer_study.md.

    python layer_study.py                  # all four families
    python layer_study.py --families bragg --n 41
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                         # noqa: E402
from hops_nl.reference import check_map, tmm        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures', 'layer_study')
RES = os.path.join(HERE, 'results', 'layer_study')
LS = (2, 3, 4, 5, 6, 7, 8)


def stack_def(family, L):
    if family in ('bragg', 'bragg_conf'):
        if L == 2:
            d = dict(layers=[1.0, R._NS], thick=[], profiles=['cosx'])
        else:
            d = R._bragg(L, family == 'bragg_conf')
        return d
    if family == 'graded':
        return R._graded(L) if L > 2 else dict(layers=[1.0, 1.6], thick=[], profiles=['cosx'])
    if family == 'hmm':
        inner = [0.05 + 2.275j if i % 2 == 0 else 2.3 for i in range(L - 2)]
        return dict(layers=[1.0] + inner + [1.5], thick=[0.15 if i % 2 == 0 else 0.3 for i in range(L - 2)],
                    profiles=['cosx'] * (L - 1))
    raise ValueError(family)


def radius(coef):
    """root-test estimate of the radius of convergence from |c_k|, k = 1 .. K (fit of log|c_k| ~ a - k log rho)"""
    k = np.arange(len(coef))
    a = np.abs(coef)
    m = (k >= len(k) // 3) & (a > 0)
    if m.sum() < 3:
        return np.inf
    slope = np.polyfit(k[m], np.log(a[m]), 1)[0]
    return float(np.exp(-slope))


def one(family, L, n=41):
    sc = dict(R._P)
    sc.update(stack_def(family, L))
    sc.update(q=(1,), desc=f'{family} L={L}')
    R.SCENARIOS[f'_{family}_{L}'] = sc
    t0 = time.time()
    res, info = R.run(f'_{family}_{L}', N_Eps=n, N_delta=n, workers=2, verbose=False)
    wall = time.time() - t0
    chk = check_map(res, info)
    lam = np.concatenate([r['omega'] for r in res])
    o = np.argsort(lam)
    cat = lambda k, i=None: np.concatenate([np.real(r[k][i] if i is not None else r[k]) for r in res])[o] if i is not None \
        else np.concatenate([np.real(r[k]).ravel() for r in res])
    ie = int(np.argmin(np.abs(res[0]['Eps'] - 0.15)))
    om = lam[o]
    Rf = np.concatenate([np.real(r['ru_flat'][0]) for r in res])[o]
    R15 = np.concatenate([np.real(r['ru'][ie]) for r in res])[o]
    D = cat('ee')
    # radii of convergence (specular amplitude of the top trace): delta at eps = 0, eps at delta = 0
    rd, re_ = [], []
    for r in res:
        c = np.fft.fft(r['ubar_n_m'], axis=0)[0] / r['ubar_n_m'].shape[0]     # (M+1, N+1)
        rd.append(radius(c[:, 0]) / 1.0)
        re_.append(radius(c[0, :]))
    s = dict(family=family, L=L, windows=len(res), size=int(res[0]['size']), t_solve=float(np.mean([r['t_solve'] for r in res])),
             wall=wall, err_max=chk['max'], err_median=chk['median'], lossless=info['lossless'],
             Rflat_max=float(Rf.max()), R15_max=float(R15.max()), RR_min=float(np.min(cat('ru') / cat('ru_flat'))),
             rho_delta=float(np.median(rd)), rho_delta_min=float(np.min(rd)), rho_eps=float(np.median(re_)),
             D_max=float(np.log10(np.max(np.abs(D)) + 1e-300)) if info['lossless'] else None,
             D_median=float(np.log10(np.median(np.abs(D)) + 1e-300)) if info['lossless'] else None,
             A_max=None if info['lossless'] else float(D.max()), stack=info['stack'])
    curves = dict(omega=om, Rflat=Rf, R15=R15)
    return s, curves, (res, info)


def run(families, n=41):
    os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
    rows = []
    for fam in families:
        allc, maps = {}, {}
        for L in LS:
            s, c, ri = one(fam, L, n)
            rows.append(s)
            allc[L], maps[L] = c, ri
            print(f"{fam:10s} L={L}: {s['windows']:2d} windows, matrix {s['size']:3d}, {s['t_solve']:.2f} s/window, "
                  f"true err {s['err_max']:.1e}/{s['err_median']:.1e}, rho_delta {s['rho_delta']:.3f} (min {s['rho_delta_min']:.3f}), "
                  f"rho_eps {s['rho_eps']:.2f}, max R_flat {s['Rflat_max']:.3f}" +
                  (f", log10|D| max {s['D_max']:.1f}" if s['lossless'] else f", A max {s['A_max']:.2f}"), flush=True)
        json.dump([r for r in rows if r['family'] == fam], open(os.path.join(RES, f'{fam}.json'), 'w'), indent=1)
        figure(fam, allc, maps)
    summary()


def figure(fam, allc, maps):
    """per family: R maps for every L (rows of a grid), R(omega) at eps = 0 and 0.15, and log10|D| or A maps"""
    nL = len(LS)
    fig, axs = plt.subplots(2, nL, figsize=(2.9 * nL, 6.6), sharey='row')
    for k, L in enumerate(LS):
        res, info = maps[L]
        ax = axs[0, k]
        for r in res:
            m = ax.pcolormesh(r['omega'], r['Eps'], np.real(r['ru']), cmap='hot', vmin=0, vmax=1, shading='auto')
        ax.set_title(f'L = {L}', fontsize=10)
        ax.set_xlabel(r'$\omega$')
        if k == 0:
            ax.set_ylabel(r'$\varepsilon$  (map of R)')
        ax2 = axs[1, k]
        c = allc[L]
        ax2.plot(c['omega'], c['Rflat'], 'k-', lw=1.2, label='flat')
        ax2.plot(c['omega'], c['R15'], 'r-', lw=1.0, label=r'$\varepsilon$ = 0.15')
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel(r'$\omega$')
        if k == 0:
            ax2.set_ylabel('R'); ax2.legend(fontsize=7)
    fig.colorbar(m, ax=axs[0, :].tolist(), shrink=0.8, pad=0.01)
    fig.suptitle(f'{fam}: reflectivity of the q = 1 band as layers are added  ({maps[LS[-1]][1]["stack"][:90]})', fontsize=10)
    fig.savefig(os.path.join(FIG, f'{fam}_maps.png'), dpi=95, bbox_inches='tight')
    plt.close(fig)
    lossless = maps[LS[-1]][1]['lossless']
    fig, axs = plt.subplots(1, nL, figsize=(2.9 * nL, 3.2), sharey=True)
    for k, L in enumerate(LS):
        res, info = maps[L]
        for r in res:
            Z = np.log10(np.abs(np.real(r['ee'])) + 1e-17) if info['lossless'] else np.real(r['ee'])
            m = axs[k].pcolormesh(r['omega'], r['Eps'], Z, cmap='hot' if info['lossless'] else 'magma',
                                  vmin=-16 if info['lossless'] else 0, vmax=0 if info['lossless'] else 1, shading='auto')
        axs[k].set_title(f'L = {L}', fontsize=10); axs[k].set_xlabel(r'$\omega$')
    axs[0].set_ylabel(r'$\varepsilon$')
    fig.colorbar(m, ax=axs.tolist(), shrink=0.9, pad=0.01, label=r'log$_{10}|D|$' if lossless else 'absorptance A')
    fig.suptitle(f'{fam}: ' + ('energy defect' if lossless else 'absorptance') + ' as layers are added', fontsize=10)
    fig.savefig(os.path.join(FIG, f'{fam}_D.png'), dpi=95, bbox_inches='tight')
    plt.close(fig)


def summary():
    rows = []
    for f in sorted(os.listdir(RES)):
        if f.endswith('.json'):
            rows += json.load(open(os.path.join(RES, f)))
    fams = sorted({r['family'] for r in rows})
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.4))
    for fam in fams:
        rs = sorted([r for r in rows if r['family'] == fam], key=lambda r: r['L'])
        Lv = [r['L'] for r in rs]
        axs[0].semilogy(Lv, [r['err_max'] for r in rs], 'o-', label=fam)
        axs[1].plot(Lv, [r['rho_delta'] for r in rs], 'o-', label=fam)
        axs[2].plot(Lv, [r['t_solve'] for r in rs], 'o-', label=fam)
        axs[3].plot(Lv, [r['Rflat_max'] for r in rs], 'o-', label=fam)
    for ax, t in zip(axs, ('true max error of the AWE map (q = 1)', 'radius of convergence in delta (median over windows)',
                           'solve time per window (s)', 'max R of the flat stack in q = 1')):
        ax.set_title(t, fontsize=10); ax.set_xlabel('number of layers L'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'layer_study_summary.png'), dpi=100)
    plt.close(fig)
    L = ['| family | L | windows | matrix | solve s/window | true error max / median | radius in delta (median, min) | '
         'radius in eps | max R flat | max R at eps 0.15 | min R/R_flat | log10\\|D\\| max / median or A max |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['family'], r['L'])):
        dd = f"{r['D_max']:.1f} / {r['D_median']:.1f}" if r['lossless'] else f"A {r['A_max']:.2f}"
        L.append(f"| {r['family']} | {r['L']} | {r['windows']} | {r['size']} | {r['t_solve']:.2f} | {r['err_max']:.1e} / "
                 f"{r['err_median']:.1e} | {r['rho_delta']:.3f}, {r['rho_delta_min']:.3f} | {r['rho_eps']:.2f} | "
                 f"{r['Rflat_max']:.3f} | {r['R15_max']:.3f} | {r['RR_min']:.3f} | {dd} |")
    open(os.path.join(HERE, 'results', 'layer_study.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--families', nargs='*', default=['bragg', 'bragg_conf', 'graded', 'hmm'])
    ap.add_argument('--n', type=int, default=41)
    ap.add_argument('--summary-only', action='store_true')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    if a.summary_only:
        summary()
    else:
        run(a.families, a.n)
