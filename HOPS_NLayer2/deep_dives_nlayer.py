"""deep_dives_nlayer.py -- detailed studies of the most interesting multilayer material cases (exact material
dispersion: every frequency solved by HOPS in eps with n_l(lambda); per-layer absorption from the volume series).

  tamm     Tamm plasmon (Ag / TiO2-SiO2 mirror) x grating-coupled surface plasmon of the Ag/air face: the
           anticrossing of the two resonances when the period is scanned, and the splitting vs grating depth eps
  lrspp    long-range / short-range surface plasmons of a thin Ag film in silica vs film thickness, compared with
           the flat-film (IMI) mode solver
  sensor   grating-assisted Kretschmann SPR sensor: Au / Ag / Cu / Al, Cr and Ti adhesion layers, protein layer;
           resonance, width, sensitivity (nm / RIU) and figure of merit, flat vs corrugated
  vo2      VO2 film on sapphire through the insulator-metal transition (Bruggeman mix, 0 ... 100 % metallic):
           near-perfect mid-IR absorption at an intermediate state (Kats et al. 2012) and the grating's effect
  ge_au    ultrathin Ge on Au (Kats et al. 2013): absorption peak and reflected colour vs Ge thickness and eps
  cavity   TiO2/SiO2 microcavity with p = 0, 1, 2 mirror pairs (L = 5, 9, 13): Q factor, and the grating-induced
           shift / broadening / guided-mode resonances
  solar    perovskite and thin c-Si cells: absorption in the active layer vs parasitic layers, light trapping vs eps
  contrast HOPS/AWE accuracy (energy defect, true error, delta-radius) of lossless (H L)^3 mirrors vs index contrast

    python deep_dives_nlayer.py                    # all
    python deep_dives_nlayer.py --study tamm sensor
Writes figures/deep_dives/*.png and results/deep_dives/*.json, results/deep_dives.md.
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

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                         # noqa: E402
from hops_nl.absorption import layer_absorption     # noqa: E402
from hops_nl.reference import tmm, check_map        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures', 'deep_dives')
RES = os.path.join(HERE, 'results', 'deep_dives')
WORKERS = 2
REPORT = {}


def n_(key, lam):
    return complex(R._index(key, lam))


def run_spec(base, lam_lo, lam_hi, n_cols=40, eps_max=0.2, n_eps=9, absorption=False, max_delta=0.05, **over):
    """exact-dispersion spectra of a scenario (with overrides) on [lam_lo, lam_hi] um:
    returns dict(lam, Eps, R[n_eps, n_lam], A[n_eps, L, n_lam] or None, info)"""
    sc = dict(R.SCENARIOS[base])
    sc.update(over)
    P = sc['period']
    key = f'_dd_{base}'
    sc.update(omega_range=R._lam(lam_lo, lam_hi, P), dispersion='exact', eps_max=eps_max, max_delta=max_delta)
    R.SCENARIOS[key] = sc
    res, info = R.run(key, N_Eps=n_eps, N_delta=n_cols, workers=WORKERS, verbose=False, keep_fields=absorption)
    sc_ = P / (2 * np.pi)
    lam = np.concatenate([r['lam'] for r in res]) * sc_
    Rr = np.concatenate([np.real(r['ru']) for r in res], axis=1)
    o = np.argsort(lam)
    keep = (lam[o] >= lam_lo * 0.999) & (lam[o] <= lam_hi * 1.001)
    o = o[keep]
    out = dict(lam=lam[o], Eps=res[0]['Eps'], R=Rr[:, o], info=info, res=res)
    if absorption:
        A = []
        for r in res:
            for j in range(len(r['delta'])):
                A.append([layer_absorption(r, info, e, r['delta'][j]) for e in r['Eps']])
        A = np.array(A)                          # (n_lam, n_eps, L)
        out['A'] = np.transpose(A, (1, 2, 0))[:, :, o]
    return out


def dips(lam, Rr, k=2, lo=None, hi=None):
    """k deepest local minima of R(lambda) inside [lo, hi]: list of dict(lam, R, fwhm) (parabolic refinement)"""
    m = np.ones_like(lam, bool)
    if lo is not None:
        m &= lam >= lo
    if hi is not None:
        m &= lam <= hi
    idx = [i for i in range(1, len(Rr) - 1) if m[i] and Rr[i] < Rr[i - 1] and Rr[i] <= Rr[i + 1]]
    idx = sorted(idx, key=lambda i: Rr[i])[:k]
    out = []
    for i in idx:
        y0, y1, y2 = Rr[i - 1], Rr[i], Rr[i + 1]
        den = y0 - 2 * y1 + y2
        sh = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        lm = lam[i] + sh * 0.5 * (lam[i + 1] - lam[i - 1])
        base = np.max(Rr[max(0, i - 25):i + 26])
        half = 0.5 * (base + y1)
        a = i
        while a > 0 and Rr[a] < half:
            a -= 1
        b = i
        while b < len(Rr) - 1 and Rr[b] < half:
            b += 1
        fa = lam[a] + (half - Rr[a]) * (lam[a + 1] - lam[a]) / (Rr[a + 1] - Rr[a] + 1e-300)
        fb = lam[b - 1] + (half - Rr[b - 1]) * (lam[b] - lam[b - 1]) / (Rr[b] - Rr[b - 1] + 1e-300)
        out.append(dict(lam=float(lm), R=float(y1), fwhm=float(abs(fb - fa))))
    return sorted(out, key=lambda d: d['lam'])


def save(study, data, md):
    os.makedirs(RES, exist_ok=True)
    json.dump(data, open(os.path.join(RES, f'{study}.json'), 'w'), indent=1, default=float)
    REPORT[study] = md
    print(md, flush=True)


def _fig(name):
    os.makedirs(FIG, exist_ok=True)
    return os.path.join(FIG, name)


# ---------------------------------------------------------------------------------------------------------------- tamm
def study_tamm():
    """Tamm plasmon x surface plasmon: period scans at eps = 0.05 ... 0.2 (anticrossing maps, minimum separation of
    the two reflection dips), and R vs eps at the crossing period"""
    t0 = time.time()
    lo, hi = 0.60, 0.80
    lam = np.linspace(lo, hi, 801)
    sc = R.SCENARIOS['tamm_ag']
    Rf = np.array([tmm([n_(k, l) for k in sc['layers']], [t * 2 * np.pi for t in sc['thick_um']], 1.0 / l, 0, 'TM')[0]
                   for l in lam])
    lam_T = float(lam[np.argmin(Rf)])

    def spp_lam(P):
        L_ = P
        for _ in range(30):
            e = n_('Ag', L_) ** 2
            L_ = P * np.real(np.sqrt(e / (e + 1)))
        return float(L_)
    periods = np.round(np.arange(0.56, 0.745, 0.01), 3)
    Rmaps = None
    for k, P in enumerate(periods):
        s = run_spec('tamm_ag', lo, hi, n_cols=30, eps_max=0.2, n_eps=5, period=float(P))
        if Rmaps is None:
            Eps = s['Eps']
            Rmaps = np.zeros((len(Eps), len(periods), lam.size))
        for i in range(len(Eps)):
            Rmaps[i, k] = np.interp(lam, s['lam'], s['R'][i])
        print(f'  tamm P = {P:.3f} done', flush=True)
    # the two branches: for every (eps, P) the two deepest local minima of R in [0.62, 0.78] with R < 0.85
    gaps = []
    for i, e in enumerate(Eps):
        sep = []
        for k, P in enumerate(periods):
            d = [x for x in dips(lam, Rmaps[i, k], k=4, lo=0.62, hi=0.78) if x['R'] < 0.85]
            d = sorted(d, key=lambda x: x['R'])[:2]
            if len(d) == 2:
                sep.append((float(P), abs(d[0]['lam'] - d[1]['lam']), d))
        if sep:
            Pm, gm, dm = min(sep, key=lambda t: t[1])
            gaps.append(dict(eps=float(e), P_min=Pm, gap_nm=1e3 * gm, dips=[(x['lam'], x['R']) for x in dm]))
    Pc = float(min(periods, key=lambda P: abs(spp_lam(P) - lam_T)))
    kc = int(np.argmin(np.abs(periods - Pc)))
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for a, i in zip(ax[:3], (1, 2, 4)):
        im = a.pcolormesh(lam, periods, Rmaps[i], cmap='hot', shading='auto', vmin=0, vmax=1)
        a.plot([spp_lam(P) for P in periods], periods, 'c--', lw=1.0, label='Ag/air SPP (flat, 1st order)')
        a.axvline(lam_T, color='w', ls=':', lw=1.0, label=f'flat Tamm state {lam_T:.3f} um')
        a.set_xlabel(r'$\lambda$ ($\mu$m)'); a.set_ylabel(r'period ($\mu$m)'); a.set_title(f'R at eps = {Eps[i]:.2f}')
        if i == 1:
            a.legend(fontsize=7, loc='lower right')
    fig.colorbar(im, ax=ax[2])
    for i in range(len(Eps)):
        ax[3].plot(lam, Rmaps[i, kc], label=f'eps = {Eps[i]:.2f}')
    ax[3].plot(lam, Rf, 'k:', lw=1, label='flat (TMM)')
    ax[3].set_title(f'period {periods[kc]:.2f} um (crossing): R vs eps'); ax[3].legend(fontsize=7)
    ax[3].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[3].grid(alpha=0.3)
    fig.suptitle('L = 8: air | Ag 30 nm | TiO2/SiO2 mirror | BK7 -- Tamm plasmon meets the grating-coupled surface plasmon',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(_fig('tamm_spp_anticrossing.png'), dpi=120); plt.close(fig)
    md = (f"### Tamm plasmon x surface plasmon (tamm_ag, L = 8)\n"
          f"* flat Tamm state at **{lam_T:.4f} um** (TMM, R = {Rf.min():.3f}); the Ag/air SPP (first grating order) crosses it at "
          f"period **{Pc:.3f} um**.\n"
          f"* the period scans show the two dips **anticross** (figures/deep_dives/tamm_spp_anticrossing.png); minimum "
          f"separation of the two dips over the scan: "
          + ', '.join(f"eps {g['eps']:.2f}: {g['gap_nm']:.1f} nm (P {g['P_min']:.2f})" for g in gaps if g['eps'] > 0) + '.\n'
          f"* at the crossing, R at the hybrid dip: " + ', '.join(f"eps {e:.2f}: {Rmaps[i, kc].min():.3f}" for i, e in enumerate(Eps))
          + f"\n* ({time.time() - t0:.0f} s)\n")
    save('tamm', dict(lam_T=lam_T, Pc=Pc, gaps=gaps, periods=periods.tolist(), Eps=Eps.tolist()), md)


# --------------------------------------------------------------------------------------------------------------- lrspp
def imi_mode(eps_m, eps_d, t, lam, branch):
    """effective index of the TM modes of a metal film (eps_m, thickness t) in eps_d:
    branch 'LR' (symmetric H_y): (k_m/eps_m) tanh(k_m t/2) = -k_d/eps_d;  'SR': coth"""
    from scipy.optimize import fsolve
    k0 = 2 * np.pi / lam
    ns = np.sqrt(eps_m * eps_d / (eps_m + eps_d))

    def F(v):
        ne = v[0] + 1j * v[1]
        km = k0 * np.sqrt(ne ** 2 - eps_m + 0j)
        kd = k0 * np.sqrt(ne ** 2 - eps_d + 0j)
        th = np.tanh(km * t / 2)
        g = (km / eps_m) * (th if branch == 'LR' else 1 / th) + kd / eps_d
        return [g.real / k0, g.imag / k0]
    best = None
    for start in ([np.sqrt(eps_d).real * 1.001, 1e-4], [ns.real, ns.imag], [ns.real * 1.3, ns.imag * 2]):
        v, info, ier, _ = fsolve(F, start, full_output=True)
        ne = v[0] + 1j * v[1]
        kd = np.sqrt(ne ** 2 - eps_d + 0j)
        if ier == 1 and np.real(kd) > 0 and ne.real > np.sqrt(eps_d).real:
            if best is None or (branch == 'LR' and ne.imag < best.imag) or (branch == 'SR' and ne.real > best.real):
                best = ne
    return best


def study_lrspp():
    t0 = time.time()
    P, lo, hi = 0.5, 0.70, 0.95
    ths = [0.008, 0.012, 0.016, 0.020, 0.030, 0.040, 0.060]
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for t in ths:
        s = run_spec('lrspp_ag20', lo, hi, n_cols=40, eps_max=0.05, n_eps=2, thick_um=[t])
        d = dips(s['lam'], s['R'][-1], k=2)
        pred = {}
        for br in ('LR', 'SR'):
            L_ = P * 1.47
            ok = True
            for _ in range(40):
                ne = imi_mode(n_('Ag', L_) ** 2, n_('fused_silica', L_) ** 2, t, L_, br)
                if ne is None:
                    ok = False
                    break
                L_ = P * ne.real
            pred[br] = (float(L_), float(ne.imag)) if ok else None
        rows.append(dict(t=t, dips=d, pred={k: v for k, v in pred.items()}))
        ax[0].plot(s['lam'], s['R'][-1], label=f'Ag {1e3 * t:.0f} nm')
        print(f'  lrspp t = {1e3 * t:.0f} nm: dips ' + ', '.join(f"{x['lam']:.4f} (R {x['R']:.2f}, fwhm {1e3 * x['fwhm']:.1f} nm)" for x in d)
              + f"  predicted LR {pred['LR']}, SR {pred['SR']}", flush=True)
    ax[0].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[0].set_ylabel('R (eps = 0.05)'); ax[0].legend(fontsize=7)
    ax[0].set_title('silica | Ag film | silica, period 0.5 um')
    for br, mk in (('LR', 'o'), ('SR', 's')):
        tt = [r['t'] * 1e3 for r in rows if r['pred'][br]]
        ax[1].plot(tt, [r['pred'][br][0] for r in rows if r['pred'][br]], mk + '--', mfc='none', label=f'{br} mode solver (flat film)')
    for r in rows:
        for d_ in r['dips']:
            ax[1].plot(r['t'] * 1e3, d_['lam'], 'k.', ms=8)
    ax[1].plot([], [], 'k.', label='HOPS dips (eps = 0.05)')
    ax[1].set_xlabel('Ag thickness (nm)'); ax[1].set_ylabel(r'resonance $\lambda$ ($\mu$m)'); ax[1].legend(fontsize=8)
    ax[1].set_title('long-range (LR) and short-range (SR) branches')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(_fig('lrspp_thickness.png'), dpi=120); plt.close(fig)
    lines = []
    for r in rows:
        dd = '; '.join(f"{x['lam']:.4f} um (R {x['R']:.2f}, FWHM {1e3 * x['fwhm']:.1f} nm)" for x in r['dips'])
        pl = ', '.join(f"{k} {v[0]:.4f}" for k, v in r['pred'].items() if v)
        lines.append(f"| {1e3 * r['t']:.0f} | {dd} | {pl} |")
    md = ("### Long- and short-range surface plasmons (silica | Ag | silica, L = 3)\n"
          "| Ag (nm) | HOPS dips at eps = 0.05 | flat-film mode solver: lambda = P Re n_eff |\n|---|---|---|\n"
          + '\n'.join(lines) + f"\n\n({time.time() - t0:.0f} s)\n")
    save('lrspp', dict(rows=rows), md)


# -------------------------------------------------------------------------------------------------------------- sensor
def study_sensor():
    t0 = time.time()
    lo, hi = 0.55, 0.95
    cases = [('Au 50 nm', ['BK7', 'Au', None], [0.05]), ('Ag 50 nm', ['BK7', 'Ag', None], [0.05]),
             ('Cu 50 nm', ['BK7', 'Cu', None], [0.05]), ('Al 20 nm', ['BK7', 'Al', None], [0.02]),
             ('Cr 2 nm + Au 48 nm', ['BK7', 'Cr', 'Au', None], [0.002, 0.048]),
             ('Ti 2 nm + Au 48 nm', ['BK7', 'Ti', 'Au', None], [0.002, 0.048]),
             ('Au 50 nm + protein 10 nm', ['BK7', 'Au', 1.45, None], [0.05, 0.01])]
    nas = [1.333, 1.343]
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, lay, th in cases:
        res_ = {}
        for na in nas:
            layers = [v if v is not None else na for v in lay]
            s = run_spec('kretschmann_au', lo, hi, n_cols=30, eps_max=0.1, n_eps=2, layers=layers, thick_um=th,
                         profiles=['cosx'] * (len(layers) - 1))
            res_[na] = s
        out = {}
        for ie, tag in ((0, 'flat'), (1, 'eps0.1')):
            d0 = dips(res_[nas[0]]['lam'], res_[nas[0]]['R'][ie], k=1)
            d1 = dips(res_[nas[1]]['lam'], res_[nas[1]]['R'][ie], k=1)
            if d0 and d1:
                S = (d1[0]['lam'] - d0[0]['lam']) / (nas[1] - nas[0]) * 1e3
                out[tag] = dict(lam=d0[0]['lam'], R=d0[0]['R'], fwhm=d0[0]['fwhm'], S=S, FOM=S / (1e3 * d0[0]['fwhm']))
        rows.append(dict(case=name, **out))
        ax[0].plot(res_[nas[0]]['lam'], res_[nas[0]]['R'][0], label=name)
        print(f'  sensor {name}: ' + str({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
    ax[0].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[0].set_ylabel('R (flat, water)'); ax[0].legend(fontsize=7)
    ax[0].set_title('Kretschmann, BK7 prism, 68 deg, TM')
    names = [r['case'] for r in rows]
    for k, (tag, mk) in enumerate((('flat', 'o'), ('eps0.1', 's'))):
        ax[1].bar(np.arange(len(rows)) + 0.38 * k, [r.get(tag, {}).get('FOM', 0) for r in rows], 0.38,
                  label='flat' if tag == 'flat' else 'grating eps = 0.1')
    ax[1].set_xticks(np.arange(len(rows)) + 0.19); ax[1].set_xticklabels(names, rotation=25, ha='right', fontsize=7)
    ax[1].set_ylabel('figure of merit S / FWHM (1/RIU)'); ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(_fig('sensor_kretschmann.png'), dpi=120); plt.close(fig)
    lines = [f"| {r['case']} | " + ' | '.join(
        (f"{r[t]['lam']:.4f} / {r[t]['R']:.3f} / {1e3 * r[t]['fwhm']:.1f} / {r[t]['S']:.0f} / {r[t]['FOM']:.1f}" if t in r else '-')
        for t in ('flat', 'eps0.1')) + ' |' for r in rows]
    md = ("### Grating-assisted Kretschmann sensor (BK7 prism, 68 deg, TM; analyte 1.333 -> 1.343)\n"
          "| stack | flat: lambda_res (um) / R_min / FWHM (nm) / S (nm/RIU) / FOM | grating eps = 0.1: same |\n|---|---|---|\n"
          + '\n'.join(lines) + f"\n\n({time.time() - t0:.0f} s)\n")
    save('sensor', dict(rows=rows), md)


# ----------------------------------------------------------------------------------------------------------------- vo2
def study_vo2():
    t0 = time.time()
    lo, hi = 8.0, 14.0
    keys = ['VO2_cold'] + [f'VO2_f{p}' for p in range(10, 100, 10)] + ['VO2_hot']
    fr = np.linspace(0, 1, len(keys))
    lam = np.linspace(lo, hi, 241)
    Af, Ag, rows = [], [], []
    for f, k in zip(fr, keys):
        s = run_spec('vo2_cold_sapphire', lo, hi, n_cols=20, eps_max=0.2, n_eps=3, layers=['air', k, 'sapphire_IR'])
        A0, A2 = 1 - s['R'][0], 1 - s['R'][-1]
        Af.append(np.interp(lam, s['lam'], A0)); Ag.append(np.interp(lam, s['lam'], A2))
        i0, i2 = np.argmax(A0), np.argmax(A2)
        rows.append(dict(f=float(f), key=k, A_max_flat=float(A0[i0]), lam_flat=float(s['lam'][i0]),
                         A_max_eps0p2=float(A2[i2]), lam_eps0p2=float(s['lam'][i2])))
        print(f'  vo2 f = {f:.1f}: max A flat {A0[i0]:.3f} at {s["lam"][i0]:.2f} um, eps 0.2: {A2[i2]:.3f} at {s["lam"][i2]:.2f} um', flush=True)
    Af, Ag = np.array(Af), np.array(Ag)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, Z, t in ((ax[0], Af, 'absorptance 1 - R, flat'), (ax[1], Ag, 'absorptance 1 - R, grating eps = 0.2')):
        im = a.pcolormesh(lam, 100 * fr, Z, cmap='inferno', vmin=0, vmax=1, shading='auto')
        a.set_xlabel(r'$\lambda$ ($\mu$m)'); a.set_ylabel('metallic fraction of VO2 (%)'); a.set_title(t)
        fig.colorbar(im, ax=a)
    ax[2].plot(100 * fr, [r['A_max_flat'] for r in rows], 'o-', label='flat')
    ax[2].plot(100 * fr, [r['A_max_eps0p2'] for r in rows], 's-', label='eps = 0.2')
    ax[2].set_xlabel('metallic fraction (%)'); ax[2].set_ylabel('peak absorptance'); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.suptitle('air | VO2 150 nm | sapphire (period 8 um): absorption through the insulator-metal transition', fontsize=11)
    fig.tight_layout(); fig.savefig(_fig('vo2_transition.png'), dpi=120); plt.close(fig)
    best = max(rows, key=lambda r: r['A_max_flat'])
    md = ("### VO2 on sapphire through the phase transition (L = 3, Bruggeman mix)\n"
          "| metallic fraction | peak A flat (lambda um) | peak A at eps = 0.2 (lambda um) |\n|---|---|---|\n"
          + '\n'.join(f"| {100 * r['f']:.0f} % | {r['A_max_flat']:.3f} ({r['lam_flat']:.2f}) | {r['A_max_eps0p2']:.3f} ({r['lam_eps0p2']:.2f}) |" for r in rows)
          + f"\n\nMaximum absorption {best['A_max_flat']:.3f} at {best['lam_flat']:.2f} um for {100 * best['f']:.0f} % metallic "
            f"domains. ({time.time() - t0:.0f} s)\n")
    save('vo2', dict(rows=rows), md)


# ---------------------------------------------------------------------------------------------------------------- ge_au
def _cmf(lam_nm):
    """CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit of Wyman, Sloan & Shirley (JCGT 2013)"""
    def g(x, mu, s1, s2):
        return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
    x = 1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


def reflected_srgb(lam_um, Rr):
    """sRGB colour of a surface with reflectance R(lambda) under an equal-energy illuminant"""
    l = np.linspace(380, 780, 401)
    Ri = np.interp(l, 1e3 * np.asarray(lam_um), Rr)
    x, y, z = _cmf(l)
    X, Y, Z = (np.trapz(Ri * c, l) / np.trapz(y, l) for c in (x, y, z))
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(M @ np.array([X, Y, Z]), 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def study_ge_au():
    t0 = time.time()
    lo, hi = 0.38, 0.85
    ths = [0.0, 0.005, 0.007, 0.010, 0.015, 0.020, 0.025]
    rows, cols = [], []
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for t in ths:
        if t == 0:
            s = run_spec('ge_on_au', lo, hi, n_cols=24, eps_max=0.2, n_eps=3, layers=['air', 'Au'], thick_um=[],
                         profiles=['cosx'])
        else:
            s = run_spec('ge_on_au', lo, hi, n_cols=24, eps_max=0.2, n_eps=3, thick_um=[t])
        A0 = 1 - s['R'][0]
        i = np.argmax(A0)
        rgb = [reflected_srgb(s['lam'], s['R'][k]) for k in range(3)]
        cols.append(rgb)
        rows.append(dict(t=t, A_max=float(A0[i]), lam=float(s['lam'][i]), rgb=[list(map(float, c)) for c in rgb],
                         A_max_eps0p2=float(np.max(1 - s['R'][-1]))))
        ax[0].plot(s['lam'], s['R'][0], label=f'Ge {1e3 * t:.0f} nm')
        ax[1].plot(s['lam'], s['R'][-1], label=f'Ge {1e3 * t:.0f} nm')
        print(f'  ge_au t = {1e3 * t:.0f} nm: max A {A0[i]:.3f} at {s["lam"][i]:.3f} um', flush=True)
    for a, t in ((ax[0], 'flat'), (ax[1], 'grating eps = 0.2 (period 0.5 um)')):
        a.set_xlabel(r'$\lambda$ ($\mu$m)'); a.set_ylabel('R'); a.set_title(t); a.legend(fontsize=7); a.grid(alpha=0.3)
    img = np.array(cols)                           # (n_t, 3 eps, 3)
    ax[2].imshow(np.transpose(img, (1, 0, 2)), aspect='auto', origin='lower',
                 extent=(-0.5, len(ths) - 0.5, -0.05, 0.25))
    ax[2].set_xticks(range(len(ths))); ax[2].set_xticklabels([f'{1e3 * t:.0f}' for t in ths])
    ax[2].set_yticks([0, 0.1, 0.2]); ax[2].set_xlabel('Ge thickness (nm)'); ax[2].set_ylabel('eps')
    ax[2].set_title('reflected colour (equal-energy white)')
    fig.suptitle('air | Ge | Au: strong interference in an ultrathin absorbing film (L = 3; t = 0 is bare Au, L = 2)', fontsize=11)
    fig.tight_layout(); fig.savefig(_fig('ge_on_au_colours.png'), dpi=120); plt.close(fig)
    md = ("### Ultrathin Ge on Au: absorption and colour (L = 3)\n| Ge (nm) | peak absorptance flat (lambda um) | peak A at eps = 0.2 |\n|---|---|---|\n"
          + '\n'.join(f"| {1e3 * r['t']:.0f} | {r['A_max']:.3f} ({r['lam']:.3f}) | {r['A_max_eps0p2']:.3f} |" for r in rows)
          + f"\n\n({time.time() - t0:.0f} s)\n")
    save('ge_au', dict(rows=rows), md)


# -------------------------------------------------------------------------------------------------------------- cavity
def study_cavity():
    t0 = time.time()
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for p, c in zip((0, 1, 2), ('C0', 'C1', 'C2')):
        cv = R._cavity(p)
        L = len(cv['layers'])
        lam = np.linspace(0.6, 0.8, 4001)
        # flat cavity transmission with the period-0.65 nondimensionalisation (omega = P / lambda, d = 2 pi t / P)
        P = 0.65
        T = np.array([tmm([n_(k, l) for k in cv['layers']], [t * 2 * np.pi / P for t in cv['thick_um']], P / l, 0, 'TM')[1]
                      for l in lam])
        ic = np.argmax(T * ((lam > 0.66) & (lam < 0.74)))
        lc = lam[ic]
        half = T[ic] / 2
        a = ic
        while a > 0 and T[a] > half:
            a -= 1
        b = ic
        while b < len(T) - 1 and T[b] > half:
            b += 1
        fw = lam[b] - lam[a]
        win = max(4 * fw, 0.004)
        key = f'_cav{p}'
        R.SCENARIOS[key] = dict(R.SCENARIOS['microcavity_L9'], **cv)
        s = run_spec(key, max(0.6, lc - win), min(0.8, lc + win), n_cols=40, eps_max=0.2, n_eps=5, max_delta=0.01)
        res_e = []
        for i, e in enumerate(s['Eps']):
            Tm = 1 - s['R'][i]
            j = np.argmax(Tm)
            res_e.append(dict(eps=float(e), lam=float(s['lam'][j]), Tmax=float(Tm[j])))
        rows.append(dict(p=p, L=L, lam_c=float(lc), fwhm=float(fw), Q=float(lc / fw), eps=res_e))
        ax[0].semilogy(lam, T + 1e-6, color=c, label=f'p = {p} (L = {L}), Q = {lc / fw:.0f}')
        for i in (0, 2, 4):
            ax[1 if p < 2 else 2].plot(1e3 * (s['lam'] - lc), 1 - s['R'][i], color=c, alpha=0.4 + 0.15 * i,
                                       label=f'p = {p}, eps = {s["Eps"][i]:.2f}')
        print(f'  cavity p = {p} (L = {L}): lambda_c {lc:.4f}, FWHM {1e3 * fw:.2f} nm, Q {lc / fw:.0f}; '
              + ', '.join(f"eps {r['eps']:.2f}: {r['lam']:.4f} T {r['Tmax']:.3f}" for r in res_e), flush=True)
    ax[0].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[0].set_ylabel('T (flat, TMM)'); ax[0].legend(fontsize=7)
    for a in ax[1:]:
        a.set_xlabel(r'$\lambda - \lambda_c$ (nm)'); a.set_ylabel('1 - R (lossless: T)'); a.legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle('TiO2/SiO2 microcavity (H L)^p H | SiO2 half-wave | H (L H)^p, top interface corrugated (period 0.65 um)', fontsize=11)
    fig.tight_layout(); fig.savefig(_fig('microcavity_Q.png'), dpi=120); plt.close(fig)
    md = ("### Microcavity: Q vs mirror pairs, and the grating (L = 5, 9, 13)\n"
          "| p | L | lambda_c (um) | FWHM (nm) | Q | cavity peak at eps = 0 / 0.1 / 0.2 (lambda um, T) |\n|---|---|---|---|---|---|\n"
          + '\n'.join(f"| {r['p']} | {r['L']} | {r['lam_c']:.4f} | {1e3 * r['fwhm']:.2f} | {r['Q']:.0f} | "
                      + ' / '.join(f"{x['lam']:.4f}, {x['Tmax']:.3f}" for x in r['eps'] if abs(x['eps'] * 10 - round(x['eps'] * 10)) < 1e-9)
                      + ' |' for r in rows) + f"\n\n({time.time() - t0:.0f} s)\n")
    save('cavity', dict(rows=rows), md)


# --------------------------------------------------------------------------------------------------------------- solar
def study_solar():
    t0 = time.time()
    out = {}
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    for row, (base, lo, hi, act) in enumerate((('perovskite_cell', 0.40, 0.85, 2), ('si_cell', 0.50, 1.10, 2))):
        s = run_spec(base, lo, hi, n_cols=20, eps_max=0.2, n_eps=5, absorption=True)
        lam, A, info = s['lam'], s['A'], s['info']
        Aact = A[:, act, :]
        mean = [float(np.trapz(Aact[i], lam) / (lam[-1] - lam[0])) for i in range(len(s['Eps']))]
        par = {info['layers'][l] if isinstance(info['layers'][l], str) else str(l):
               [float(np.trapz(A[i, l], lam) / (lam[-1] - lam[0])) for i in range(len(s['Eps']))]
               for l in range(info['L']) if l != act and np.max(A[:, l]) > 1e-4}
        out[base] = dict(eps=list(map(float, s['Eps'])), active_mean=mean, parasitic_mean=par,
                         R_mean=[float(np.trapz(s['R'][i], lam) / (lam[-1] - lam[0])) for i in range(len(s['Eps']))])
        for k, i in enumerate((0, len(s['Eps']) - 1)):
            a = ax[row, k]
            bottom = np.zeros_like(lam)
            for l in range(info['L']):
                if np.max(A[i, l]) < 1e-4:
                    continue
                a.fill_between(lam, bottom, bottom + A[i, l], alpha=0.6, lw=0, label=f"{info['layers'][l]}")
                bottom = bottom + A[i, l]
            a.plot(lam, s['R'][i], 'k-', lw=1.2, label='R')
            a.set_title(f"{base}: eps = {s['Eps'][i]:.2f}"); a.set_ylim(0, 1); a.legend(fontsize=7); a.grid(alpha=0.3)
            a.set_xlabel(r'$\lambda$ ($\mu$m)')
        ax[row, 2].plot(s['Eps'], mean, 'o-', label=f'active layer ({info["layers"][act]})')
        for k_, v in par.items():
            ax[row, 2].plot(s['Eps'], v, 's--', label=f'parasitic: {k_}')
        ax[row, 2].set_xlabel('eps'); ax[row, 2].set_ylabel('spectrally averaged absorptance'); ax[row, 2].legend(fontsize=7)
        ax[row, 2].grid(alpha=0.3)
        print(f'  solar {base}: active mean A vs eps ' + ', '.join(f'{m:.3f}' for m in mean), flush=True)
    fig.tight_layout(); fig.savefig(_fig('solar_light_trapping.png'), dpi=120); plt.close(fig)
    md = "### Thin-film solar cells: where the light is absorbed (per-layer absorption from the volume series)\n"
    for base, v in out.items():
        md += (f"* **{base}**: active-layer absorptance (spectral mean) " +
               ', '.join(f"eps {e:.2f}: {m:.3f}" for e, m in zip(v['eps'], v['active_mean'])) +
               '; parasitic ' + '; '.join(f"{k}: {v_[0]:.3f} -> {v_[-1]:.3f}" for k, v_ in v['parasitic_mean'].items()) + '\n')
    md += f"({time.time() - t0:.0f} s)\n"
    save('solar', out, md)


# ------------------------------------------------------------------------------------------------------------ contrast
def study_contrast():
    """lossless quarter-wave (H L)^3 mirrors, n_L = 1.45, glass substrate: HOPS/AWE accuracy vs n_H"""
    t0 = time.time()
    rows = []
    for nH in (1.6, 2.0, 2.4, 3.0, 3.5, 4.0):
        tH, tL = np.pi / (2 * nH * 1.5), np.pi / (2 * 1.45 * 1.5)
        key = f'_contrast{nH}'
        R.SCENARIOS[key] = R._s('contrast', layers=[1.0] + [nH, 1.45] * 3 + [1.52], thick=[tH, tL] * 3,
                                profiles=['cosx'] + ['flat'] * 6, q=(1,), windows='joint', max_delta=0.1)
        res, info = R.run(key, N_Eps=30, N_delta=30, workers=WORKERS, verbose=False)
        D = np.concatenate([np.abs(np.real(r['ee'])).ravel() for r in res])
        chk = check_map(res, info, cols=(0.0, -0.5, 0.5, -0.95, 0.95))
        Rf = np.concatenate([np.real(r['ru_flat'][0]) for r in res])
        rows.append(dict(nH=nH, contrast=(nH - 1.45) / (nH + 1.45), D_max=float(np.log10(D.max())),
                         D_median=float(np.log10(np.median(D))), err_max=chk['max'], err_median=chk['median'],
                         Rflat_max=float(Rf.max())))
        print(f'  contrast n_H = {nH}: log10|D| max {rows[-1]["D_max"]:.1f} median {rows[-1]["D_median"]:.1f}, true err '
              f'{chk["max"]:.1e} / {chk["median"]:.1e}, R_flat max {Rf.max():.3f}', flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    c = [r['contrast'] for r in rows]
    ax[0].plot(c, [r['D_max'] for r in rows], 'o-', label='max'); ax[0].plot(c, [r['D_median'] for r in rows], 's-', label='median')
    ax[0].set_ylabel(r'log$_{10}$|D|'); ax[1].semilogy(c, [r['err_max'] for r in rows], 'o-', label='max')
    ax[1].semilogy(c, [r['err_median'] for r in rows], 's-', label='median'); ax[1].set_ylabel('true error of R')
    for a in ax:
        a.set_xlabel('index contrast (n_H - n_L)/(n_H + n_L)'); a.legend(); a.grid(alpha=0.3)
    fig.suptitle('HOPS/AWE accuracy for lossless (H L)^3 mirrors (L = 8), band q = 1, |delta| <= 0.1', fontsize=11)
    fig.tight_layout(); fig.savefig(_fig('accuracy_vs_contrast.png'), dpi=120); plt.close(fig)
    md = ("### HOPS/AWE accuracy vs index contrast (lossless (H L)^3 mirrors, L = 8)\n"
          "| n_H | contrast | max R_flat | log10\\|D\\| max / median | true error max / median |\n|---|---|---|---|---|\n"
          + '\n'.join(f"| {r['nH']} | {r['contrast']:.3f} | {r['Rflat_max']:.3f} | {r['D_max']:.1f} / {r['D_median']:.1f} | "
                      f"{r['err_max']:.1e} / {r['err_median']:.1e} |" for r in rows) + f"\n\n({time.time() - t0:.0f} s)\n")
    save('contrast', dict(rows=rows), md)


STUDIES = dict(tamm=study_tamm, lrspp=study_lrspp, sensor=study_sensor, vo2=study_vo2, ge_au=study_ge_au,
               cavity=study_cavity, solar=study_solar, contrast=study_contrast)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', nargs='*', default=list(STUDIES))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    WORKERS = a.workers
    warnings.filterwarnings('ignore')
    for st in a.study:
        try:
            STUDIES[st]()
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'{st}: FAILED {e!r}', flush=True)
    # collect every study's markdown (also those of earlier runs)
    md = ['# Deep dives (deep_dives_nlayer.py)\n']
    for st in STUDIES:
        fn = os.path.join(RES, f'{st}.md')
        if st in REPORT:
            open(fn, 'w').write(REPORT[st])
        if os.path.exists(fn):
            md.append(open(fn).read())
    open(os.path.join(HERE, 'results', 'deep_dives.md'), 'w').write('\n'.join(md))
