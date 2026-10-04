"""deep_dives_db_nlayer.py -- detailed studies of the multilayer stacks built from the WHOLE refractiveindex.info database
(materials chosen by screen_rii_database.py).  Exact material dispersion; flat-stack sweeps by the transfer matrix
(validated against HOPS at eps = 0), corrugated stacks by HOPS.

  ws2       WS2 monolayer (0.6 nm, Hsu 2019) in a TiO2/SiO2 microcavity (L = 15): exciton-polaritons, angle dispersion,
            coupled-oscillator Rabi splitting; the grating (period 1 um) imprints the large-angle polaritons on the
            normal-incidence spectrum
  sqib      SQIB squaraine film (Funke 2016) in a silver cavity (L = 6): ultrastrong coupling, spacer-thickness
            anticrossing, absorption in the dye vs the silver
  gst       Ge2Sb2Te5 (Frantz 2023) amorphous -> crystalline (Lorentz-Lorenz mix): colour pixels (ITO/GST/ITO/Pt, L = 5)
            and a mid-IR switchable absorber (GST on Au, L = 3)
  vo2       thermochromic window Si3N4/VO2/Si3N4 on glass through the measured VO2 temperature series (Oguntoye 2023,
            20 ... 80 C): luminous and solar transmittance
  lc        5CB liquid-crystal cavity between silver mirrors, TE (exact for a uniaxial LC with its axis along y or z)
  thermo    Kretschmann sensor with hot gold (Magnozzi 25 / 225 / 350 C) and hot silver (Ferrera 298 ... 600 K)
  largeL    polystyrene / PMMA Bragg mirrors with 5 ... 25 pairs (L = 12 ... 52 layers): HOPS/AWE cost and accuracy

    python deep_dives_db_nlayer.py --study ws2 sqib
Writes figures/deep_dives/db_*.png, results/deep_dives/db_*.{json,md}, results/deep_dives_db.md.
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
import deep_dives_nlayer as D                       # noqa: E402
from hops_nl.reference import tmm, check_map        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'deep_dives')
REPORT = {}
HC = 1.23984193                                     # eV um


def n_(k, l):
    return complex(R._index(k, l))


def tmm_spec(layers, th_um, lams, mode='TM', theta=0.0, layers_out=False):
    """flat stack (thicknesses in um) with exact dispersion; returns R, T (and A_l)"""
    out = []
    for l in lams:
        ns = [n_(k, l) for k in layers]
        a = np.real(ns[0]) / l * np.sin(np.radians(theta))
        r = tmm(ns, [t * 2 * np.pi for t in th_um], 1.0 / l, a, mode, layers=True)
        out.append(np.r_[r[0], r[1], r[2]])
    out = np.array(out)
    return out if layers_out else (out[:, 0], out[:, 1])


def minima(x, y, thr=0.9):
    return [i for i in range(1, len(y) - 1) if y[i] < y[i - 1] and y[i] <= y[i + 1] and y[i] < thr]


def save(study, data, md):
    os.makedirs(RES, exist_ok=True)
    json.dump(data, open(os.path.join(RES, f'db_{study}.json'), 'w'), indent=1, default=float)
    open(os.path.join(RES, f'db_{study}.md'), 'w').write(md)
    REPORT[study] = md
    print(md, flush=True)


# ------------------------------------------------------------------------------------------------------------------ ws2
def study_ws2():
    t0 = time.time()
    sc = R.SCENARIOS['ws2_polariton_L15']
    lay, th = sc['layers'], sc['thick_um']
    empty_lay = [k for k in lay if k != 'WS2_1L']
    i_ws = lay.index('WS2_1L')
    empty_th = th[:i_ws - 1] + th[i_ws:]           # drop the monolayer (thickness index = layer index - 1)
    lams = np.linspace(0.56, 0.66, 801)
    thetas = np.arange(0, 41, 2.0)
    maps, rows = {}, []
    for tag, (L_, T_) in (('with WS2', (lay, th)), ('empty', (empty_lay, empty_th))):
        Rm = np.array([tmm_spec(L_, T_, lams, 'TM', th_)[0] for th_ in thetas])
        maps[tag] = Rm
    Ex = HC / R._LX
    for k, th_ in enumerate(thetas):
        ic = minima(lams, maps['empty'][k], 0.95)
        ic = [i for i in ic if 0.56 < lams[i] < 0.62]
        Ec = HC / lams[min(ic, key=lambda i: maps['empty'][k][i])] if ic else np.nan
        ip = sorted(minima(lams, maps['with WS2'][k], 0.97), key=lambda i: maps['with WS2'][k][i])[:2]
        Es = sorted(HC / lams[i] for i in ip)
        rows.append(dict(theta=float(th_), E_cav=float(Ec), E_pol=[float(e) for e in Es]))
    # coupled-oscillator fit: E +/- = (Ec + Ex)/2 +/- sqrt(g^2 + (Ec - Ex)^2 / 4)  ->  g from every angle with 2 branches
    gs = []
    for r in rows:
        if len(r['E_pol']) == 2 and np.isfinite(r['E_cav']):
            Ep, Em = r['E_pol'][1], r['E_pol'][0]
            gs.append(np.sqrt(max(((Ep - Em) / 2) ** 2 - ((r['E_cav'] - Ex) / 2) ** 2, 0)))
    # the splitting at the smallest cavity-exciton detuning (the median over all angles is biased where one branch
    # becomes exciton-like and its dip fades)
    both = [r for r in rows if len(r['E_pol']) == 2 and np.isfinite(r['E_cav'])]
    r0 = min(both, key=lambda r: abs(r['E_cav'] - Ex)) if both else None
    g = float(np.sqrt(max(((r0['E_pol'][1] - r0['E_pol'][0]) / 2) ** 2 - ((r0['E_cav'] - Ex) / 2) ** 2, 0))) if r0 else np.nan
    # HOPS: normal incidence with the grating (period 1 um): the +-1 orders travel at sin(theta) = lambda / P
    s = D.run_spec('ws2_polariton_L15', 0.575, 0.655, n_cols=30, eps_max=0.2, n_eps=5)
    lam_h = s['lam']
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, tag in zip(ax[:2], ('empty', 'with WS2')):
        im = a.pcolormesh(HC / lams, thetas, maps[tag], cmap='magma', shading='auto', vmin=0, vmax=1)
        a.set_xlabel('photon energy (eV)'); a.set_ylabel('angle in air (deg)'); a.set_title(f'flat cavity, {tag}: R (TM)')
        a.axvline(Ex, color='c', ls=':', lw=1)
    for r in rows:
        for e in r['E_pol']:
            ax[1].plot(e, r['theta'], 'w.', ms=3)
    fig.colorbar(im, ax=ax[1])
    for i, e in enumerate(s['Eps']):
        ax[2].plot(HC / lam_h, s['R'][i], label=f'eps = {e:.2f}')
    ax[2].set_xlabel('photon energy (eV)'); ax[2].set_ylabel('R (normal incidence, HOPS)'); ax[2].legend(fontsize=7)
    ax[2].set_title('grating P = 1 um: R vs eps (L = 15)'); ax[2].grid(alpha=0.3)
    fig.suptitle(f'WS2 monolayer in a TiO2/SiO2 microcavity: polariton splitting {2e3 * g:.0f} meV at normal incidence '
                 '(white dots: reflection minima)', fontsize=11)
    fig.tight_layout(); fig.savefig(D._fig('db_ws2_polariton.png'), dpi=120); plt.close(fig)
    d0 = D.dips(lam_h, s['R'][0], k=3)
    d2 = D.dips(lam_h, s['R'][-1], k=4)
    chk = check_map(s['res'], s['info'], cols=(0.0, -0.9, 0.9))
    md = ("### WS2 monolayer exciton-polaritons (L = 15)\n"
          f"* exciton {Ex:.3f} eV ({R._LX} um); flat cavity polaritons at normal incidence: "
          + ', '.join(f'{e:.3f} eV' for e in rows[0]['E_pol']) + f"; splitting at the smallest cavity-exciton detuning: **2g = {2e3 * g:.0f} meV** (the monolayer's background index also red-shifts the cavity mode, so the two branches are not symmetric about the exciton).\n"
          f"* HOPS with the 1 um grating: normal-incidence dips at eps = 0: " + ', '.join(f"{HC / x['lam']:.3f} eV (R {x['R']:.2f})" for x in d0)
          + "; at eps = 0.2: " + ', '.join(f"{HC / x['lam']:.3f} eV (R {x['R']:.2f})" for x in d2)
          + f".  True error of the HOPS map {chk['max']:.1e}.\n({time.time() - t0:.0f} s)\n")
    save('ws2', dict(rows=rows, g_eV=g, dips_eps0=d0, dips_eps02=d2, err=chk['max']), md)


# ----------------------------------------------------------------------------------------------------------------- sqib
def study_sqib():
    t0 = time.time()
    lams = np.linspace(0.48, 0.9, 841)
    sps = np.round(np.arange(0.05, 0.121, 0.005), 3)
    M1, M0, rows = [], [], []
    for sp in sps:
        M1.append(tmm_spec(['air', 'Ag', 'PMMA_Z', 'SQIB', 'PMMA_Z', 'Ag'], [0.03, sp, 0.015, sp], lams)[0])
        M0.append(tmm_spec(['air', 'Ag', 'PMMA_Z', 'Ag'], [0.03, 2 * sp + 0.015], lams)[0])
        i1 = sorted(minima(lams, M1[-1], 0.8), key=lambda i: M1[-1][i])[:2]
        i0 = sorted(minima(lams, M0[-1], 0.8), key=lambda i: M0[-1][i])[:1]
        rows.append(dict(spacer=float(sp), E_pol=sorted(float(HC / lams[i]) for i in i1),
                         E_cav=float(HC / lams[i0[0]]) if i0 else np.nan))
    Ex = HC / 0.642
    sp_c = min(rows, key=lambda r: abs(r['E_cav'] - Ex) if np.isfinite(r['E_cav']) else 9)
    split = sp_c['E_pol'][1] - sp_c['E_pol'][0] if len(sp_c['E_pol']) == 2 else np.nan
    s = D.run_spec('sqib_polariton', 0.5, 0.85, n_cols=24, eps_max=0.2, n_eps=3, absorption=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, M, t in ((ax[0], M0, 'empty Ag cavity'), (ax[1], M1, 'with 15 nm SQIB')):
        im = a.pcolormesh(HC / lams, 2 * sps * 1e3 + 15, np.array(M), cmap='magma', shading='auto', vmin=0, vmax=1)
        a.axvline(Ex, color='c', ls=':'); a.set_xlabel('photon energy (eV)'); a.set_ylabel('cavity thickness (nm)'); a.set_title(f'flat: R, {t}')
    fig.colorbar(im, ax=ax[1])
    A = s['A']
    for i, ls in zip((0, 2), ('-', '--')):
        ax[2].plot(HC / s['lam'], s['R'][i], 'k' + ls, label=f"R, eps {s['Eps'][i]:.1f}")
        ax[2].plot(HC / s['lam'], A[i, 3], 'C1' + ls, label=f"A in SQIB, eps {s['Eps'][i]:.1f}")
        ax[2].plot(HC / s['lam'], A[i, 1] + A[i, 5], 'C0' + ls, label=f"A in Ag, eps {s['Eps'][i]:.1f}")
    ax[2].set_xlabel('photon energy (eV)'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    ax[2].set_title('HOPS (period 0.5 um), cavity 175 nm: where the light is absorbed')
    fig.suptitle(f'SQIB in a silver cavity (L = 6): polariton splitting {1e3 * split:.0f} meV at zero detuning', fontsize=11)
    fig.tight_layout(); fig.savefig(D._fig('db_sqib_polariton.png'), dpi=120); plt.close(fig)
    md = ("### SQIB dye in a silver cavity (L = 6): ultrastrong coupling\n"
          f"* exciton {Ex:.3f} eV (0.642 um); at zero detuning (PMMA spacers {1e3 * sp_c['spacer']:.0f} nm) the polaritons are at "
          + ', '.join(f'{e:.3f} eV' for e in sp_c['E_pol']) + f": **splitting {1e3 * split:.0f} meV = {100 * split / Ex:.0f} % of the "
          "exciton energy** (ultrastrong coupling regime > 10 %).\n"
          f"* HOPS at the design (175 nm cavity): absorbed in SQIB (spectral max) {A[0, 3].max():.2f} flat -> {A[-1, 3].max():.2f} at "
          f"eps = 0.2; in Ag {np.max(A[0, 1] + A[0, 5]):.2f} -> {np.max(A[-1, 1] + A[-1, 5]):.2f}.\n({time.time() - t0:.0f} s)\n")
    save('sqib', dict(rows=rows, split_eV=split, spacer=sp_c['spacer']), md)


# ------------------------------------------------------------------------------------------------------------------ gst
def study_gst():
    t0 = time.time()
    states = [('GST_a', 0.0), ('GST_x25', 0.25), ('GST_x50', 0.5), ('GST_x75', 0.75), ('GST_c', 1.0)]
    lams = np.linspace(0.38, 0.8, 211)
    ds = np.arange(0.06, 0.221, 0.02)
    cols = np.zeros((len(ds), len(states), 3))
    for i, d in enumerate(ds):
        for j, (k, _) in enumerate(states):
            cols[i, j] = D.reflected_srgb(lams, tmm_spec(['air', 'ITO', k, 'ITO', 'Pt_W'], [0.02, 0.007, d], lams)[0])
    hops_cols = []
    for k, f in states:
        s = D.run_spec('gst_colour_a', 0.4, 0.8, n_cols=16, eps_max=0.2, n_eps=2, layers=['air', 'ITO', k, 'ITO', 'Pt_W'])
        hops_cols.append([D.reflected_srgb(s['lam'], s['R'][0]), D.reflected_srgb(s['lam'], s['R'][1])])
    # mid IR: GST thickness scan on Au, amorphous vs crystalline
    lir = np.linspace(2.0, 6.0, 401)
    ts = np.arange(0.2, 1.01, 0.1)
    mir = []
    for t in ts:
        Ra = tmm_spec(['air', 'GST_a', 'Au_IR'], [t], lir)[0]
        Rc = tmm_spec(['air', 'GST_c', 'Au_IR'], [t], lir)[0]
        i = int(np.argmin(Rc))
        mir.append(dict(t=float(t), lam=float(lir[i]), R_c=float(Rc[i]), R_a=float(Ra[i]), contrast=float(Ra[i] - Rc[i])))
    sa = D.run_spec('gst_mir_a', 2.0, 5.0, n_cols=20, eps_max=0.2, n_eps=2)
    scc = D.run_spec('gst_mir_c', 2.0, 5.0, n_cols=20, eps_max=0.2, n_eps=2)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    ax[0].imshow(cols, aspect='auto', origin='lower', extent=(-0.5, len(states) - 0.5, 1e3 * ds[0] - 10, 1e3 * ds[-1] + 10))
    ax[0].set_xticks(range(len(states))); ax[0].set_xticklabels([f'{int(100 * f)} %' for _, f in states])
    ax[0].set_xlabel('crystalline fraction of GST'); ax[0].set_ylabel('ITO spacer (nm)')
    ax[0].set_title('reflected colour, flat: air | ITO 20 | GST 7 nm | ITO | Pt')
    hc = np.array(hops_cols)                      # (states, 2, 3)
    ax[1].imshow(np.transpose(hc, (1, 0, 2)), aspect='auto', origin='lower', extent=(-0.5, len(states) - 0.5, -0.1, 0.3))
    ax[1].set_xticks(range(len(states))); ax[1].set_xticklabels([f'{int(100 * f)} %' for _, f in states])
    ax[1].set_yticks([0, 0.2]); ax[1].set_ylabel('eps'); ax[1].set_title('HOPS, ITO 200 nm: flat (eps 0) vs grating eps 0.2')
    ax[2].plot(sa['lam'], sa['R'][0], 'b-', label='amorphous, flat'); ax[2].plot(scc['lam'], scc['R'][0], 'r-', label='crystalline, flat')
    ax[2].plot(sa['lam'], sa['R'][1], 'b--', label='amorphous, eps 0.2'); ax[2].plot(scc['lam'], scc['R'][1], 'r--', label='crystalline, eps 0.2')
    ax[2].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[2].set_ylabel('R'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    ax[2].set_title('GST 0.6 um on Au (L = 3), period 1.5 um')
    fig.tight_layout(); fig.savefig(D._fig('db_gst_switching.png'), dpi=120); plt.close(fig)
    best = max(mir, key=lambda r: r['contrast'])
    md = ("### Ge2Sb2Te5 phase-change stacks (L = 5 colour pixel, L = 3 mid-IR absorber)\n"
          "* colour pixels: the reflected colour changes with crystallisation for every ITO spacer (figures/deep_dives/db_gst_switching.png); "
          "the grating (eps 0.2) adds a small hue shift.\n"
          f"* mid-IR: best switching contrast for GST {1e3 * best['t']:.0f} nm on Au: R {best['R_a']:.2f} (amorphous) -> {best['R_c']:.3f} "
          f"(crystalline) at {best['lam']:.2f} um.  HOPS (0.6 um, P 1.5 um): R_min crystalline flat {scc['R'][0].min():.3f}, "
          f"eps 0.2 {scc['R'][1].min():.3f}; amorphous {sa['R'][0].min():.2f} / {sa['R'][1].min():.2f}.\n"
          "| GST (nm) | lambda (um) | R amorphous | R crystalline |\n|---|---|---|---|\n"
          + '\n'.join(f"| {1e3 * r['t']:.0f} | {r['lam']:.2f} | {r['R_a']:.3f} | {r['R_c']:.3f} |" for r in mir)
          + f"\n\n({time.time() - t0:.0f} s)\n")
    save('gst', dict(mir=mir, colours=cols.tolist(), hops_colours=hc.tolist()), md)


# ------------------------------------------------------------------------------------------------------------------ vo2
def _planck(lam_um, T=5778.0):
    c2 = 14387.77
    return 1.0 / (lam_um ** 5 * (np.exp(c2 / (lam_um * T)) - 1))


def study_vo2():
    t0 = time.time()
    Ts = [20, 30, 40, 50, 55, 60, 70, 80]
    lams = np.linspace(0.4, 2.45, 206)
    ybar = D._cmf(1e3 * lams)[1]
    sol = _planck(lams)
    rows, spec = [], {}
    for T in Ts:
        s = D.run_spec('vo2_window_T20', 0.4, 2.45, n_cols=12, eps_max=0.2, n_eps=2, absorption=True,
                       layers=['air', 'Si3N4', f'VO2_T{T}', 'Si3N4', 'BK7'])
        lam = s['lam']
        out = {}
        for i, e in enumerate(s['Eps']):
            Tt = 1 - s['R'][i] - s['A'][i, :4].sum(axis=0)     # transmitted into the glass
            Ti = np.interp(lams, lam, Tt)
            out[f'eps{e:.1f}'] = dict(T_lum=float(np.trapz(Ti * ybar, lams) / np.trapz(ybar, lams)),
                                      T_sol=float(np.trapz(Ti * sol, lams) / np.trapz(sol, lams)),
                                      T_nir=float(np.mean(Ti[lams > 0.78])), A_VO2=float(np.trapz(np.interp(lams, lam, s['A'][i, 2]) * sol, lams) / np.trapz(sol, lams)))
            spec[(T, i)] = (lam, Tt)
        rows.append(dict(T=T, **out))
        print(f"  vo2 {T} C: " + ', '.join(f"{k}: Tlum {v['T_lum']:.3f} Tsol {v['T_sol']:.3f}" for k, v in out.items()), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cm = plt.get_cmap('coolwarm')
    for k, T in enumerate(Ts):
        lam, Tt = spec[(T, 0)]
        ax[0].plot(lam, Tt, color=cm(k / (len(Ts) - 1)), label=f'{T} C')
    ax[0].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[0].set_ylabel('transmittance into the glass (flat)'); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3); ax[0].set_title('air | Si3N4 40 | VO2 50 nm | Si3N4 40 | BK7')
    for key, mk in (('eps0.0', 'o-'), ('eps0.2', 's--')):
        ax[1].plot(Ts, [r[key]['T_lum'] for r in rows], mk, color='C2', label=f'T_lum ({key})')
        ax[1].plot(Ts, [r[key]['T_sol'] for r in rows], mk, color='C3', label=f'T_sol ({key})')
    ax[1].set_xlabel('temperature (C)'); ax[1].set_ylabel('weighted transmittance'); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(D._fig('db_vo2_thermochromic.png'), dpi=120); plt.close(fig)
    a, b = rows[0]['eps0.0'], rows[-1]['eps0.0']
    a2, b2 = rows[0]['eps0.2'], rows[-1]['eps0.2']
    md = ("### VO2 thermochromic window through the measured temperature series (L = 5)\n"
          f"* flat: T_lum {a['T_lum']:.3f} -> {b['T_lum']:.3f}, T_sol {a['T_sol']:.3f} -> {b['T_sol']:.3f} (**dT_sol = "
          f"{100 * (a['T_sol'] - b['T_sol']):.1f} %**), NIR {a['T_nir']:.3f} -> {b['T_nir']:.3f} from 20 to 80 C.\n"
          f"* grating eps = 0.2: T_lum {a2['T_lum']:.3f} -> {b2['T_lum']:.3f}, dT_sol = {100 * (a2['T_sol'] - b2['T_sol']):.1f} %.\n"
          "| T (C) | T_lum flat | T_sol flat | T_lum eps 0.2 | T_sol eps 0.2 |\n|---|---|---|---|---|\n"
          + '\n'.join(f"| {r['T']} | {r['eps0.0']['T_lum']:.3f} | {r['eps0.0']['T_sol']:.3f} | {r['eps0.2']['T_lum']:.3f} | "
                      f"{r['eps0.2']['T_sol']:.3f} |" for r in rows) + f"\n\n(solar weighting: 5778 K black body; luminous: CIE y-bar)  ({time.time() - t0:.0f} s)\n")
    save('vo2', dict(rows=rows), md)


# ------------------------------------------------------------------------------------------------------------------- lc
def study_lc():
    t0 = time.time()
    out = {}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, c in (('o', 'C0'), ('e', 'C3')):
        s = D.run_spec(f'lc_cavity_{k}', 0.62, 1.5, n_cols=20, eps_max=0.2, n_eps=2)
        out[k] = [D.dips(s['lam'], s['R'][i], k=4) for i in range(2)]
        ax.plot(s['lam'], s['R'][0], color=c, label=f'n_{k}, flat')
        ax.plot(s['lam'], s['R'][1], color=c, ls='--', label=f'n_{k}, eps 0.2')
    ax.set_xlabel(r'$\lambda$ ($\mu$m)'); ax.set_ylabel('R (TE)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Ag | 5CB 1 um | Ag: switching the director (n_o -> n_e) tunes every cavity mode')
    fig.tight_layout(); fig.savefig(D._fig('db_lc_cavity.png'), dpi=120); plt.close(fig)
    pairs = []
    for a in out['o'][0]:
        b = min(out['e'][0], key=lambda b: abs(b['lam'] / a['lam'] - 1.1), default=None)
        if b:
            pairs.append((a['lam'], b['lam']))
    md = ("### Liquid-crystal (5CB) tunable silver cavity, TE (L = 4)\n"
          "* cavity dips with the director along z (n_o): " + ', '.join(f"{x['lam']:.3f} um (R {x['R']:.2f})" for x in out['o'][0])
          + "; along y (n_e): " + ', '.join(f"{x['lam']:.3f} um (R {x['R']:.2f})" for x in out['e'][0]) + ".\n"
          "* grating eps = 0.2 splits the modes and adds grating-coupled guided-mode dips (TE: no surface plasmons): n_o " + ', '.join(f"{x['lam']:.3f}" for x in out['o'][1])
          + "; n_e " + ', '.join(f"{x['lam']:.3f}" for x in out['e'][1]) + f" um.\n({time.time() - t0:.0f} s)\n")
    save('lc', out, md)


# --------------------------------------------------------------------------------------------------------------- thermo
def study_thermo():
    t0 = time.time()
    cases = [('Au 25 C', 'main/Au/nk/Magnozzi-25C.yml'), ('Au 225 C', 'main/Au/nk/Magnozzi-225C.yml'),
             ('Au 350 C', 'main/Au/nk/Magnozzi-350C.yml'), ('Ag 298 K', 'main/Ag/nk/Ferrera-298K.yml'),
             ('Ag 404 K', 'main/Ag/nk/Ferrera-404K.yml'), ('Ag 501 K', 'main/Ag/nk/Ferrera-501K.yml'),
             ('Ag 600 K', 'main/Ag/nk/Ferrera-600K.yml')]
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, key in cases:
        res_ = {}
        for na in (1.333, 1.343):
            res_[na] = D.run_spec('kretschmann_au', 0.55, 0.95, n_cols=24, eps_max=0.1, n_eps=2, layers=['BK7', key, na])
        d0 = D.dips(res_[1.333]['lam'], res_[1.333]['R'][0], k=1)
        d1 = D.dips(res_[1.343]['lam'], res_[1.343]['R'][0], k=1)
        if d0 and d1:
            S = 1e3 * (d1[0]['lam'] - d0[0]['lam']) / 0.01
            rows.append(dict(case=name, lam=d0[0]['lam'], R=d0[0]['R'], fwhm=d0[0]['fwhm'], S=S, FOM=S / (1e3 * d0[0]['fwhm'])))
        ax[0].plot(res_[1.333]['lam'], res_[1.333]['R'][0], label=name)
    ax[0].set_xlabel(r'$\lambda$ ($\mu$m)'); ax[0].set_ylabel('R (Kretschmann, 68 deg, water)'); ax[0].legend(fontsize=7)
    ax[1].bar(range(len(rows)), [r['FOM'] for r in rows], color=['C1'] * 3 + ['C0'] * 4)
    ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels([r['case'] for r in rows], rotation=25, fontsize=8)
    ax[1].set_ylabel('FOM = S / FWHM (1/RIU)')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(D._fig('db_thermoplasmonics.png'), dpi=120); plt.close(fig)
    md = ("### Thermoplasmonics: Kretschmann sensor with hot metals (L = 3, measured temperature pages)\n"
          "| metal | lambda_res (um) | R_min | FWHM (nm) | S (nm/RIU) | FOM |\n|---|---|---|---|---|---|\n"
          + '\n'.join(f"| {r['case']} | {r['lam']:.4f} | {r['R']:.3f} | {1e3 * r['fwhm']:.1f} | {r['S']:.0f} | {r['FOM']:.1f} |" for r in rows)
          + f"\n\n({time.time() - t0:.0f} s)\n")
    save('thermo', dict(rows=rows), md)


# --------------------------------------------------------------------------------------------------------------- largeL
def study_largeL():
    t0 = time.time()
    rows = []
    for P in (5, 10, 15, 20, 25, 30):
        key = f'_poly{P}'
        # thin quarter-wave polymer layers (n ~ 1.5, 0.1 um) are resolved by Nz = 12 (checked against the transfer
        # matrix below); the dense block inverse costs (13 L)^2 Nx complex numbers, ~0.33 GB at L = 62
        R.SCENARIOS[key] = R._s('polymer DBR', **R._polymer_dbr(P), period=0.5, omega_range=R._lam(0.5, 0.75, 0.5),
                                windows='joint', max_delta=0.05, q=(0,), M=12, Nz=12)
        t1 = time.time()
        res, info = R.run(key, N_Eps=20, N_delta=20, workers=D.WORKERS, verbose=False)
        wall = time.time() - t1
        Dd = np.concatenate([np.abs(np.real(r['ee'])).ravel() for r in res])
        chk = check_map(res, info, cols=(0.0, -0.9, 0.9))
        Rf = np.concatenate([np.real(r['ru_flat'][0]) for r in res])
        tm_err = 0.0
        for r in res:
            for j in range(0, len(r['delta']), 5):
                Rt = tmm(list(r['n_layers']), info['thick'], r['omega'][j], 0.0, 'TM')[0]
                tm_err = max(tm_err, abs(Rt - np.real(r['ru_flat'][0, j])))
        rows.append(dict(P=P, L=2 * P + 2, flat_vs_tmm=float(tm_err), size=int(res[0]['size']), windows=len(res), t_solve=float(np.mean([r['t_solve'] for r in res])),
                         wall=wall, D_max=float(np.log10(Dd.max())), D_median=float(np.log10(np.median(Dd))),
                         err_max=chk['max'], err_median=chk['median'], Rflat_max=float(Rf.max())))
        print(f"  largeL P = {P} (L = {2 * P + 2}): matrix {res[0]['size']}, {rows[-1]['t_solve']:.1f} s / window, log10|D| "
              f"{rows[-1]['D_max']:.1f}/{rows[-1]['D_median']:.1f}, true err {chk['max']:.1e}/{chk['median']:.1e}, R_flat max {Rf.max():.3f}, "
              f"flat vs TMM {tm_err:.1e}", flush=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    Ls = [r['L'] for r in rows]
    ax[0].plot(Ls, [r['t_solve'] for r in rows], 'o-'); ax[0].set_ylabel('solve time per window (s)')
    ax[1].plot(Ls, [r['D_max'] for r in rows], 'o-', label='max'); ax[1].plot(Ls, [r['D_median'] for r in rows], 's-', label='median')
    ax[1].set_ylabel('log10 |energy defect|'); ax[1].legend()
    ax[2].semilogy(Ls, [r['err_max'] for r in rows], 'o-', label='max'); ax[2].semilogy(Ls, [r['err_median'] for r in rows], 's-', label='median')
    ax[2].set_ylabel('true error of R'); ax[2].legend()
    for a in ax:
        a.set_xlabel('number of layers L'); a.grid(alpha=0.3)
    fig.suptitle('polystyrene / PMMA Bragg mirrors with 5 ... 25 pairs, HOPS/AWE (N = M = 12, Nz = 24 per layer)', fontsize=11)
    fig.tight_layout(); fig.savefig(D._fig('db_large_L.png'), dpi=120); plt.close(fig)
    md = ("### Very many layers: polymer Bragg mirrors, L = 12 ... 52 (HOPS/AWE, lossless)\n"
          "| pairs | L | matrix | s / window | log10\\|D\\| max / median | true error max / median | flat R vs TMM | max R_flat |\n|---|---|---|---|---|---|---|---|\n"
          + '\n'.join(f"| {r['P']} | {r['L']} | {r['size']} | {r['t_solve']:.1f} | {r['D_max']:.1f} / {r['D_median']:.1f} | "
                      f"{r['err_max']:.1e} / {r['err_median']:.1e} | {r['flat_vs_tmm']:.1e} | {r['Rflat_max']:.3f} |" for r in rows) + f"\n\n({time.time() - t0:.0f} s)\n")
    save('largeL', dict(rows=rows), md)


STUDIES = dict(ws2=study_ws2, sqib=study_sqib, gst=study_gst, vo2=study_vo2, lc=study_lc, thermo=study_thermo,
               largeL=study_largeL)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', nargs='*', default=list(STUDIES))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    D.WORKERS = a.workers
    warnings.filterwarnings('ignore')
    for st in a.study:
        try:
            STUDIES[st]()
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'{st}: FAILED {e!r}', flush=True)
    md = ['# Deep dives with the full refractiveindex.info database (deep_dives_db_nlayer.py)\n']
    for st in STUDIES:
        fn = os.path.join(RES, f'db_{st}.md')
        if os.path.exists(fn):
            md.append(open(fn).read())
    open(os.path.join(HERE, 'results', 'deep_dives_db.md'), 'w').write('\n'.join(md))
