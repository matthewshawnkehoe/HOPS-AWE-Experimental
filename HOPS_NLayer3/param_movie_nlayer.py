"""param_movie_nlayer.py -- movies in which a MATERIAL or GEOMETRY PARAMETER changes (not the wavelength):
the whole spectrum (R and the absorption of every layer, exact dispersion, HOPS with a corrugated interface) and the
field through the stack at a chosen wavelength are recomputed for every frame.

Presets (figures/movies/):
  vo2_heating     thermochromic window Si3N4 / VO2 50 nm / Si3N4 on glass, VO2 heated 20 -> 80 C (measured Oguntoye
                  pages, permittivity interpolated between them); field at 1.5 um
  gst_crystal     ITO / GST 7 nm / ITO / Pt colour pixel while GST crystallises 0 -> 100 % (Lorentz-Lorenz mix of the
                  Frantz amorphous / crystalline pages); the frame shows the reflected colour
  ws2_angle       WS2-monolayer microcavity (L = 15): angle of incidence 0 -> 35 deg (polariton dispersion)
  sqib_eps        SQIB polariton cavity (L = 6): grating depth eps 0 -> 0.2

    python param_movie_nlayer.py --preset vo2_heating gst_crystal
"""
import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                         # noqa: E402
import refl_movie_nlayer as MV                      # noqa: E402
import deep_dives_nlayer as D                       # noqa: E402
from hops_nl import Stack, multilayer_solve         # noqa: E402
from hops_nl.absorption import layer_absorption     # noqa: E402
from hops_nl.materials_ext import register_param, vo2_temperature, gst_mix   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'movies')


def _vo2(T):
    key = f'_VO2_{T:.2f}'
    register_param(key, vo2_temperature(T), (0.21, 2.5), f'VO2 at {T:.1f} C')
    return dict(layers=['air', 'Si3N4', key, 'Si3N4', 'BK7'])


def _gst(f):
    key = f'_GST_{f:.3f}'
    register_param(key, gst_mix(f), (0.35, 29.6), f'GST {100 * f:.0f} % crystalline')
    return dict(layers=['air', 'ITO', key, 'ITO', 'Pt_W'])


PRESETS = {
    'vo2_heating': dict(scenario='vo2_window_T20', values=np.linspace(20, 80, 31), make=_vo2, lam=(0.4, 2.45), lam_field=1.5,
                        eps=0.1, label='VO2 temperature (C)', n_cols=24),
    'gst_crystal': dict(scenario='gst_colour_a', values=np.linspace(0, 1, 26), make=_gst, lam=(0.4, 0.8), lam_field=0.55,
                        eps=0.1, label='GST crystalline fraction', n_cols=30, colour=True),
    'ws2_angle': dict(scenario='ws2_polariton_L15', values=np.linspace(0, 35, 22), make=lambda t: dict(theta=float(t)),
                      lam=(0.575, 0.65), lam_field=0.6105, eps=0.05, label='angle of incidence (deg)', n_cols=40, max_delta=0.05),
    'sqib_eps': dict(scenario='sqib_polariton', values=np.linspace(0, 0.2, 9), make=None, lam=(0.5, 0.85), lam_field=0.642,
                     eps=None, label='grating depth eps', n_cols=40),
}


def frame_data(p, v):
    over = p['make'](v) if p['make'] else {}
    eps = p['eps'] if p['eps'] is not None else float(v)
    # exact dispersion: every column is an independent eps-only solve, so long windows cost nothing in accuracy
    s = D.run_spec(p['scenario'], p['lam'][0], p['lam'][1], n_cols=p['n_cols'], eps_max=max(eps, 1e-6), n_eps=2,
                   absorption=True, max_delta=p.get('max_delta', 0.25), **over)
    info = s['info']
    lam, Rr, A = s['lam'], s['R'][-1], s['A'][-1]
    # field at lam_field: one eps-only HOPS solve with the volume fields
    lf = p['lam_field']
    P = info['period']
    ns = [R._index(k, lf) for k in info['layers']]
    st = Stack(ns, info['thick'], info['f'], info['fx'], info['a'], info['b'], info['Nz'], info['mode'])
    om = P / lf
    th = info.get('theta')
    al = np.real(ns[0]) * om * np.sin(np.radians(th)) if th else 0.0
    res = multilayer_solve(st, om, al, info['N'], 0, keep_volume=True)
    r = dict(vol=res['vol'], n_layers=np.array(ns), omega_bar=om, alpha_bar=al, delta=np.array([0.0]))
    info0 = dict(info, M=0)
    SF = MV.StackField(info0)
    fld = SF.total(r, eps, 0.0)
    Af = layer_absorption(r, info0, eps, 0.0)
    return dict(v=v, lam=lam, R=Rr, A=A, info=info, field=fld, SF=SF, Af=Af, eps=eps, alpha=al)


def make(preset, fps=8, dpi=80):
    p = PRESETS[preset]
    t0 = time.time()
    frames = []
    for v in p['values']:
        frames.append(frame_data(p, v))
    print(f'{preset}: {len(frames)} frames computed in {time.time() - t0:.0f} s', flush=True)
    info = frames[0]['info']
    sc = info['period'] / (2 * np.pi)
    lam = frames[0]['lam']
    Rmap = np.array([np.interp(lam, f['lam'], f['R']) for f in frames])
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.32, wspace=0.28)
    aM, aS, aC = (fig.add_subplot(gs[0, i]) for i in range(3))
    aF = fig.add_subplot(gs[1, :2]); aP = fig.add_subplot(gs[1, 2])
    vals = np.array(p['values'])
    aM.pcolormesh(lam, vals, Rmap, cmap='magma', vmin=0, vmax=1, shading='auto')
    aM.set_xlabel(r'$\lambda$ ($\mu$m)'); aM.set_ylabel(p['label']); aM.set_title('R over the sweep')
    aM.axvline(p['lam_field'], color='c', ls=':', lw=1)
    cur, = aM.plot([], [], 'c-', lw=1.5)
    if p.get('colour'):
        cols = np.array([D.reflected_srgb(f['lam'], f['R']) for f in frames])
        aC.imshow(cols[:, None, :], aspect='auto', origin='lower', extent=(0, 1, vals[0], vals[-1]))
        aC.set_xticks([]); aC.set_ylabel(p['label']); aC.set_title('reflected colour')
        mk, = aC.plot([], [], 'w<', ms=12, transform=aC.get_yaxis_transform())
    else:
        Af = np.array([f['Af'] for f in frames])
        for l in range(Af.shape[1]):
            if Af[:, l].max() > 1e-3:
                nm = info['layers'][l]
                aC.plot(vals, Af[:, l], label=f'A in {nm if isinstance(nm, str) and not nm.startswith("_") else "layer " + str(l)}')
        aC.set_xlabel(p['label']); aC.set_ylabel(f"absorption at {p['lam_field']} um"); aC.legend(fontsize=7); aC.grid(alpha=0.3)
        mk = aC.axvline(vals[0], color='k', lw=1)
    smF = plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap='RdBu_r')
    fig.colorbar(smF, ax=aF, pad=0.01)
    title = fig.suptitle('')
    art = []

    def update(k):
        f = frames[k]
        for a_ in art:
            a_.remove()
        art.clear()
        cur.set_data(lam, np.full_like(lam, vals[k]))
        if p.get('colour'):
            mk.set_data([1.0], [vals[k]])
        else:
            mk.set_xdata([vals[k], vals[k]])
        aS.cla()
        bottom = np.zeros_like(f['lam'])
        cm = plt.get_cmap('tab10')
        for l in range(f['A'].shape[0]):
            if f['A'][l].max() < 1e-3:
                continue
            nm = info['layers'][l]
            nm = nm if isinstance(nm, str) and not nm.startswith('_') else ('active layer' if isinstance(nm, str) else f'{complex(nm).real:.3g}')
            aS.fill_between(f['lam'], bottom, bottom + f['A'][l], color=cm(l % 10), alpha=0.55, lw=0, label=f'A {nm}')
            bottom = bottom + f['A'][l]
        Tt = 1 - f['R'] - bottom
        if Tt.max() > 1e-3:
            aS.fill_between(f['lam'], bottom, bottom + Tt, color='0.8', lw=0, label='T')
        aS.plot(f['lam'], f['R'], 'k-', lw=1.5, label='R')
        aS.axvline(p['lam_field'], color='c', ls=':', lw=1)
        aS.set_ylim(0, 1.02); aS.set_xlabel(r'$\lambda$ ($\mu$m)'); aS.legend(fontsize=6, loc='upper right'); aS.grid(alpha=0.3)
        aS.set_title(f"{p['label']} = {f['v']:.3g}, eps = {f['eps']:.2f}", fontsize=10)
        layers = f['field']
        vmax = np.percentile(np.abs(np.concatenate([np.real(U).ravel() for _, _, U in layers])), 99.5) + 1e-12
        smF.set_clim(-vmax, vmax)
        prof_z, prof_I = [], []
        for X, Z, U in layers:
            for shift in (0.0, 2 * np.pi):
                art.append(aF.pcolormesh((X + shift) * sc, Z * sc, np.real(U * np.exp(1j * f['alpha'] * shift)), cmap='RdBu_r',
                                         vmin=-vmax, vmax=vmax, shading='gouraud'))
            prof_z.append(Z.mean(axis=0)); prof_I.append(np.mean(np.abs(U) ** 2, axis=0))
        SF = f['SF']
        for l in range(1, SF.L):
            zz = SF.h[l - 1] + f['eps'] * SF.f[l - 1]
            art.append(aF.plot(np.concatenate([SF.x, SF.x + 2 * np.pi]) * sc, np.tile(zz, 2) * sc, 'k-', lw=0.7)[0])
        aF.set_xlim(0, 4 * np.pi * sc); aF.set_ylim((SF.h[-1] - info['b']) * sc, (SF.h[0] + info['a']) * sc)
        aF.set_xlabel(r'x ($\mu$m)'); aF.set_ylabel(r'z ($\mu$m)'); aF.set_title(f"Re u_tot at {p['lam_field']} um", fontsize=10)
        zc = np.concatenate(prof_z); Ic = np.concatenate(prof_I); oo = np.argsort(zc)
        art.append(aP.plot(Ic[oo], zc[oo] * sc, 'k-', lw=1.3)[0])
        aP.set_ylim(*aF.get_ylim()); aP.set_xlim(0, max(1.0, 1.05 * Ic.max())); aP.set_title('intensity profile', fontsize=10)
        import re
        stk = re.sub(r'_GST_([\d.]+)', lambda m: f'GST (x = {float(m.group(1)):.2f})', f['info']['stack'])
        stk = re.sub(r'_VO2_([\d.]+)', lambda m: f'VO2 ({float(m.group(1)):.0f} C)', stk)
        title.set_text(f"L = {info['L']}: {stk[:130]}")
        return art

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    os.makedirs(OUT, exist_ok=True)
    if not animation.writers.is_available('ffmpeg'):
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    path = os.path.join(OUT, f'movie_{preset}.mp4')
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2800), dpi=dpi)
    update(len(frames) // 2)
    fig.savefig(path.replace('.mp4', '_frame.png'), dpi=dpi)
    plt.close(fig)
    print(f'saved {path} ({time.time() - t0:.0f} s)', flush=True)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', nargs='*', default=list(PRESETS))
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    D.WORKERS = 2
    for pr in a.preset:
        make(pr)
