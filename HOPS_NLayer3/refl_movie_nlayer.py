"""refl_movie_nlayer.py -- movies of multi-layer gratings: reflectivity map, energy defect / absorptance, spectrum and
the field THROUGH THE WHOLE STACK (the n-layer analogue of refl_movie.py).

The HOPS/AWE coefficients (including the volume fields u_{l, n, m} of every layer) are computed once per frequency
window; every frame is only a Pade summation at the current (eps, omega).  Each frame shows
  top row     R(lambda, eps) with a cursor | log10|D| (lossless) or absorptance A | spectrum R, T, A with marker
  bottom      Re u_tot in the (x, z) plane over two periods, every layer and interface drawn (incident + reflected
              above, transmitted below, Rayleigh-extended beyond the artificial boundaries) | intensity profile
              <|u|^2>_x (z): where the light sits in the stack (guided modes, gap plasmons, Bragg decay)

Presets (figures/movies/):
  gmr            TiO2 waveguide grating on silica (L=3): guided-mode resonance -- R jumps to ~1 and the field is
                 trapped in the 120 nm film
  silver_film    thin silver film in vacuum (L=3, IMI): coupled surface plasmons on both faces of the film
  mim            Ag / SiO2 / Ag gap-plasmon absorber (L=4): absorption peak with the field in the 60 nm gap
  bragg          quarter-wave Bragg mirror (H L)x3 on glass (L=8): the stop band (R -> 1, field decays into the
                 mirror) and the grating anomalies at its edges
  hmm            hyperbolic metal/dielectric stack (L=8)
  film_eps       dielectric film (L=3): eps grows 0 -> 0.2 at a fixed wavelength
    python refl_movie_nlayer.py --preset bragg
"""
import argparse
import os
import sys
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refl_map_nlayer as R                    # noqa: E402
from hops_nl.summation_ext import sum_series    # noqa: E402
from hops import cheb                          # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'movies')
PRESETS = {
    'gmr': dict(scenario='gmr_filter', q=(0,), eps=0.05, lam_range=None, sweep='lambda'),
    'silver_film': dict(scenario='silver_film', q=(1, 2), eps=0.12, lam_range=None, sweep='lambda'),
    'mim': dict(scenario='mim_absorber', q=(0, 1), eps=0.12, lam_range=None, sweep='lambda'),
    'bragg': dict(scenario='bragg_L8', q=(1,), eps=0.15, lam_range=None, sweep='lambda'),
    'hmm': dict(scenario='hmm_L8', q=(1, 2), eps=0.12, lam_range=None, sweep='lambda'),
    'film_eps': dict(scenario='film_dielectric', q=(1,), eps=None, lam=4.5, lam_range=None, sweep='eps'),
    # ---- materials gallery (exact material dispersion; per-layer absorption in the spectrum panel)
    'tamm': dict(scenario='tamm_ag', eps=0.1, lam_range=None, sweep='lambda', n_grid=24, workers=2,
                 over=dict(omega_range=(0.65 / 0.82, 0.65 / 0.58))),
    'lrspp': dict(scenario='lrspp_ag20', eps=0.05, lam_range=None, sweep='lambda', n_grid=30, workers=2,
                  over=dict(omega_range=(0.5 / 0.95, 0.5 / 0.66))),
    'kretschmann': dict(scenario='kretschmann_au', eps=0.1, lam_range=None, sweep='lambda', n_grid=30, workers=2),
    'ge_au': dict(scenario='ge_on_au', eps=0.1, lam_range=None, sweep='lambda', n_grid=30, workers=2),
    'vo2': dict(scenario='vo2_f30_sapphire', eps=0.15, lam_range=None, sweep='lambda', n_grid=30, workers=2),
    'perovskite': dict(scenario='perovskite_cell', eps=0.15, lam_range=None, sweep='lambda', n_grid=24, workers=2),
    'cavity': dict(scenario='microcavity_L9', eps=0.1, lam_range=None, sweep='lambda', n_grid=24, workers=2,
                   over=dict(omega_range=(0.65 / 0.78, 0.65 / 0.62))),
    'berreman': dict(scenario='berreman_ito', eps=0.1, lam_range=None, sweep='lambda', n_grid=30, workers=2),
    # ---- database-wide stacks
    'ws2_polariton': dict(scenario='ws2_polariton_L15', eps=0.15, lam_range=None, sweep='lambda', n_grid=24, workers=2,
                          over=dict(omega_range=(1.0 / 0.645, 1.0 / 0.58))),
    'sqib': dict(scenario='sqib_polariton', eps=0.15, lam_range=None, sweep='lambda', n_grid=24, workers=2),
    'gst_mir': dict(scenario='gst_mir_c', eps=0.15, lam_range=None, sweep='lambda', n_grid=24, workers=2),
    'vdw_mirror': dict(scenario='vdw_mirror_TE', eps=0.1, lam_range=None, sweep='lambda', n_grid=24, workers=2),
    'polymer_dbr': dict(scenario='polymer_dbr_L22', eps=0.15, lam_range=None, sweep='lambda', n_grid=24, workers=2),
}


def _cheb_interp(Nz, nfine):
    t = np.cos(np.pi * np.arange(Nz + 1) / Nz)
    tf = np.linspace(1, -1, nfine)
    V = np.polynomial.chebyshev.chebvander(t, Nz)
    return np.polynomial.chebyshev.chebvander(tf, Nz) @ np.linalg.inv(V), tf


def _up(u, nf):
    """periodic spectral upsampling along axis 0"""
    Nx = u.shape[0]
    U = np.fft.fft(u, axis=0)
    Uf = np.zeros((nf,) + U.shape[1:], complex)
    h = Nx // 2
    Uf[:h] = U[:h]
    Uf[-h:] = U[-h:]
    return np.fft.ifft(Uf, axis=0) * (nf / Nx)


class StackField:
    """total field of a multilayer from the HOPS/AWE volume series (Pade-summed at (eps, delta))"""

    def __init__(self, info, nx=128, nz=40, ext=1.0):
        self.info, self.nx, self.nz = info, nx, nz
        self.L, self.thick = info['L'], info['thick']
        h = [0.0]
        for t in self.thick:
            h.append(h[-1] - t)
        self.h = h
        Nz = info['Nz']
        self.C, tf = _cheb_interp(Nz, nz)
        self.s = (tf + 1) / 2                                      # 1 (top) ... 0 (bottom)
        self.x = 2 * np.pi * np.arange(nx) / nx
        self.f = [_up(np.asarray(f, complex)[:, None], nx)[:, 0].real for f in info['f']]
        self.ext = ext

    def bounds(self, l):
        a, b, L = self.info['a'], self.info['b'], self.L
        if l == 0:
            return self.h[0], self.h[0] + a, self.f[0], 0 * self.x
        if l == L - 1:
            return self.h[-1] - b, self.h[-1], 0 * self.x, self.f[-1]
        return self.h[l], self.h[l - 1], self.f[l], self.f[l - 1]

    def total(self, r, eps, delta, kind=2):
        """list over layers of (X, Z, U) arrays (nx, nz) of the total Bloch field e^{i alpha x} u"""
        from hops_nl.absorption import column_view
        r, delta, M = column_view(r, self.info, delta)
        N = self.info['N']
        s1 = 1 + delta
        alpha = r.get('alpha_bar', self.info['alpha']) * s1
        k0 = np.real(r['n_layers'][0]) * r['omega_bar'] * s1
        g0 = np.sqrt(complex(k0 ** 2 - alpha ** 2))
        out = []
        for l in range(self.L):
            v = r['vol'][l]                                                    # (Nx, Nz+1, M+1, N+1)
            Nx, Nz1 = v.shape[:2]
            c = np.transpose(v.reshape(Nx * Nz1, M + 1, N + 1), (0, 2, 1))       # (P, N+1, M+1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                u = sum_series(kind, c, np.array([[eps]]), np.array([[delta]]), N, M).reshape(Nx, Nz1)
            u = _up(u, self.nx) @ self.C.T                                     # (nx, nz)
            zb, zt, fb, ft = self.bounds(l)
            Z = (zb + eps * fb)[:, None] + self.s[None, :] * ((zt - zb) + eps * (ft - fb))[:, None]
            if l == 0:
                u = u + np.exp(-1j * g0 * Z)                                   # + incident wave
            U = np.exp(1j * alpha * self.x)[:, None] * u
            X = np.broadcast_to(self.x[:, None], Z.shape)
            # close the period (no seam between the two copies)
            X = np.vstack([X, X[:1] + 2 * np.pi]); Z = np.vstack([Z, Z[:1]])
            U = np.vstack([U, U[:1] * np.exp(2j * np.pi * alpha)])
            out.append((X, Z, U))
        return out


def make_movie(preset, frames=120, fps=15, n_grid=40, dpi=80, out=None):
    p = dict(PRESETS[preset])
    t0 = time.time()
    n_grid = p.get('n_grid', n_grid)
    res, info = R.run(p['scenario'], qq=p.get('q'), N_Eps=n_grid, N_delta=n_grid, workers=p.get('workers', 1),
                      verbose=False, keep_fields=True, **p.get('over', {}))
    res = sorted(res, key=lambda r: r['omega_bar'])
    print(f"{preset}: {len(res)} windows in {time.time() - t0:.0f} s", flush=True)
    period = info['period']
    sc = period / (2 * np.pi) if period else 1.0
    unit = r' ($\mu$m)' if period else ''
    lam_all = np.concatenate([r['lam'] for r in res]) * sc
    lo, hi = (p['lam_range'] or (lam_all.min(), lam_all.max()))
    if p['sweep'] == 'lambda':
        lams, epss = np.linspace(hi, lo, frames), np.full(frames, p['eps'])
    else:
        lams, epss = np.full(frames, p['lam']), np.linspace(0, info['eps_max'], frames)
    SF = StackField(info)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.28)
    aR, aD, aS = (fig.add_subplot(gs[0, i]) for i in range(3))
    aF = fig.add_subplot(gs[1, :2])
    aP = fig.add_subplot(gs[1, 2])
    lossless = info['lossless']
    amax = max(1e-3, np.percentile(np.concatenate([np.real(r['ee']).ravel() for r in res]), 99.9))
    for r in res:
        L_ = r['lam'] * sc
        aR.pcolormesh(L_, r['Eps'], np.clip(np.real(r['ru']), 0, 1), cmap='hot', vmin=0, vmax=1, shading='gouraud')
        if lossless:
            aD.pcolormesh(L_, r['Eps'], np.log10(np.abs(np.real(r['ee'])) + 1e-17), cmap='hot', vmin=-16, vmax=0, shading='gouraud')
        else:
            aD.pcolormesh(L_, r['Eps'], np.clip(np.real(r['ee']), 0, 1), cmap='magma', vmin=0, vmax=amax, shading='gouraud')
    for ax, t in ((aR, 'Reflectivity R'), (aD, r'Energy defect $\log_{10}|D|$' if lossless else 'Absorptance A = 1 - R - T')):
        ax.set_xlim(lo, hi); ax.set_ylim(0, info['eps_max']); ax.set_xlabel(r'$\lambda$' + unit); ax.set_ylabel(r'$\varepsilon$')
        ax.set_title(t, fontsize=10)
    fig.colorbar(aR.collections[0], ax=aR, pad=0.01); fig.colorbar(aD.collections[0], ax=aD, pad=0.01)
    cur = [ax.plot([], [], 'o', color='c', mec='k', ms=6)[0] for ax in (aR, aD)]
    # spectrum (or eps-scan) along the sweep: Pade values at the frame points
    def locate(lam_):
        om = 2 * np.pi / (lam_ / sc)
        best = min(res, key=lambda r: 0 if r['omega'].min() <= om <= r['omega'].max() else
                   min(abs(om - r['omega'].min()), abs(om - r['omega'].max())))
        return best, float(np.clip(om / best['omega_bar'] - 1, best['delta'].min(), best['delta'].max()))
    xs = lams if p['sweep'] == 'lambda' else epss
    RT = np.zeros((frames, 3))
    for k in range(frames):
        r, dl = locate(lams[k])
        i = int(np.argmin(np.abs(r['Eps'] - epss[k]))); j = int(np.argmin(np.abs(r['delta'] - dl)))
        RT[k] = np.real(r['ru'][i, j]), np.real(r['rl'][i, j]), np.real(r['ee'][i, j])
    o = np.argsort(xs)
    if not lossless:
        from hops_nl.absorption import layer_absorption
        AL = np.array([layer_absorption(*((lambda rr, dd: (rr, info, float(epss[k]), dd))(*locate(lams[k]))))
                       for k in range(frames)])
        bottom = np.zeros(frames)
        cm_ = plt.get_cmap('tab10')
        for l in range(info['L']):
            if np.max(AL[:, l]) < 1e-3:
                continue
            nm = info['layers'][l]
            nm = nm if isinstance(nm, str) else f'{complex(nm).real:.3g}'
            aS.fill_between(xs[o], bottom[o], (bottom + AL[:, l])[o], color=cm_(l % 10), alpha=0.5, lw=0,
                            label=f'A in {nm}' + (' (+ transmitted)' if l == info['L'] - 1 else ''))
            bottom = bottom + AL[:, l]
    aS.plot(xs[o], RT[o, 0], 'k-', lw=1.6, label='R')
    if np.max(np.abs(RT[:, 1])) > 1e-6:
        aS.plot(xs[o], RT[o, 1], 'b-', lw=1.2, label='T')
    aS.set_ylim(-0.02, 1.02); aS.grid(alpha=0.3); aS.legend(fontsize=7)
    aS.set_xlabel(r'$\lambda$' + unit if p['sweep'] == 'lambda' else r'$\varepsilon$')
    aS.set_title(f"along the sweep ({'eps = %g' % p['eps'] if p['sweep'] == 'lambda' else 'lambda = %g' % p['lam']})", fontsize=10)
    mark, = aS.plot([], [], 'o', color='c', mec='k', ms=8)
    title = fig.suptitle('', fontsize=11)
    art = []
    smF = plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap='RdBu_r')
    fig.colorbar(smF, ax=aF, pad=0.01)

    def update(k):
        for a_ in art:
            a_.remove()
        art.clear()
        r, dl = locate(lams[k])
        e = float(epss[k])
        layers = SF.total(r, e, dl)
        a_eff = (r['alpha_cols'][int(np.argmin(np.abs(r['delta'] - dl)))] if 'alpha_cols' in r
                 else r.get('alpha_bar', info['alpha']) * (1 + dl))
        vmax = np.percentile(np.abs(np.concatenate([np.real(U).ravel() for _, _, U in layers])), 99.5) + 1e-12
        smF.set_clim(-vmax, vmax)
        prof_z, prof_I = [], []
        for l, (X, Z, U) in enumerate(layers):
            for shift in (0.0, 2 * np.pi):
                art.append(aF.pcolormesh((X + shift) * sc, Z * sc, np.real(U * np.exp(1j * a_eff * shift)),
                                         cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud'))
            prof_z.append(Z.mean(axis=0)); prof_I.append(np.mean(np.abs(U) ** 2, axis=0))
        for l in range(1, SF.L):
            zz = SF.h[l - 1] + e * SF.f[l - 1]
            xx2 = np.concatenate([SF.x, SF.x + 2 * np.pi])
            art.append(aF.plot(xx2 * sc, np.tile(zz, 2) * sc, 'k-', lw=0.8)[0])
        aF.set_xlim(0, 4 * np.pi * sc)
        aF.set_ylim((SF.h[-1] - info['b']) * sc, (SF.h[0] + info['a']) * sc)
        aF.set_xlabel('x' + unit); aF.set_ylabel('z' + unit); aF.set_title(r'Re $u_{tot}$ (two periods)', fontsize=10)
        zc = np.concatenate(prof_z); Ic = np.concatenate(prof_I); oo = np.argsort(zc)
        art.append(aP.plot(Ic[oo], zc[oo] * sc, 'k-', lw=1.4)[0])
        for l in range(1, SF.L):
            art.append(aP.axhline(SF.h[l - 1] * sc, color='0.6', lw=0.6))
        aP.set_ylim(*aF.get_ylim()); aP.set_xlabel(r'$\langle |u|^2 \rangle_x$'); aP.set_title('intensity profile', fontsize=10)
        aP.set_xlim(0, max(1.0, 1.05 * Ic.max()))
        for c_ in cur:
            c_.set_data([lams[k]], [e])
        mark.set_data([xs[k]], [RT[k, 0]])
        title.set_text(f"L = {info['L']}: {info['stack'][:120]}\n"
                       rf"$\lambda$ = {lams[k]:.4g}{unit},  $\varepsilon$ = {e:.3f}:  R = {RT[k, 0]:.4f}" +
                       ('' if lossless else f",  A = {RT[k, 2]:.3f}"))
        return art

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(OUTDIR, exist_ok=True)
    if not animation.writers.is_available('ffmpeg'):
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    path = os.path.join(OUTDIR, (out or f'movie_{preset}') + '.mp4')
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2800), dpi=dpi)
    update(frames // 2)
    fig.savefig(path.replace('.mp4', '_frame.png'), dpi=dpi)
    plt.close(fig)
    print(f'saved {path} ({time.time() - t0:.0f} s)', flush=True)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', nargs='*', default=list(PRESETS))
    ap.add_argument('--n-grid', type=int, default=40)
    ap.add_argument('--frames', type=int, default=120)
    a = ap.parse_args()
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    for pr in a.preset:
        make_movie(pr, frames=a.frames)
