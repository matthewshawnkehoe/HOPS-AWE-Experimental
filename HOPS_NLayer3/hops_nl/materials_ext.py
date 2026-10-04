"""Extra material models for the n-layer studies, registered into hops.materials (keys usable in every scenario).

The curated refractiveindex.info pages bundled with HOPS_Python are used as they are; the models below add
materials whose physics the multilayer studies need but whose pages are not bundled:

  VO2_fXX   VO2 in the middle of its insulator-metal transition: Bruggeman effective medium of metallic
            (VO2_hot, Beaini 100 C) inclusions with filling fraction XX % in insulating VO2 (VO2_cold, 25 C)
            -- the "natural disordered metamaterial" of Kats et al., PRL 110, 2013.  Keys VO2_f10 ... VO2_f90.
  hBN_ip    hexagonal boron nitride, IN-PLANE (ordinary) permittivity, Lorentz oscillator of the upper
            Reststrahlen band (Caldwell et al., Nat. Commun. 5, 5221 (2014)): eps_inf = 4.87, w_TO = 1370,
            w_LO = 1610, gamma = 5 cm^-1.  hBN is uniaxial with its c axis along z, so for TE polarisation
            (E along y) the scalar model with the in-plane permittivity is exact.
"""
import os

import numpy as np

from . import _paths  # noqa: F401
from hops import materials as mat

# full refractiveindex.info database (python fetch_rii_database.py, or git clone ... and set RII_DB):
# looked for in $RII_DB, ../rii and HOPS_NLayer/rii_database (the pages used by the scenarios are shipped there)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get('RII_DB'):
    for _d in (os.path.join(_HERE, '..', 'rii'), os.path.join(_HERE, 'rii_database')):
        if os.path.exists(os.path.join(_d, 'data')):
            os.environ['RII_DB'] = os.path.abspath(_d)
            break

# short keys for refractiveindex.info pages used by the database-wide scenarios (path relative to database/data)
EXTRA = {
    'GST_a': ('main/Ge2Sb2Te5/nk/Frantz-amorphous.yml', 'phase change', 'Ge2Sb2Te5 amorphous, Frantz 2021 (0.35-30 um)'),
    'GST_c': ('main/Ge2Sb2Te5/nk/Frantz-crystal.yml', 'phase change', 'Ge2Sb2Te5 crystalline, Frantz 2021 (0.35-30 um)'),
    'WS2_1L': ('main/WS2/nk/Hsu-1L.yml', 'excitonic 2D', 'WS2 monolayer (0.618 nm), Hsu 2019: A exciton ~ 0.615 um'),
    'WS2_o': ('main/WS2/nk/Munkhbat-o.yml', 'van der Waals', 'bulk WS2 in-plane (ordinary), Munkhbat 2022 (0.3-1.69 um)'),
    'hBN_o': ('main/BN/nk/Grudinin-o.yml', 'van der Waals', 'hBN in-plane (ordinary), Grudinin 2023 (0.25-1.7 um)'),
    'SQIB': ('organic/C32H44N2O6 - SQIB/nk/Funke-alpha.yml', 'excitonic organic',
             'squaraine SQIB film, Funke 2016: strong exciton at 0.65 um, Re eps < 0'),
    'LC5CB_o': ('other/liquid crystals/5CB/nk/Tkachenko-o.yml', 'liquid crystal', '5CB ordinary index, Tkachenko'),
    'LC5CB_e': ('other/liquid crystals/5CB/nk/Tkachenko-e.yml', 'liquid crystal', '5CB extraordinary index, Tkachenko'),
    'PS': ('organic/(C8H8)n - polystyrene/nk/Zhang.yml', 'polymer', 'polystyrene, Zhang 2020 (0.4-20 um)'),
    'PMMA_Z': ('organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Zhang-Tomson.yml', 'polymer', 'PMMA, Zhang 2020 (0.4-20 um)'),
    'Au_25C': ('main/Au/nk/Magnozzi-25C.yml', 'plasmonic metal', 'gold at 25 C, Magnozzi 2019'),
    'Au_350C': ('main/Au/nk/Magnozzi-350C.yml', 'plasmonic metal', 'gold at 350 C, Magnozzi 2019'),
    'Ag_298K': ('main/Ag/nk/Ferrera-298K.yml', 'plasmonic metal', 'silver at 298 K, Ferrera 2019'),
    'Ag_600K': ('main/Ag/nk/Ferrera-600K.yml', 'plasmonic metal', 'silver at 600 K, Ferrera 2019'),
    'Na_liq': ('main/Na/nk/Inagaki-liquid.yml', 'plasmonic metal', 'liquid sodium, Inagaki 1976'),
    'Pt_W': ('main/Pt/nk/Werner.yml', 'lossy metal', 'platinum, Werner 2009'),
    'CsPbBr3': ('other/perovskite/CsPbBr3/nk/Brennan.yml', 'semiconductor', 'CsPbBr3 perovskite, Brennan 2020'),
    'AlN_IR': ('main/AlN/nk/Kischkat.yml', 'phonon polariton', 'AlN film, Kischkat 2012 (1.5-14.3 um)'),
    'SiO2_K': ('main/SiO2/nk/Kischkat.yml', 'phonon polariton', 'SiO2 film, Kischkat 2012 (1.5-14.3 um)'),
}
for _t in (20, 30, 40, 50, 55, 60, 70, 80):
    EXTRA[f'VO2_T{_t}'] = (f'main/VO2/nk/Oguntoye-{_t}C.yml', 'phase change', f'VO2 film at {_t} C, Oguntoye 2023 (0.21-2.5 um)')


class _Fn:
    def __init__(self, fn, rng, ref):
        self.fn, self.range, self.references, self.comments, self.path = fn, rng, ref, '', ref

    def __call__(self, lam_um, warn=True):
        return self.fn(np.asarray(lam_um, float))


def bruggeman(eps_m, eps_d, f):
    """3D Bruggeman effective permittivity of spherical inclusions eps_m (fraction f) in eps_d"""
    b = (3 * f - 1) * eps_m + (2 - 3 * f) * eps_d
    r = np.sqrt(b ** 2 + 8 * eps_m * eps_d + 0j)
    e1, e2 = (b + r) / 4, (b - r) / 4
    return np.where(np.imag(e1) >= np.imag(e2), e1, e2)


def _vo2_mix(f):
    def fn(L):
        ec = mat.refractive_index('VO2_cold', L, warn=False) ** 2
        eh = mat.refractive_index('VO2_hot', L, warn=False) ** 2
        n = np.sqrt(bruggeman(eh, ec, f))
        return np.where(np.imag(n) < 0, -n, n)
    return fn


def _hbn_ip(L):
    w = 1e4 / L
    eps_inf, wTO, wLO, g = 4.87, 1370.0, 1610.0, 5.0
    e = eps_inf * (1 + (wLO ** 2 - wTO ** 2) / (wTO ** 2 - w ** 2 - 1j * g * w))
    n = np.sqrt(e)
    return np.where(np.imag(n) < 0, -n, n)


def gst_mix(f):
    """Ge2Sb2Te5 partially crystallised (fraction f): Lorentz-Lorenz effective medium of the amorphous and
    crystalline phases (the standard model of GST crystallisation, e.g. Chu et al. 2016)"""
    def fn(L):
        ea = mat.refractive_index('GST_a', L, warn=False) ** 2
        ec = mat.refractive_index('GST_c', L, warn=False) ** 2
        g = f * (ec - 1) / (ec + 2) + (1 - f) * (ea - 1) / (ea + 2)
        n = np.sqrt((1 + 2 * g) / (1 - g))
        return np.where(np.imag(n) < 0, -n, n)
    return fn


def vo2_temperature(T):
    """VO2 film at temperature T (20 ... 80 C): permittivity interpolated linearly between the measured Oguntoye pages"""
    Ts = [20, 30, 40, 50, 55, 60, 70, 80]
    T = float(np.clip(T, 20, 80))
    k = max(0, min(len(Ts) - 2, int(np.searchsorted(Ts, T)) - 1))
    a, b = Ts[k], Ts[k + 1]
    w = (T - a) / (b - a)

    def fn(L):
        e = (1 - w) * mat.refractive_index(f'VO2_T{a}', L, warn=False) ** 2 + w * mat.refractive_index(f'VO2_T{b}', L, warn=False) ** 2
        n = np.sqrt(e)
        return np.where(np.imag(n) < 0, -n, n)
    return fn


def register_param(key, fn, rng, note, period=1.0):
    mat.MODELS[key] = _Fn(fn, rng, note)
    mat.MATERIALS[key] = dict(path=None, category='model', period=period, note=note)
    mat.load.cache_clear()


def register():
    for k, (path, cat, note) in EXTRA.items():
        mat.MATERIALS[k] = dict(path=path, category=cat, period=1.0, note=note)
    for p in (25, 50, 75):
        register_param(f'GST_x{p}', gst_mix(p / 100), (0.35, 29.6), f'GST {p} % crystallised (Lorentz-Lorenz)')
    for p in range(10, 100, 10):
        key = f'VO2_f{p}'
        mat.MODELS[key] = _Fn(_vo2_mix(p / 100), (0.5, 25.0), f'Bruggeman mix of VO2 100 C ({p} %) in VO2 25 C')
        mat.MATERIALS[key] = dict(path=None, category='phase change', period=10.0,
                                  note=f'VO2 during the transition: {p} % metallic domains (Bruggeman EMA)')
    mat.MODELS['hBN_ip'] = _Fn(_hbn_ip, (5.0, 9.0), 'Lorentz model, Caldwell et al. 2014 (in-plane)')
    mat.MATERIALS['hBN_ip'] = dict(path=None, category='phonon polariton', period=6.0,
                                   note='hBN in-plane permittivity (upper Reststrahlen band 6.2-7.3 um); exact for TE')
    mat.load.cache_clear()


register()
