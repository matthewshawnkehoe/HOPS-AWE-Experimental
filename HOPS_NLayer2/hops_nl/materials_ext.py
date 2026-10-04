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
import numpy as np

from . import _paths  # noqa: F401
from hops import materials as mat


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


def register():
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
