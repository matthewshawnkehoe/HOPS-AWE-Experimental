"""HOPS/AWE for 2D gratings with L >= 2 layers (L-1 independently corrugated interfaces).

Extends the two-layer solver of HOPS_Python/hops (Kehoe & Nicholls, J. Sci. Comput. 2024) to stacks of
any number of layers; reuses hops (Chebyshev, T_dno, incidence data, summation, energy, materials).
"""
from . import _paths  # noqa: F401
from .solver import Stack, multilayer_solve, energies

__version__ = '1.0.0'
from . import materials_ext  # noqa: F401,E402  (registers VO2_fXX, hBN_ip in hops.materials)
