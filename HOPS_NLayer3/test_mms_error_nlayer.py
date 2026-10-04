"""test_mms_error_nlayer.py -- validation driver of the n-layer HOPS/AWE code (as test_mms_error.py for 2 layers).

Runs every check of tests/test_nlayer.py and prints a PASS/FAIL table.  Equivalent: python -m pytest tests -q
"""
import time
import traceback

import matplotlib
matplotlib.use('Agg')

from tests import test_nlayer as T
from tests.test_nlayer import *  # noqa: F401,F403,E402  (pytest collects these when it runs this file)

CHECKS = [
    ('N1 L = 2 equals the 2D coupled solver (normal, oblique, metal)', T.test_two_layers_equal_2d_solver),
    ('N2 flat stacks (3, 4, 6 layers, TE/TM, oblique) equal the transfer-matrix R over a delta window',
     T.test_flat_stacks_vs_transfer_matrix),
    ('N3 a fictitious flat interface inside the substrate changes nothing', T.test_fictitious_interface_invariance),
    ('N4 energy conservation R + T = 1, lossless 4-layer stack, 3 different profiles', T.test_energy_conservation_lossless_four_layers),
    ('N5 manufactured solution recovered order by order (L = 3, 4, 6)', T.test_manufactured_solution_all_orders),
    ('N6 test_single_eps_delta: traces and volume fields of every layer (L = 3, 8)', T.test_single_eps_delta_volume_fields),
    ('N7 oblique incidence alpha = 0.15 (L = 5) manufactured solution', T.test_mms_oblique_incidence),
    ('N8 refl_map_nlayer with L = 2 reproduces refl_map.py (silver)', T.test_refl_map_two_layers_equals_refl_map_py),
    ('N9 per-layer absorption: equals TMM layer by layer (flat) and sums to 1 - R - T (corrugated)', T.test_layer_absorption),
    ('N10 exact material dispersion + fixed incidence angle (Kretschmann) vs TMM and pointwise HOPS', T.test_exact_dispersion_fixed_angle),
    ('N11 refractiveindex.info database pages: aliases, GST mixing, L = 15 WS2 cavity vs TMM', T.test_database_materials),
]

if __name__ == '__main__':
    results = []
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            fn()
            status = 'PASS'
        except Exception:                                  # noqa: BLE001
            status = 'FAIL'
            traceback.print_exc()
        results.append((status, name))
        print(f'[{status}] {name}  ({time.time() - t0:.1f} s)', flush=True)
    n_fail = sum(r[0] == 'FAIL' for r in results)
    print(f'\n{len(results) - n_fail}/{len(results)} checks passed' + ('' if n_fail == 0 else f', {n_fail} FAILED'))
