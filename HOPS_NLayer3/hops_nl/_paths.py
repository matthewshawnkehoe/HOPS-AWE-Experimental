"""Put ../HOPS_Python (or $HOPS_PYTHON) on sys.path: the n-layer code reuses the 2D package `hops`."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOPS_PYTHON = os.environ.get('HOPS_PYTHON', os.path.join(_ROOT, 'HOPS_Python'))
if HOPS_PYTHON not in sys.path:
    sys.path.insert(0, HOPS_PYTHON)
