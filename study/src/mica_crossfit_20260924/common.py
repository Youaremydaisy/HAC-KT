import os
import sys
from pathlib import Path

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '2')

ROOT = Path(os.environ.get('MICA_ROOT', '/root/autodl-tmp/research'))
sys.path.insert(0, str(ROOT / 'src/mica_transparent_20260923'))
from core import OUT as FROZEN, DATA, data, positions, cognitive, write, sha, validation_metrics

DEST = ROOT / 'reports/mica_crossfit_20260924'
