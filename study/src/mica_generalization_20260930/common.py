"""Isolated paths and reproducibility utilities for external refitting."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

ROOT = Path(os.environ.get('MICA_ROOT', str(Path(__file__).resolve().parents[2])))
OUT = Path(os.environ.get('MICA_OUT', str(ROOT / 'reports/mica_generalization_20260930')))
SRC = ROOT / 'src/mica_generalization_20260930'
RAW = ROOT / 'data/raw/generalization_20260930'
DATA_ROOT = Path(os.environ.get('MICA_DATA_ROOT', str(ROOT / 'data/processed/mica_generalization_20260930')))
SEEDS = (42, 43, 44)

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def rank(value, purpose):
    return hashlib.sha256(f'20260930:{purpose}:{value}'.encode()).hexdigest()

def data_path(dataset):
    if dataset not in ('assist09', 'ednet', 'assist17_diagnostic'): raise ValueError(dataset)
    return DATA_ROOT / dataset

def positions(data, split):
    o = np.load(data / f'{split}_offsets.npy')
    w = np.load(data / f'{split}_windows.npy')
    return np.concatenate([np.arange(o[u] + s + 1, o[u] + s + n) for u, s, n in w])

def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7); y = np.asarray(y)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(float(y[bins == k].sum()) - float(p[bins == k].sum())) for k in range(10)) / len(y)
    return {'count': int(len(y)), 'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'ap': float(average_precision_score(y, p)), 'log_loss': float(log_loss(y, p, labels=[0,1])),
            'brier': float(np.mean((p-y)**2)), 'ece10': float(ece)}

def folds(users):
    return (((np.asarray(users, np.int64)*1103515245+12345) % 2147483647) % 10).astype(np.int8)
