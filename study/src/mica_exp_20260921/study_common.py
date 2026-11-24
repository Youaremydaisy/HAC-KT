"""Shared conventions for the declared follow-up study."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

ROOT = Path(os.environ.get('MICA_ROOT', '/root/autodl-tmp/research'))
OUT = ROOT / 'reports/mica_exp_20260921'
DATA12 = ROOT / 'data/processed/eckt_sequences_full'
DATA17 = ROOT / 'data/processed/mica_assist2017_v1'
STATES = ['Concentration', 'Confusion', 'Frustration', 'Boredom']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def positions(data,split):
    offsets=np.load(data/f'{split}_offsets.npy')
    windows=np.load(data/f'{split}_windows.npy')
    return np.concatenate([np.arange(offsets[i]+s+1,offsets[i]+s+n) for i,s,n in windows])

def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7);y=np.asarray(y)
    return {'count':len(y),'auc':float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
            'ap':float(average_precision_score(y,p)), 'brier':float(np.mean((p-y)**2)),
            'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(np.mean((p>=.5)==y))}

def affect_metrics(y,p):
    mse=np.mean((y-p)**2,axis=0);var=np.var(y,axis=0)
    return {'rmse':float(np.sqrt(mse.mean())),'rmse_by_state':np.sqrt(mse).tolist(),
            'mae_by_state':np.mean(np.abs(y-p),axis=0).tolist(),
            'r2_by_state':[float(1-mse[i]/var[i]) if var[i]>1e-12 else None for i in range(4)]}
