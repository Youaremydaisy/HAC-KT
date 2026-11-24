"""Domain-specific, causal CF features using the frozen recurrence equations."""
import os
import sys
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.special import expit
from common import ROOT, OUT, data_path, positions, write, sha
from prepare import prefix_features

os.environ.setdefault('MICA_ROOT',str(ROOT))
sys.path.insert(0,str(ROOT/'src/mica_transparent_20260923'))
sys.path.insert(0,str(ROOT/'reports/mica_transparent_20260923/dependencies'))
import core
import uncertainty_search as uncertainty
from gam_search import ExplainableBoostingClassifier, export_tables, predict_tables, table_terms

core.OUT=OUT/'compiled'; core.OUT.mkdir(parents=True,exist_ok=True)
uncertainty.DEST=core.OUT

def recurrences(o,s,q,y,b,meta):
    m={'num_skills_with_padding':meta['num_state_skills'],'num_problems_with_padding':meta['num_state_items']}
    trace=core.trace_features(o,s,q,y,b,m)
    belief=uncertainty.calculate(uncertainty.library(),o,s,q,y,b,meta['num_state_skills'],meta['num_state_items'])
    return trace,belief

def history_signals(raw,o):
    parts=[]
    for decay in (None,.5,.9,.98):
        result=np.zeros_like(raw,dtype='float32')
        for a,b in zip(o[:-1],o[1:]):
            if b-a>1:
                result[a+1:b]=raw[a:b-1] if decay is None else lfilter([1-decay],[1,-decay],raw[a:b-1],axis=0)
        parts.append(result)
    return np.column_stack(parts)

def compute(dataset,split,window=False,neutral=False):
    data=data_path(dataset);meta=json.loads((data/'metadata.json').read_text())
    a={k:np.load(data/f'{split}_{k}.npy') for k in ['offsets','state_skills','state_items','correct','behavior_features',
        'measurement_missing','crossfit_difficulty','user_ids','skill_missing','skill_unseen','item_unseen']}
    o=a['offsets']; target=positions(data,split)
    if window:
        windows=np.load(data/f'{split}_windows.npy')
        selected=np.concatenate([np.arange(o[u]+s,o[u]+s+n) for u,s,n in windows])
        lengths=windows[:,2];o=np.r_[0,np.cumsum(lengths)].astype('int64')
        a={k:v[selected] for k,v in a.items() if k!='offsets'}
        target=np.concatenate([np.arange(x+1,y) for x,y in zip(o[:-1],o[1:])])
        # Discard global-history behavior profiles when testing the finite-window control.
        raw=a['behavior_features'];stats=json.loads((OUT/dataset/'data_audit.json').read_text())
        tr=np.load(data/'train_behavior_features.npy',mmap_mode='r');rtmean=float(tr[:,0].mean());rtscale=max(float(tr[:,0].std()),.25)
        dtmean=float(tr[:,3].mean());dtscale=max(float(tr[:,3].std()),.5)
        for x,z in zip(o[:-1],o[1:]):
            r=raw[x:z,0];dt=raw[x:z,3];y=a['correct'][x:z];n=np.arange(z-x)
            prior=(np.r_[0,np.cumsum(y)[:-1]]+10)/(n+20)
            raw[x:z,4]=prior;raw[x:z,5]=np.abs(y-prior)
            raw[x:z,6]=expit((r-(np.r_[0,np.cumsum(r)[:-1]]+20*rtmean)/(n+20))/rtscale)
            raw[x:z,7]=expit((dt-(np.r_[0,np.cumsum(dt)[:-1]]+20*dtmean)/(n+20))/dtscale)
        raw[a['measurement_missing'][:,0].astype(bool),6]=.5
        raw[a['measurement_missing'][:,3].astype(bool),7]=.5
    y=a['correct'];s=a['state_skills'];q=a['state_items'];b=a['crossfit_difficulty']
    trace,belief=recurrences(o,s,q,y,np.zeros_like(b) if neutral else b,meta)
    if neutral:
        # Preserve current-item difficulty and direct contexts; ablate only state updates.
        trace[:,21]=b;trace[:,22]=trace[:,1]*np.tanh(b)
        for start,end in zip(o[:-1],o[1:]):
            trace[start,27]=0;trace[start+1:end,27]=np.diff(b[start:end])
    prefix=prefix_features(o,s,q,y)
    raw=np.column_stack([a['behavior_features'],y,a['measurement_missing']]).astype('float32')
    tr=np.column_stack([np.load(data/'train_behavior_features.npy',mmap_mode='r'),
                        np.load(data/'train_correct.npy',mmap_mode='r'),np.load(data/'train_measurement_missing.npy',mmap_mode='r')])
    center=tr.mean(0,dtype=np.float64);scale=np.maximum(tr.std(0,dtype=np.float64),.05)
    hist=history_signals(((raw-center)/scale).astype('float32'),o)
    history=np.arange(len(y))-np.repeat(o[:-1],np.diff(o))
    raw_names=meta['behavior_feature_names']+['answer']+[f'missing_measurement_{j}' for j in range(4)]
    names=[f'{tag}_{n}' for tag in ['last','ema_05','ema_09','ema_098'] for n in raw_names]+core.PREFIX_NAMES+core.TRACE_NAMES+['log_observed_history','known_skill']+uncertainty.NAMES
    x=np.column_stack([hist,prefix,trace,np.log1p(history),~a['skill_missing'],belief]).astype('float32')[target]
    forbidden=set()
    for j in meta['absent_measurements']:
        forbidden.add(meta['behavior_feature_names'][j])
        if j==0:forbidden.add('time_efficiency_discrepancy')
        if j==3:forbidden.add('persistence_deviation')
    keep=[i for i,n in enumerate(names) if not any(n==f'{tag}_{bad}' for tag in ['last','ema_05','ema_09','ema_098'] for bad in forbidden)]
    x=x[:,keep];names=[names[i] for i in keep]
    assert np.isfinite(x).all()
    return x,names,{'target':y[target],'users':a['user_ids'][target],'known':~a['skill_missing'][target],
                   'history':history[target],'skill_unseen':a['skill_unseen'][target],'item_unseen':a['item_unseen'][target]}

def cached(dataset,split,window=False,neutral=False):
    if split=='test':assert (OUT/dataset/'freeze.json').exists(),'Test locked until all models freeze'
    tag='window' if window else 'neutral' if neutral else 'full'
    folder=OUT/dataset/'features';folder.mkdir(parents=True,exist_ok=True)
    xp=folder/f'{split}_{tag}.npy';jp=folder/f'{split}_{tag}.json';zp=folder/f'{split}_{tag}_targets.npz'
    if not xp.exists():
        x,names,d=compute(dataset,split,window,neutral)
        np.save(xp,x);np.savez_compressed(zp,**d)
        write(jp,{'names':names,'source_sha256':sha(__file__),'recurrence_source':[sha(ROOT/'src/mica_transparent_20260923'/n) for n in ['state_features.cpp','uncertainty_features.cpp']],
                  'timing':'Emit before target answer. Four history summaries contain previous observations only.'})
    record=json.loads(jp.read_text());assert record['source_sha256']==sha(__file__),'Cache source changed; regenerate explicit feature cache'
    return np.load(xp,mmap_mode='r'),record['names'],dict(np.load(zp))
