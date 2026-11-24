"""Same-budget transparent readouts and predeclared mechanism controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import gc
import json
import time
import numpy as np
from common import OUT, data_path, write, sha, folds, metrics
from features import cached, ExplainableBoostingClassifier, export_tables, predict_tables, table_terms

KINDS=('cf','raw_history','neutral_update','no_offset','window_no_offset')

def design(dataset,split,kind):
    x,names,d=cached(dataset,split,window=kind=='window_no_offset',neutral=kind=='neutral_update')
    if kind=='raw_history':
        keep=[i for i,n in enumerate(names) if n.startswith(('last_','ema_')) or n in ('log_observed_history','known_skill','training_item_difficulty')
              or any(n==f'{g}_{v}' for g in ('student','skill','item') for v in ('log_success','log_failure','beta_log_odds','previous_answer','log_gap','seen'))]
        x=x[:,keep];names=[names[i] for i in keep]
    return x,names,d

def offset(dataset,split,kind,seed,n):
    if kind in ('raw_history','no_offset','window_no_offset'):return np.zeros(n)
    return np.load(OUT/dataset/'models'/f'cognitive_seed{seed}'/f'{split}_predictions.npz')['logit']

def fit(dataset,kind,seed):
    dest=OUT/dataset/'models'/f'{kind}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists():return
    started=time.time();x,names,tr=design(dataset,'train',kind);v,vnames,va=design(dataset,'val',kind);assert names==vnames
    # Drop constant columns using training only; retain missing indicators when variable.
    keep=np.flatnonzero(np.ptp(x,axis=0)>1e-9);x=np.asarray(x[:,keep],float);v=np.asarray(v[:,keep],float);names=[names[i] for i in keep]
    ot=offset(dataset,'train',kind,seed,len(x));ov=offset(dataset,'val',kind,seed,len(v));pred=np.empty(len(v));heads={};replays=[]
    for known in [False,True]:
        tm=tr['known']==known;vm=va['known']==known
        if tm.sum()<500 or len(np.unique(tr['target'][tm]))<2:continue
        f=folds(tr['users'][tm]);bags=np.column_stack([np.where(f==j,-1,1) for j in (0,1)]).astype('int8')
        model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),
            interactions=64,max_bins=256,max_interaction_bins=32,outer_bags=2,inner_bags=0,learning_rate=.03,
            max_rounds=2000,early_stopping_rounds=80,min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=2,
            random_state=seed,smoothing_rounds=50,interaction_smoothing_rounds=50)
        print('FIT',dataset,kind,seed,known,int(tm.sum()),flush=True)
        model.fit(x[tm],tr['target'][tm],bags=bags,init_score=ot[tm])
        file=dest/f'{known}_tables.npz';export_tables(model,file);table=dict(np.load(file))
        ix=np.flatnonzero(vm)[:1000]
        if len(ix):
            error=float(np.max(np.abs(predict_tables(v[ix],ov[ix],table)-model.predict_proba(v[ix],init_score=ov[ix])[:,1])))
            assert error<1e-10;replays.append(error)
            pred[vm]=predict_tables(v[vm],ov[vm],table)
        heads[str(known)]={'file':file.name,'sha256':sha(file),'terms':len(model.term_features_),
                           'best_iterations':np.asarray(model.best_iteration_).tolist()}
        del model;gc.collect()
    for known in (False,True):
        if str(known) not in heads:
            fallback=next(iter(heads.values()));heads[str(known)]={**fallback,'fallback':True}
            vm=va['known']==known
            if vm.any():pred[vm]=predict_tables(v[vm],ov[vm],dict(np.load(dest/fallback['file'])))
    np.savez_compressed(dest/'val_predictions.npz',target=va['target'],full=pred,student_id=va['users'])
    write(dest/'results.json',{'dataset':dataset,'kind':kind,'seed':seed,'validation':metrics(va['target'],pred),
        'columns':keep.tolist(),'feature_names':names,'heads':heads,'max_native_replay_error':max(replays,default=0),
        'elapsed_seconds':time.time()-started,'source_sha256':sha(__file__),'test_access':False,
        'offset_is_end_to_end_oof':False,'neutral_update_scope':'Neutral difficulty in recurrent updates only; current difficulty inputs retained',
        'raw_history_scope':'Observed behavior, prefix history and training difficulty; same EBM capacity, zero cognitive offset'})
    print('DONE',dataset,kind,seed,metrics(va['target'],pred),flush=True)

def infer(dataset,kind,seed):
    assert (OUT/dataset/'freeze.json').exists()
    dest=OUT/dataset/'models'/f'{kind}_seed{seed}';p=dest/'test_predictions.npz'
    if p.exists():return
    result=json.loads((dest/'results.json').read_text());x,names,d=design(dataset,'test',kind)
    keep=result['columns'];assert [names[i] for i in keep]==result['feature_names'];x=x[:,keep]
    o=offset(dataset,'test',kind,seed,len(x));pred=np.empty(len(x));max_replay=0.
    for known in (False,True):
        m=d['known']==known
        if not m.any():continue
        head=result['heads'][str(known)];assert sha(dest/head['file'])==head['sha256']
        table=dict(np.load(dest/head['file']));pred[m]=predict_tables(x[m],o[m],table)
        # Explanations partition each native term once. Interactions are not assigned arbitrary causal credit.
        ix=np.flatnonzero(m)[:500];terms=table_terms(x[ix],table)
        groups={}
        for j in range(terms.shape[1]):
            features=table[f'features_{j}']; label='interaction' if len(features)>1 else 'main_effect'
            groups.setdefault(label,np.zeros(len(ix)));groups[label]+=terms[:,j]
        z=o[ix]+float(table['intercept'].ravel()[0])+sum(groups.values())
        from scipy.special import expit
        max_replay=max(max_replay,float(np.max(np.abs(expit(z)-pred[ix]))))
    assert max_replay<1e-10
    np.savez_compressed(p,target=d['target'],full=pred,student_id=d['users'])
    write(dest/'test_replay.json',{'group_probability_max_error':max_replay,'scope':'Exact numerical replay, not teacher or causal validation'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('kind',choices=KINDS);p.add_argument('--seed',type=int,default=42);p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.test:infer(a.dataset,a.kind,a.seed)
    else:fit(a.dataset,a.kind,a.seed)
