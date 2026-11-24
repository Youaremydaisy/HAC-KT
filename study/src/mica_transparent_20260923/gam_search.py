"""MICA v2 with a transparent one/two-dimensional response function.

Uses the published EBM/GA2M fitting algorithm, not a novel training algorithm.
The educational state framework remains ECKT/MICA and no neural KT baseline
predictions enter training. Inference can use exported lookup tables alone.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, json, pickle, sys, time
import numpy as np
from scipy.special import expit
from core import OUT, SRC, data, cognitive, validation_metrics, TRACE_NAMES, PREFIX_NAMES, STATES, write, sha
sys.path.insert(0,str(OUT/'dependencies'))
from interpret.glassbox import ExplainableBoostingClassifier

DEST=OUT/'gam_search'

def features(d, include_state=True):
    b=np.asarray(d['behavior'],np.float64);pref=np.asarray(d['prefix'],np.float64)
    tr=np.asarray(d['trace'],np.float64)
    names=[f'behavior_history_{i}' for i in range(b.shape[1])]+PREFIX_NAMES+TRACE_NAMES+['log_observed_history','known_skill']
    parts=[b,pref,tr,np.log1p(d['history'])[:,None],np.asarray(d['known'],float)[:,None]]
    if include_state:
        names += ['affect_'+s for s in STATES];parts += [d['state']]
    x=np.column_stack(parts).astype(np.float64)
    assert np.isfinite(x).all()
    return x,names

def interactions(names):
    pairs=[('student_rasch_medium','training_item_difficulty'),('skill_rasch_medium','training_item_difficulty'),
        ('student_beta_log_odds','training_item_difficulty'),('skill_beta_log_odds','skill_log_gap'),
        ('student_rasch_medium','known_skill'),('behavior_history_0','behavior_history_8'),
        ('item_seen','training_item_difficulty'),('skill_seen','skill_rasch_medium')]
    pairs += [('affect_'+s,'training_item_difficulty') for s in STATES if 'affect_'+s in names]
    return [(names.index(a),names.index(b)) for a,b in pairs]

def export_tables(model,path):
    arrays={'intercept':np.asarray(model.intercept_),'feature_names':np.asarray(model.feature_names_in_),
            'term_names':np.asarray(model.term_names_)}
    for i,bins in enumerate(model.bins_):
        for j,cuts in enumerate(bins):
            if isinstance(cuts,dict):raise ValueError('numeric transparent features required')
            arrays[f'cuts_{i}_{j}']=cuts
        arrays[f'levels_{i}']=np.array(len(bins))
    for t,(indexes,scores) in enumerate(zip(model.term_features_,model.term_scores_)):
        arrays[f'features_{t}']=np.asarray(indexes);arrays[f'scores_{t}']=scores
    np.savez_compressed(path,**arrays)

def table_terms(x,table):
    nterms=len(table['term_names']);out=np.empty((len(x),nterms))
    for t in range(nterms):
        ix=table[f'features_{t}'];indices=[]
        for i in ix:
            level=min(len(ix),int(table[f'levels_{i}']))-1
            bins=np.searchsorted(table[f'cuts_{i}_{level}'],x[:,i],side='right')+1
            bins[np.isnan(x[:,i])]=0;indices.append(bins)
        out[:,t]=table[f'scores_{t}'][tuple(indices)]
    return out

def predict_tables(x,offset,table):
    out=np.empty(len(x))
    for a in range(0,len(x),30000):
        b=min(a+30000,len(x));out[a:b]=expit(offset[a:b].astype(np.float64)+float(table['intercept'].ravel()[0])+table_terms(x[a:b],table).sum(1))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',type=int,default=600000);p.add_argument('--rounds',type=int,default=2000)
    a=p.parse_args();DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'selection.json').exists():raise FileExistsError('completed')
    sample=np.sort(np.random.default_rng(20260923).choice(len(data('train')['target']),a.sample,replace=False))
    np.save(DEST/'sample.npy',sample);tr=data('train',sample);va=data('val')
    x,names=features(tr);xv,_=features(va);ot=np.asarray(cognitive(42,'train')[sample]);ov=np.asarray(cognitive(42,'val'))
    # Hold out whole training students for the fitting algorithm's early stop.
    fold=((tr['users']*1103515245+12345)%2147483647)%10
    bags=np.column_stack([np.where(fold==j,-1,1) for j in (0,1)]).astype(np.int8)
    protocol={'created':time.time(),'sample':a.sample,'max_rounds':a.rounds,'outer_bags':2,
        'candidates':['additive','pairwise'],'pairwise_feature_names':[[names[i],names[j]] for i,j in interactions(names)],
        'selection':'Maximum validation development AUC; check half reported separately; four detector states required.',
        'early_stop':'Two fixed held-out groups of training students; frozen cognitive and affect components were trained on all training students.',
        'test_access':False,'historical_validation_reuse':True,'state_estimators_changed':False,
        'training_library':'interpret-core 0.7.7, EBM; existing method, not claimed as our invention',
        'literature':'https://doi.org/10.1145/2487575.2487579','source_sha256':sha(__file__)}
    write(DEST/'protocol.json',protocol);rows={}
    for kind in ['additive','pairwise']:
        start=time.time();print('FIT',kind,flush=True)
        model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),
            interactions=0 if kind=='additive' else interactions(names),max_bins=64,max_interaction_bins=16,
            outer_bags=2,inner_bags=0,learning_rate=.03,max_rounds=a.rounds,early_stopping_rounds=80,
            min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=2,random_state=42,
            smoothing_rounds=50,interaction_smoothing_rounds=50)
        if (DEST/f'{kind}.pkl').exists():
            with (DEST/f'{kind}.pkl').open('rb') as f:model=pickle.load(f)
        else:
            model.fit(x,tr['target'],bags=bags,init_score=ot)
            with (DEST/f'{kind}.pkl').open('wb') as f:pickle.dump(model,f)
        export_tables(model,DEST/f'{kind}_tables.npz')
        table=dict(np.load(DEST/f'{kind}_tables.npz'))
        pred=predict_tables(xv,ov,table)
        native=model.predict_proba(xv[:5000],init_score=ov[:5000])[:,1]
        err=float(np.max(np.abs(pred[:5000]-native)));assert err<1e-10,err
        np.save(DEST/f'{kind}_validation.npy',pred)
        rows[kind]={'metrics':validation_metrics(va,pred),'seconds':time.time()-start,
            'table_vs_library_max_error':err,'best_iteration':np.asarray(model.best_iteration_).tolist(),
            'num_terms':len(model.term_features_),'max_term_dimensions':max(map(len,model.term_features_)),
            'tables_sha256':sha(DEST/f'{kind}_tables.npz')}
        write(DEST/f'{kind}.json',rows[kind]);print(json.dumps({'candidate':kind,**rows[kind]}),flush=True)
    best=max(rows,key=lambda k:rows[k]['metrics']['development']['auc'])
    write(DEST/'selection.json',{'candidate':best,**rows[best],'test_access':False})

if __name__=='__main__':main()
