"""Repeat the fixed pairwise configuration, freeze, then evaluate once.

No model selection is performed on test data. All preregistered repeats and
the no-affect control are reported even if they perform worse.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,pickle,sys,time
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from core import ROOT,OUT,SRC,LEGACY,data,prepare,cognitive,metrics,validation_metrics,write,sha
from gam_search import features,interactions,export_tables,predict_tables,table_terms,ExplainableBoostingClassifier

DEST=OUT/'confirmation'

def calibrate(y,p):
    z=logit(np.clip(p,1e-7,1-1e-7))
    def objective(t):
        a=np.exp(t[0]);logits=a*z+t[1];e=expit(logits)-y
        return np.mean(np.logaddexp(0,logits)-y*logits),np.array([np.mean(e*a*z),np.mean(e)])
    f=minimize(objective,[0.,0.],jac=True,method='L-BFGS-B',bounds=[(-3,3),(-5,5)],options={'ftol':1e-12,'gtol':1e-8})
    if not f.success:raise RuntimeError(f.message)
    return {'scale':float(np.exp(f.x[0])),'intercept':float(f.x[1]),'fit_split':'validation'}

def state_metrics(truth,p):
    mse=np.mean((truth-p)**2,axis=0);var=np.var(truth,axis=0)
    return {'rmse':float(np.sqrt(mse.mean())),'rmse_by_state':np.sqrt(mse).tolist(),'r2_by_state':(1-mse/var).tolist()}

def train_repeats():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'selection_freeze.json').exists():return
    protocol={'created':time.time(),'primary':'seed42 pairwise MICA transparent extension',
        'fixed_configuration':'gam_search pairwise: 600000 rows, 64 bins, 16 interaction bins, 2 bags, .03 learning rate, 2000 max rounds, 100 min leaf, lambda10, 3 leaves',
        'repeats':[43,44],'control':'seed42 no-affect state columns or interactions',
        'selection':'configuration already selected with validation development AUC; repeats do not change it',
        'bootstrap_replicates':1000,'test_used_for_selection':False,'historical_test_reuse':True,
        'source_sha256':sha(__file__),'retraining_scope':'Fixed detector estimator shared. Each seed uses an already independently trained cognitive checkpoint, then fits a new transparent response function.'}
    if not (DEST/'protocol.json').exists():write(DEST/'protocol.json',protocol)
    sample=np.load(OUT/'gam_search/sample.npy');tr=data('train',sample);va=data('val');results={};frozen={}
    for seed,kind in [(42,'joint'),(43,'joint'),(44,'joint'),(42,'no_affect')]:
        name=f'seed{seed}_{kind}';start=time.time();path=DEST/f'{name}_tables.npz'
        print('REPEAT',name,flush=True)
        x,names=features(tr,kind=='joint');xv,_=features(va,kind=='joint')
        ot=np.asarray(cognitive(seed,'train')[sample]);ov=np.asarray(cognitive(seed,'val'))
        if seed==42 and kind=='joint':
            path=OUT/'gam_search/pairwise_tables.npz';table=dict(np.load(path))
            pred=np.load(OUT/'gam_search/pairwise_validation.npy');iterations=json.loads((OUT/'gam_search/pairwise.json').read_text())['best_iteration']
        elif path.exists():
            table=dict(np.load(path));pred=predict_tables(xv,ov,table)
            iterations=json.loads((DEST/f'{name}.json').read_text())['best_iteration']
        else:
            fold=((tr['users']*1103515245+12345)%2147483647)%10
            bags=np.column_stack([np.where(fold==j,-1,1) for j in (0,1)]).astype(np.int8)
            model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),
                interactions=interactions(names),max_bins=64,max_interaction_bins=16,outer_bags=2,inner_bags=0,
                learning_rate=.03,max_rounds=2000,early_stopping_rounds=80,min_samples_leaf=100,reg_lambda=10.,
                max_leaves=3,n_jobs=2,random_state=seed,smoothing_rounds=50,interaction_smoothing_rounds=50)
            model.fit(x,tr['target'],bags=bags,init_score=ot)
            export_tables(model,path);table=dict(np.load(path));pred=predict_tables(xv,ov,table)
            check=model.predict_proba(xv[:5000],init_score=ov[:5000])[:,1]
            assert np.max(np.abs(pred[:5000]-check))<1e-10
            iterations=np.asarray(model.best_iteration_).tolist()
        np.save(DEST/f'{name}_validation.npy',pred)
        results[name]={'seed':seed,'kind':kind,'validation':validation_metrics(va,pred),'tables':str(path.relative_to(ROOT)),
            'best_iteration':iterations,'seconds':time.time()-start,'calibration':calibrate(va['target'],pred)}
        write(DEST/f'{name}.json',results[name]);print(json.dumps({'repeat':name,**results[name]}),flush=True)
        frozen[str(path.relative_to(ROOT))]=sha(path)
        cognitive_checkpoint=(ROOT/'checkpoints/mica_confirm_20260917_v2/no_affect_gain/best.pt') if seed==42 else ROOT/f'checkpoints/mica_meeting_20260917_v1/no_gain_seed{seed}/best.pt'
        frozen[str(cognitive_checkpoint.relative_to(ROOT))]=sha(cognitive_checkpoint)
    for name in ['state_selection.npz','memory_gam_model.npz','history_gam_model.npz','history_mean_model.npz']:
        frozen[str((LEGACY/name).relative_to(ROOT))]=sha(LEGACY/name)
    for directory,filenames in [(SRC,['core.py','state_features.cpp','gam_search.py','confirm.py']),
        (ROOT/'reports/mica_confirm_20260917_v2/source',['train_eckt.py','train_model.py']),
        (ROOT/'src/mica_history_20260917_v2',['develop.py','predictor.py']),
        (ROOT/'src/mica_opt_20260917_v1',['run_optimization.py'])]:
        for name in filenames:frozen[str((directory/name).relative_to(ROOT))]=sha(directory/name)
    write(DEST/'validation_repeats.json',results)
    primary=results['seed42_joint']['validation']
    reference=json.loads((OUT/'cuda_search/selection.json').read_text())['metrics']
    audit=json.loads((OUT/'trace_audit.json').read_text())
    if not audit['passed']:raise RuntimeError('trace audit did not pass')
    # This is a promotion guard against the immediately preceding joint candidate,
    # not a significance claim or an untouched validation cohort.
    if not all(primary[k]['auc']>=reference[k]['auc'] for k in ['development','check']):
        raise RuntimeError('joint candidate did not improve both validation groups')
    write(DEST/'selection_freeze.json',{'created':time.time(),'primary':'seed42_joint','models':results,
        'files':frozen,'source_protocol_sha256':sha(DEST/'protocol.json'),'test_used_for_selection':False,
        'historical_test_reuse':True,'no_further_test_driven_tuning':True,'trace_audit_sha256':sha(OUT/'trace_audit.json')})

def evaluate():
    if (DEST/'test_results.json').exists():return
    freeze=json.loads((DEST/'selection_freeze.json').read_text())
    for path,digest in freeze['files'].items():assert sha(ROOT/path)==digest,path
    prepare('test');d=data('test');y=d['target'];users=d['users'];rows={};probs={}
    for name,conf in freeze['models'].items():
        x,names=features(d,conf['kind']=='joint');offset=cognitive(conf['seed'],'test')
        table=dict(np.load(ROOT/conf['tables']));pred=predict_tables(x,offset,table)
        c=conf['calibration'];cal=expit(c['scale']*logit(np.clip(pred,1e-7,1-1e-7))+c['intercept'])
        rows[name]={'raw':metrics(y,pred),'calibrated':metrics(y,cal),
            'known_skill':metrics(y[d['known']],pred[d['known']]),'unknown_skill':metrics(y[~d['known']],pred[~d['known']])}
        probs[name]=pred;np.savez_compressed(DEST/f'{name}_test.npz',target=y,student_id=users,full=pred,calibrated=cal)
        if name=='seed42_joint':
            index=np.linspace(0,len(y)-1,200,dtype=int);terms=table_terms(x[index],table)
            np.savez_compressed(DEST/'test_explanation_examples.npz',positions=index,features=x[index],cognitive_logit=offset[index],
                terms=terms,term_names=table['term_names'],intercept=table['intercept'],probability=pred[index],state=d['state'][index])
        print('TEST',name,rows[name]['raw']['auc'],flush=True)
    sources={'LBKT':'reports/mica_exp_20260921/baselines/assist12/lbkt_seed42/test_predictions.npz',
             'MICA_v2':'reports/mica_history_20260917_v2/evaluation/seed42/test_behavior_states.npz',
             'AKT':'reports/mica_exp_20260921/baselines/assist12/akt_seed42/test_predictions.npz',
             'MICA_prefix':'reports/mica_completion_20260922/prefix_readout/test/seed42_prefix_behavior.npz'}
    for name,path in sources.items():
        z=np.load(ROOT/path);np.testing.assert_array_equal(z['target'],y);np.testing.assert_array_equal(z['student_id'],users)
        probs[name]=z['full'];rows[name]={'raw':metrics(y,probs[name]),'source':path,'sha256':sha(ROOT/path)}
    truth=np.load(OUT/'cache/test/truth.npy',mmap_mode='r')
    result={'models':rows,'primary':'seed42_joint','affect':state_metrics(truth,d['state']),
        'three_seed_auc_mean':float(np.mean([rows[f'seed{s}_joint']['raw']['auc'] for s in (42,43,44)])),
        'three_seed_auc_std_ddof1':float(np.std([rows[f'seed{s}_joint']['raw']['auc'] for s in (42,43,44)],ddof=1)),
        'freeze_sha256':sha(DEST/'selection_freeze.json'),'test_used_for_selection':False,'historical_test_reuse':True,
        'status':'metrics_complete_bootstrap_pending'}
    write(DEST/'test_point_results.json',result)
    sys.path.insert(0,str(ROOT/'src'))
    from bootstrap_eckt_student_auc import _prepared,_weighted_auc
    names=['seed42_joint','seed42_no_affect','LBKT','MICA_v2','MICA_prefix']
    prepared={n:_prepared(probs[n]) for n in names};bootstrap={n:np.empty(1000) for n in names}
    _,group=np.unique(users,return_inverse=True);count=int(group.max()+1);rng=np.random.default_rng(20260923)
    for i in range(1000):
        weights=np.bincount(rng.integers(count,size=count),minlength=count)[group]
        for name in names:bootstrap[name][i]=_weighted_auc(prepared[name],y,weights)
        if i%100==0:write(DEST/'status.json',{'stage':'bootstrap','replicates':i,'total':1000})
    intervals={}
    for name in names[1:]:
        delta=bootstrap['seed42_joint']-bootstrap[name]
        intervals['primary_minus_'+name]={'delta_auc':rows['seed42_joint']['raw']['auc']-rows[name]['raw']['auc'],
            'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))}
    for path,digest in freeze['files'].items():assert sha(ROOT/path)==digest,path
    result.update(student_bootstrap=intervals,bootstrap_replicates=1000,students=count,status='complete',
        boundary='Repeated cognitive backbones and newly fitted transparent readouts share a detector estimator and dataset. Intervals are fixed-model student resampling, not independent confirmatory evidence or acceptance probabilities.')
    write(DEST/'test_results.json',result);write(DEST/'status.json',{'stage':'complete'})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    train_repeats();evaluate()
