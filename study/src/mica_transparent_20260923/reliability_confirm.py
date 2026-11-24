"""Freeze the literature-driven candidate by validation, repeat, evaluate once.

The earlier test results were known before this branch. This is explicitly an
exploratory reuse of that cohort, not an independent confirmatory experiment.
No post-test search or candidate substitution is performed by this script.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import json,sys,time
import numpy as np
from scipy.special import expit,logit
from core import ROOT,OUT,SRC,DATA,STATES,data,cognitive,positions,write,sha,metrics,validation_metrics
from gam_search import features,interactions,export_tables,predict_tables,ExplainableBoostingClassifier
from uncertainty_search import NAMES,library,calculate
from affect_reliability import matrix,regression
from confirm import calibrate,state_metrics

DEST=OUT/'reliability_confirmation'

def get_selection():
    path=DEST/'selection.json'
    if path.exists():return json.loads(path.read_text())
    capacity=json.loads((OUT/'contextual_capacity_validation/results.json').read_text())['candidates']
    fixed=json.loads((OUT/'contextual_validation/joint.json').read_text())
    choices={'fixed_1800k':{'result':fixed,'mode':'fixed','sample_path':'uncertainty_validation/sample_1800k.npy','bins':128}}
    for key,r in capacity.items():choices[key]={'result':r,'mode':'auto16','sample_path':f'contextual_capacity_validation/{key}_sample.npy','bins':r['bins']}
    baseline=json.loads((OUT/'gam_search/pairwise.json').read_text())['metrics']
    eligible=[k for k,v in choices.items() if all(v['result']['metrics'][s]['auc']>=baseline[s]['auc'] for s in ('development','check'))]
    assert eligible
    best=max(eligible,key=lambda k:choices[k]['result']['metrics']['development']['auc']);selected=choices[best]
    if best=='fixed_1800k':
        selected['primary_tables']={str(k):f'contextual_validation/joint_'+('known' if k else 'missing')+'_tables.npz' for k in (False,True)}
        selected['primary_validation']='contextual_validation/joint_validation.npy'
    else:
        selected['primary_tables']={str(r['known']):r['tables'] for r in selected['result']['regimes']}
        selected['primary_validation']=f'contextual_capacity_validation/{best}_validation.npy'
    selection={'created':time.time(),'candidate':best,**selected,'all_candidates':choices,
        'criterion':'Highest validation development AUC; check half must also improve the preceding transparent model.',
        'test_used_for_selection':False,'historical_test_reuse':True,'test_performance_of_earlier_branch_known':True,
        'remaining_evaluations':'Three fixed seeds plus one no-affect control; no selection by their test values',
        'source_sha256':sha(__file__)}
    write(path,selection);return selection

def response_matrix(d,belief,state,include_affect=True):
    x,names=matrix(d,belief)
    if include_affect:x=np.column_stack((x,state));names+=['affect_'+s for s in STATES]
    return x,names

def repeat(selection,seed,kind):
    name=f'seed{seed}_{kind}';meta_path=DEST/f'{name}.json'
    if meta_path.exists():return json.loads(meta_path.read_text())
    va=data('val');bv=np.load(OUT/'uncertainty_validation/val_belief.npy',mmap_mode='r')
    sv=np.load(OUT/'affect_reliability_validation/validation_state.npy',mmap_mode='r')
    v,names=response_matrix(va,bv,sv,kind=='joint');ov=np.asarray(cognitive(seed,'val'))
    if seed==42 and kind=='joint':
        tables=selection['primary_tables'];pred=np.load(OUT/selection['primary_validation'])
    elif seed==42 and kind=='no_affect' and selection['mode']=='fixed':
        tables={str(k):f'contextual_validation/no_affect_'+('known' if k else 'missing')+'_tables.npz' for k in (False,True)}
        pred=np.load(OUT/'contextual_validation/no_affect_validation.npy')
    else:
        sample=np.load(OUT/selection['sample_path']);tr=data('train',sample)
        bt=np.load(OUT/'uncertainty_validation/train_belief.npy',mmap_mode='r')[sample]
        st=np.load(OUT/'affect_reliability_validation/train_state.npy',mmap_mode='r')[sample]
        x,_=response_matrix(tr,bt,st,kind=='joint');ot=np.asarray(cognitive(seed,'train')[sample]);pred=np.empty(len(v));tables={}
        pairwise=16
        if selection['mode']=='fixed':
            pairwise=interactions(names)+[(names.index(a),names.index(b)) for a,b in [
                ('student_belief_medium','training_item_difficulty'),('skill_belief_medium','training_item_difficulty'),
                ('student_belief_medium','student_belief_variance_medium'),('skill_belief_medium','skill_belief_variance_medium')]]
        for known in (False,True):
            file=DEST/(name+('_known' if known else '_missing')+'_tables.npz');tables[str(known)]=str(file.relative_to(OUT));tm=tr['known']==known;vm=va['known']==known
            print('REPEAT',name,known,flush=True);write(DEST/'status.json',{'stage':'repeat','candidate':name,'known_skill':known})
            if not file.exists():
                folds=((tr['users'][tm]*1103515245+12345)%2147483647)%10
                bags=np.column_stack([np.where(folds==j,-1,1) for j in (0,1)]).astype(np.int8)
                model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),interactions=pairwise,
                    max_bins=selection['bins'],max_interaction_bins=32 if selection['mode']=='auto16' else 16,
                    outer_bags=2,inner_bags=0,learning_rate=.03,max_rounds=2000,early_stopping_rounds=80,
                    min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=3,random_state=seed,
                    smoothing_rounds=50,interaction_smoothing_rounds=50)
                model.fit(x[tm],tr['target'][tm],bags=bags,init_score=ot[tm]);export_tables(model,file)
                pv=predict_tables(v[vm],ov[vm],dict(np.load(file)))
                err=np.max(np.abs(pv[:1000]-model.predict_proba(v[vm][:1000],init_score=ov[vm][:1000])[:,1]));assert err<1e-10
            pred[vm]=predict_tables(v[vm],ov[vm],dict(np.load(file)))
    np.save(DEST/f'{name}_validation.npy',pred)
    result={'seed':seed,'kind':kind,'tables':tables,'validation':validation_metrics(va,pred),'calibration':calibrate(va['target'],pred)}
    write(meta_path,result);return result

def freeze(selection):
    path=DEST/'freeze.json'
    if path.exists():return json.loads(path.read_text())
    models={f'seed{s}_{k}':repeat(selection,s,k) for s,k in [(42,'joint'),(43,'joint'),(44,'joint'),(42,'no_affect')]}
    paths=[DEST/'selection.json',SRC/'reliability_confirm.py',SRC/'core.py',SRC/'gam_search.py',SRC/'uncertainty_features.cpp',
        SRC/'uncertainty_search.py',SRC/'affect_reliability.py',SRC/'contextual_capacity.py',SRC/'contextual_search.py',
        OUT/'affect_reliability_validation/state_parameters.npz',ROOT/'reports/mica_opt_20260917_v1/state_selection.npz',
        DATA/'metadata.json']
    paths += [OUT/'affect_reliability_validation'/f'{s}_tables.npz' for s in STATES]
    paths += [OUT/p for m in models.values() for p in m['tables'].values()]
    old=json.loads((OUT/'confirmation/selection_freeze.json').read_text())['files']
    paths += [ROOT/p for p in old if p.endswith('best.pt') or '/source/' in p or p.startswith('src/')]
    record={'created':time.time(),'candidate':selection['candidate'],'primary':'seed42_joint','models':models,
        'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'historical_test_reuse':True,'test_used_for_selection':False,
        'no_post_test_tuning':True,'retraining_scope':'Previously independent cognitive checkpoints and newly trained response tables; fixed shared new detector estimator and fixed state filters.',
        'validation_state_results':json.loads((OUT/'affect_reliability_validation/state_results.json').read_text())}
    write(path,record);return record

def test_features():
    d=data('test');folder=DEST/'cache';folder.mkdir(exist_ok=True)
    bpath=folder/'test_belief.npy'
    if not bpath.exists():
        fn=library();m=json.loads((DATA/'metadata.json').read_text());args=[np.load(DATA/f'test_{n}.npy') for n in ('offsets','skills','problem_index','correct','problem_difficulty')]
        belief=calculate(fn,*args,m['num_skills_with_padding'],m['num_problems_with_padding'])[positions('test')];np.save(bpath,belief)
    else:belief=np.load(bpath,mmap_mode='r')
    x,names=matrix(d,belief)
    with np.load(OUT/'affect_reliability_validation/state_parameters.npz') as z:
        means=z['means'];rho=z['rho'];choice=z['choice'];bins=z['history_bins']
    g=np.searchsorted(bins,d['history'],side='right');mu=means[g];state=np.empty((len(g),4))
    for j,s in enumerate(STATES):
        table=dict(np.load(OUT/'affect_reliability_validation'/f'{s}_tables.npz'))
        assert list(table['feature_names'])==names
        raw=np.clip(mu[:,j]+regression(x,table),0,1)
        state[:,j]=mu[:,j] if choice[j]==0 else raw if choice[j]==1 else mu[:,j]+rho[g,j]*(raw-mu[:,j])
    np.save(folder/'test_state.npy',state);return d,belief,state

def evaluate(frozen):
    result_path=DEST/'test_results.json'
    if result_path.exists():return
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    d,belief,state=test_features();y=d['target'];users=d['users'];rows={};preds={}
    for name,conf in frozen['models'].items():
        path=DEST/f'{name}_test.npz'
        if path.exists():
            z=np.load(path);np.testing.assert_array_equal(z['target'],y);p=z['full'];cal=z['calibrated']
        else:
            x,names=response_matrix(d,belief,state,conf['kind']=='joint');offset=cognitive(conf['seed'],'test');p=np.empty(len(y))
            for known in (False,True):
                m=d['known']==known;table=dict(np.load(OUT/conf['tables'][str(known)]));assert list(table['feature_names'])==names
                p[m]=predict_tables(x[m],offset[m],table)
            c=conf['calibration'];cal=expit(c['scale']*logit(np.clip(p,1e-7,1-1e-7))+c['intercept'])
            np.savez_compressed(path,target=y,student_id=users,full=p,calibrated=cal)
        rows[name]={'raw':metrics(y,p),'calibrated':metrics(y,cal)};preds[name]=p;print('TEST',name,rows[name]['raw']['auc'],flush=True)
    refs={'LBKT':ROOT/'reports/mica_exp_20260921/baselines/assist12/lbkt_seed42/test_predictions.npz',
        'previous_transparent':OUT/'confirmation/seed42_joint_test.npz'}
    for name,path in refs.items():
        z=np.load(path);np.testing.assert_array_equal(y,z['target']);np.testing.assert_array_equal(users,z['student_id'])
        preds[name]=z['full'];rows[name]={'raw':metrics(y,preds[name]),'sha256':sha(path)}
    result={'candidate':frozen['candidate'],'primary':'seed42_joint','models':rows,
        'three_seed_auc_mean':float(np.mean([rows[f'seed{s}_joint']['raw']['auc'] for s in (42,43,44)])),
        'three_seed_auc_std_ddof1':float(np.std([rows[f'seed{s}_joint']['raw']['auc'] for s in (42,43,44)],ddof=1)),
        'affect':state_metrics(np.load(OUT/'cache/test/truth.npy',mmap_mode='r'),state),
        'freeze_sha256':sha(DEST/'freeze.json'),'test_used_for_selection':False,'historical_test_reuse':True,
        'status':'points_complete','post_test_tuning':False}
    write(DEST/'test_point_results.json',result)
    sys.path.insert(0,str(ROOT/'src'));from bootstrap_eckt_student_auc import _prepared,_weighted_auc
    names=['seed42_joint','seed42_no_affect','LBKT','previous_transparent'];prepared={n:_prepared(preds[n]) for n in names}
    _,group=np.unique(users,return_inverse=True);count=int(group.max()+1);rng=np.random.default_rng(20260929);boot={n:np.empty(1000) for n in names}
    for i in range(1000):
        w=np.bincount(rng.integers(count,size=count),minlength=count)[group]
        for n in names:boot[n][i]=_weighted_auc(prepared[n],y,w)
        if i%100==0:write(DEST/'status.json',{'stage':'bootstrap','completed':i,'total':1000})
    intervals={}
    for name in names[1:]:
        delta=boot['seed42_joint']-boot[name];intervals[name]={'delta_auc':rows['seed42_joint']['raw']['auc']-rows[name]['raw']['auc'],
            'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))}
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    result.update(status='complete',student_bootstrap=intervals,bootstrap_replicates=1000,students=count,
        boundary='Exploratory reuse of an already inspected test cohort; fixed-model resampling, shared detector estimator, not fresh confirmation or acceptance probability.')
    write(result_path,result);write(DEST/'status.json',{'stage':'complete'});print(json.dumps(result),flush=True)

def main():
    DEST.mkdir(parents=True,exist_ok=True);selection=get_selection();frozen=freeze(selection);evaluate(frozen)

if __name__=='__main__':main()
