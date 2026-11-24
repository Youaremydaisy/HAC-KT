"""Full difficulty-and-belief cross-fit validation control.

The earlier sensitivity audit changed only the trace difficulty. This script
reuses its causal trace and recomputes the uncertainty filter with the same
student-fold difficulty, then fits the identical no-affect readout budget.
It still opens train/validation arrays only.
"""
import gc, json, time
import numpy as np
from common import ROOT, FROZEN, DEST, DATA, data, positions, cognitive, write, sha, validation_metrics
from crossfit_readout import crossfit_difficulty, student_folds
from gam_search import ExplainableBoostingClassifier, export_tables, predict_tables
from reliability_confirm import response_matrix
from uncertainty_search import library, calculate

def main():
    out=DEST/'full';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists(): print((out/'results.json').read_text());return
    started=time.time();meta=json.loads((DATA/'metadata.json').read_text());idx=positions('train')
    raw_diff=crossfit_difficulty('train');raw_sk=np.load(DATA/'train_skills.npy',mmap_mode='r');raw_it=np.load(DATA/'train_problem_index.npy',mmap_mode='r');raw_y=np.load(DATA/'train_correct.npy',mmap_mode='r');offsets=np.load(DATA/'train_offsets.npy',mmap_mode='r')
    fn=library();belief=calculate(fn,offsets,raw_sk,raw_it,raw_y,raw_diff,meta['num_skills_with_padding'],meta['num_problems_with_padding'])[idx]
    bpath=out/'train_belief_crossfit.npy';np.save(bpath,belief);del belief,raw_diff;gc.collect()
    sample=np.load(FROZEN/'contextual_capacity_validation/auto16_3000k_sample.npy');tr,va=data('train',sample),data('val');tr['trace']=np.load(DEST/'train_trace_crossfit.npy',mmap_mode='r')[sample];bt=np.load(bpath,mmap_mode='r')[sample];bv=np.load(FROZEN/'uncertainty_validation/val_belief.npy',mmap_mode='r')
    x,names=response_matrix(tr,bt,np.empty((len(tr['target']),4)),False);v,vnames=response_matrix(va,bv,np.empty((len(va['target']),4)),False);assert names==vnames
    ot=np.asarray(cognitive(42,'train')[sample]);ov=np.asarray(cognitive(42,'val'));pred=np.empty(len(v));rows=[]
    for known in (False,True):
        tag='known' if known else 'missing';tm=tr['known']==known;vm=va['known']==known;path=out/f'{tag}_tables.npz';folds=student_folds(tr['users'][tm]);bags=np.column_stack([np.where(folds==j,-1,1) for j in (0,1)]).astype(np.int8);print('FIT',tag,flush=True)
        model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),interactions=64,max_bins=256,max_interaction_bins=32,outer_bags=2,inner_bags=0,learning_rate=.03,max_rounds=2000,early_stopping_rounds=80,min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=2,random_state=42,smoothing_rounds=50,interaction_smoothing_rounds=50);model.fit(x[tm],tr['target'][tm],bags=bags,init_score=ot[tm]);export_tables(model,path);table=dict(np.load(path));pred[vm]=predict_tables(v[vm],ov[vm],table);rows.append({'known':known,'tables':str(path.relative_to(ROOT)),'sha256':sha(path),'terms':len(model.term_features_),'best_iteration':np.asarray(model.best_iteration_).tolist()});del model;gc.collect()
    np.save(out/'validation.npy',pred);frozen=np.load(FROZEN/'extended_confirmation/seed42_no_affect_validation.npy',mmap_mode='r');partial=np.load(DEST/'validation.npy',mmap_mode='r');joint=np.load(FROZEN/'extended_confirmation/seed42_joint_validation.npy',mmap_mode='r')
    result={'metrics':validation_metrics(va,pred),'partial_trace_crossfit_metrics':validation_metrics(va,partial),'frozen_no_affect_metrics':validation_metrics(va,frozen),'frozen_joint_metrics':validation_metrics(va,joint),'regimes':rows,'scope':'Both item/skill difficulty and uncertainty belief are student-fold cross-fitted for training target rows; validation uses all training students. Affect remains excluded from this prediction control and is retained as an auxiliary weak-label output.','test_access':False,'source_sha256':sha(__file__),'seconds':time.time()-started}
    result['auc_deltas_vs_frozen_no_affect']={k:result['metrics'][k]['auc']-result['frozen_no_affect_metrics'][k]['auc'] for k in ('all','development','check')};result['eligible']=bool(result['auc_deltas_vs_frozen_no_affect']['all']>=.0002 and result['auc_deltas_vs_frozen_no_affect']['development']>=0 and result['auc_deltas_vs_frozen_no_affect']['check']>=0)
    write(out/'results.json',result);write(out/'status.json',{'stage':'complete'});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
