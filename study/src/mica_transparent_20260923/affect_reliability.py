"""Transparent detector forecasts with student-held-out residual shrinkage.

Detector probabilities are weak labels. The shrinkage factor is a regression
coefficient, not the probability that a learner's psychological state is true.
The entire branch is train/validation only; no final-test paths are used.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time
import numpy as np
from core import OUT,data,write,sha,STATES,cognitive,validation_metrics
from gam_search import features,interactions,export_tables,table_terms,predict_tables,ExplainableBoostingClassifier
from interpret.glassbox import ExplainableBoostingRegressor
from uncertainty_search import NAMES

DEST=OUT/'affect_reliability_validation'
HISTORY_BINS=np.array([10,50,100,200,400,800,1600])

def fold(d):return ((d['users']*1103515245+12345)%2147483647)%10
def group(d):return np.searchsorted(HISTORY_BINS,d['history'],side='right')
def matrix(d,belief):
    x,names=features(d,include_state=False)
    return np.column_stack((x,belief)),names+NAMES
def regression(x,table):
    result=np.empty(len(x))
    for a in range(0,len(x),40000):
        b=min(a+40000,len(x));result[a:b]=float(table['intercept'].ravel()[0])+table_terms(x[a:b],table).sum(1)
    return result
def scores(y,p):
    mse=np.mean((y-p)**2,axis=0);var=np.var(y,axis=0)
    return {'rmse':float(np.sqrt(mse.mean())),'rmse_by_state':np.sqrt(mse).tolist(),'r2_by_state':(1-mse/var).tolist()}
def shrink(mu,raw,truth,groups):
    # Each history group has thousands of held-out events. The +1 in the
    # denominator regularizes unstable residual coefficients toward zero.
    rho=np.zeros((8,4));counts=[]
    for g in range(8):
        m=groups==g;delta=raw[m]-mu[m]
        rho[g]=np.clip(np.sum(delta*(truth[m]-mu[m]),axis=0)/(np.sum(delta*delta,axis=0)+1.),0.,1.)
        counts.append(int(m.sum()))
    return rho,counts

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'results.json').exists():raise FileExistsError('completed branch')
    protocol={'created':time.time(),'source_sha256':sha(__file__),'test_access':False,
        'hypothesis':'Forecast detector variation beyond observed-history means; use held-out students to suppress unreliable residuals.',
        'fit_students':'folds 3-9; fold 2 for EBM early stopping; fold 1 for shrinkage; fold 0 for internal audit',
        'frozen_feature_caveat':'Existing training-only item difficulty uses all training responses; the internal audit is not an independently preprocessed cohort.',
        'state_models':'One 1D/2D additive regressor per detector, same fixed training settings, clip after history mean plus predicted residual.',
        'selection':'State model chosen per dimension by internal fold-0 RMSE; validation used for reporting and downstream response-model development only.',
        'state_candidates':['history_mean','unshrunk_additive_residual','reliability_shrunk_residual'],
        'claim_boundary':'Shrinkage, additive regression and approximate Bayesian filtering are existing methods; no novelty or real-emotion validity assumed.'}
    write(DEST/'protocol.json',protocol)
    tr=data('train');va=data('val');bt=np.load(OUT/'uncertainty_validation/train_belief.npy',mmap_mode='r');bv=np.load(OUT/'uncertainty_validation/val_belief.npy',mmap_mode='r')
    ft=fold(tr);gt=group(tr);gv=group(va)
    yt=np.load(OUT/'cache/train/truth.npy',mmap_mode='r');yv=np.load(OUT/'cache/val/truth.npy',mmap_mode='r')
    rng=np.random.default_rng(20260925)
    fitix=np.sort(rng.choice(np.flatnonzero(ft>=2),600000,replace=False))
    calix=np.flatnonzero(ft==1);auditix=np.flatnonzero(ft==0)
    np.savez_compressed(DEST/'sample.npz',fit=fitix,calibration=calix,audit=auditix)
    means=np.array([np.mean(yt[(ft>=3)&(gt==g)],axis=0) for g in range(8)])
    fit={k:v[fitix] for k,v in tr.items()};x,names=matrix(fit,bt[fitix]);xv,_=matrix(va,bv)
    xc,_=matrix({k:v[calix] for k,v in tr.items()},bt[calix]);xa,_=matrix({k:v[auditix] for k,v in tr.items()},bt[auditix])
    pc=np.empty((len(calix),4));pa=np.empty((len(auditix),4));pv=np.empty((len(yv),4))
    pairs=[('student_belief_last_innovation','behavior_history_0'),('log_observed_history','behavior_history_0'),
           ('student_belief_medium','training_item_difficulty'),('behavior_history_0','behavior_history_8')]
    pairix=[(names.index(a),names.index(b)) for a,b in pairs]
    bags=np.where(ft[fitix]==2,-1,1).astype(np.int8)[:,None]
    for j,state in enumerate(STATES):
        start=time.time();print('FIT AFFECT',state,flush=True);write(DEST/'status.json',{'stage':'state_fit','state':state})
        model=ExplainableBoostingRegressor(feature_names=names,feature_types=['continuous']*len(names),
            interactions=pairix,max_bins=64,max_interaction_bins=16,outer_bags=1,inner_bags=0,
            learning_rate=.03,max_rounds=1200,early_stopping_rounds=60,min_samples_leaf=150,reg_lambda=10.,
            max_leaves=3,n_jobs=2,random_state=42,smoothing_rounds=50,interaction_smoothing_rounds=50)
        model.fit(x,yt[fitix,j]-means[gt[fitix],j],bags=bags)
        path=DEST/f'{state}_tables.npz';export_tables(model,path);table=dict(np.load(path))
        pc[:,j]=np.clip(means[gt[calix],j]+regression(xc,table),0,1)
        pa[:,j]=np.clip(means[gt[auditix],j]+regression(xa,table),0,1)
        pv[:,j]=np.clip(means[gv,j]+regression(xv,table),0,1)
        err=float(np.max(np.abs(regression(xa[:1000],table)-model.predict(xa[:1000]))));assert err<1e-10
        write(DEST/f'{state}.json',{'seconds':time.time()-start,'table_error':err,'sha256':sha(path)})
    rho,counts=shrink(means[gt[calix]],pc,yt[calix],gt[calix])
    estimates_a=[means[gt[auditix]],pa,means[gt[auditix]]+rho[gt[auditix]]*(pa-means[gt[auditix]])]
    estimates_v=[means[gv],pv,means[gv]+rho[gv]*(pv-means[gv])]
    labels=['history_mean','unshrunk_additive_residual','reliability_shrunk_residual']
    audit_scores={label:scores(yt[auditix],p) for label,p in zip(labels,estimates_a)}
    choice=np.argmin(np.array([audit_scores[label]['rmse_by_state'] for label in labels]),axis=0)
    selected=np.column_stack([estimates_v[choice[j]][:,j] for j in range(4)])
    report={'internal_audit':audit_scores,'choice_per_state':[labels[c] for c in choice],
        'validation':{label:scores(yv,p) for label,p in zip(labels,estimates_v)},
        'selected_validation':scores(yv,selected),'old_detector_validation':scores(yv,va['state']),
        'rho':rho.tolist(),'calibration_counts':counts,'test_access':False}
    np.savez_compressed(DEST/'state_parameters.npz',means=means,rho=rho,choice=choice,history_bins=HISTORY_BINS)
    np.save(DEST/'validation_state.npy',selected);write(DEST/'state_results.json',report);print(json.dumps(report),flush=True)
    # Predict training rows for the downstream fixed response comparison. These
    # are not cross-fitted predictions; do not claim double machine learning.
    pt=np.empty((len(yt),4),np.float32)
    tables=[dict(np.load(DEST/f'{s}_tables.npz')) for s in STATES]
    for a in range(0,len(yt),60000):
        b=min(a+60000,len(yt));batch={k:v[a:b] for k,v in tr.items()};xx,_=matrix(batch,bt[a:b]);mu=means[gt[a:b]]
        for j in range(4):
            raw=np.clip(mu[:,j]+regression(xx,tables[j]),0,1)
            pt[a:b,j]=mu[:,j] if choice[j]==0 else raw if choice[j]==1 else mu[:,j]+rho[gt[a:b],j]*(raw-mu[:,j])
    np.save(DEST/'train_state.npy',pt)
    # The matched response budget and filter features are fixed before scoring.
    sample=np.load(OUT/'uncertainty_validation/sample_1800k.npy')
    fit={k:v[sample] for k,v in tr.items()};xx,names=matrix(fit,bt[sample]);vv,_=matrix(va,bv)
    ot=np.asarray(cognitive(42,'train')[sample]);ov=np.asarray(cognitive(42,'val'))
    response={}
    for kind in ('new_affect','no_affect'):
        print('FIT RESPONSE',kind,flush=True);write(DEST/'status.json',{'stage':'response_fit','candidate':kind})
        xh=xx;vh=vv;nh=list(names)
        if kind=='new_affect':xh=np.column_stack((xx,pt[sample]));vh=np.column_stack((vv,selected));nh+=['affect_'+s for s in STATES]
        pairs=interactions(nh)
        pairs += [(nh.index(a),nh.index(b)) for a,b in [('student_belief_medium','training_item_difficulty'),
            ('skill_belief_medium','training_item_difficulty'),('student_belief_medium','student_belief_variance_medium'),
            ('skill_belief_medium','skill_belief_variance_medium')]]
        bags=np.column_stack([np.where(ft[sample]==j,-1,1) for j in (0,1)]).astype(np.int8)
        model=ExplainableBoostingClassifier(feature_names=nh,feature_types=['continuous']*len(nh),interactions=pairs,
            max_bins=128,max_interaction_bins=16,outer_bags=2,inner_bags=0,learning_rate=.03,max_rounds=2000,
            early_stopping_rounds=80,min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=2,
            random_state=42,smoothing_rounds=50,interaction_smoothing_rounds=50)
        model.fit(xh,fit['target'],bags=bags,init_score=ot)
        path=DEST/f'{kind}_tables.npz';export_tables(model,path);pred=predict_tables(vh,ov,dict(np.load(path)))
        err=float(np.max(np.abs(pred[:1000]-model.predict_proba(vh[:1000],init_score=ov[:1000])[:,1])));assert err<1e-10
        np.save(DEST/f'{kind}_validation.npy',pred);response[kind]={'metrics':validation_metrics(va,pred),'table_error':err,'sha256':sha(path)}
        write(DEST/f'{kind}.json',response[kind]);print(json.dumps({kind:response[kind]}),flush=True)
    write(DEST/'results.json',{'state':report,'response':response,'test_access':False,'state_predictions_cross_fitted':False})
    write(DEST/'status.json',{'stage':'complete'})

if __name__=='__main__':main()
