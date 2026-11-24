"""Validation-only filter extension; the already evaluated model is immutable.

Fixed feature and budget controls separate the new state update from a larger
readout budget. No test arrays, predictions, or labels are opened by this file.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import ctypes,json,subprocess,time
import numpy as np
from core import ROOT,OUT,SRC,DATA,data,positions,cognitive,validation_metrics,write,sha
from gam_search import features,interactions,export_tables,predict_tables,ExplainableBoostingClassifier

DEST=OUT/'uncertainty_validation'
NAMES=[f'{g}_belief_{n}' for g in ('student','skill','item') for n in (
    'slow','medium','fast','variance_slow','variance_medium','variance_fast','last_innovation')]

def library():
    lib=DEST/'uncertainty.so';source=SRC/'uncertainty_features.cpp'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).mica_beliefs
    fn.argtypes=[ctypes.c_int64,np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),
        ctypes.c_int32,ctypes.c_int32,np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')]
    fn.restype=ctypes.c_int;return fn

def calculate(fn,offsets,skills,items,answers,difficulty,ns,ni):
    arrays=[np.ascontiguousarray(a,dtype=t) for a,t in zip((offsets,skills,items,answers,difficulty),
        (np.int64,np.int32,np.int32,np.float32,np.float32))]
    result=np.zeros((len(answers),21),np.float32)
    assert fn(len(offsets)-1,*arrays,ns,ni,result)==0
    return result

def reference(offsets,skills,items,answers,difficulty):
    result=np.zeros((len(answers),21),np.float64)
    for u in range(len(offsets)-1):
        states={('student',0):[np.zeros(3),np.ones(3),0.]}
        for t in range(offsets[u],offsets[u+1]):
            for g,k,base in [('student',0,0),('skill',int(skills[t]),7),('item',int(items[t]),14)]:
                if g!='student' and k==0:continue
                key=(g,k)
                if key not in states:states[key]=[np.zeros(3),np.ones(3),0.]
                m,v,last=states[key];result[t,base:base+7]=np.r_[m,v,last]
                prior=np.minimum(4.,v+np.array([.001,.02,.1]));p=1/(1+np.exp(-np.clip(m-float(difficulty[t]),-25,25)))
                fisher=p*(1-p);v=1/(1/prior+fisher)
                last=(float(answers[t])-p[1])/np.sqrt(max(.01,fisher[1]+prior[1]*fisher[1]**2))
                m=np.clip(m+v*(float(answers[t])-p),-8,8);states[key]=[m,v,last]
    return result

def audit(fn):
    rng=np.random.default_rng(9023);o=np.array([0,37,91,150]);k=rng.integers(0,8,150);q=rng.integers(0,11,150)
    a=rng.integers(0,2,150).astype(np.float32);b=rng.normal(size=150).astype(np.float32)
    r=calculate(fn,o,k,q,a,b,8,11);ref=reference(o,k,q,a,b)
    err=float(np.max(np.abs(r-ref)));assert err<5e-7,err
    changed=a.copy();changed[20:37]=1-changed[20:37]
    z=calculate(fn,o,k,q,changed,b,8,11)
    assert np.array_equal(z[:21],r[:21]);assert np.array_equal(z[37:],r[37:])
    assert not r[k==0,7:14].any();assert not r[q==0,14:21].any()
    write(DEST/'audit.json',{'passed':True,'independent_reference_max_error':err,
        'target_and_future_isolation':True,'student_isolation':True,'unknown_concept_state_zero':True})

def prepare(fn,split):
    if split not in ('train','val'):raise ValueError('test access disabled')
    dest=DEST/f'{split}_belief.npy'
    if dest.exists():return np.load(dest,mmap_mode='r')
    m=json.loads((DATA/'metadata.json').read_text())
    args=[np.load(DATA/f'{split}_{name}.npy') for name in ('offsets','skills','problem_index','correct','problem_difficulty')]
    result=calculate(fn,*args,m['num_skills_with_padding'],m['num_problems_with_padding'])[positions(split)]
    np.save(dest,result);return result

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'results.json').exists():raise FileExistsError('completed validation branch')
    configs=[('belief_600k',600000,64,True),('budget_1800k',1800000,128,False),('belief_1800k',1800000,128,True)]
    protocol={'created':time.time(),'source_sha256':sha(__file__),'filter_sha256':sha(SRC/'uncertainty_features.cpp'),
        'purpose':'Test whether explicit evidence-dependent cognitive update adds validation performance, with a matched larger-budget control.',
        'configs':configs,'fixed_process_variance':[.001,.02,.1],'test_access':False,
        'selection':'development validation AUC, report check AUC; no additional historical test evaluation in this branch',
        'novelty':'Logistic approximate Bayesian filtering and GA2M are existing methods. Their combination alone is not claimed as a new algorithm.',
        'detector':'unchanged; this experiment alone makes no new affect-validity claim'}
    write(DEST/'protocol.json',protocol);fn=library();audit(fn)
    beliefs={s:prepare(fn,s) for s in ('train','val')};va=data('val');ov=np.asarray(cognitive(42,'val'))
    sample_big=np.sort(np.random.default_rng(20260924).choice(len(data('train')['target']),1800000,replace=False))
    np.save(DEST/'sample_1800k.npy',sample_big);rows={}
    for kind,count,bins,use in configs:
        start=time.time();print('FIT',kind,flush=True);write(DEST/'status.json',{'stage':'fitting','candidate':kind})
        sample=np.load(OUT/'gam_search/sample.npy') if count==600000 else sample_big
        tr=data('train',sample);x,names=features(tr);xv,_=features(va)
        if use:x=np.column_stack((x,beliefs['train'][sample]));xv=np.column_stack((xv,beliefs['val']));names=names+NAMES
        pairs=interactions(names)
        if use:
            extra=[('student_belief_medium','training_item_difficulty'),('skill_belief_medium','training_item_difficulty'),
                ('student_belief_medium','student_belief_variance_medium'),('skill_belief_medium','skill_belief_variance_medium')]
            pairs += [(names.index(a),names.index(b)) for a,b in extra]
        fold=((tr['users']*1103515245+12345)%2147483647)%10
        bags=np.column_stack([np.where(fold==j,-1,1) for j in (0,1)]).astype(np.int8)
        model=ExplainableBoostingClassifier(feature_names=names,feature_types=['continuous']*len(names),interactions=pairs,
            max_bins=bins,max_interaction_bins=16,outer_bags=2,inner_bags=0,learning_rate=.03,max_rounds=2000,
            early_stopping_rounds=80,min_samples_leaf=100,reg_lambda=10.,max_leaves=3,n_jobs=3,
            random_state=42,smoothing_rounds=50,interaction_smoothing_rounds=50)
        model.fit(x,tr['target'],bags=bags,init_score=np.asarray(cognitive(42,'train')[sample]))
        path=DEST/f'{kind}_tables.npz';export_tables(model,path);table=dict(np.load(path));pred=predict_tables(xv,ov,table)
        err=float(np.max(np.abs(pred[:5000]-model.predict_proba(xv[:5000],init_score=ov[:5000])[:,1])));assert err<1e-10
        np.save(DEST/f'{kind}_validation.npy',pred)
        rows[kind]={'metrics':validation_metrics(va,pred),'seconds':time.time()-start,'table_error':err,
            'tables_sha256':sha(path),'num_terms':len(model.term_features_)}
        write(DEST/f'{kind}.json',rows[kind]);print(json.dumps({'candidate':kind,**rows[kind]}),flush=True)
    write(DEST/'results.json',{'candidates':rows,'test_access':False,
        'best_validation_development':max(rows,key=lambda k:rows[k]['metrics']['development']['auc'])})
    write(DEST/'status.json',{'stage':'complete'})

if __name__=='__main__':main()
