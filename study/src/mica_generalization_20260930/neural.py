"""Train-only/validation-only neural runners with target-aligned frozen inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT, OUT, SRC, data_path, positions, write, sha, metrics

sys.path.insert(0,str(ROOT/'src/mica_exp_20260921'))
import neural_bench as nb

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

module('train_model',SRC/'frozen_train_model.py')
frozen=module('generalization_frozen_eckt',SRC/'frozen_train_eckt.py')

def setup(dataset,kind,seed):
    data=data_path(dataset);meta=json.loads((data/'metadata.json').read_text());device=torch.device('cuda')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    stats=np.load(data/'train_behavior_features.npy',mmap_mode='r')
    center=torch.tensor(stats.mean(0),device=device);scale=torch.tensor(np.maximum(stats.std(0),.05),device=device)
    if kind=='cognitive':
        model=frozen.ECKT(meta['num_skills_with_padding'],meta['num_problems_with_padding'],behavior_dim=8,
            mode='cognitive',history=8,use_discrimination=True,response_mode='hinge',weight_mode='dynamic',weight_tau=1.,
            behavior_gain=True,behavior_gain_hinge=True,pfa_signed_learning=True,normalize_behavior=True,
            practice_readout=True,revisit_forgetting=False,hawkes_readout=True).to(device)
        with torch.no_grad():model.behavior_center.copy_(center);model.behavior_scale.copy_(scale)
        factory=lambda split:frozen.ECKTWindowDataset(data,split,128,meta['num_skills_with_padding'],False)
        batch_size=256
    else:
        factory=lambda split:nb.Windows(data,split)
        if kind=='lbkt':
            if set(meta['absent_measurements']) & {1,2}:raise ValueError('LBKT full behavior baseline requires observed hints and attempts')
            upstream=module('generalization_lbkt',ROOT/'third_party/eduktm_lbkt_20260921/model.py')
            nc=meta['num_skills_with_padding'];nq=meta['num_problems_with_padding']
            q=np.load(data/'train_problem_index.npy');c=np.load(data/'train_skills.npy')
            qm=np.full((nq+10,nc),.01,'float32');qm[q,c]=1.;qm[0]=.01;qm[0,0]=1.
            model=upstream.Recurrent(nq,32,2,32,.1,16,nc,64,torch.tensor(qm,device=device)).to(device)
        else:model=nb.make_model(kind,meta,device)
        batch_size=64 if kind=='lbkt' else 128
    return model,factory,batch_size,device,center[:3],scale[:3]

def forward(model,kind,b,center,scale,training=False):
    if kind=='cognitive':
        out=model(b);return torch.sigmoid(out['cognitive_logits']),0.,out
    p,reg=nb.predict(model,kind,b,center,scale,training);return p,reg,None

def target_mask(kind,b):
    return (b['target_correct'],b['mask']>0) if kind=='cognitive' else (b['r'][:,1:].float(),b['mask'])

def collect(model,kind,ds,batch_size,device,center,scale):
    model.eval();ys=[];ps=[];zs=[];max_replay=0.
    with torch.no_grad():
        for b in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            b={k:v.to(device) for k,v in b.items()};p,_,out=forward(model,kind,b,center,scale)
            y,m=target_mask(kind,b);ys.append(y[m].cpu().numpy());ps.append(p[m].cpu().numpy())
            if out is not None:
                zs.append(out['cognitive_logits'][m].cpu().numpy())
                max_replay=max(max_replay,float((out['cognitive_logits']-out['cognitive_parts'].sum(-1))[m].abs().max()))
    result={'target':np.concatenate(ys),'full':np.concatenate(ps)}
    if zs:result['logit']=np.concatenate(zs)
    return result,max_replay

def preflight(model,kind,ds,device,center,scale):
    b=next(iter(DataLoader(ds,batch_size=3)));b={k:v.to(device) for k,v in b.items()}
    model.eval()
    with torch.no_grad():
        p,_,_=forward(model,kind,b,center,scale);other={k:v.clone() for k,v in b.items()}
        if kind=='cognitive':
            other['correct'][:,6:]=1-other['correct'][:,6:];other['behavior_features'][:,6:]+=7
            # Target labels are never an input to the cognitive calculation.
            other['target_correct']=1-other['target_correct']
        else:
            other['r'][:,6:]=1-other['r'][:,6:];other['b'][:,6:]+=7
        alt,_,_=forward(model,kind,other,center,scale)
        # Cognitive prediction index5 targets event6 but consumes only event5.
        error=float((p[:,:6]-alt[:,:6]).abs().max());assert error<1e-6,(kind,error)
    model.train();p,reg,_=forward(model,kind,b,center,scale,True);y,m=target_mask(kind,b)
    loss=F.binary_cross_entropy(p[m],y[m])+reg;assert torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    model.zero_grad(set_to_none=True)
    return {'future_answer_behavior_isolation_max_error':error,'finite_gradients':True}

def train(dataset,kind,seed,smoke=False):
    dest=OUT/dataset/('smoke' if smoke else 'models')/f'{kind}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists():return
    model,factory,bs,device,center,scale=setup(dataset,kind,seed);tr=factory('train');va=factory('val')
    if smoke:tr.windows=tr.windows[:8];va.windows=va.windows[:8]
    audit=preflight(model,kind,tr,device,center,scale);write(dest/'audit.json',audit)
    import inspect
    source=Path(inspect.getfile(type(model)))
    protocol={'dataset':dataset,'kind':kind,'seed':seed,'epochs_max':30,'patience':5,'batch_size':bs,
        'learning_rate':.002 if kind=='cognitive' else .001,'parameters':sum(v.numel() for v in model.parameters()),
        'model_source_sha256':sha(source),'runner_sha256':sha(__file__),'data_sha256':sha(data_path(dataset)/'metadata.json'),
        'mode':'external refit, no affect loss','test_access':False,'window':128,
        'cognitive_adaptation':'Cognitive branch from frozen source. No event gate, affect loss or wall-clock forgetting; absent time remains unavailable.',
        'history_scope':'Cognitive model carries skill success/failure prefix; AKT/LBKT are finite-window baselines'}
    write(dest/'protocol.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=1e-4) if kind=='cognitive' else torch.optim.Adam(model.parameters(),lr=.001)
    best=-1.;stale=0;rows=[];start=time.time()
    for epoch in range(1,(1 if smoke else 30)+1):
        model.train();losses=[];started=time.time()
        for b in DataLoader(tr,batch_size=bs,shuffle=True,num_workers=0):
            b={k:v.to(device) for k,v in b.items()};optimizer.zero_grad(set_to_none=True)
            p,reg,_=forward(model,kind,b,center,scale,True);y,m=target_mask(kind,b)
            loss=F.binary_cross_entropy(p[m],y[m])+reg
            if kind=='cognitive':loss=loss*1.25
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5. if kind=='cognitive' else 1.)
            optimizer.step();losses.append(float(loss.detach()))
        arrays,replay=collect(model,kind,va,bs,device,center,scale);m=metrics(arrays['target'],arrays['full'])
        if not smoke:np.testing.assert_array_equal(arrays['target'],np.load(data_path(dataset)/'val_correct.npy')[positions(data_path(dataset),'val')])
        improved=m['auc']>best+1e-5
        if m['auc']>best:
            best=m['auc'];torch.save(model.state_dict(),dest/'best.pt');np.savez_compressed(dest/'val_predictions.npz',**arrays)
        stale=0 if improved else stale+1
        row={'epoch':epoch,'loss':float(np.mean(losses)),'validation':m,'seconds':time.time()-started,'cognitive_replay_error':replay}
        rows.append(row);write(dest/'progress.json',{'history':rows,'best_auc':best});print(json.dumps({'dataset':dataset,'kind':kind,'seed':seed,**row}),flush=True)
        if stale>=5:break
    bestrow=max(rows,key=lambda r:r['validation']['auc'])
    write(dest/'results.json',{'validation':bestrow['validation'],'best_epoch':bestrow['epoch'],'history':rows,
        'at_epoch_cap':len(rows)==30,'elapsed_seconds':time.time()-start,'checkpoint_sha256':sha(dest/'best.pt'),'protocol':protocol})

def infer(dataset,kind,seed,split):
    if split=='test':assert (OUT/dataset/'freeze.json').exists()
    dest=OUT/dataset/'models'/f'{kind}_seed{seed}'
    target=dest/f'{split}_predictions.npz'
    if target.exists():return
    model,factory,bs,device,center,scale=setup(dataset,kind,seed)
    model.load_state_dict(torch.load(dest/'best.pt',map_location=device))
    arrays,replay=collect(model,kind,factory(split),bs,device,center,scale)
    ix=positions(data_path(dataset),split)
    np.testing.assert_array_equal(arrays['target'],np.load(data_path(dataset)/f'{split}_correct.npy')[ix])
    arrays['student_id']=np.load(data_path(dataset)/f'{split}_user_ids.npy')[ix]
    np.savez_compressed(target,**arrays);write(dest/f'{split}_replay.json',{'max_error':replay})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('kind',choices=['cognitive','akt','lbkt']);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true');p.add_argument('--infer',choices=['train','val','test']);a=p.parse_args()
    # Serialize duplicate requests for one model while allowing distinct models
    # to use the under-utilized GPU concurrently. Training equations are unchanged.
    import fcntl
    lockdir=OUT/a.dataset/('smoke' if a.smoke else 'models')/f'{a.kind}_seed{a.seed}'
    lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.infer:infer(a.dataset,a.kind,a.seed,a.infer)
        else:train(a.dataset,a.kind,a.seed,a.smoke)
