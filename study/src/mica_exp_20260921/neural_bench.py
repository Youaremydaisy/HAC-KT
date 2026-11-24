"""Matched-target, train/validation-only screening using upstream KT models."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from study_common import ROOT,OUT,DATA12,DATA17,read,write,sha,positions,metrics

class Windows(Dataset):
    def __init__(self,data,split,limit=0):
        self.meta=read(data/'metadata.json');self.length=self.meta['max_seq_len']
        self.windows=np.load(data/f'{split}_windows.npy')
        if limit:self.windows=self.windows[:limit]
        self.offsets=np.load(data/f'{split}_offsets.npy')
        self.arrays={k:np.load(data/f'{split}_{v}.npy',mmap_mode='r') for k,v in [('q','problem_index'),('c','skills'),('r','correct'),('b','behavior_features')]}
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):
        student,start,length=self.windows[i];a=self.offsets[student]+start;b=a+length
        result={}
        for k,x in self.arrays.items():
            shape=(self.length,8) if k=='b' else (self.length,)
            y=np.zeros(shape,np.float32 if k=='b' else np.int64);y[:length]=x[a:b];result[k]=torch.from_numpy(y)
        result['mask']=torch.arange(self.length-1)<int(length-1)
        return result

def make_model(name,meta,device):
    nc=meta['num_skills_with_padding'];nq=meta['num_problems_with_padding'];length=meta['max_seq_len']
    if name=='dkt':
        from pykt.models.dkt import DKT
        return DKT(nc,64,dropout=.1).to(device)
    if name=='dktplus':
        from pykt.models.dkt_plus import DKTPlus
        return DKTPlus(nc,64,.1,.01,.1,dropout=.1).to(device)
    if name=='dkvmn':
        from pykt.models.dkvmn import DKVMN
        return DKVMN(nc,64,32,dropout=.1).to(device)
    if name=='sakt':
        from pykt.models.sakt import SAKT
        return SAKT(nc,length,64,4,.1,num_en=2).to(device)
    if name=='simplekt':
        from pykt.models.simplekt import simpleKT
        return simpleKT(nc,nq,64,2,.1,d_ff=128,num_attn_heads=4,final_fc_dim=128,final_fc_dim2=64,seq_len=length).to(device)
    if name=='akt':
        from pykt.models.akt import AKT
        return AKT(nc,nq,64,2,.1,d_ff=128,num_attn_heads=4,final_fc_dim=128).to(device)
    if name=='dtransformer':
        from pykt.models.dtransformer import DTransformer
        model=DTransformer(nc,nq,d_model=64,d_ff=128,num_attn_heads=4,n_know=8,n_blocks=3,dropout=.1,lambda_cl=.1,proj=True,emb_type='qid_cl').to(device)
        embedding=model.embedding;cl_loss=model.get_cl_loss
        def safe_embedding(q,r,pid=None):
            qemb,remb,_,diff=embedding(q.clamp_min(0),r.clamp_min(0),pid.clamp_min(0) if pid is not None else None)
            return qemb,remb,(r>=0).sum(1),diff
        model.embedding=safe_embedding
        def safe_cl(q,r,pid=None):
            if int((r>=0).sum(1).min())<5:return model.get_loss(q,r,pid)
            return cl_loss(q,r,pid)
        model.get_cl_loss=safe_cl
        return model
    if name=='lbkt':
        path=ROOT/'third_party/eduktm_lbkt_20260921/model.py'
        spec=importlib.util.spec_from_file_location('upstream_lbkt',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        data=DATA12 if meta['max_seq_len']==128 else DATA17
        q=np.load(data/'train_problem_index.npy');c=np.load(data/'train_skills.npy')
        matrix=np.full((nq+10,nc),.01,np.float32)
        matrix[q,c]=1.;matrix[0]=.01;matrix[0,0]=1.
        return module.Recurrent(nq,32,2,32,.1,16,nc,64,torch.tensor(matrix,device=device)).to(device)
    raise ValueError(name)

def predict(model,name,b,behavior_center,behavior_scale,training=False):
    q,c,r=b['q'],b['c'],b['r'];mask=b['mask']
    reg=0.
    if name in ['dkt','dktplus']:
        all_p=model(c[:,:-1],r[:,:-1]);p=all_p.gather(-1,c[:,1:,None]).squeeze(-1)
        if training and name=='dktplus':
            current=all_p.gather(-1,c[:,:-1,None]).squeeze(-1)
            reg=.1*F.binary_cross_entropy(current[mask],r[:,:-1][mask].float())
            smooth=all_p[:,1:]-all_p[:,:-1];m=mask[:,1:]&mask[:,:-1]
            if m.any():reg=reg+.01*smooth[m].abs().mean()+.1*smooth[m].square().mean()
    elif name=='dkvmn':p=model(c,r)[:,1:]
    elif name=='sakt':p=model(c[:,:-1],r[:,:-1],c[:,1:])
    elif name=='simplekt':
        d={'qseqs':q[:,:-1],'shft_qseqs':q[:,1:],'cseqs':c[:,:-1],'shft_cseqs':c[:,1:],'rseqs':r[:,:-1],'shft_rseqs':r[:,1:]}
        p=model(d)[:,1:]
    elif name=='akt':
        prob,reg=model(c,r,q);p=prob[:,1:]
    elif name=='dtransformer':
        valid=torch.cat([torch.ones_like(mask[:,:1]),mask],1)
        cr=torch.where(valid,r,-torch.ones_like(r));cq=torch.where(valid,c,-torch.ones_like(c));qp=torch.where(valid,q,-torch.ones_like(q))
        if training:prob,reg=model.get_cl_loss(cq,cr,qp)
        else:
            logits,*_=model.predict(cq,cr,qp);prob=torch.sigmoid(logits)
        p=prob[:,1:]
    elif name=='lbkt':
        factors=torch.sigmoid((b['b'][:,:,:3]-behavior_center)/behavior_scale)
        p=model(q,r,factors[:,:,0],factors[:,:,2],factors[:,:,1])[:,1:]
    else:raise ValueError(name)
    return p,reg

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['assist12','assist17'],required=True)
    parser.add_argument('--model',choices=['dkt','dktplus','dkvmn','sakt','simplekt','akt','dtransformer','lbkt'],required=True)
    parser.add_argument('--epochs',type=int,default=8);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();data=DATA12 if args.dataset=='assist12' else DATA17
    dest=OUT/'baselines'/args.dataset/f'{args.model}_seed{args.seed}'
    if args.smoke:dest=OUT/'smoke'/args.dataset/f'{args.model}_seed{args.seed}'
    if (dest/'results.json').exists():print('Already complete',dest);return
    dest.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=Windows(data,'train',8 if args.smoke else 0);va=Windows(data,'val',8 if args.smoke else 0)
    model=make_model(args.model,tr.meta,device)
    stats=np.load(data/'train_behavior_features.npy',mmap_mode='r')[:,:3]
    center=torch.tensor(np.mean(stats,axis=0,dtype=np.float64),device=device,dtype=torch.float32)
    scale=torch.tensor(np.maximum(np.std(stats,axis=0,dtype=np.float64),.01),device=device,dtype=torch.float32)
    batch_size=64 if args.model in ['lbkt','dtransformer','dkvmn'] else 128
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    source=Path(inspect.getfile(type(model)))
    protocol={'model':args.model,'seed':args.seed,'dataset':args.dataset,'data_metadata_sha256':sha(data/'metadata.json'),
              'source_path':str(source),'source_sha256':sha(source),'runner_sha256':sha(Path(__file__)),
              'epochs_max':args.epochs,'patience':3,'emb_size':32 if args.model=='lbkt' else 64,'batch_size':batch_size,
              'learning_rate':.001,'dropout':.1,'selection':'validation AUC, strict best; test never loaded',
              'test_access':False,'parameter_count':sum(p.numel() for p in model.parameters()),'train_windows':len(tr),'val_windows':len(va),
              'lbkt_adaptation':'Official recurrent architecture; Q links from training only; log-behavior factors standardized with training statistics then sigmoid. Practical adaptation, not paper-optimal reproduction.' if args.model=='lbkt' else None,
              'dtransformer_adapter':'Clamp padded embedding indices to zero while retaining true lengths; short batches use non-contrastive regularization path; target loss still masks padding.' if args.model=='dtransformer' else None,
              'scope':'Finite single-configuration screen, no fivefold claim; original population and all target masks retained.'}
    write(dest/'protocol.json',protocol)
    sample=next(iter(DataLoader(va,batch_size=2)))
    sample={k:v.to(device) for k,v in sample.items()};model.eval()
    with torch.no_grad():
        p,_=predict(model,args.model,sample,center,scale)
        alternate={k:v.clone() for k,v in sample.items()}
        alternate['r'][:,6:]=1-alternate['r'][:,6:];alternate['b'][:,6:]+=5
        other,_=predict(model,args.model,alternate,center,scale)
        leak_error=float((p[:,:6]-other[:,:6]).abs().max())
        assert leak_error<1e-6,(args.model,'future leakage',leak_error)
        assert torch.isfinite(p).all()
    model.train();p,reg=predict(model,args.model,sample,center,scale,True)
    loss=F.binary_cross_entropy(p[sample['mask']],sample['r'][:,1:][sample['mask']].float())+reg
    assert torch.isfinite(loss);loss.backward();optimizer.zero_grad()
    write(dest/'preflight.json',{'passed':True,'future_label_and_behavior_isolation_max_error':leak_error,'finite_backward':True})

    def collect():
        model.eval();targets=[];probs=[];started=time.time()
        with torch.no_grad():
            for b in DataLoader(va,batch_size=batch_size,shuffle=False):
                b={k:v.to(device) for k,v in b.items()};p,_=predict(model,args.model,b,center,scale)
                targets.append(b['r'][:,1:][b['mask']].cpu().numpy());probs.append(p[b['mask']].cpu().numpy())
        y=np.concatenate(targets);p=np.concatenate(probs)
        if not args.smoke:np.testing.assert_array_equal(y,np.load(data/'val_correct.npy',mmap_mode='r')[positions(data,'val')])
        return y,p,time.time()-started

    best=-1.;stale=0;history=[];started=time.time()
    for epoch in range(1,(1 if args.smoke else args.epochs)+1):
        model.train();start=time.time();loss_sum=0.;n=0
        for b in DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=0):
            b={k:v.to(device) for k,v in b.items()};optimizer.zero_grad(set_to_none=True)
            p,reg=predict(model,args.model,b,center,scale,True)
            loss=F.binary_cross_entropy(p[b['mask']],b['r'][:,1:][b['mask']].float())+reg
            assert torch.isfinite(loss),args.model
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach());n+=1
        y,p,seconds=collect();m=metrics(y,p);previous=best
        if m['auc']>best:best=m['auc'];torch.save(model.state_dict(),dest/'best.pt');np.savez_compressed(dest/'val_predictions.npz',target=y,full=p)
        stale=0 if m['auc']>previous+1e-5 else stale+1
        row={'epoch':epoch,'loss':loss_sum/n,'validation':m,'epoch_seconds':time.time()-start,'inference_seconds':seconds};history.append(row)
        write(dest/'progress.json',{'best_auc':best,'history':history,'elapsed_seconds':time.time()-started})
        print(json.dumps({'model':args.model,'dataset':args.dataset,**row}),flush=True)
        if stale>=3:break
    best_row=max(history,key=lambda row:row['validation']['auc'])
    write(dest/'results.json',{'protocol':protocol,'best_epoch':best_row['epoch'],'validation':best_row['validation'],
                             'history':history,'elapsed_seconds':time.time()-started,'new_test_evaluation':False,'preflight_passed':True})

if __name__=='__main__':
    main()
