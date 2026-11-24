"""Feature and EBM routines from the study code, with portable imports."""
import numpy as np
from scipy.special import expit


SETTINGS = dict(interactions=64, max_bins=256, max_interaction_bins=32, outer_bags=2, inner_bags=0,
                learning_rate=.03, max_rounds=2000, early_stopping_rounds=80, min_samples_leaf=100,
                reg_lambda=10., max_leaves=3, n_jobs=2, smoothing_rounds=50, interaction_smoothing_rounds=50)


def folds(users):
    return (((np.asarray(users, np.int64)*1103515245+12345) % 2147483647) % 10).astype(np.int8)


def difficulty(q,c,y,use,nq,nc):
    yy=y[use]; p=float(np.clip(yy.mean(),1e-4,1-1e-4))
    def rates(ids,n):
        cnt=np.bincount(ids[use],minlength=n); sums=np.bincount(ids[use],weights=yy,minlength=n)
        pp=np.clip((sums+20*p)/(cnt+20),1e-4,1-1e-4)
        return np.log((1-pp)/pp).astype(np.float32)
    return rates(q,nq), rates(c,nc), float(np.log((1-p)/p))


def apply_difficulty(q,c,stats):
    a,b,g=stats
    return np.where(q>0,a[q],np.where(c>0,b[c],g)).astype(np.float32)


def sigmoid(x):return 1./(1.+np.exp(-x))


def feature_row(f, names):
    y=f["answers"];skills=f["skills"];items=f["items"];b=f["difficulty"]
    raw=np.column_stack((f["behavior"],y,f["measurement_missing"]))
    z=((raw-f["center"])/f["scale"]).astype(np.float32)
    n=len(y);hist=[]
    for decay in (None,.5,.9,.98):
        out=np.zeros_like(z,dtype=np.float32)
        if decay is None:out[1:]=z[:-1]
        else:
            state=np.zeros(z.shape[1],dtype=float)
            for t in range(1,n):
                state=(1-decay)*z[t-1].astype(float)+decay*state
                out[t]=state
        hist.append(out)
    prefix=np.zeros((n,18),dtype=np.float32)
    for j,keys in enumerate((np.ones(n,int),skills,items)):
        states={}
        for t,key in enumerate(keys):
            if key==0:continue
            count,success,last,at=states.get(int(key),(0,0,.5,t))
            failure=count-success
            prefix[t,6*j:6*j+6]=(np.log1p(success),np.log1p(failure),np.log((success+2)/(failure+2)),last-.5,np.log1p(t-at) if count else 0,count>0)
            states[int(key)]=(count+1,success+float(y[t]),float(y[t]),t)
    def trace_state():
        return dict(theta=np.zeros(3),success=np.zeros(3),expected=np.zeros(3),weight=np.zeros(3),all_success=0.,all_expected=0.,count=0.)
    def belief_state():
        return dict(mean=np.zeros(3),variance=np.ones(3),last=0.)
    traces=[{}, {}, {}];beliefs=[{}, {}, {}]
    trace=np.zeros((n,28),np.float32);belief=np.zeros((n,21),np.float32)
    eta=np.array([.05,.2,.8]);decays=np.array([.5,.9,.98]);process=np.array([.001,.02,.1])
    for t in range(n):
        keys=(1,int(skills[t]),int(items[t]))
        for h,key in enumerate(keys):
            if key==0:continue
            ts=traces[h].setdefault(key,trace_state());bs=beliefs[h].setdefault(key,belief_state())
            s,e,w=ts["success"],ts["expected"],ts["weight"]
            gap=np.log((s+1)/(w-s+1))-np.log((e+1)/(w-e+1))
            S,E,N=ts["all_success"],ts["all_expected"],ts["count"]
            trace[t,h*7:h*7+7]=np.r_[ts["theta"],gap,np.log((S+1)/(N-S+1))-np.log((E+1)/(N-E+1))]
            belief[t,h*7:h*7+7]=np.r_[bs["mean"],bs["variance"],bs["last"]]
        trace[t,21]=b[t]
        trace[t,22]=traces[0][1]["theta"][1]*np.tanh(float(b[t]))
        trace[t,23]=traces[0][1]["theta"][1]*traces[1][keys[1]]["theta"][1] if keys[1]>0 else 0
        if t:
            trace[t,24]=keys[1]>0 and skills[t]==skills[t-1]
            trace[t,25]=keys[2]>0 and items[t]==items[t-1]
            trace[t,26]=trace[t,24]*(y[t-1]-.5)
            trace[t,27]=b[t]-b[t-1]
        # Only after emitting t do we incorporate the answer at t.
        for h,key in enumerate(keys):
            if key==0:continue
            ts=traces[h][key];bs=beliefs[h][key]
            ts["theta"]+=eta*(float(y[t])-sigmoid(ts["theta"]-float(b[t])))
            ts["success"]=decays*ts["success"]+y[t]
            ts["expected"]=decays*ts["expected"]+sigmoid(-float(b[t]))
            ts["weight"]=decays*ts["weight"]+1.
            ts["all_success"]+=float(y[t]);ts["all_expected"]+=sigmoid(-float(b[t]));ts["count"]+=1
            prior=np.minimum(4.,bs["variance"]+process)
            p=sigmoid(np.clip(bs["mean"]-float(b[t]),-25,25));I=p*(1-p)
            bs["variance"]=1./(1./prior+I)
            bs["last"]=float((y[t]-p[1])/np.sqrt(max(.01,I[1]+prior[1]*I[1]*I[1])))
            bs["mean"]=np.clip(bs["mean"]+bs["variance"]*(y[t]-p),-8,8)
    all_x=np.column_stack((*hist,prefix,trace,np.log1p(np.arange(n)),skills>0,belief)).astype(np.float32)
    return all_x[-1,f["retained_raw_columns"]]


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
