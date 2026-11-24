"""Auditable public-data adapters. No fitting or reporting of test metrics."""
import argparse
import json
import zipfile
import io
import time
import re
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from common import ROOT, OUT, RAW, write, sha, rank, folds, data_path

BEHAVIOR = ['log_response_seconds', 'log1p_hint_count', 'log1p_attempt_count', 'log_delta_seconds',
            'prefix_correctness', 'correctness_discrepancy', 'time_efficiency_discrepancy', 'persistence_deviation']

def canonical(value):
    if pd.isna(value) or str(value).strip().lower() in ('', 'nan', 'none', '-1'): return ''
    tokens = [str(int(float(x))) for x in re.split(r'[_;]', str(value)) if x and x != '-1']
    return '_'.join(sorted(set(tokens), key=int))

def assist09():
    path = RAW/'assist09_corrected.csv'
    f = pd.read_csv(path, encoding='latin1', low_memory=False)
    count = len(f); f = f[f['correct'].isin([0,1])].copy()
    assert not f.duplicated(['user_id','order_id']).any(), 'Investigate event duplicates; never silently repeat an answer'
    out = pd.DataFrame({'user':f.user_id.astype('int64'), 'event':f.order_id.astype('int64'),
                        'item':f.problem_id.astype(str), 'skill':f.skill_id.map(canonical),
                        'answer':f.correct.astype('int8'), 'response_ms':pd.to_numeric(f.ms_first_response, errors='coerce'),
                        'hints':pd.to_numeric(f.hint_count, errors='coerce'),
                        'attempts':pd.to_numeric(f.attempt_count, errors='coerce'), 'timestamp_ms':np.nan})
    return out, {'raw_sha256':sha(path), 'raw_rows':count, 'invalid_answer_rows':count-len(f),
                 'event_key':'user_id + order_id', 'ordering':'order_id (no global timestamps available)',
                 'multi_skill_events':int(out.skill.str.contains('_').sum()),
                 'time_semantics':'milliseconds to first response', 'global_time_available':False,
                 'skill_policy':'Preserve complete sorted skill-set as atomic ID; does not estimate individual multi-skill mastery'}

def bundle_representatives(f):
    """Choose without inspecting answer or elapsed time; retain source order."""
    f = f.sort_values('timestamp',kind='stable').drop_duplicates(['solving_id','question_id']).copy()
    f['_rank'] = f['question_id'].map(lambda x: rank(x, 'bundle-question'))
    f = f.sort_values(['solving_id','_rank','question_id'], kind='stable').drop_duplicates('solving_id')
    return f.sort_values(['timestamp','solving_id'], kind='stable').drop(columns='_rank')

def visible_bundle_representatives(f):
    """Conservative sequential subset: no prior selected response is unfinished.

    KT1 does not supply explicit submit events. Timestamp+elapsed is a duration-
    based availability bound, not a measured submission timestamp. Using the
    latest bound within a bundle and its earliest presentation is conservative.
    """
    first=f.sort_values('timestamp',kind='stable').drop_duplicates(['solving_id','question_id']).copy()
    first['_valid_duration']=np.isfinite(first.elapsed_time) & (first.elapsed_time>0)
    first['_available']=first.timestamp+first.elapsed_time.clip(lower=0)
    intervals=first.groupby('solving_id').agg(_start=('timestamp','min'),_end=('_available','max'),_valid=('_valid_duration','all'))
    chosen=bundle_representatives(f).join(intervals,on='solving_id').sort_values(['_start','solving_id'],kind='stable')
    keep=[];last=-np.inf;overlap=0;invalid=0
    for i,row in chosen.iterrows():
        if not row['_valid'] or not np.isfinite(row['_end']):invalid+=1;continue
        if row['_start']<last:overlap+=1;continue
        keep.append(i);last=float(row['_end'])
    chosen=chosen.loc[keep].copy()
    chosen['timestamp']=chosen['_start']
    if len(chosen)>1:assert np.all(chosen['_start'].to_numpy()[1:]>=chosen['_end'].to_numpy()[:-1])
    return chosen,overlap,invalid

def ednet():
    path = RAW/'ednet_kt1.zip'; z = zipfile.ZipFile(path)
    candidates = [n for n in z.namelist() if re.fullmatch(r'KT1/u\d+\.csv', n)]
    selected = sorted(candidates, key=lambda n: rank(n, 'student-sample'))[:5000]
    write(OUT/'ednet/selected_students_before_labels.json', {'files':selected, 'population_files':len(candidates),
        'rule':'SHA256 filename rank, seed20260930; selected before reading any answer', 'time_unix':time.time()})
    qz = zipfile.ZipFile(RAW/'ednet_contents.zip')
    questions = pd.read_csv(qz.open('contents/questions.csv')).set_index('question_id')
    parts=[]; raw_rows=0; representatives=0; invalid=0; unknown=0; repeated_keys=0; overlap=0; unknown_duration=0
    for i, name in enumerate(selected):
        f=pd.read_csv(z.open(name)); raw_rows+=len(f)
        repeated_keys+=int(f.duplicated(['solving_id','question_id']).sum())
        f,n_overlap,n_invalid=visible_bundle_representatives(f);representatives+=len(f)
        overlap+=n_overlap;unknown_duration+=n_invalid
        f=f.join(questions[['correct_answer','tags','bundle_id']], on='question_id')
        unknown+=int(f.correct_answer.isna().sum())
        valid=f.user_answer.isin(['a','b','c','d']) & f.correct_answer.isin(['a','b','c','d'])
        invalid+=int((~valid).sum()); f=f[valid]
        u=int(re.search(r'u(\d+)\.csv', name).group(1))
        parts.append(pd.DataFrame({'user':u,'event':f.solving_id.astype('int64'), 'item':f.question_id,
            'skill':f.tags.map(canonical), 'answer':(f.user_answer==f.correct_answer).astype('int8'),
            'response_ms':f.elapsed_time, 'hints':np.nan, 'attempts':np.nan, 'timestamp_ms':f.timestamp}))
        if i%1000==0: print('READ_EDNET',i,flush=True)
    z.close()
    return pd.concat(parts,ignore_index=True), {'raw_sha256':sha(path), 'questions_sha256':sha(RAW/'ednet_contents.zip'),
        'raw_rows_in_sample':raw_rows, 'bundle_representatives':representatives,
        'discarded_same_bundle_rows':raw_rows-representatives, 'invalid_selected_answer_rows':invalid,
        'repeated_solving_question_rows':repeated_keys,
        'repeated_key_policy':'Keep earliest timestamp for question within solving_id, then select one question by ID hash; first-response evidence only, no label-based choice',
        'overlapping_group_exclusions':overlap,'unknown_duration_group_exclusions':unknown_duration,
        'availability_rule':'Each retained bundle starts no earlier than the previous retained bundle max(timestamp+elapsed_time). Groups with unknown/nonpositive durations are excluded. This is a conservative KT1 duration bound, not observed KT2 submission times.',
        'missing_question_metadata_rows':unknown, 'sampled_students':5000,
        'sample_manifest_sha256':sha(OUT/'ednet/selected_students_before_labels.json'),
        'event_key':'original user filename + solving_id', 'ordering':'timestamp then solving_id',
        'time_semantics':'elapsed milliseconds for bundle-associated question; not equated with ASSIST first-response duration',
        'global_time_available':True, 'skill_policy':'Complete tag set as atomic ID; one deterministic question per bundle attempt'}

def prefix_features(o, skill, item, y):
    """Fully vectorized pre-response sufficient statistics with explicit state keys."""
    n=len(y); user=np.repeat(np.arange(len(o)-1), np.diff(o)); t=np.arange(n)-o[user]
    result=[]
    for k in [np.ones(n,dtype=int), skill, item]:
        f=pd.DataFrame({'u':user,'k':k,'y':y.astype(float),'t':t})
        g=f.groupby(['u','k'],sort=False)
        count=g.cumcount().to_numpy(); success=g.y.cumsum().to_numpy()-y; fail=count-success
        last=g.y.shift().fillna(.5).to_numpy()-.5
        previous_t=g.t.shift().to_numpy()
        gap=np.log1p(np.where(count>0,t-previous_t,0))
        part=np.column_stack([np.log1p(success),np.log1p(fail),np.log((success+2)/(fail+2)),last,gap,count>0])
        part[k==0]=0; result.append(part)
    return np.column_stack(result).astype(np.float32)

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

def prepare(dataset):
    dest=data_path(dataset); report=OUT/dataset; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metadata.json').exists(): raise FileExistsError('Prepared dataset is immutable; create a new version for changes')
    f,audit=assist09() if dataset=='assist09' else ednet()
    lengths=f.groupby('user').size(); short=lengths[lengths<2].index
    f=f[~f.user.isin(short)].copy()
    split_salt = os.environ.get('MICA_SPLIT_SALT')
    users=sorted(f.user.unique().tolist(),key=lambda x:rank(f'{dataset}:{x}',split_salt or 'student-split'))
    inherited=OUT/'ednet_pre_availability_v1/split_students.json'
    if dataset=='ednet' and inherited.exists() and not split_salt:
        original=json.loads(inherited.read_text());splits={int(u):s for s,us in original.items() for u in us}
        assert set(users)<=set(splits)
        audit['inherited_student_split_sha256']=sha(inherited)
    else:
        n=len(users);a=int(.70*n);b=int(.85*n)
        splits={u:s for s,us in [('train',users[:a]),('val',users[a:b]),('test',users[b:])] for u in us}
    write(report/'split_students.json', {s:[u for u in users if splits[u]==s] for s in ('train','val','test')})
    audit['split_salt'] = split_salt or 'original-inherited-policy'
    f['split']=f.user.map(splits)
    # Vocabulary for trainable parameters is fitted exclusively to train students.
    train=f[f.split=='train']; learned={}; state={}
    for name in ('skill','item'):
        learned[name]={v:i+1 for i,v in enumerate(sorted(set(train[name])-{''}))}
        state[name]={v:i+1 for i,v in enumerate(sorted(set(f[name])-{''}))}
        f[name+'_index']=f[name].map(learned[name]).fillna(0).astype('int32')
        f['state_'+name]=f[name].map(state[name]).fillna(0).astype('int32')
    write(dest/'vocabulary.json', {'learned_from_training':learned,'state_keys_no_learned_parameters':state})
    ordering=['user','event'] if dataset=='assist09' else ['user','timestamp_ms','event']
    f=f.sort_values(ordering,kind='stable').reset_index(drop=True)
    tm=f.split.eq('train').to_numpy()
    response=pd.to_numeric(f.response_ms,errors='coerce').to_numpy(float)/1000.
    response[(response<=0)|(~np.isfinite(response))]=np.nan
    median=float(np.nanmedian(response[tm])); upper=float(np.nanquantile(response[tm],.999))
    missing=np.column_stack([~np.isfinite(response),~np.isfinite(f.hints.to_numpy(float)),
                             ~np.isfinite(f.attempts.to_numpy(float)),~np.isfinite(f.timestamp_ms.to_numpy(float))])
    r=np.log1p(np.clip(np.nan_to_num(response,nan=median),0,upper))
    h=np.log1p(np.clip(f.hints.fillna(0).to_numpy(float),0,50));att=np.log1p(np.clip(f.attempts.fillna(0).to_numpy(float),0,50))
    dt=(f.groupby('user').timestamp_ms.diff().fillna(0).to_numpy(float)/1000.).clip(0,7*86400)
    dt=np.log1p(dt); dt[missing[:,3]]=0
    f['r']=r; f['d']=dt
    g=f.groupby('user',sort=False); count=g.cumcount().to_numpy()
    pre_y=(g.answer.cumsum().to_numpy()-f.answer.to_numpy()+10)/(count+20)
    rtmean=float(r[tm].mean()); rtscale=max(float(r[tm].std()),.25)
    prior_rt=(g.r.cumsum().to_numpy()-r+20*rtmean)/(count+20)
    dtmean=float(dt[tm].mean()); dtscale=max(float(dt[tm].std()),.5)
    prior_dt=(g.d.cumsum().to_numpy()-dt+20*dtmean)/(count+20)
    eff=expit((r-prior_rt)/rtscale); eff[missing[:,0]]=.5
    persist=expit((dt-prior_dt)/dtscale); persist[missing[:,3]]=.5
    behavior=np.column_stack([r,h,att,dt,pre_y,np.abs(f.answer.to_numpy()-pre_y),eff,persist]).astype(np.float32)
    q=f.item_index.to_numpy(); c=f.skill_index.to_numpy(); y=f.answer.to_numpy()
    nq=len(learned['item'])+1;nc=len(learned['skill'])+1
    fullstats=difficulty(q,c,y,tm,nq,nc); diff=apply_difficulty(q,c,fullstats)
    cf=diff.copy();fold=folds(f.user.to_numpy())
    for j in range(10):
        m=tm & (fold==j); stats=difficulty(q,c,y,tm & (fold!=j),nq,nc)
        cf[m]=apply_difficulty(q[m],c[m],stats)
    np.savez_compressed(dest/'training_statistics.npz',item_difficulty=fullstats[0],skill_difficulty=fullstats[1],global_difficulty=fullstats[2])
    splitrows={}
    for split in ('train','val','test'):
        mask=f.split.eq(split).to_numpy(); sf=f.loc[mask]; counts=sf.groupby('user',sort=False).size().to_numpy()
        o=np.r_[0,np.cumsum(counts)].astype('int64'); windows=[]
        # Contiguous windows with a one-event overlap: every answer after a student's first is predicted exactly once.
        for u,length in enumerate(counts):
            for start in range(0,int(length)-1,127):windows.append((u,start,min(128,int(length)-start)))
        values={'skills':c[mask],'problem_index':q[mask],'state_skills':sf.state_skill.to_numpy('int32'),
                'state_items':sf.state_item.to_numpy('int32'),'correct':y[mask],'response':r[mask].astype('float32'),
                'delta':dt[mask].astype('float32'),'behavior_features':behavior[mask],
                'measurement_missing':missing[mask].astype('float32'),'affects':np.zeros((mask.sum(),4),'float32'),
                'behavior_affects':behavior[mask,4:],'problem_difficulty':diff[mask],
                'crossfit_difficulty':cf[mask],'user_ids':sf.user.to_numpy('int64'),
                'event_ids':sf.event.to_numpy('int64'),'offsets':o,'windows':np.asarray(windows,dtype='int64'),
                'skill_missing':sf.skill.eq('').to_numpy(),'skill_unseen':((sf.skill_index==0)&sf.skill.ne('')).to_numpy(),
                'item_unseen':sf.item_index.eq(0).to_numpy()}
        values['prefix_features']=prefix_features(o,values['state_skills'],values['state_items'],values['correct'])
        for key,val in values.items():np.save(dest/f'{split}_{key}.npy',val,allow_pickle=False)
        splitrows[split]={'students':len(counts),'events':len(sf),'scored_targets':len(sf)-len(counts),'windows':len(windows),
                          'missing_skill_events':int(values['skill_missing'].sum()),'unseen_skill_events':int(values['skill_unseen'].sum()),
                          'unseen_item_events':int(values['item_unseen'].sum())}
    audit.update(excluded_short_students=len(short),split=splitrows,response_median_seconds=median,
        response_upper_clip_seconds=upper,measurement_missing_fraction=missing.mean(0).tolist(),
        absent_feature_semantics='Zero/0.5 carriers only; missing indicators retained and globally absent channels excluded from CF readout',
        test_metrics_accessed=False,source_sha256=sha(__file__))
    write(report/'data_audit.json',audit)
    (report/'prepare_source.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    write(dest/'metadata.json',{'dataset':dataset,'max_seq_len':128,'num_skills_with_padding':nc,'num_problems_with_padding':nq,
        'num_state_skills':len(state['skill'])+1,'num_state_items':len(state['item'])+1,'behavior_feature_names':BEHAVIOR,
        'absent_measurements':[j for j in range(4) if missing[tm,j].all()], 'split':splitrows,
        'affect_labels_available':False,'split_sha256':sha(report/'split_students.json'),'audit_sha256':sha(report/'data_audit.json'),
        'raw_sha256':audit['raw_sha256'],'source_sha256':sha(__file__)})
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['assist09','ednet']);a=p.parse_args();prepare(a.dataset)
