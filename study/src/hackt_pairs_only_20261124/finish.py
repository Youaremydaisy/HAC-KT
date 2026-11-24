"""Freeze nine completed fits, evaluate the original targets and compare the four corrections."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import fcntl
import json
import os
import subprocess
import sys
import time
import experiment as e
import numpy as np

def write_rows(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def intervals(d):
    from sklearn.metrics import roc_auc_score
    path=e.folder(d,42)/"test.npz"
    with np.load(path) as z:
        y=z["target"]
        students,g=np.unique(z["student_id"],return_inverse=True)
        scores=[np.clip(z[k].astype(float),1e-7,1-1e-7) for k in ("full","pairs_only","mains")]
    n=len(students)
    lib=e.ROOT/"reports/hackt_bd2006_extension_20261123/statistics/weighted_auc.so"
    fun=ctypes.CDLL(str(lib)).cluster_auc
    ar=np.ctypeslib.ndpointer
    fun.argtypes=[ctypes.c_int64,ar(np.float64,flags="C_CONTIGUOUS"),ar(np.uint8,flags="C_CONTIGUOUS"),ar(np.int32,flags="C_CONTIGUOUS"),ar(np.int32,flags="C_CONTIGUOUS")]
    fun.restype=ctypes.c_double
    arrays=[]
    for s in scores:
        ix=np.argsort(s,kind="stable")
        arrays.append((s[ix].copy(),y[ix].astype(np.uint8),g[ix].astype(np.int32)))
    def calculate(w):
        return np.array([fun(len(y),*a,w) for a in arrays])
    point=calculate(np.ones(n,np.int32))
    np.testing.assert_allclose(point,[roc_auc_score(y,s) for s in scores],rtol=0,atol=1e-12)
    draws=np.empty((10000,3))
    rng=np.random.default_rng(20261124)
    for i in range(len(draws)):
        w=np.bincount(rng.integers(n,size=n),minlength=n).astype(np.int32)
        draws[i]=calculate(w)
    assert np.isfinite(draws).all()
    np.savez_compressed(e.OUT/d/"auc_draws.npz",auc=draws,order=["full","pairs_only","mains"],student_ids=students)
    rows=[]
    for j,k,label in ((0,1,"full_minus_pairs"),(1,2,"pairs_minus_mains")):
        lo,hi=np.quantile(draws[:,j]-draws[:,k],[.025/6,1-.025/6])
        rows.append(dict(dataset=d,contrast=label,seed=42,point=point[j]-point[k],adjusted_low=lo,adjusted_high=hi,students=n,targets=len(y),draws=len(draws),family=6))
    return rows

def score(d):
    cmd=[sys.executable,"-Bu",str(Path(__file__).with_name("bd_driver.py")),"test"] if d=="bd2006" else [sys.executable,"-Bu",e.__file__,"evaluate",d]
    with (e.OUT/"logs"/f"test_{d}.log").open("a") as log:
        subprocess.run(cmd,stdout=log,stderr=log,check=True)
    return intervals(d)

def finish():
    lock=(e.OUT/"finish.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while any(not (e.folder(d,s)/"validation.json").exists() for d in e.DOMAINS for s in e.SEEDS):
        time.sleep(10)
    e.freeze()
    with ThreadPoolExecutor(3) as pool:
        ci=[r for group in pool.map(score,e.DOMAINS) for r in group]
    write_rows(e.OUT/"intervals.csv",ci)
    records=[]
    summary=[]
    for d in e.DOMAINS:
        for s in e.SEEDS:
            m=e.read(e.folder(d,s)/"test.json")["metrics"]
            records.extend(dict(dataset=d,seed=s,method=k,**v) for k,v in m.items())
        for k in ("base","mains","pairs_only","full"):
            rows=[r for r in records if r["dataset"]==d and r["method"]==k]
            line=dict(dataset=d,method=k,n=len(rows))
            for metric in ("auc","ap","log_loss","brier","ece10"):
                vals=[r[metric] for r in rows]
                line[metric+"_mean"]=np.mean(vals);line[metric+"_sample_sd"]=np.std(vals,ddof=1)
            summary.append(line)
    write_rows(e.OUT/"per_seed.csv",records)
    write_rows(e.OUT/"summary.csv",summary)
    paths=sorted(p for p in e.OUT.rglob("*") if p.is_file() and p.suffix in (".json",".csv",".npz") and p.name!="delivery.json")
    e.save(e.OUT/"delivery.json",dict(files={p.relative_to(e.OUT).as_posix():e.sha(p) for p in paths},
           datasets=list(e.DOMAINS),seeds=list(e.SEEDS),n_validation_fits=9,n_heads=18,n_pure_two_feature_terms_per_head=64,bootstrap_rows=len(ci),created=time.time()))
    print("DELIVERY READY",flush=True)

if __name__=="__main__":
    finish()

