"""Paired student bootstrap. The student-pair matrix counts AUC wins, with half credit for ties."""
import csv
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "2"
from runtime import SEEDS, ARMS, SINGLES, BASELINES, OUT, RUN, DATA, read, save, sha, setup
setup()
import numpy as np
from sklearn.metrics import roc_auc_score
from common import metrics, positions

MODELS = (*BASELINES, "pfa", *ARMS)
CONTRASTS = [("correction","full","cognitive",1), *[("single",a,"full",6) for a in SINGLES],
             ("joint","drop_all","full",1), ("direct_with_updates","full","direct_off",4),
             ("direct_without_updates","neutral_update","both_off",4),
             ("updates_with_direct","full","neutral_update",4), ("updates_without_direct","direct_off","both_off",4)]


def csv_rows(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def draws(p, y, group, weights):
    p = np.clip(np.asarray(p,float),1e-7,1-1e-7)
    n = weights.shape[1]
    neg = y == 0
    negatives, neg_group = p[neg], group[neg]
    positive_counts = np.bincount(group[~neg],minlength=n)
    negative_counts = np.bincount(neg_group,minlength=n)
    wins = np.zeros((n,n))
    for k in range(n):
        pos = np.sort(p[(group==k) & ~neg])
        less = pos.searchsorted(negatives,side="left")
        upto = pos.searchsorted(negatives,side="right")
        wins[k] = np.bincount(neg_group,weights=len(pos)-(less+upto)*0.5,minlength=n)
    denom = (weights @ positive_counts) * (weights @ negative_counts)
    assert (denom > 0).all()
    result = np.einsum("ij,ij->i", weights @ wins, weights) / denom
    assert abs(wins.sum() / (positive_counts.sum()*negative_counts.sum()) - roc_auc_score(y,p)) < 2e-12
    for i in range(3):
        assert abs(result[i] - roc_auc_score(y,p,sample_weight=weights[i,group])) < 2e-12
    return result


def item_history_analysis(folder, y, users, group, weights):
    supplement = OUT/"item_history_addendum.md"
    assert read(OUT/"item_history_registration.json")["plan_sha256"] == sha(supplement)
    data = DATA/"bd2006"
    ix = positions(data,"test")
    repeated = np.load(data/"test_prefix_features.npy",mmap_mode="r")[ix,17].astype(bool)
    assert int(repeated.sum()) == read(RUN/"coverage.json")["test"]["prior_identical_step"]
    rows = []
    for name,mask in (("prior_identical_step",repeated),("first_identical_step",~repeated)):
        yy, gg = y[mask], group[mask]
        n = weights.shape[1]
        positives = np.bincount(gg[yy==1],minlength=n)
        negatives = np.bincount(gg[yy==0],minlength=n)
        usable = ((weights @ positives)>0) & ((weights @ negatives)>0)
        for seed in SEEDS:
            with np.load(RUN/"predictions"/f"seed{seed}.npz") as z:
                full, dropped = (np.clip(z[k][mask],1e-7,1-1e-7) for k in ("full","drop_item"))
            row = dict(stratum=name,seed=seed,students=len(np.unique(users[mask])),targets=int(mask.sum()),
                       full_auc="",drop_item_auc="",drop_minus_full="",valid_draws="",ci95_low="",ci95_high="",
                       family2_low="",family2_high="")
            if len(np.unique(yy)) == 2:
                row["full_auc"] = float(roc_auc_score(yy,full))
                row["drop_item_auc"] = float(roc_auc_score(yy,dropped))
                row["drop_minus_full"] = row["drop_item_auc"] - row["full_auc"]
                if seed == 42:
                    row["valid_draws"] = int(usable.sum())
                    if usable.sum() >= 3:
                        diff = draws(dropped,yy,gg,weights[usable])-draws(full,yy,gg,weights[usable])
                        row["ci95_low"],row["ci95_high"] = np.quantile(diff,[.025,.975])
                        row["family2_low"],row["family2_high"] = np.quantile(diff,[.0125,.9875])
            rows.append(row)
    csv_rows(folder/"item_history_strata.csv",rows)


def main():
    folder = RUN/"analysis"
    folder.mkdir(exist_ok=True, parents=True)
    all_metrics, predictors, rows, summary, effects = {}, {}, [], [], []
    for seed in SEEDS:
        path = RUN/"predictions"/f"seed{seed}.npz"
        receipt = read(path.with_suffix(".json"))
        assert sha(path) == receipt["sha256"] and sha(RUN/"freeze.json") == receipt["freeze"]
        with np.load(path) as z:
            if seed == SEEDS[0]:
                y, users, events = (z[k] for k in ("target","student_id","event_id"))
            else:
                for key,a in (("target",y),("student_id",users),("event_id",events)):
                    np.testing.assert_array_equal(z[key],a)
            all_metrics[seed] = {}
            for k in MODELS:
                vals = metrics(y,z[k])
                assert all(abs(v-receipt["metrics"][k][m])<2e-12 for m,v in vals.items())
                all_metrics[seed][k] = vals
                if k != "pfa" or seed == 42:
                    rows.append(dict(model=k, seed=seed, **vals))
                if seed == 42:
                    predictors[k] = z[k]
                elif k == "pfa":
                    np.testing.assert_array_equal(z[k],predictors[k])
    csv_rows(folder/"metrics_per_seed.csv",rows)
    for k in MODELS:
        for m in ("auc","ap","log_loss","brier","ece10"):
            v = [all_metrics[s][k][m] for s in ((42,) if k=="pfa" else SEEDS)]
            summary.append(dict(model=k,metric=m,independent_training_runs=len(v),mean=float(np.mean(v)),
                                sample_sd=float(np.std(v,ddof=1)) if len(v)>1 else ""))
    csv_rows(folder/"metrics_summary.csv",summary)
    student_ids, g = np.unique(users,return_inverse=True)
    rng = np.random.default_rng(20261123)
    weights = np.asarray([np.bincount(rng.integers(len(student_ids),size=len(student_ids)),minlength=len(student_ids))
                          for _ in range(10000)],dtype=np.float64)
    selected = dict.fromkeys(k for _,a,b,_ in CONTRASTS for k in (a,b))
    auc_draws = {k:draws(predictors[k],y,g,weights) for k in selected}
    np.savez_compressed(folder/"student_bootstrap.npz", student_ids=student_ids, first_weights=weights[:3], **auc_draws)
    for family,a,b,size in CONTRASTS:
        per = [all_metrics[s][a]["auc"]-all_metrics[s][b]["auc"] for s in SEEDS]
        effect = auc_draws[a]-auc_draws[b]
        l95,h95 = np.quantile(effect,[.025,.975])
        lo,hi = np.quantile(effect,[.025/size,1-.025/size])
        effects.append(dict(family=family,first=a,second=b,definition="first minus second",mean_auc_delta=float(np.mean(per)),
                            seed42=per[0],seed43=per[1],seed44=per[2],seed42_ci95_low=l95,seed42_ci95_high=h95,
                            family_size=size,seed42_adjusted_low=lo,seed42_adjusted_high=hi))
    csv_rows(folder/"paired_auc.csv",effects)
    item_history_analysis(folder, y, users, g, weights)
    save(folder/"complete.json",dict(students=len(student_ids),targets=len(y),seeds=SEEDS,
                                     bootstrap_draws=10000,rng_seed=20261123,checked_weighted_auc_against_sklearn=True,
                                     files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!="complete.json"},
                                     protocol=sha(OUT/"protocol.md"),freeze=sha(RUN/"freeze.json")))


if __name__ == "__main__":
    main()
