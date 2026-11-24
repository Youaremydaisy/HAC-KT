"""Fit the registered two-feature-only corrections to the three existing student splits."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.path.insert(0, str(ROOT / "src/mica_hist_completion_20261014"))
import historical_ablation as H
import numpy as np

OUT = ROOT / "reports/hackt_pairs_only_20261124"
BD = ROOT / "reports/hackt_bd2006_20261123/bd2006"
FAST = ROOT / "reports/hackt_bd2006_extension_20261123/fast_inputs"
DOMAINS = ("assist12", "assist09", "bd2006")
SEEDS = (42, 43, 44)
N = dict(assist12=(105, 3000000, 898446, 924442), assist09=(99, 239423, 53574, 49646),
         bd2006=(107, 2334305, 511136, 504443))
read, sha = H.read, H.sha


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf8")
    temp.replace(path)


def folder(d, seed):
    return OUT / d / f"seed{seed}"


def original(d, seed):
    if d != "bd2006":
        _, heads = H.old_heads(d, "cf", seed)
        return {k: ROOT / v["path"] for k, v in heads.items()}
    f = BD / "models" / f"full_seed{seed}"
    m = read(f / "results.json")
    heads = {k: f / v["file"] for k, v in m["heads"].items()}
    assert all(sha(path) == m["heads"][k]["sha256"] for k, path in heads.items())
    return heads


def register():
    import importlib.metadata as packages
    settings = dict(H.SETTINGS, exclude="mains")
    del settings["interactions"]
    tasks = {}
    for d in DOMAINS:
        for seed in SEEDS:
            heads = {}
            for k, path in original(d, seed).items():
                with np.load(path, allow_pickle=False) as z:
                    terms = [tuple(map(int, z[f"features_{i}"])) for i in range(len(z["term_names"]))]
                    pairs = [t for t in terms if len(t) == 2]
                    names = z["feature_names"].tolist()
                assert len(names) == N[d][0] and len(terms) == N[d][0] + 64 and len(set(pairs)) == 64
                heads[k] = dict(original=str(path.relative_to(ROOT)), sha256=sha(path), pairs=pairs, names=names)
            assert set(heads) == {"True", "False"}
            tasks[f"{d}/{seed}"] = heads
    plan = dict(datasets=DOMAINS, seeds=SEEDS, settings=settings, heads=tasks,
                rule="In each frozen upstream seed and known-skill branch, retain the 64 feature pairs selected by the original full model on its training data. Refit their surfaces and the intercept from scratch with no univariate term. Never copy fitted contributions or select using new test performance.",
                comparison="Unchanged intercept eligibility, frozen coefficient-one upstream logit, continuous features, student bags, train sample and early stopping. A pair surface can also express some one-feature dependence. Compares term parameterizations, not removal of all marginal information.",
                outcomes="Primary test AUC, AP, log loss, Brier and ECE10 descriptive. Three training seeds share the students. At fixed seed 42 compare full minus pairs, and pairs minus mains on the same students; percentile bootstrap 10000 draws RNG 20261124, Bonferroni six differences, report every direction.",
                history="Selection of three displayed datasets and this experiment followed earlier test results. Freeze all nine branches-pairs models before comparative test inference.",
                source=sha(Path(__file__)), packages={k:packages.version(k) for k in ("interpret-core","numpy","scikit-learn","scipy")})
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == json.loads(json.dumps(plan))
    else:
        save(path, plan)
    print("registered", sha(path), flush=True)


def design(d, split):
    if split == "test":
        assert (OUT / "freeze.json").is_file()
    if d == "bd2006":
        path = FAST / f"{split}_active.npy"
        assert sha(path) == read(FAST / "ready.json")["matrices"][split]
        x = np.load(path, mmap_mode="r")
        meta = read(BD / "feature_columns.json")
        names = [meta["names"][i] for i in meta["active"]]
        with np.load(BD / "features" / f"{split}_full_targets.npz") as z:
            a = dict(z)
    else:
        x, names, a = H.design(d, split)
        ref = read(H.DEST / d / "training_design.json")
        cols = ref["baseline_columns"]
        if cols != list(range(len(names))):
            x = x[:, cols]
            names = [names[i] for i in cols]
        assert names == ref["baseline_feature_names"]
        if split in ("train", "val"):
            pre = "training" if split == "train" else "validation"
            assert H.signature(a["target"]) == ref[pre+"_targets"]
            assert H.signature(a["users"]) == ref[pre+"_students"]
        a = {k:a[k] for k in ("target","users","known")}
    assert x.shape == (N[d][1 + ("train", "val", "test").index(split)], N[d][0])
    return x, names, a


def offset(d, split, seed, a):
    if d != "bd2006":
        return H.load_offset(d, split, seed, a)
    with np.load(BD / "models" / f"cognitive_seed{seed}" / f"{split}_predictions.npz") as z:
        np.testing.assert_array_equal(z["target"], a["target"])
        np.testing.assert_array_equal(z["student_id"], a["users"])
        return z["logit"].astype(float)


def predict(x, a, o, tables, names):
    out = np.full(len(o), np.nan)
    for branch, path in tables.items():
        with np.load(path) as z:
            t = dict(z)
        assert t["feature_names"].tolist() == names
        ix = np.flatnonzero(np.asarray(a["known"], bool) == (branch == "True"))
        for start in range(0, len(ix), 25000):
            at = ix[start:start+25000]
            out[at] = H.predict_tables(x[at], o[at], t)
    assert np.isfinite(out).all() and np.all((out >= 0) & (out <= 1))
    return out


def preds(path, a, arr):
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, student_id=a["users"], target=a["target"], **arr)
    temp.replace(path)


def fit(d, seed):
    p = read(OUT / "protocol.json")
    assert sha(Path(__file__)) == p["source"]
    dest = folder(d,seed)
    dest.mkdir(exist_ok=True, parents=True)
    done = dest / "validation.json"
    if done.exists():
        check_fit(d, seed)
        return
    started = time.time()
    x, names, a = design(d, "train")
    v, vn, va = design(d, "val")
    assert names == vn
    ot, ov = offset(d, "train", seed, a), offset(d, "val", seed, va)
    prior = original(d, seed)
    old = predict(v, va, ov, prior, names)
    if d == "assist12":
        expected = np.load(H.old_folder(d, "cf", seed) / "validation.npy")
    else:
        f = BD / "models" / f"full_seed{seed}" if d == "bd2006" else H.old_folder(d, "cf", seed)
        with np.load(f/"val_predictions.npz") as z:
            expected = z["full"]
            np.testing.assert_array_equal(z["target"], va["target"])
    replay = float(np.max(np.abs(old - expected)))
    assert replay < 1e-10, replay
    records = {}
    for known in (False, True):
        key = str(known)
        src = p["heads"][f"{d}/{seed}"][key]
        pairs = [tuple(x) for x in src["pairs"]]
        assert names == src["names"] and sha(prior[key]) == src["sha256"]
        m = np.asarray(a["known"], bool) == known
        f = H.folds(a["users"][m])
        bags = np.column_stack([np.where(f == j, -1, 1) for j in (0,1)]).astype("int8")
        assert len(np.unique(a["target"][m])) == 2 and np.all((bags<0).any(0)) and np.all((bags>0).any(0))
        model = H.ExplainableBoostingClassifier(feature_names=names, feature_types=["continuous"]*len(names),
                  random_state=seed, interactions=pairs, **p["settings"])
        print("FIT", d, seed, key, int(m.sum()), flush=True)
        model.fit(np.asarray(x[m], dtype=np.float64), a["target"][m], bags=bags, init_score=ot[m])
        assert sorted(model.term_features_) == sorted(pairs)
        target = dest / f"{key}_tables.npz"
        temp = dest / f"{key}_partial.npz"
        H.export_tables(model, temp)
        val_ix = np.flatnonzero(np.asarray(va["known"],bool) == known)[:1200]
        with np.load(temp) as z:
            terms = dict(z)
        err = float(np.max(np.abs(H.predict_tables(v[val_ix], ov[val_ix], terms) - model.predict_proba(v[val_ix], init_score=ov[val_ix])[:,1])))
        assert err < 1e-10
        temp.replace(target)
        records[key] = dict(file=target.name, sha256=sha(target), n=int(m.sum()), terms=len(pairs),
                            offset_sha256=H.signature(ot[m]), bags_sha256=H.signature(bags), native_max_error=err,
                            best_iterations=np.asarray(model.best_iteration_).tolist())
        del model, terms
        gc.collect()
    new = predict(v, va, ov, {k:dest/t["file"] for k,t in records.items()}, names)
    preds(dest/"validation.npz", va, {"pairs_only":new})
    save(done, dict(seed=seed,dataset=d,heads=records,seconds=time.time()-started,protocol=sha(OUT/"protocol.json"),
                    upsteam_full_validation_replay=replay, validation=H.scores(va["target"],new), validation_sha256=sha(dest/"validation.npz")))
    print("DONE",d,seed,flush=True)


def check_fit(d, seed):
    dest=folder(d,seed)
    record=read(dest/"validation.json")
    assert record["protocol"] == sha(OUT/"protocol.json") and sha(dest/"validation.npz")==record["validation_sha256"]
    for key,h in record["heads"].items():
        assert sha(dest/h["file"]) == h["sha256"]
        with np.load(dest/h["file"]) as z:
            assert len(z["term_names"])==64 and all(len(z[f"features_{i}"])==2 for i in range(64))
    return sha(dest/"validation.json")


def freeze():
    hashes = {f"{d}/{s}":check_fit(d,s) for d in DOMAINS for s in SEEDS}
    if (OUT/"freeze.json").exists():
        assert read(OUT/"freeze.json")["validation_hashes"]==hashes
    else:
        save(OUT/"freeze.json",dict(time=time.time(),protocol=sha(OUT/"protocol.json"),validation_hashes=hashes))


def evaluate(d):
    frozen=read(OUT/"freeze.json")
    x, names, a = design(d,"test")
    for seed in SEEDS:
        assert check_fit(d,seed) == frozen["validation_hashes"][f"{d}/{seed}"]
        dest=folder(d,seed)
        o=offset(d,"test",seed,a)
        full=predict(x,a,o,original(d,seed),names)
        if d=="bd2006":
            with np.load(BD/"predictions"/f"seed{seed}.npz") as z:
                np.testing.assert_array_equal(z["target"],a["target"])
                np.testing.assert_array_equal(z["student_id"],a["users"])
                old,uni=z["full"],z["main_effects"]
        else:
            path=ROOT/"reports/mica_joint_state_ablation_20261111/runs"/d/f"seed{seed}/test_predictions.npz"
            with np.load(path) as z:
                np.testing.assert_array_equal(z["target"],a["target"])
                np.testing.assert_array_equal(z["student_id"],a["users"])
                old=z["baseline"]
            with np.load(H.DEST/d/f"main_effects_seed{seed}"/"test_predictions.npz") as z:
                np.testing.assert_array_equal(z["target"],a["target"])
                np.testing.assert_array_equal(z["student_id"],a["users"])
                uni=z["full"]
        replay=float(np.max(np.abs(old-full)))
        assert replay<1e-10
        model=read(dest/"validation.json")
        q=predict(x,a,o,{k:dest/h["file"] for k,h in model["heads"].items()},names)
        from scipy.special import expit
        values=dict(base=expit(o.astype(float)), mains=uni, pairs_only=q, full=old)
        preds(dest/"test.npz",a,values)
        save(dest/"test.json",dict(freeze=sha(OUT/"freeze.json"), metrics={k:H.scores(a["target"],p) for k,p in values.items()},
                                  sha256=sha(dest/"test.npz"), n=len(q),students=len(np.unique(a["users"])),old_full_error=replay))
    print("TEST",d,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["register","fit","freeze","evaluate"])
    parser.add_argument("dataset",nargs="?",choices=DOMAINS)
    parser.add_argument("seed",nargs="?",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=="fit": fit(args.dataset,args.seed)
    elif args.action=="evaluate": evaluate(args.dataset)
    elif args.action=="register": register()
    else: freeze()

