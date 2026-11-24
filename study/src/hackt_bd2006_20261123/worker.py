"""Prepare, fit and evaluate the fourth-dataset replication with the original model code."""
import argparse
import gc
import importlib.util
import platform
from pathlib import Path
import time
from runtime import (ROOT, HERE, OUT, RUN, VENDOR, GEN, RAW, DATA, SEEDS, SINGLES, ARMS, BASELINES,
                     setup, create_runtime, read, save, sha, model_dir)

setup()
import numpy as np

DIRECT = ("training_item_difficulty", "student_ability_x_difficulty", "difficulty_change")


def original(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    from acquire import validate
    from bd_data import audit_coverage
    validate(RAW)
    create_runtime()
    import common
    import prepare as adapter
    import torch, sklearn, pandas
    from features import ExplainableBoostingClassifier
    import interpret
    paths = [GEN / n for n in ("common.py", "prepare.py", "neural.py", "features.py", "frozen_train_model.py",
                              "frozen_train_eckt.py", "pfa.py", "readout.py")]
    paths += [ROOT / "src/mica_transparent_20260923" / name for name in
              ("core.py", "gam_search.py", "state_features.cpp", "uncertainty_features.cpp", "uncertainty_search.py")]
    paths += [ROOT / "src/mica_hist_completion_20261014/historical_ablation.py",
              ROOT / "src/mica_exp_20260921/neural_bench.py", ROOT / "src/mica_revision_20260928/weighted_auc.cpp",
              ROOT / "src/mica_joint_state_ablation_20261111/analyze.py",
              OUT / "protocol.md", VENDOR / "common.py", VENDOR / "prepare.py", HERE / "bd_data.py", HERE / "runtime.py"]
    paths += [ROOT / "third_party/pykt-toolkit/pykt/models" / name for name in ("akt.py", "simplekt.py")]
    fixed = dict(dataset="bd2006", raw_sha256=sha(RAW), protocol_sha256=sha(OUT/"protocol.md"),
                 seeds=list(SEEDS), baselines=list(BASELINES)+["pfa"], arms=list(ARMS),
                 sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
                 versions=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                               sklearn=sklearn.__version__, pandas=pandas.__version__, interpret=interpret.__version__,
                               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    marker = OUT / "registration.json"
    if marker.exists():
        assert read(marker) == fixed, "Registered data, method or environment changed"
    else:
        save(marker, fixed)
    data = common.data_path("bd2006")
    if not (data / "metadata.json").exists():
        adapter.prepare("bd2006")
    assert read(data / "metadata.json")["raw_sha256"] == fixed["raw_sha256"]
    audit_coverage(data, common.positions)
    # Derived matrices are on the system volume to leave the 60 GB data volume for permanent records.
    cache = Path("/root/hackt_bd2006_20261123_cache/features")
    cache.mkdir(parents=True, exist_ok=True)
    feature_dir = RUN / "features"
    if not feature_dir.exists():
        feature_dir.symlink_to(cache, target_is_directory=True)
    assert feature_dir.resolve() == cache
    save(RUN/"data_ready.json", dict(metadata=sha(data/"metadata.json"), coverage=sha(RUN/"coverage.json"),
                                     split=sha(RUN/"split_students.json"), registration=sha(marker)))


def build_features():
    from features import cached
    from common import positions
    data = DATA / "bd2006"
    for split in ("train","val"):
        full, names, y = cached("bd2006", split)
        neutral, other_names, nd = cached("bd2006", split, neutral=True)
        assert names == other_names and len(names) == len(set(names))
        for k in ("target","users","known"):
            np.testing.assert_array_equal(y[k], nd[k])
        ix = positions(data, split)
        np.testing.assert_array_equal(y["target"], np.load(data / f"{split}_correct.npy", mmap_mode="r")[ix])
        np.testing.assert_array_equal(y["users"], np.load(data / f"{split}_user_ids.npy", mmap_mode="r")[ix])
        for name in ("training_item_difficulty","difficulty_change"):
            np.testing.assert_array_equal(full[:,names.index(name)], neutral[:,names.index(name)])
        if split == "train":
            keep = np.flatnonzero(np.ptp(full, axis=0) > 1e-9).tolist()
            save(RUN / "feature_columns.json", dict(names=names, active=keep))
    save(RUN / "features_ready.json", {"columns":sha(RUN/"feature_columns.json")})


def register_item_history():
    plan = OUT/"item_history_addendum.md"
    out = OUT/"item_history_registration.json"
    fixed = dict(plan_sha256=sha(plan),coverage_sha256=sha(RUN/"coverage.json"),
                 comparison="drop_item minus full, separately for prior identical problem-step and first visit")
    if out.exists():
        assert all(read(out)[k] == v for k,v in fixed.items())
        return
    assert not (RUN/"freeze.json").exists() and not (RUN/"predictions").exists()
    assert not any((RUN/"models").glob("*/test_predictions.npz"))
    save(out, dict(**fixed,registered_unix=time.time(),before_test_predictions=True))


def get_design(arm, split):
    import features
    previous = original("bd_historical_ablation", ROOT/"src/mica_hist_completion_20261014/historical_ablation.py")
    neutral = arm in ("neutral_update","both_off")
    base, names, d = features.cached("bd2006", split, neutral=neutral)
    schema = read(RUN / "feature_columns.json")
    assert names == schema["names"]
    drops = set()
    if arm == "drop_all":
        for level in ("drop_student","drop_skill","drop_item"):
            drops.update(previous.removed(level))
        assert len(drops) == 44
    elif arm.startswith("drop_"):
        drops.update(previous.removed(arm))
    assert drops.issubset(names)
    cols = [i for i in schema["active"] if names[i] not in drops]
    selected = [names[i] for i in cols]
    # Materialize only the selected columns once, in the format consumed by InterpretML.
    x = np.empty((len(base),len(cols)), dtype=np.float64, order="F")
    for dest, src in enumerate(cols):
        x[:,dest] = 0. if arm in ("direct_off","both_off") and names[src] in DIRECT else base[:,src]
    return x, selected, d, cols, sorted(drops)


def offsets(arm, seed, split, d):
    if arm == "no_offset":
        return np.zeros(len(d["target"]), dtype=np.float64)
    with np.load(model_dir("cognitive", seed)/f"{split}_predictions.npz") as z:
        np.testing.assert_array_equal(z["target"], d["target"])
        if "student_id" in z:
            np.testing.assert_array_equal(z["student_id"], d["users"])
        result = z["logit"].astype(float)
        assert np.isfinite(result).all()
        return result


def predict_heads(dest, x, d, offset, heads):
    from features import predict_tables
    pred = np.full(len(x), np.nan)
    for known in (False, True):
        mask = d["known"] == known
        if not mask.any():
            continue
        h = heads[str(known)]
        path = dest / h["file"]
        assert sha(path) == h["sha256"]
        with np.load(path) as tables:
            t = dict(tables)
        ids = np.flatnonzero(mask)
        for start in range(0, len(ids), 32768):
            at = ids[start:start + 32768]
            pred[at] = predict_tables(x[at], offset[at], t)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    return pred


def fit(arm, seed):
    import features
    from common import folds, metrics
    historical = original("bd_settings", ROOT/"src/mica_hist_completion_20261014/historical_ablation.py")
    dest = model_dir(arm,seed)
    dest.mkdir(exist_ok=True, parents=True)
    if (dest/"results.json").exists():
        old = read(dest/"results.json")
        assert old["registration"] == sha(OUT/"registration.json")
        assert all(sha(dest/h["file"]) == h["sha256"] for h in old["heads"].values())
        return
    started = time.time()
    x, names, tr, cols, drops = get_design(arm, "train")
    v, vn, va, vc, _ = get_design(arm, "val")
    assert names == vn and cols == vc
    ot, ov = offsets(arm, seed, "train", tr), offsets(arm, seed, "val", va)
    settings = dict(historical.SETTINGS, random_state=seed)
    if arm == "main_effects":
        settings["interactions"] = 0
    heads, errors = {}, []
    for known in (False, True):
        mask = tr["known"] == known
        if mask.sum() < 500 or len(np.unique(tr["target"][mask])) < 2:
            continue
        fold = folds(tr["users"][mask])
        bags = np.column_stack([np.where(fold == j, -1, 1) for j in (0,1)]).astype("int8")
        assert all((bags[:,i] < 0).any() and (bags[:,i] > 0).any() for i in (0,1))
        model = features.ExplainableBoostingClassifier(feature_names=names, feature_types=["continuous"]*len(names), **settings)
        print("Fitting", arm, seed, "known skill", known, "targets", int(mask.sum()), flush=True)
        model.fit(x[mask], tr["target"][mask], bags=bags, init_score=ot[mask])
        path = dest / f"{known}_tables.npz"
        features.export_tables(model, path)
        ix = np.flatnonzero(va["known"] == known)[:1000]
        if len(ix):
            with np.load(path) as f:
                p = features.predict_tables(v[ix], ov[ix], dict(f))
            error = float(np.max(np.abs(model.predict_proba(v[ix], init_score=ov[ix])[:,1] - p)))
            assert error < 1e-10
            errors.append(error)
        heads[str(known)] = dict(file=path.name, sha256=sha(path), train_rows=int(mask.sum()),
                                 terms=len(model.term_features_), interactions=sum(len(t)>1 for t in model.term_features_))
        del model
        gc.collect()
    assert heads
    for known in (False, True):
        if str(known) not in heads:
            heads[str(known)] = dict(next(iter(heads.values())), fallback=True)
    p = predict_heads(dest, v, va, ov, heads)
    np.savez_compressed(dest/"val_predictions.npz", target=va["target"], student_id=va["users"], full=p)
    save(dest/"results.json", dict(arm=arm, seed=seed, settings=settings, columns=cols, feature_names=names,
         removed_feature_definitions=drops, heads=heads, validation=metrics(va["target"],p), rows=len(x),
         registration=sha(OUT/"registration.json"), fit_seconds=time.time()-started,
         native_replay_error=max(errors, default=None), test_access=False))


def neural_job(kind, seed):
    import neural
    neural.train("bd2006", kind, seed)
    if kind == "cognitive":
        neural.infer("bd2006", kind, seed, "train")


def freeze():
    files = {}
    def register(p):
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    for seed in SEEDS:
        for kind in (*BASELINES, "pfa", *ARMS):
            dest = model_dir(kind,seed)
            meta = read(dest/"results.json")
            register(dest/"results.json")
            register(dest/"val_predictions.npz")
            if kind in BASELINES:
                assert sha(dest/"best.pt") == meta["checkpoint_sha256"]
                register(dest/"best.pt")
            if kind in ARMS:
                assert meta["registration"] == sha(OUT/"registration.json")
                for h in meta["heads"].values():
                    assert sha(dest/h["file"]) == h["sha256"]
                    register(dest/h["file"])
    for path in (OUT/"registration.json", OUT/"item_history_addendum.md", OUT/"item_history_registration.json",
                 RUN/"feature_columns.json", HERE/"worker.py", HERE/"run.py", HERE/"analysis.py"):
        register(path)
    record = dict(files=files, before_test_evaluation=True)
    if (RUN/"freeze.json").exists():
        assert read(RUN/"freeze.json") == record
    else:
        save(RUN/"freeze.json", record)


def evaluate():
    import neural, pfa
    from common import positions, metrics
    frozen = read(RUN/"freeze.json")
    assert all(sha(ROOT/name) == digest for name, digest in frozen["files"].items())
    data = DATA/"bd2006"
    ix = positions(data, "test")
    users, events, y = (np.load(data/f"test_{k}.npy", mmap_mode="r")[ix] for k in ("user_ids","event_ids","correct"))
    assert len(set(events.tolist())) == len(events)
    pfa.infer("bd2006")
    for seed in SEEDS:
        result = {}
        for kind in (*BASELINES,"pfa"):
            if kind in BASELINES:
                neural.infer("bd2006", kind, seed, "test")
            with np.load(model_dir(kind,seed)/"test_predictions.npz") as z:
                np.testing.assert_array_equal(z["target"],y)
                np.testing.assert_array_equal(z["student_id"],users)
                result[kind] = z["full"]
        for arm in ARMS:
            dest = model_dir(arm,seed)
            x,names,d,cols,_ = get_design(arm,"test")
            meta = read(dest/"results.json")
            assert meta["columns"] == cols and meta["feature_names"] == names
            np.testing.assert_array_equal(d["users"],users)
            np.testing.assert_array_equal(d["target"],y)
            result[arm] = predict_heads(dest,x,d,offsets(arm,seed,"test",d),meta["heads"])
            del x
        dest = RUN/"predictions"
        dest.mkdir(exist_ok=True)
        path = dest/f"seed{seed}.npz"
        tmp = dest/f"seed{seed}.partial.npz"
        np.savez_compressed(tmp, student_id=users, event_id=events, target=y, **result)
        tmp.replace(path)
        save(dest/f"seed{seed}.json", dict(metrics={k:metrics(y,p) for k,p in result.items()},
                                           sha256=sha(path), freeze=sha(RUN/"freeze.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","register_item_history","features","fit","neural","pfa","freeze","evaluate"))
    p.add_argument("kind",nargs="?")
    p.add_argument("seed",type=int,nargs="?",choices=SEEDS)
    a = p.parse_args()
    if a.action == "fit":
        assert a.kind in ARMS and a.seed in SEEDS
        from parallel import run_locked
        run_locked("fit",a.kind,a.seed,fit)
    elif a.action == "neural":
        assert a.kind in BASELINES and a.seed in SEEDS
        from parallel import run_locked
        run_locked("neural",a.kind,a.seed,neural_job)
    elif a.action == "pfa":
        import pfa
        pfa.fit("bd2006")
    else:
        {"prepare":prepare,"register_item_history":register_item_history,"features":build_features,
         "freeze":freeze,"evaluate":evaluate}[a.action]()
