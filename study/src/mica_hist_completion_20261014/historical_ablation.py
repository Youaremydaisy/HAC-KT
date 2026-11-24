"""Complete the historical C-HIST ablations without changing historical files.

prepare opens train/validation only; fit changes only the specified readout;
infer requires a parent freeze containing the exact result and head hashes.
ASSIST12 uses the original 3M sample and the corresponding historical seed
offset. Its repetitions are not newly refitted end-to-end cognitive models.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(os.environ.get("MICA_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "reports/mica_hist_completion_20261014"
DEST = OUT / "historical"
CACHE = Path(os.environ.get("MICA_HIST_CACHE", "/root/mica_hist_completion_cache_20261014")) / "historical"
FROZEN = ROOT / "reports/mica_transparent_20260923"
CF = ROOT / "reports/mica_crossfit_20260924"
REV = ROOT / "reports/mica_revision_20260928/response_ablation"
GEN = ROOT / "reports/mica_generalization_20260930"
A12DATA = ROOT / "data/processed/eckt_sequences_full"
GEN_DATA = ROOT / "data/processed/mica_generalization_20260930"
SOURCE = Path(__file__).resolve()
SEEDS = (42, 43, 44)
REUSED = ("no_cognitive_offset", "neutral_updates_only", "main_effects")
ARMS = {"assist12": REUSED + ("drop_student", "drop_skill", "drop_item"),
        "assist09": ("main_effects",), "ednet": ("main_effects",)}
TRACE = ("rasch_slow", "rasch_medium", "rasch_fast", "evidence_ema2", "evidence_ema10", "evidence_ema50", "cumulative_difficulty_adjusted_evidence")
BELIEF = ("slow", "medium", "fast", "variance_slow", "variance_medium", "variance_fast", "last_innovation")
TRACE_NAMES = [f"{g}_{n}" for g in ("student", "skill", "item") for n in TRACE] + [
    "training_item_difficulty", "student_ability_x_difficulty", "student_x_skill_ability", "same_skill_as_previous",
    "same_item_as_previous", "same_skill_previous_answer", "difficulty_change"]
PREFIX_NAMES = [f"{g}_{n}" for g in ("student", "skill", "item") for n in (
    "log_success", "log_failure", "beta_log_odds", "previous_answer", "log_gap", "seen")]
BELIEF_NAMES = [f"{g}_belief_{n}" for g in ("student", "skill", "item") for n in BELIEF]
SETTINGS = dict(interactions=64, max_bins=256, max_interaction_bins=32, outer_bags=2, inner_bags=0,
                learning_rate=.03, max_rounds=2000, early_stopping_rounds=80, min_samples_leaf=100,
                reg_lambda=10., max_leaves=3, n_jobs=2, smoothing_rounds=50, interaction_smoothing_rounds=50)
SOURCES = ["src/mica_transparent_20260923/core.py", "src/mica_transparent_20260923/gam_search.py",
           "src/mica_transparent_20260923/affect_reliability.py", "src/mica_transparent_20260923/uncertainty_search.py",
           "src/mica_transparent_20260923/state_features.cpp", "src/mica_transparent_20260923/uncertainty_features.cpp",
           "src/mica_crossfit_20260924/crossfit_readout.py", "src/mica_crossfit_20260924/full_crossfit_readout.py",
           "src/mica_crossfit_20260924/seed_validation.py", "src/mica_revision_20260928/response_ablation.py",
           "src/mica_revision_20260928/difficulty_update_control.py", "src/mica_generalization_20260930/features.py",
           "src/mica_generalization_20260930/readout.py", "src/mica_generalization_20260930/common.py"]

# These imports are read-only. In particular do not import generalization.features,
# whose import mutates core.OUT and creates an old compiled-cache directory.
os.environ["MICA_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src/mica_transparent_20260923"))
sys.path.insert(0, str(FROZEN / "dependencies"))
from gam_search import ExplainableBoostingClassifier, export_tables, predict_tables, table_terms


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def record(path):
    p = Path(path)
    return {"path": relative(p), "sha256": sha(p), "bytes": p.stat().st_size}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def signature(a):
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def folds(users):
    return (((np.asarray(users, np.int64) * 1103515245 + 12345) % 2147483647) % 10).astype(np.int8)


def scores(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    bins = np.minimum((p * 10).astype(int), 9)
    return {"count": len(y), "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "ap": float(average_precision_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(np.mean((y - p) ** 2)),
            "ece10": sum(abs(float(y[bins == k].sum()) - float(p[bins == k].sum())) for k in range(10)) / len(y)}


def removed(arm):
    if not arm.startswith("drop_"):
        return []
    level = arm[5:]
    result = [f"{level}_{n}" for n in TRACE] + [f"{level}_belief_{n}" for n in BELIEF]
    if level == "student":
        result += ["student_ability_x_difficulty", "student_x_skill_ability"]
    elif level == "skill":
        result += ["student_x_skill_ability"]
    return result


def sample_path():
    return FROZEN / "contextual_capacity_validation/auto16_3000k_sample.npy"


def offset_path(dataset, split, seed):
    if dataset != "assist12":
        return GEN / dataset / "models" / f"cognitive_seed{seed}" / f"{split}_predictions.npz"
    folder = ROOT / ("reports/mica_opt_20260917_v1" if seed == 42 else f"reports/mica_meeting_20260917_v1/seed{seed}")
    path = folder / f"{split}_cognitive_logit.npy"
    if split == "test" and not path.exists():
        path = ROOT / f"reports/mica_meeting_20260917_v1/seed{seed}/test_cognitive_logit.npy"
    return path


def target_positions(data, split):
    off = np.load(data / f"{split}_offsets.npy")
    win = np.load(data / f"{split}_windows.npy")
    return np.concatenate([np.arange(off[u] + s + 1, off[u] + s + n) for u, s, n in win])


def load_offset(dataset, split, seed, d, zero=False):
    if zero:
        return np.zeros(len(d["target"]), np.float64)
    path = offset_path(dataset, split, seed)
    if dataset == "assist12":
        values = np.load(path, mmap_mode="r")
        sample = np.load(sample_path()) if split == "train" else None
        if sample is not None:
            values = values[sample]
        # Seed42 train/val exported labels and IDs are explicit. The other
        # historical logits share this frozen order, checked against raw windows.
        p = A12DATA
        ix = target_positions(p, split)
        if sample is not None:
            ix = ix[sample]
        assert np.array_equal(np.load(p / f"{split}_correct.npy", mmap_mode="r")[ix], d["target"])
        assert np.array_equal(np.load(p / f"{split}_user_ids.npy", mmap_mode="r")[ix], d["users"])
        if seed == 42 and split in ("train", "val"):
            for name, key in (("target", "target"), ("users", "users")):
                a = np.load(path.parent / f"{split}_{name}.npy", mmap_mode="r")
                if sample is not None:
                    a = a[sample]
                assert np.array_equal(a, d[key])
    else:
        a = dict(np.load(path))
        assert np.array_equal(a["target"], d["target"])
        if "student_id" in a:
            assert np.array_equal(a["student_id"], d["users"])
        else:
            assert split == "val", "Historical omission applies only to validation"
            p = GEN_DATA / dataset
            ix = target_positions(p, split)
            assert np.array_equal(np.load(p / f"{split}_correct.npy")[ix], d["target"])
            assert np.array_equal(np.load(p / f"{split}_user_ids.npy")[ix], d["users"])
        values = a["logit"]
    assert len(values) == len(d["target"]) and np.isfinite(values).all()
    return np.asarray(values)


def neutral_test():
    """Build only the missing neutral test traces in a new, rebuildable cache."""
    folder = CACHE / "assist12_neutral_test"
    receipt = folder / "receipt.json"
    if receipt.exists():
        r = read(receipt)
        for name, expected in r["files"].items():
            assert sha(folder / name) == expected
        return np.load(folder / "trace.npy", mmap_mode="r"), np.load(folder / "belief.npy", mmap_mode="r")
    folder.mkdir(parents=True, exist_ok=True)
    off, sk, it, y = [np.load(A12DATA / f"test_{n}.npy", mmap_mode="r") for n in ("offsets", "skills", "problem_index", "correct")]
    args = [np.ascontiguousarray(a, dtype=t) for a, t in zip((off, sk, it, y, np.zeros(len(y), np.float32)),
                                                         (np.int64, np.int32, np.int32, np.float32, np.float32))]
    meta = read(A12DATA / "metadata.json")
    ix = target_positions(A12DATA, "test")
    info = {}
    for name, stem, function, columns in (("trace", "state_features", "mica_states", 28), ("belief", "uncertainty_features", "mica_beliefs", 21)):
        source = ROOT / f"src/mica_transparent_20260923/{stem}.cpp"
        lib = folder / f"{stem}.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)], check=True)
        fn = getattr(ctypes.CDLL(str(lib)), function)
        fn.argtypes = [ctypes.c_int64, np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
                       np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"), np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
                       np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
                       ctypes.c_int32, ctypes.c_int32, np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")]
        fn.restype = ctypes.c_int
        result = np.zeros((len(y), columns), np.float32)
        assert fn(len(off) - 1, *args, meta["num_skills_with_padding"], meta["num_problems_with_padding"], result) == 0
        path = folder / f"{name}.npy"
        assert not path.exists(), "Partial cache needs explicit investigation, never implicit overwrite"
        np.save(path, result[ix])
        info[path.name] = sha(path)
        del result
        gc.collect()
    write_new(receipt, {"created_utc": utc(), "files": info, "runner_sha256": sha(SOURCE),
                        "scope": "Zero difficulty in recurrence only; direct difficulty restored during matrix assembly."})
    return np.load(folder / "trace.npy", mmap_mode="r"), np.load(folder / "belief.npy", mmap_mode="r")


def design(dataset, split, arm="cf"):
    assert split in ("train", "val", "test")
    if split == "test":
        assert (OUT / "freeze.json").exists(), "Parent freeze required before test access"
    if dataset != "assist12":
        folder = GEN / dataset / "features"
        meta = read(folder / f"{split}_full.json")
        assert meta["source_sha256"] == sha(ROOT / "src/mica_generalization_20260930/features.py")
        x = np.load(folder / f"{split}_full.npy", mmap_mode="r")
        return x, meta["names"], dict(np.load(folder / f"{split}_full_targets.npz"))
    keys = ("trace", "prefix", "known", "target", "users", "behavior", "history")
    d = {n: np.load(FROZEN / f"cache/{split}/{n}.npy", mmap_mode="r") for n in keys}
    sample = np.load(sample_path()) if split == "train" else None
    if sample is not None:
        assert len(sample) == 3000000
        d = {k: a[sample] for k, a in d.items()}
        d["trace"] = np.load(CF / "train_trace_crossfit.npy", mmap_mode="r")[sample]
        belief = np.load(CF / "full/train_belief_crossfit.npy", mmap_mode="r")[sample]
    else:
        belief = np.load(FROZEN / ("uncertainty_validation/val_belief.npy" if split == "val" else "reliability_confirmation/cache/test_belief.npy"), mmap_mode="r")
    original_trace = d["trace"]
    if arm == "neutral_updates_only":
        if split == "test":
            d["trace"], belief = neutral_test()
        else:
            d["trace"] = np.load(REV / f"neutral/{split}_trace.npy", mmap_mode="r")
            belief = np.load(REV / f"neutral/{split}_belief.npy", mmap_mode="r")
    names = [f"behavior_history_{j}" for j in range(d["behavior"].shape[1])] + PREFIX_NAMES + TRACE_NAMES + ["log_observed_history", "known_skill"] + BELIEF_NAMES
    x = np.column_stack((np.asarray(d["behavior"], np.float64), np.asarray(d["prefix"], np.float64),
                         np.asarray(d["trace"], np.float64), np.log1p(d["history"]), np.asarray(d["known"], float), belief))
    if arm == "neutral_updates_only":
        for name in ("training_item_difficulty", "difficulty_change"):
            x[:, names.index(name)] = original_trace[:, TRACE_NAMES.index(name)]
        x[:, names.index("student_ability_x_difficulty")] = (
            x[:, names.index("student_rasch_medium")] * np.tanh(x[:, names.index("training_item_difficulty")])).astype(np.float32)
        assert np.array_equal(x[:, names.index("training_item_difficulty")], original_trace[:, 21])
        assert np.array_equal(x[:, names.index("difficulty_change")], original_trace[:, 27])
    assert len(names) == 105 and x.shape == (len(d["target"]), 105) and np.isfinite(x).all()
    return x, names, {k: d[k] for k in ("known", "target", "users", "history")}


def old_folder(dataset, arm, seed):
    if dataset == "assist12":
        return CF / ("full" if seed == 42 else f"full_seed{seed}") if arm == "cf" else REV / f"{arm}_seed{seed}"
    return GEN / dataset / "models" / f"{arm}_seed{seed}"


def old_heads(dataset, arm, seed):
    folder = old_folder(dataset, arm, seed)
    r = read(folder / "results.json")
    expected_source = ("src/mica_crossfit_20260924/full_crossfit_readout.py" if seed == 42 else "src/mica_crossfit_20260924/seed_validation.py") if arm == "cf" else "src/mica_revision_20260928/response_ablation.py"
    if dataset != "assist12":
        expected_source = "src/mica_generalization_20260930/readout.py"
    assert r["source_sha256"] == sha(ROOT / expected_source), (dataset, arm, seed, "training source changed")
    heads = {}
    if dataset == "assist12":
        for head in r["regimes"]:
            value = head.get("table", head.get("tables"))
            path = ROOT / (value["path"] if isinstance(value, dict) else value)
            expected = value["sha256"] if isinstance(value, dict) else head["sha256"]
            assert sha(path) == expected
            heads[str(head["known"])] = {**record(path), "terms": head["terms"]}
        if arm == "neutral_updates_only":
            assert r["control_adapter"]["sha256"] == sha(ROOT / "src/mica_revision_20260928/difficulty_update_control.py")
    else:
        for known, head in r["heads"].items():
            path = folder / head["file"]
            assert sha(path) == head["sha256"]
            heads[known] = {**record(path), "terms": head["terms"], "fallback": head.get("fallback", False)}
    return r, heads


def prediction(x, names, d, offset, heads):
    p = np.empty(len(x), float)
    maximum = 0.
    for known in (False, True):
        m = np.asarray(d["known"], bool) == known
        if not m.any():
            continue
        head = heads[str(known)]
        path = ROOT / head["path"]
        assert sha(path) == head["sha256"]
        table = dict(np.load(path))
        assert list(table["feature_names"]) == names
        p[m] = predict_tables(x[m], offset[m], table)
        ix = np.flatnonzero(m)[:500]
        terms = table_terms(x[ix], table)
        z = np.asarray(offset[ix], np.float64) + float(table["intercept"].ravel()[0]) + terms.sum(1)
        maximum = max(maximum, float(np.max(np.abs(expit(z) - p[ix]))))
    assert maximum < 1e-10 and np.isfinite(p).all()
    return p, maximum


def input_files(dataset):
    paths = []
    if dataset == "assist12":
        paths.append(sample_path())
        for split in ("train", "val", "test"):
            paths += [FROZEN / f"cache/{split}/{n}.npy" for n in ("trace", "prefix", "known", "target", "users", "behavior", "history")]
            paths += [A12DATA / f"{split}_{n}.npy" for n in ("offsets", "windows", "correct", "user_ids")]
            if split in ("train", "val"):
                paths += [REV / f"neutral/{split}_{n}.npy" for n in ("trace", "belief")]
        paths += [CF / "train_trace_crossfit.npy", CF / "full/train_belief_crossfit.npy",
                  FROZEN / "uncertainty_validation/val_belief.npy", FROZEN / "reliability_confirmation/cache/test_belief.npy",
                  A12DATA / "metadata.json", A12DATA / "test_skills.npy", A12DATA / "test_problem_index.npy"]
    else:
        for split in ("train", "val", "test"):
            folder = GEN / dataset / "features"
            paths += [folder / f"{split}_full{suffix}" for suffix in (".npy", ".json", "_targets.npz")]
            paths += [GEN_DATA / dataset / f"{split}_{n}.npy" for n in ("offsets", "windows", "correct", "user_ids")]
    for seed in SEEDS:
        paths += [offset_path(dataset, split, seed) for split in ("train", "val", "test")]
    return sorted(set(paths))


def prepared(dataset):
    path = DEST / dataset / "prepare_receipt.json"
    p = read(path)
    assert p["runner_sha256"] == sha(SOURCE), "Runner changed after prepare"
    assert p["parent_protocol_sha256"] == sha(OUT / "protocol.json"), "Parent protocol changed"
    for name, digest in p["source_hashes"].items():
        assert sha(ROOT / name) == digest, (name, "source changed")
    return p


def prepare(dataset):
    assert (OUT / "protocol.json").exists(), "Register parent protocol before preparing formal models"
    dest = DEST / dataset
    assert not (dest / "prepare_receipt.json").exists(), "Preparation already completed"
    start = time.time()
    x, names, tr = design(dataset, "train")
    v, vn, va = design(dataset, "val")
    assert names == vn
    keep = list(range(len(names))) if dataset == "assist12" else np.flatnonzero(np.ptp(x, axis=0) > 1e-9).tolist()
    design_record = {"dataset": dataset, "train_rows": len(x), "validation_rows": len(v), "all_feature_names": names,
                     "baseline_columns": keep, "baseline_feature_names": [names[i] for i in keep],
                     "training_targets": signature(tr["target"]), "training_students": signature(tr["users"]),
                     "validation_targets": signature(va["target"]), "validation_students": signature(va["users"]),
                     "folds": signature(folds(tr["users"])), "arms": {}, "baseline_validation_replays": []}
    for arm in ARMS[dataset]:
        drop = removed(arm)
        assert set(drop).issubset(names)
        columns = [i for i in keep if names[i] not in drop]
        design_record["arms"][arm] = {"columns": columns, "feature_names": [names[i] for i in columns],
                                     "removed_feature_names": drop, "interactions": 0 if arm == "main_effects" else 64,
                                     "reuse_historical_head": dataset == "assist12" and arm in REUSED}
    del x
    gc.collect()
    for seed in SEEDS:
        load_offset(dataset, "train", seed, tr)
        o = load_offset(dataset, "val", seed, va)
        old, heads = old_heads(dataset, "cf", seed)
        if dataset != "assist12":
            assert old["columns"] == keep and old["feature_names"] == [names[i] for i in keep]
        p, reconstruction = prediction(v[:, keep], [names[i] for i in keep], va, o, heads)
        folder = old_folder(dataset, "cf", seed)
        saved = np.load(folder / "validation.npy") if dataset == "assist12" else np.load(folder / "val_predictions.npz")["full"]
        error = float(np.max(np.abs(p - saved)))
        assert error < 1e-10, (dataset, seed, "baseline validation mismatch", error)
        m = scores(va["target"], p)
        expected = old["metrics"]["all"] if dataset == "assist12" else old["validation"]
        for metric in ("auc", "log_loss", "brier"):
            assert abs(m[metric] - expected[metric]) < 1e-12
        design_record["baseline_validation_replays"].append({"seed": seed, "max_error": error,
            "reconstruction_error": reconstruction, "metrics": m, "heads": heads,
            "original_result": record(folder / "results.json"), "offset": record(offset_path(dataset, "val", seed))})
    del v
    gc.collect()
    # Reused heads are admitted only after recreating each exact historical
    # design and verifying all its saved validation probabilities.
    if dataset == "assist12":
        for arm in REUSED:
            v, names, va = design(dataset, "val", arm)
            for seed in SEEDS:
                old, heads = old_heads(dataset, arm, seed)
                assert old["features"] == names
                o = load_offset(dataset, "val", seed, va, arm == "no_cognitive_offset")
                p, reconstruction = prediction(v, names, va, o, heads)
                saved_path = old_folder(dataset, arm, seed) / "validation.npy"
                error = float(np.max(np.abs(p - np.load(saved_path))))
                assert error < 1e-10, (arm, seed, "saved validation mismatch", error)
                metrics = scores(va["target"], p)
                for metric in ("auc", "log_loss", "brier"):
                    assert abs(metrics[metric] - old["metrics"]["all"][metric]) < 1e-12
                target = dest / f"{arm}_seed{seed}"
                target.mkdir(parents=True, exist_ok=True)
                assert not (target / "results.json").exists()
                np.savez_compressed(target / "val_predictions.npz", target=va["target"], full=p, student_id=va["users"])
                write_new(target / "results.json", {"dataset": dataset, "arm": arm, "seed": seed,
                    "model_version": "C-HIST", "new_readout_fit": False, "historical_validation_head_reused": True,
                    "original_result": record(old_folder(dataset, arm, seed) / "results.json"),
                    "original_validation": record(saved_path), "columns": list(range(len(names))), "feature_names": names,
                    "heads": heads, "validation": metrics, "saved_validation_max_error": error,
                    "reconstruction_max_error": reconstruction, "offset_zeroed": arm == "no_cognitive_offset",
                    "cognitive_offsets": {s: record(offset_path(dataset, s, seed)) for s in ("train", "val", "test")},
                    "runner_sha256": sha(SOURCE), "test_access": False, "scope": old["scope"],
                    "upstream_cognitive_refit_this_study": False, "fully_end_to_end_oof": False})
                print("REUSED", dataset, arm, seed, error, flush=True)
            del v
            gc.collect()
    design_path = dest / "training_design.json"
    write_new(design_path, design_record)
    # Hash test bytes for identity only; no test arrays or outcomes are opened.
    files = {relative(p): sha(p) for p in input_files(dataset)}
    receipt = {"created_utc": utc(), "dataset": dataset, "runner_sha256": sha(SOURCE),
               "parent_protocol_sha256": sha(OUT / "protocol.json"), "design": record(design_path),
               "source_hashes": {p: sha(ROOT / p) for p in SOURCES}, "input_hashes": files,
               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "interpret-core")},
               "test_arrays_opened": False, "new_training_started": False, "elapsed_seconds": time.time() - start}
    write_new(dest / "prepare_receipt.json", receipt)
    print("PREPARED", dataset, json.dumps({"rows": design_record["train_rows"], "columns": len(keep), "seconds": receipt["elapsed_seconds"]}), flush=True)


def verify_inputs(p, split):
    for name, digest in p["input_hashes"].items():
        filename = Path(name).name
        if filename.startswith(split + "_") or f"/cache/{split}/" in name or filename == "auto16_3000k_sample.npy":
            assert sha(ROOT / name) == digest, (name, "input changed")


def fit(dataset, arm, seed):
    p = prepared(dataset)
    assert not (dataset == "assist12" and arm in REUSED), "This arm has a reusable frozen historical head"
    verify_inputs(p, "train")
    verify_inputs(p, "val")
    dest = DEST / dataset / f"{arm}_seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    write_new(dest / "attempt.json", {"created_utc": utc(), "runner_sha256": sha(SOURCE), "pid": os.getpid(), "no_automatic_retry": True})
    start = time.time()
    info = read(DEST / dataset / "training_design.json")["arms"][arm]
    x, names, tr = design(dataset, "train")
    v, vn, va = design(dataset, "val")
    assert names == vn and [names[i] for i in info["columns"]] == info["feature_names"]
    x = np.asarray(x[:, info["columns"]], float)
    v = np.asarray(v[:, info["columns"]], float)
    names = info["feature_names"]
    ot = load_offset(dataset, "train", seed, tr)
    ov = load_offset(dataset, "val", seed, va)
    settings = {**SETTINGS, "interactions": info["interactions"]}
    heads, native_errors = {}, []
    for known in (False, True):
        tm = tr["known"] == known
        vm = va["known"] == known
        if dataset != "assist12" and (tm.sum() < 500 or len(np.unique(tr["target"][tm])) < 2):
            continue
        f = folds(tr["users"][tm])
        bags = np.column_stack([np.where(f == j, -1, 1) for j in (0, 1)]).astype(np.int8)
        model = ExplainableBoostingClassifier(feature_names=names, feature_types=["continuous"] * len(names), random_state=seed, **settings)
        print("FIT", dataset, arm, seed, known, int(tm.sum()), flush=True)
        model.fit(x[tm], tr["target"][tm], bags=bags, init_score=ot[tm])
        path = dest / f"{known}_tables.npz"
        assert not path.exists()
        export_tables(model, path)
        table = dict(np.load(path))
        ix = np.flatnonzero(vm)[:1000]
        if len(ix):
            error = float(np.max(np.abs(predict_tables(v[ix], ov[ix], table) - model.predict_proba(v[ix], init_score=ov[ix])[:, 1])))
            assert error < 1e-10
            native_errors.append(error)
        heads[str(known)] = {**record(path), "terms": len(model.term_features_), "best_iterations": np.asarray(model.best_iteration_).tolist(),
                             "training_rows": int(tm.sum()), "bags_signature": signature(bags)}
        del model
        gc.collect()
    for known in (False, True):
        if str(known) not in heads:
            assert dataset != "assist12" and heads
            heads[str(known)] = {**next(iter(heads.values())), "fallback": True}
    pred, reconstruction = prediction(v, names, va, ov, heads)
    np.savez_compressed(dest / "val_predictions.npz", target=va["target"], full=pred, student_id=va["users"])
    result = {"dataset": dataset, "arm": arm, "seed": seed, "model_version": "C-HIST", "new_readout_fit": True,
              "historical_validation_head_reused": False, "columns": info["columns"], "feature_names": names,
              "heads": heads, "validation": scores(va["target"], pred), "max_native_replay_error": max(native_errors, default=0.),
              "reconstruction_max_error": reconstruction, "settings": settings, "train_rows": len(x),
              "cognitive_offsets": {s: record(offset_path(dataset, s, seed)) for s in ("train", "val", "test")},
              "runner_sha256": sha(SOURCE), "prepare_receipt": record(DEST / dataset / "prepare_receipt.json"),
              "test_access": False, "upstream_cognitive_refit_this_study": False, "fully_end_to_end_oof": False,
              "scope": "Readout feature-block deletion only; behavior, prefix history, other states, and corresponding historical cognitive offset remain." if arm.startswith("drop_") else "EBM refitted with interactions=0; same original full-history input columns, training rows and student bags.",
              "elapsed_seconds": time.time() - start}
    write_new(dest / "results.json", result)
    print("DONE", dataset, arm, seed, json.dumps(result["validation"]), flush=True)


def infer(dataset, arm, seed):
    p = prepared(dataset)
    dest = DEST / dataset / f"{arm}_seed{seed}"
    result_path = dest / "results.json"
    r = read(result_path)
    assert r["runner_sha256"] == sha(SOURCE)
    freeze = read(OUT / "freeze.json")
    assert freeze["protocol_sha256"] == sha(OUT / "protocol.json")
    frozen_files = {entry["path"]: entry["sha256"] for entry in freeze["files"]}
    for path in [result_path] + [ROOT / h["path"] for h in r["heads"].values()]:
        assert frozen_files.get(relative(path)) == sha(path), (relative(path), "model absent or changed in parent freeze")
    assert not (dest / "test_predictions.npz").exists(), "Never overwrite a completed test prediction"
    verify_inputs(p, "test")
    x, names, d = design(dataset, "test", arm)
    assert [names[i] for i in r["columns"]] == r["feature_names"]
    x = x[:, r["columns"]]
    o = load_offset(dataset, "test", seed, d, arm == "no_cognitive_offset")
    pred, reconstruction = prediction(x, r["feature_names"], d, o, r["heads"])
    np.savez_compressed(dest / "test_predictions.npz", target=d["target"], full=pred, student_id=d["users"])
    write_new(dest / "test_results.json", {"dataset": dataset, "arm": arm, "seed": seed, "model_version": "C-HIST",
        "test": scores(d["target"], pred), "reconstruction_max_error": reconstruction,
        "test_predictions": record(dest / "test_predictions.npz"), "result": record(result_path),
        "freeze_sha256": sha(OUT / "freeze.json"), "target_signature": signature(d["target"]),
        "student_signature": signature(d["users"]), "historical_test_reuse": True,
        "test_used_for_selection": False, "runner_sha256": sha(SOURCE), "created_utc": utc()})
    print("TEST", dataset, arm, seed, json.dumps(scores(d["target"], pred)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "infer"))
    parser.add_argument("--dataset", choices=tuple(ARMS), required=True)
    parser.add_argument("--arm", choices=tuple(dict.fromkeys(a for group in ARMS.values() for a in group)))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "prepare":
        assert args.arm is None and args.seed is None
        prepare(args.dataset)
    else:
        assert args.arm in ARMS[args.dataset] and args.seed in SEEDS
        (fit if args.stage == "fit" else infer)(args.dataset, args.arm, args.seed)


if __name__ == "__main__":
    main()
