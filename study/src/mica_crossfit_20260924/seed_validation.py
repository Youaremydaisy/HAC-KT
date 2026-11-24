"""Validation-only seed stability check for the frozen full cross-fit features."""
import gc, json, time
from pathlib import Path

import numpy as np

from common import ROOT, FROZEN, DEST, DATA, data, cognitive, write, sha, validation_metrics
from crossfit_readout import student_folds
from gam_search import ExplainableBoostingClassifier, export_tables, predict_tables
from reliability_confirm import response_matrix


def run(seed):
    out = DEST / f"full_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    sample = np.load(FROZEN / "contextual_capacity_validation/auto16_3000k_sample.npy")
    tr, va = data("train", sample), data("val")
    tr["trace"] = np.load(DEST / "train_trace_crossfit.npy", mmap_mode="r")[sample]
    bt = np.load(DEST / "full/train_belief_crossfit.npy", mmap_mode="r")[sample]
    bv = np.load(FROZEN / "uncertainty_validation/val_belief.npy", mmap_mode="r")
    x, names = response_matrix(tr, bt, np.empty((len(tr["target"]), 4)), False)
    v, vnames = response_matrix(va, bv, np.empty((len(va["target"]), 4)), False)
    assert names == vnames
    ot = np.asarray(cognitive(seed, "train")[sample]); ov = np.asarray(cognitive(seed, "val"))
    pred = np.empty(len(v)); regimes = []
    for known in (False, True):
        tag = "known" if known else "missing"
        tm, vm = tr["known"] == known, va["known"] == known
        table_path = out / f"{tag}_tables.npz"
        folds = student_folds(tr["users"][tm])
        bags = np.column_stack([np.where(folds == j, -1, 1) for j in (0, 1)]).astype(np.int8)
        model = ExplainableBoostingClassifier(
            feature_names=names, feature_types=["continuous"] * len(names), interactions=64,
            max_bins=256, max_interaction_bins=32, outer_bags=2, inner_bags=0,
            learning_rate=.03, max_rounds=2000, early_stopping_rounds=80,
            min_samples_leaf=100, reg_lambda=10., max_leaves=3, n_jobs=2,
            random_state=seed, smoothing_rounds=50, interaction_smoothing_rounds=50)
        print(f"FIT seed={seed} known={known}", flush=True)
        model.fit(x[tm], tr["target"][tm], bags=bags, init_score=ot[tm])
        export_tables(model, table_path)
        pred[vm] = predict_tables(v[vm], ov[vm], dict(np.load(table_path)))
        regimes.append({"known": known, "table": str(table_path.relative_to(ROOT)), "sha256": sha(table_path), "terms": len(model.term_features_)})
        del model
        gc.collect()
    np.save(out / "validation.npy", pred)
    baseline_path = ROOT / f"reports/mica_evidence_2012_20260924/response/no_affect_seed{seed}/validation.npy"
    baseline = np.load(baseline_path, mmap_mode="r")
    result = {"seed": seed, "metrics": validation_metrics(va, pred), "frozen_no_affect_metrics": validation_metrics(va, baseline), "regimes": regimes, "test_access": False, "source_sha256": sha(__file__), "seconds": time.time() - start}
    result["auc_deltas_vs_frozen_no_affect"] = {k: result["metrics"][k]["auc"] - result["frozen_no_affect_metrics"][k]["auc"] for k in ("all", "development", "check")}
    result["eligible"] = bool(result["auc_deltas_vs_frozen_no_affect"]["all"] >= .0002 and result["auc_deltas_vs_frozen_no_affect"]["development"] >= 0 and result["auc_deltas_vs_frozen_no_affect"]["check"] >= 0)
    write(path, result)
    return result


if __name__ == "__main__":
    start = time.time()
    print(json.dumps({"seed43": run(43), "seed44": run(44)}, indent=2), flush=True)
