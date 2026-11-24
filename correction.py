"""Fit and apply the HAC-KT correction to target-aligned upstream logits."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from core_functions import SETTINGS, export_tables, folds, predict_tables


def read_inputs(path, training=False):
    fields = {"x", "known", "logit", "feature_names"}
    if training:
        fields.update(("target", "students"))
    with np.load(path, allow_pickle=False) as data:
        missing = fields.difference(data.files)
        if missing:
            raise ValueError(f"Missing input arrays: {sorted(missing)}")
        arrays = {key: data[key] for key in fields}

    x = arrays["x"]
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty, finite, two-dimensional numeric matrix")
    rows, columns = x.shape
    names = arrays["feature_names"]
    if (names.shape != (columns,) or names.dtype.kind != "U"
            or len(set(names.tolist())) != columns):
        raise ValueError("feature_names must contain one unique Unicode name per column")
    for key in fields.difference(("x", "feature_names")):
        if arrays[key].shape != (rows,):
            raise ValueError(f"{key} must contain one value per row")
    if not np.isfinite(arrays["logit"]).all():
        raise ValueError("An upstream logit must be finite")
    if not np.isin(arrays["known"], [0, 1]).all():
        raise ValueError("known must mark each row with zero or one")
    arrays["known"] = arrays["known"].astype(bool)
    if training:
        if not np.isin(arrays["target"], [0, 1]).all():
            raise ValueError("target must contain zero or one")
        if arrays["students"].dtype.kind not in "iu":
            raise ValueError("students must contain integer student identifiers")
    return arrays


def fit(training_file, model_dir, seed=42, example_settings=None):
    """Fit the two skill-label branches and verify the exported lookup tables."""
    from interpret.glassbox import ExplainableBoostingClassifier

    data = read_inputs(training_file, training=True)
    model_dir = Path(model_dir)
    if model_dir.exists():
        raise FileExistsError(f"Choose a new model directory: {model_dir}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    keep = np.flatnonzero(np.ptp(data["x"], axis=0) > 1e-9)
    if not keep.size:
        raise ValueError("Training contains no varying feature columns")
    names = data["feature_names"][keep].tolist()
    settings = dict(SETTINGS, random_state=seed)
    if example_settings is not None:
        settings.update(example_settings)
    branches = {}
    max_error = 0.0
    with tempfile.TemporaryDirectory(dir=model_dir.parent) as scratch:
        stage = Path(scratch) / "model"
        stage.mkdir()
        for known in (False, True):
            selected = data["known"] == known
            y = data["target"][selected]
            if len(y) < 500 or len(np.unique(y)) < 2:
                continue
            x = data["x"][selected][:, keep].astype(np.float64)
            z = data["logit"][selected].astype(np.float64)
            student_folds = folds(data["students"][selected])
            bags = np.column_stack([np.where(student_folds == k, -1, 1) for k in (0, 1)]).astype(np.int8)
            if not (np.any(bags < 0, axis=0).all() and np.any(bags > 0, axis=0).all()):
                raise ValueError("Each branch must have students in its fitting and validation folds")
            model = ExplainableBoostingClassifier(feature_names=names, feature_types=["continuous"] * len(names), **settings)
            model.fit(x, y, init_score=z, bags=bags)
            filename = f"known_{int(known)}.npz"
            export_tables(model, stage / filename)
            n = min(128, len(y))
            native = model.predict_proba(x[:n], init_score=z[:n])[:, 1]
            with np.load(stage / filename, allow_pickle=False) as table:
                replay = predict_tables(x[:n], z[:n], dict(table))
            np.testing.assert_allclose(replay, native, atol=1e-10, rtol=0)
            max_error = max(max_error, float(np.max(np.abs(replay - native))))
            branches[str(int(known))] = filename
        if not branches:
            raise ValueError("Need at least one branch with 500 training targets of both classes")
        trained = sorted(branches)
        for key in ("0", "1"):
            branches.setdefault(key, branches[trained[0]])
        manifest = dict(input_names=data["feature_names"].tolist(), retained_columns=keep.tolist(),
                        retained_names=names, branches=branches, trained_branches=trained,
                        settings=settings, native_table_max_error=max_error)
        (stage / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(model_dir)
    return manifest


def predict(input_file, model_dir):
    """Add saved function values and the intercept to the fixed upstream logit."""
    data = read_inputs(input_file)
    model_dir = Path(model_dir)
    info = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if data["feature_names"].tolist() != info["input_names"]:
        raise ValueError("Input column names and order must match the training file")
    p = np.empty(len(data["logit"]), dtype=float)
    for known in (False, True):
        chosen = data["known"] == known
        if not chosen.any():
            continue
        with np.load(model_dir / info["branches"][str(int(known))], allow_pickle=False) as f:
            table = dict(f)
        if table["feature_names"].tolist() != info["retained_names"]:
            raise ValueError("Stored table and model.json feature names differ")
        p[chosen] = predict_tables(data["x"][chosen][:, info["retained_columns"]], data["logit"][chosen], table)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    training = commands.add_parser("fit", help="fit on aligned training rows")
    training.add_argument("input", type=Path)
    training.add_argument("model_directory", type=Path)
    training.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict", help="apply saved tables to aligned rows")
    inference.add_argument("input", type=Path)
    inference.add_argument("model_directory", type=Path)
    inference.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.input, args.model_directory, seed=args.seed)
    else:
        p = predict(args.input, args.model_directory)
        with args.output.open("xb") as stream:
            np.save(stream, p, allow_pickle=False)


if __name__ == "__main__":
    main()
