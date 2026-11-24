"""Exercise feature timing, correction fitting and prediction on generated inputs."""
import tempfile
from pathlib import Path
import numpy as np
from scipy.special import expit

from correction import fit, predict
from core_functions import apply_difficulty, difficulty, feature_row


def check_history(rng):
    n = 14
    item = rng.integers(1, 6, n)
    skill = rng.integers(1, 4, n)
    y = rng.integers(0, 2, n).astype(np.float32)
    training = np.arange(n) < 8
    stats = difficulty(item, skill, y, training, 6, 4)
    b = apply_difficulty(item, skill, stats)
    raw = rng.normal(size=(n, 8)).astype(np.float32)
    missing = np.zeros((n, 4), dtype=np.float32)
    observed = np.column_stack((raw, y, missing))
    row = dict(answers=y.copy(), skills=skill, items=item, difficulty=b, behavior=raw,
               measurement_missing=missing, center=observed[training].mean(axis=0),
               scale=np.maximum(observed[training].std(axis=0), .05),
               retained_raw_columns=np.arange(121))
    features = feature_row(row, None)
    row["answers"][-1] = 1 - row["answers"][-1]
    row["behavior"][-1] += 7
    np.testing.assert_array_equal(feature_row(row, None), features)
    return len(features)


def make_inputs(path, rng, student_ids):
    ids = np.repeat(student_ids, 16)
    x = rng.normal(size=(len(ids), 13))
    offset = rng.normal(0, .6, size=len(ids))
    known = (np.arange(len(ids)) % 3 != 0)
    target = (rng.random(len(ids)) < expit(offset + .55 * x[:, 0] - .3 * x[:, 1])).astype(np.int8)
    np.savez(path, x=x, known=known, logit=offset, students=ids, target=target,
             feature_names=np.array([f"example_{i}" for i in range(x.shape[1])]))


def main():
    rng = np.random.default_rng(42)
    width = check_history(rng)
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        make_inputs(work / "train.npz", rng, np.arange(180))
        make_inputs(work / "test.npz", rng, np.arange(180, 215))
        # Shorten the demonstration run. The fit CLI uses the paper settings.
        info = fit(work / "train.npz", work / "model",
                   example_settings=dict(max_rounds=12, interactions=3, smoothing_rounds=3,
                                         interaction_smoothing_rounds=3))
        assert info["trained_branches"] == ["0", "1"]
        predictions = predict(work / "test.npz", work / "model")
        assert predictions.shape == (560,) and np.isfinite(predictions).all()
        assert ((predictions >= 0) & (predictions <= 1)).all()
        print(f"{width} pre-response features checked; fitted both branches; predicted {len(predictions)} generated rows.")
        print(f"Maximum absolute difference from the fitted EBM: {info['native_table_max_error']:.3g}")


if __name__ == "__main__":
    main()
