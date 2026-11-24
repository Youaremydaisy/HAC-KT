#!/usr/bin/env python3
"""Prepare strictly causal student/skill/item history features.

The arrays are aligned with the flattened prepared interaction arrays.  For an
event at position ``t`` every count, previous answer, and gap is computed from
positions strictly before ``t`` in that same student sequence.  The training
dataset can then gather the row for the next target interaction without
looking at its answer or any later response.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def past_counts(group: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return count, success, previous answer and event gap before each row."""
    n = len(correct)
    order = np.argsort(group, kind="stable")
    key = group[order]
    beginnings = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[beginnings[1:], n]
    start = np.repeat(beginnings, ends - beginnings)
    count_sorted = np.arange(n, dtype=np.int64) - start
    total = np.r_[0, np.cumsum(correct[order], dtype=np.int64)]
    success_sorted = total[:-1] - total[start]
    previous_sorted = np.r_[0, correct[order][:-1]].astype(np.float32)
    previous_sorted[count_sorted == 0] = 0.5
    previous_position = np.r_[0, order[:-1]]
    gap_sorted = order - previous_position
    gap_sorted[count_sorted == 0] = 0
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    return (
        count_sorted[inverse],
        success_sorted[inverse],
        previous_sorted[inverse],
        gap_sorted[inverse],
    )


def group_features(
    group: np.ndarray, correct: np.ndarray, known: np.ndarray | None = None
) -> np.ndarray:
    count, success, previous, gap = past_counts(group, correct)
    failure = count - success
    out = np.column_stack(
        (
            np.log1p(success),
            np.log1p(failure),
            np.log((success + 2.0) / (failure + 2.0)),
            previous - 0.5,
            np.log1p(gap),
            (count > 0).astype(np.float32),
        )
    ).astype(np.float32)
    if known is not None:
        out[~known] = 0.0
    return out


def prepare_split(data_dir: Path, split: str) -> dict[str, object]:
    skills = np.load(data_dir / f"{split}_skills.npy", mmap_mode="r")
    correct = np.asarray(np.load(data_dir / f"{split}_correct.npy", mmap_mode="r"), dtype=np.int64)
    problems_path = data_dir / f"{split}_problem_index.npy"
    problems = np.asarray(
        np.load(problems_path, mmap_mode="r") if problems_path.exists() else np.zeros_like(skills),
        dtype=np.int64,
    )
    offsets = np.asarray(np.load(data_dir / f"{split}_offsets.npy", mmap_mode="r"), dtype=np.int64)
    sequence = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
    if len(sequence) != len(correct):
        raise ValueError(f"{split}: offsets do not cover correct array")
    skill_base = int(max(int(skills.max()), 0) + 1)
    problem_base = int(max(int(problems.max()), 0) + 1)
    student = group_features(sequence, correct)
    skill = group_features(sequence * skill_base + np.asarray(skills), correct, np.asarray(skills) > 0)
    item = group_features(sequence * problem_base + problems, correct, problems > 0)
    features = np.column_stack((student, skill, item)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{split}: non-finite prefix feature")
    path = data_dir / f"{split}_prefix_features.npy"
    np.save(path, features)
    return {
        "split": split,
        "rows": int(len(features)),
        "columns": int(features.shape[1]),
        "path": str(path),
        "groups": ["student", "skill", "item"],
        "causal": "strictly earlier responses within the same student sequence",
        "unknown_skill_item_zeroed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    report = [prepare_split(args.data_dir, split) for split in args.splits]
    (args.data_dir / "prefix_features_protocol.json").write_text(
        json.dumps({"features": report, "feature_names": [
            f"{g}_{n}" for g in ("student", "skill", "item")
            for n in ("log_success", "log_failure", "beta_log_odds", "previous_answer", "log_gap", "seen")
        ]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
