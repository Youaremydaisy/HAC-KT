#!/usr/bin/env python3
"""Create deterministic student-level sequence artifacts from the audit Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "row_id", "user_id", "skill_id", "problem_id", "start_time", "correct",
    "log_response_seconds", "hint_count", "attempt_count",
    "ms_first_response", "response_seconds",
    "affect_frustrated", "affect_confused",
    "affect_concentrating", "affect_bored",
]
AFFECT_COLUMNS = [
    "affect_frustrated", "affect_confused",
    "affect_concentrating", "affect_bored",
]
BEHAVIOR_AFFECT_COLUMNS = [
    "behavior_confidence",
    "correctness_discrepancy",
    "time_efficiency_discrepancy",
    "persistence_deviation",
]
BEHAVIOR_FEATURE_NAMES = [
    "log_response_seconds",
    "log1p_hint_count",
    "log1p_attempt_count",
    "log_delta_seconds",
    *BEHAVIOR_AFFECT_COLUMNS,
]


def _stable_bucket(user_ids: np.ndarray) -> np.ndarray:
    """A reproducible integer hash independent of Python's hash randomization."""
    values = user_ids.astype(np.uint64, copy=False)
    return ((values * np.uint64(2654435761)) % np.uint64(100)).astype(np.int64)


def _sequence_windows(length: int, max_seq_len: int, min_seq_len: int) -> list[tuple[int, int]]:
    if length < min_seq_len:
        return []
    # Non-overlapping windows keep every next-event target unique in each split.
    starts = list(range(0, length, max_seq_len))
    # A one-event remainder cannot form a target; rebalance the final boundary
    # so the last window has two events without dropping or duplicating events.
    if length > max_seq_len and length % max_seq_len == 1:
        starts[-1] = length - 2
    ends = starts[1:] + [length]
    return [
        (start, end - start)
        for start, end in zip(starts, ends)
        if end - start >= min_seq_len
    ]


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -30.0, 30.0))
    return 1.0 / (1.0 + float(np.exp(-value)))


def _add_behavior_affect_profiles(frame: pd.DataFrame, train_frame: pd.DataFrame,
                                  prior_strength: float = 20.0) -> dict[str, float]:
    """Add causal, low-dimensional behavior-derived affect-related profiles.

    Each profile at event t is computed from the current response/time and
    statistics available strictly before t for the same student.  Global
    fallbacks are estimated from training students only.  These variables are
    interpretable proxies, not human emotion labels.
    """
    correct = pd.to_numeric(frame["correct"], errors="coerce").fillna(0.0)
    response = pd.to_numeric(frame["log_response_seconds"], errors="coerce")
    train_correct = pd.to_numeric(train_frame["correct"], errors="coerce").fillna(0.0)
    global_correct = float(np.clip(train_correct.mean(), 1e-3, 1.0 - 1e-3))
    train_response = pd.to_numeric(
        train_frame["log_response_seconds"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    global_response = float(train_response.mean()) if len(train_response) else 0.0
    response_scale = float(train_response.std(ddof=0)) if len(train_response) > 1 else 1.0
    response_scale = max(response_scale, 0.25)

    previous_time = frame.groupby("user_id", sort=False)["start_time"].shift(1)
    gap_seconds = (frame["start_time"] - previous_time).dt.total_seconds()
    gap_seconds = gap_seconds.fillna(0.0).clip(lower=0.0, upper=7.0 * 24.0 * 3600.0)
    behavior_delta = np.log1p(gap_seconds.to_numpy(np.float64, copy=False))
    frame["_behavior_delta"] = behavior_delta.astype(np.float32)
    train_mask = frame["split"].eq("train").to_numpy()
    train_delta = behavior_delta[train_mask]
    global_delta = float(np.mean(train_delta)) if train_delta.size else 0.0
    delta_scale = float(np.std(train_delta)) if train_delta.size > 1 else 1.0
    delta_scale = max(delta_scale, 0.5)

    response_values = response.fillna(global_response).to_numpy(np.float64, copy=False)
    correct_values = correct.to_numpy(np.float64, copy=False)
    profile = np.zeros((len(frame), 4), dtype=np.float32)

    # The prior is deliberately shared across splits, but its parameters are
    # estimated only from training students; per-student state is causal.
    for _, positions in frame.groupby("user_id", sort=False).groups.items():
        positions = np.asarray(positions, dtype=np.int64)
        count = 0.0
        correct_sum = prior_strength * global_correct
        response_sum = prior_strength * global_response
        delta_sum = prior_strength * global_delta
        for position in positions:
            prior_correct = correct_sum / (prior_strength + count)
            prior_response = response_sum / (prior_strength + count)
            prior_delta = delta_sum / (prior_strength + count)
            current_correct = correct_values[position]
            current_response = response_values[position]
            current_delta = behavior_delta[position]
            profile[position] = (
                np.clip(prior_correct, 0.0, 1.0),
                np.clip(abs(current_correct - prior_correct), 0.0, 1.0),
                _sigmoid((current_response - prior_response) / response_scale),
                _sigmoid((current_delta - prior_delta) / delta_scale),
            )
            count += 1.0
            correct_sum += current_correct
            response_sum += current_response
            delta_sum += current_delta

    for index, name in enumerate(BEHAVIOR_AFFECT_COLUMNS):
        frame[name] = profile[:, index]
    return {
        "prior_strength": float(prior_strength),
        "global_correct": global_correct,
        "global_log_response_mean": global_response,
        "global_log_response_scale": response_scale,
        "global_log_delta_mean": global_delta,
        "global_log_delta_scale": delta_scale,
        "causal_rule": "current response/time plus per-student expanding statistics before current event",
    }


def _save_split(frame: pd.DataFrame, split: str, output_dir: Path,
                max_seq_len: int, min_seq_len: int) -> dict:
    skill_parts: list[np.ndarray] = []
    correct_parts: list[np.ndarray] = []
    response_parts: list[np.ndarray] = []
    affect_parts: list[np.ndarray] = []
    behavior_affect_parts: list[np.ndarray] = []
    behavior_feature_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    problem_parts: list[np.ndarray] = []
    problem_difficulty_parts: list[np.ndarray] = []
    user_parts: list[np.ndarray] = []
    offsets = [0]
    lengths: list[int] = []
    windows: list[tuple[int, int, int]] = []

    sequence_index = 0
    for user_id, group in frame.groupby("user_id", sort=False):
        if len(group) < min_seq_len:
            continue
        skills = group["skill_index"].to_numpy(np.int32, copy=True)
        problems = group["problem_index"].to_numpy(np.int32, copy=True)
        problem_difficulty = group["problem_difficulty"].to_numpy(
            np.float32, copy=True
        )
        correct = pd.to_numeric(group["correct"], errors="coerce").fillna(0).to_numpy(np.int8)
        response = pd.to_numeric(group["log_response_seconds"], errors="coerce").fillna(0).to_numpy(np.float32)
        affects = group[AFFECT_COLUMNS].to_numpy(np.float32, copy=True)
        affects = np.nan_to_num(affects, nan=0.0, posinf=0.0, neginf=0.0)
        behavior_affect = group[BEHAVIOR_AFFECT_COLUMNS].to_numpy(
            np.float32, copy=True
        )
        behavior_affect = np.nan_to_num(
            behavior_affect, nan=0.0, posinf=1.0, neginf=0.0
        )
        behavior_affect = np.clip(behavior_affect, 0.0, 1.0)
        time_ns = group["start_time"].astype("int64").to_numpy(np.int64)
        delta_seconds = np.zeros(len(group), dtype=np.float32)
        if len(group) > 1:
            gaps = np.diff(time_ns).astype(np.float64) / 1_000_000_000.0
            gaps = np.clip(gaps, 0.0, 7.0 * 24.0 * 3600.0)
            delta_seconds[1:] = np.log1p(gaps).astype(np.float32)

        hint_count = pd.to_numeric(group["hint_count"], errors="coerce").fillna(0.0)
        hint_count = np.log1p(hint_count.clip(lower=0.0, upper=50.0)).to_numpy(
            np.float32, copy=False
        )
        attempt_count = pd.to_numeric(group["attempt_count"], errors="coerce").fillna(0.0)
        attempt_count = np.log1p(attempt_count.clip(lower=0.0, upper=50.0)).to_numpy(
            np.float32, copy=False
        )
        behavior_features = np.column_stack(
            (response, hint_count, attempt_count, delta_seconds, behavior_affect)
        ).astype(np.float32, copy=False)

        base = offsets[-1]
        skill_parts.append(skills)
        problem_parts.append(problems)
        problem_difficulty_parts.append(problem_difficulty)
        correct_parts.append(correct)
        response_parts.append(response)
        affect_parts.append(affects)
        behavior_affect_parts.append(behavior_affect)
        behavior_feature_parts.append(behavior_features)
        delta_parts.append(delta_seconds)
        user_parts.append(np.full(len(group), int(user_id), dtype=np.int64))
        offsets.append(base + len(group))
        lengths.append(len(group))
        for start, window_length in _sequence_windows(len(group), max_seq_len, min_seq_len):
            windows.append((sequence_index, start, window_length))
        sequence_index += 1

    if skill_parts:
        arrays = {
            "skills": np.concatenate(skill_parts),
            "problem_index": np.concatenate(problem_parts),
            "problem_difficulty": np.concatenate(problem_difficulty_parts),
            "correct": np.concatenate(correct_parts),
            "response": np.concatenate(response_parts),
            "affects": np.concatenate(affect_parts),
            "behavior_affects": np.concatenate(behavior_affect_parts),
            "behavior_features": np.concatenate(behavior_feature_parts),
            "delta": np.concatenate(delta_parts),
            "user_ids": np.concatenate(user_parts),
        }
    else:
        arrays = {
            "skills": np.empty(0, dtype=np.int32),
            "problem_index": np.empty(0, dtype=np.int32),
            "problem_difficulty": np.empty(0, dtype=np.float32),
            "correct": np.empty(0, dtype=np.int8),
            "response": np.empty(0, dtype=np.float32),
            "affects": np.empty((0, 4), dtype=np.float32),
            "behavior_affects": np.empty((0, 4), dtype=np.float32),
            "behavior_features": np.empty((0, len(BEHAVIOR_FEATURE_NAMES)), dtype=np.float32),
            "delta": np.empty(0, dtype=np.float32),
            "user_ids": np.empty(0, dtype=np.int64),
        }
    for name, value in arrays.items():
        np.save(output_dir / f"{split}_{name}.npy", value, allow_pickle=False)
    np.save(output_dir / f"{split}_offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    np.save(output_dir / f"{split}_lengths.npy", np.asarray(lengths, dtype=np.int32), allow_pickle=False)
    np.save(output_dir / f"{split}_windows.npy", np.asarray(windows, dtype=np.int64).reshape(-1, 3), allow_pickle=False)
    return {
        "students": len(lengths),
        "interactions": int(arrays["skills"].shape[0]),
        "windows": len(windows),
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        },
    }


def run(input_path: Path, output_dir: Path, max_seq_len: int,
        min_seq_len: int, max_users: int | None,
        require_skill: bool = False,
        pykt_metadata_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pykt_metadata = None
    if pykt_metadata_path is not None:
        pykt_metadata = json.loads(pykt_metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(input_path, columns=FEATURE_COLUMNS)
    frame = frame.sort_values(["user_id", "start_time", "row_id"], kind="mergesort")
    frame = frame[(frame["user_id"] >= 0) & (frame["correct"].isin([0, 1]))].reset_index(drop=True)
    if require_skill:
        frame = frame[frame["skill_id"] >= 0].reset_index(drop=True)
    input_users = int(frame["user_id"].nunique())
    eligible_user_ids = frame.groupby("user_id", sort=False).size()
    eligible_user_ids = eligible_user_ids[eligible_user_ids >= min_seq_len].index.to_numpy(np.int64)
    frame = frame[frame["user_id"].isin(eligible_user_ids)].reset_index(drop=True)
    if pykt_metadata is not None:
        # PyKT stores concepts as 0..num_c-1.  The array pipeline reserves 0
        # for padding, so known concepts become 1..num_c and the PyKT unknown
        # concept gets its own final slot.
        pykt_skill_vocab = {
            int(raw): int(index) for raw, index in pykt_metadata["skill_vocab"].items()
        }
        unknown_skill_index = int(
            pykt_metadata.get("unknown_skill_index", len(pykt_skill_vocab))
        )
        frame["skill_index"] = (
            frame["skill_id"].map(pykt_skill_vocab)
            .fillna(unknown_skill_index)
            .add(1)
            .astype(np.int32)
        )
        skill_vocab = {raw: index + 1 for raw, index in pykt_skill_vocab.items()}
        skill_vocab_size = unknown_skill_index + 2
    else:
        raw_skills = np.sort(pd.to_numeric(frame["skill_id"], errors="coerce").dropna().unique().astype(np.int64))
        valid_raw_skills = raw_skills[raw_skills >= 0]
        skill_vocab = {int(raw): index + 1 for index, raw in enumerate(valid_raw_skills)}
        frame["skill_index"] = frame["skill_id"].map(skill_vocab).fillna(0).astype(np.int32)
        skill_vocab_size = len(skill_vocab) + 1
    all_users = np.sort(frame["user_id"].unique().astype(np.int64))
    if max_users is not None:
        all_users = all_users[:max_users]
        frame = frame[frame["user_id"].isin(all_users)].reset_index(drop=True)

    buckets = _stable_bucket(all_users)
    split_by_user = {
        int(user_id): ("train" if bucket < 70 else "val" if bucket < 85 else "test")
        for user_id, bucket in zip(all_users, buckets)
    }
    frame["split"] = frame["user_id"].map(split_by_user)

    # Item identities are known at inference, but a randomly initialized
    # embedding for an item absent from the training students is not useful.
    # Build the item vocabulary and empirical-Bayes difficulty estimates from
    # training interactions only; unseen items back off to their skill estimate.
    train_frame = frame[frame["split"] == "train"]
    if pykt_metadata is not None:
        # Reuse the exact item ids from the PyKT baseline, including unknown 0.
        problem_vocab = {
            int(raw): int(index) for raw, index in pykt_metadata["problem_vocab"].items()
        }
    else:
        train_problem_ids = np.sort(
            train_frame.loc[train_frame["problem_id"] >= 0, "problem_id"]
            .unique()
            .astype(np.int64)
        )
        problem_vocab = {
            int(raw): index + 1 for index, raw in enumerate(train_problem_ids)
        }
    frame["problem_index"] = (
        frame["problem_id"].map(problem_vocab).fillna(0).astype(np.int32)
    )

    global_correct = float(train_frame["correct"].mean())
    prior_strength = 20.0

    def smoothed_difficulty(group_column: str) -> dict[int, float]:
        grouped = train_frame.groupby(group_column, sort=False)["correct"].agg(
            ["sum", "count"]
        )
        probability = (
            grouped["sum"] + prior_strength * global_correct
        ) / (grouped["count"] + prior_strength)
        probability = probability.clip(1e-4, 1.0 - 1e-4)
        return {
            int(index): float(np.log((1.0 - value) / value))
            for index, value in probability.items()
            if int(index) >= 0
        }

    problem_difficulty = smoothed_difficulty("problem_id")
    skill_difficulty = smoothed_difficulty("skill_id")
    global_difficulty = float(np.log((1.0 - global_correct) / global_correct))
    fallback_difficulty = frame["skill_id"].map(skill_difficulty).fillna(
        global_difficulty
    )
    frame["problem_difficulty"] = frame["problem_id"].map(
        problem_difficulty
    ).fillna(fallback_difficulty).astype(np.float32)
    behavior_profile_meta = _add_behavior_affect_profiles(frame, train_frame)

    split_stats = {}
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split].drop(columns="split")
        stats = _save_split(
            split_frame, split, output_dir, max_seq_len=max_seq_len, min_seq_len=min_seq_len
        )
        stats["known_train_problem_rate"] = float(
            (split_frame["problem_index"] > 0).mean()
        )
        split_stats[split] = stats

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len,
        "split_rule": "stable_hash(user_id): train < 70, val < 85, test >= 85",
        "skill_filter": "skill_id >= 0" if require_skill else "none",
        "users_in_input": input_users,
        "users_eligible": int(len(all_users)),
        "users_excluded_for_short_history": input_users - int(len(all_users)),
        "num_skills_with_padding": skill_vocab_size,
        "num_problems_with_padding": (
            int(pykt_metadata.get("num_q", len(problem_vocab) + 1))
            if pykt_metadata is not None else len(problem_vocab) + 1
        ),
        "pykt_metadata": str(pykt_metadata_path) if pykt_metadata_path is not None else None,
        "problem_vocab_scope": "training students only; 0 is unknown",
        "problem_difficulty": {
            "source": "training responses only",
            "definition": "smoothed log-odds of incorrect response",
            "prior_strength": prior_strength,
            "unknown_problem_backoff": "skill difficulty, then global difficulty",
        },
        "behavior_affect": {
            "columns": BEHAVIOR_AFFECT_COLUMNS,
            "interpretation": {
                "behavior_confidence": "expanding correctness estimate before the current response",
                "correctness_discrepancy": "absolute deviation of current correctness from prior confidence",
                "time_efficiency_discrepancy": "sigmoid of current log response time relative to prior student time",
                "persistence_deviation": "sigmoid of current log inter-event gap relative to prior student gap",
            },
            "label_status": "behavior-derived affect-related proxies, not human emotion labels",
            **behavior_profile_meta,
        },
        "skill_vocab": {str(raw): index for raw, index in skill_vocab.items()},
        "feature_names": [
            "skill", "problem", "correct", "log_response_seconds",
            "affect_detector[4]", "affect_behavior[4]", "log_delta_seconds",
            "behavior_features[8]",
        ],
        "behavior_feature_names": BEHAVIOR_FEATURE_NAMES,
        "splits": split_stats,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--max-users", type=int, default=None,
                        help="Optional deterministic prefix for a fast smoke test.")
    parser.add_argument("--require-skill", action="store_true",
                        help="Drop interactions without a valid skill_id before splitting.")
    parser.add_argument(
        "--pykt-metadata", type=Path, default=None,
        help="Reuse skill/problem vocabularies from a PyKT metadata.json file.",
    )
    args = parser.parse_args()
    if args.max_seq_len < 2 or args.min_seq_len < 2 or args.min_seq_len > args.max_seq_len:
        parser.error("require 2 <= min-seq-len <= max-seq-len")
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    if args.pykt_metadata is not None and not args.pykt_metadata.exists():
        parser.error(f"pykt metadata does not exist: {args.pykt_metadata}")
    metadata = run(args.input, args.output_dir, args.max_seq_len, args.min_seq_len,
                   args.max_users, args.require_skill, args.pykt_metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
