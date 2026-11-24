#!/usr/bin/env python3
"""Chunked, leakage-aware audit and compaction of ASSISTments 2012-13 data.

The source CSV contains a quoted ``actions`` field with embedded newlines,
so this script always uses the CSV parser and never splits lines manually.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AFFECT_SOURCE_COLUMNS = {
    "Average_confidence(FRUSTRATED)": "affect_frustrated",
    "Average_confidence(CONFUSED)": "affect_confused",
    "Average_confidence(CONCENTRATING)": "affect_concentrating",
    "Average_confidence(BORED)": "affect_bored",
}

CORE_COLUMNS = [
    "user_id", "skill_id", "problem_id", "student_class_id", "school_id",
    "assignment_id", "sequence_id", "start_time", "end_time", "correct",
    "hint_count", "attempt_count", "ms_first_response", *AFFECT_SOURCE_COLUMNS,
]

OUTPUT_COLUMNS = [
    "row_id", "user_id", "skill_id", "problem_id", "student_class_id", "school_id",
    "assignment_id", "sequence_id", "start_time", "end_time", "correct", "hint_count",
    "attempt_count", "ms_first_response", "response_seconds", "log_response_seconds",
    "affect_frustrated", "affect_confused", "affect_concentrating", "affect_bored",
    "affect_max", "affect_label", "affect_missing", "core_valid",
]

INTEGER_COLUMNS = [
    "user_id", "skill_id", "problem_id", "student_class_id", "school_id",
    "assignment_id", "sequence_id", "correct", "hint_count", "attempt_count",
    "ms_first_response",
]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_source(path: Path) -> tuple[TextIO, zipfile.ZipFile | None]:
    """Return a text stream and keep a zip handle alive when needed."""
    if path.suffix.lower() != ".zip":
        return path.open("r", encoding="utf-8", errors="replace", newline=""), None
    zf = zipfile.ZipFile(path)
    csv_members = [
        n for n in zf.namelist()
        if n.lower().endswith((".csv", ".txt")) and not n.endswith("/")
    ]
    if not csv_members:
        zf.close()
        raise RuntimeError(f"no CSV/TXT member found in {path}")
    member = max(csv_members, key=lambda n: zf.getinfo(n).file_size)
    raw: BinaryIO = zf.open(member, "r")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""), zf


def _normalise_chunk(df: pd.DataFrame, row_start: int) -> pd.DataFrame:
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"source is missing expected columns: {missing}")
    df = df[CORE_COLUMNS].copy()
    df.insert(0, "row_id", np.arange(row_start, row_start + len(df), dtype=np.int64))
    df.rename(columns=AFFECT_SOURCE_COLUMNS, inplace=True)

    for col in INTEGER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")

    for col in AFFECT_SOURCE_COLUMNS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        # Detector confidence is defined on [0, 1]; invalid values become NA.
        df.loc[~df[col].between(0.0, 1.0), col] = np.nan

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", format="mixed")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", format="mixed")
    response = (df["ms_first_response"].astype("float64") / 1000.0).where(
        df["ms_first_response"] >= 0
    )
    df["response_seconds"] = response.astype("float32")
    df["log_response_seconds"] = np.log1p(response).astype("float32")

    affect_cols = list(AFFECT_SOURCE_COLUMNS.values())
    affect_matrix = df[affect_cols].to_numpy(dtype="float32")
    affect_max = np.nanmax(np.where(np.isnan(affect_matrix), -np.inf, affect_matrix), axis=1)
    affect_max[~np.isfinite(affect_max)] = np.nan
    winners = np.nanargmax(
        np.where(np.isnan(affect_matrix), -np.inf, affect_matrix), axis=1
    )
    labels = np.array(["frustrated", "confused", "concentrating", "bored"], dtype=object)
    label_values = labels[winners]
    label_values[(np.isnan(affect_max)) | (affect_max <= 0.0)] = "unknown"
    df["affect_max"] = affect_max.astype("float32")
    df["affect_label"] = label_values
    df["affect_missing"] = np.isnan(affect_matrix).all(axis=1).astype("int8")
    df["core_valid"] = (
        (df["user_id"] >= 0) & (df["problem_id"] >= 0)
        & df["start_time"].notna() & df["correct"].isin([0, 1])
    ).astype("int8")
    return df[OUTPUT_COLUMNS]


def _arrow_schema() -> pa.Schema:
    return pa.schema([
        ("row_id", pa.int64()), ("user_id", pa.int64()), ("skill_id", pa.int64()),
        ("problem_id", pa.int64()), ("student_class_id", pa.int64()),
        ("school_id", pa.int64()), ("assignment_id", pa.int64()),
        ("sequence_id", pa.int64()), ("start_time", pa.timestamp("ns")),
        ("end_time", pa.timestamp("ns")), ("correct", pa.int64()),
        ("hint_count", pa.int64()), ("attempt_count", pa.int64()),
        ("ms_first_response", pa.int64()), ("response_seconds", pa.float32()),
        ("log_response_seconds", pa.float32()), ("affect_frustrated", pa.float32()),
        ("affect_confused", pa.float32()), ("affect_concentrating", pa.float32()),
        ("affect_bored", pa.float32()), ("affect_max", pa.float32()),
        ("affect_label", pa.string()), ("affect_missing", pa.int8()),
        ("core_valid", pa.int8()),
    ])


def _write_markdown(summary: dict, path: Path) -> None:
    counts = summary["counts"]
    lines = [
        "# ASSISTments 2012–13 affect audit", "",
        f"Generated (UTC): `{summary['generated_at_utc']}`",
        f"Input: `{summary['input']}`", "", "## Counts", "",
        f"Input SHA-256: `{summary['input_sha256']}`", "",
        f"- Rows read: **{counts['rows_read']:,}**",
        f"- Core-valid rows: **{counts['core_valid_rows']:,}** ({counts['core_valid_rate']:.3%})",
        f"- Users: **{counts['users']:,}**",
        f"- Skills (excluding -1): **{counts['skills']:,}**",
        f"- Problems: **{counts['problems']:,}**",
        f"- Schools: **{counts['schools']:,}**",
        f"- Classes: **{counts['classes']:,}**",
        f"- Assignments: **{counts['assignments']:,}**", "", "## Missingness", "",
    ]
    for key, value in summary["missing_rate"].items():
        lines.append(f"- `{key}`: {value:.3%}")
    lines += [
        "", "## Affect estimates", "",
        "The four affect columns are detector confidence estimates supplied by the dataset; they are not independent human labels.",
        "", "| State | Winner count |", "|---|---:|",
    ]
    for key, value in summary["affect_label_counts"].items():
        lines.append(f"| {key} | {value:,} |")
    lines += [
        "", "## Leakage controls", "",
        "- Same-interaction affect estimates must not predict that interaction's correctness.",
        "- For next-event prediction, shift behavior and affect features by one event per student.",
        "- Use student-level splits; do not randomly split rows from the same student.",
        "- Report these variables as predicted affect states or affect estimates, not ground truth.",
        "", "## Time range", "",
        f"- Earliest start: `{summary['time_range']['min_start']}`",
        f"- Latest start: `{summary['time_range']['max_start']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_path: Path, output_dir: Path, chunksize: int, sample_rows: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "interactions.parquet"
    if parquet_path.exists():
        parquet_path.unlink()
    stream, zip_handle = _open_source(input_path)
    counts = Counter()
    unique_users: set[int] = set()
    unique_skills: set[int] = set()
    unique_problems: set[int] = set()
    unique_schools: set[int] = set()
    unique_classes: set[int] = set()
    unique_assignments: set[int] = set()
    affect_label_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    student_counts: Counter[int] = Counter()
    min_start = max_start = None
    writer = None
    sample_parts: list[pd.DataFrame] = []
    row_start = 0
    try:
        reader = pd.read_csv(
            stream, usecols=CORE_COLUMNS, chunksize=chunksize,
            low_memory=False, on_bad_lines="warn",
        )
        for raw in reader:
            chunk = _normalise_chunk(raw, row_start)
            row_start += len(chunk)
            counts["rows_read"] += len(chunk)
            counts["core_valid_rows"] += int(chunk["core_valid"].sum())
            unique_users.update(int(x) for x in chunk.loc[chunk.user_id >= 0, "user_id"].unique())
            unique_skills.update(int(x) for x in chunk.loc[chunk.skill_id >= 0, "skill_id"].unique())
            unique_problems.update(int(x) for x in chunk.loc[chunk.problem_id >= 0, "problem_id"].unique())
            unique_schools.update(int(x) for x in chunk.loc[chunk.school_id >= 0, "school_id"].unique())
            unique_classes.update(int(x) for x in chunk.loc[chunk.student_class_id >= 0, "student_class_id"].unique())
            unique_assignments.update(int(x) for x in chunk.loc[chunk.assignment_id >= 0, "assignment_id"].unique())
            # Counter.update must receive the per-user counts, not only the keys.
            student_counts.update({int(user_id): int(count) for user_id, count in chunk["user_id"].value_counts().items()})
            affect_label_counts.update(chunk["affect_label"].value_counts().to_dict())
            if not sample_parts and sample_rows > 0:
                sample_parts.append(chunk.head(sample_rows).copy())
            for col in INTEGER_COLUMNS:
                if col == "correct":
                    missing_counts[col] += int((~chunk[col].isin([0, 1])).sum())
                else:
                    missing_counts[col] += int((chunk[col] < 0).sum())
            for col in ["start_time", "end_time", *AFFECT_SOURCE_COLUMNS.values()]:
                missing_counts[col] += int(chunk[col].isna().sum())
            start_nonnull = chunk["start_time"].dropna()
            if len(start_nonnull):
                cmin, cmax = start_nonnull.min(), start_nonnull.max()
                min_start = cmin if min_start is None or cmin < min_start else min_start
                max_start = cmax if max_start is None or cmax > max_start else max_start
            table = pa.Table.from_pandas(chunk, schema=_arrow_schema(), preserve_index=False, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        stream.close()
        if zip_handle is not None:
            zip_handle.close()

    total = counts["rows_read"]
    lengths = list(student_counts.values())
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "output": str(parquet_path),
        "input_sha256": _sha256(input_path),
        "counts": {
            **counts, "users": len(unique_users), "skills": len(unique_skills),
            "problems": len(unique_problems), "schools": len(unique_schools),
            "classes": len(unique_classes), "assignments": len(unique_assignments),
            "core_valid_rate": counts["core_valid_rows"] / total if total else 0.0,
            "student_sequence_length": {
                "min": min(lengths) if lengths else 0,
                "median": float(np.median(lengths)) if lengths else 0.0,
                "p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
                "max": max(lengths) if lengths else 0,
            },
        },
        "missing_rate": {k: v / total if total else 0.0 for k, v in missing_counts.items()},
        "affect_label_counts": dict(affect_label_counts),
        "time_range": {
            "min_start": min_start.isoformat() if min_start is not None else None,
            "max_start": max_start.isoformat() if max_start is not None else None,
        },
        "leakage_notes": [
            "Affect columns are detector estimates and may use same-interaction behavior.",
            "Shift affect and behavior features by one event for next-event prediction.",
            "Use student-level splits; do not randomly split rows from the same student.",
        ],
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, output_dir / "audit_summary.md")
    if sample_parts:
        pd.concat(sample_parts, ignore_index=True).to_parquet(
            output_dir / "sample.parquet", index=False, compression="zstd"
        )
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    summary = run(args.input, args.output_dir, args.chunksize, args.sample_rows)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2, default=_json_default))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
