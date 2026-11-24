"""Convert complete, observable BD2006 steps into the original event schema."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from runtime import RAW, RUN, save, sha

FIELDS = ["Row", "Anon Student Id", "Problem Name", "Step Name", "KC(SubSkills)",
          "Step Start Time", "First Transaction Time", "Step End Time", "Correct First Attempt",
          "Incorrects", "Corrects", "Hints"]


def times(values):
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return ((parsed - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(milliseconds=1)).to_numpy(float)


def normalized_kcs(text):
    parts = {piece.strip() for piece in str(text).split("~~") if piece.strip()}
    return json.dumps(sorted(parts), ensure_ascii=True, separators=(",", ":")) if parts else ""


def load_events():
    names = [RAW.stem + suffix + ".txt" for suffix in ("_train", "_master")]
    parts, source_counts, excluded = [], {}, {}
    # Read original train and labelled master only. The masked test file repeats the master steps.
    with zipfile.ZipFile(RAW) as archive:
        for name in names:
            count = 0
            with archive.open(name) as stream:
                for chunk in pd.read_csv(stream, sep="\t", usecols=FIELDS, encoding="latin1",
                                         dtype=str, keep_default_na=False, quoting=3, chunksize=100000):
                    count += len(chunk)
                    y = pd.to_numeric(chunk["Correct First Attempt"], errors="coerce")
                    ident = chunk["Anon Student Id"].str.strip()
                    problem = chunk["Problem Name"].str.strip()
                    step = chunk["Step Name"].str.strip()
                    start, first, end = (times(chunk[c]) for c in ("Step Start Time", "First Transaction Time", "Step End Time"))
                    valid = y.isin([0, 1]) & ident.ne("") & problem.ne("") & step.ne("")
                    valid &= np.isfinite(start) & np.isfinite(end) & (end >= start)
                    key = "invalid_label_identity_or_observation_times"
                    excluded[key] = excluded.get(key, 0) + int((~valid).sum())
                    # Escape-free serialization of ordered pairs. Do not split a multi-KC event.
                    if problem.str.contains("\x1f", regex=False).any() or step.str.contains("\x1f", regex=False).any():
                        raise ValueError("Unexpected item identifier delimiter")
                    skills = chunk["KC(SubSkills)"]
                    mapped = {v: normalized_kcs(v) for v in skills.unique()}
                    first_ms = np.where((first >= start) & (first <= end) & (first > start), first-start, np.nan)
                    incorrect, correct, hints = (pd.to_numeric(chunk[c], errors="coerce") for c in ("Incorrects", "Corrects", "Hints"))
                    attempts = (incorrect + correct).where((incorrect >= 0) & (correct >= 0))
                    frame = pd.DataFrame({"user_raw": ident, "event": pd.to_numeric(chunk["Row"], errors="raise"),
                                          "item": problem + "\x1f" + step, "skill": skills.map(mapped), "answer": y,
                                          "response_ms": first_ms, "hints": hints.where(hints >= 0), "attempts": attempts,
                                          "timestamp_ms": start, "end_ms": end})
                    parts.append(frame.loc[valid].copy())
            source_counts[name] = count
    if not all(source_counts.values()):
        raise ValueError("Empty labelled source member")
    df = pd.concat(parts, ignore_index=True)
    del parts
    if df["event"].duplicated().any():
        raise ValueError("Repeated original Row across the two labelled members")
    df["event"] = df["event"].astype("int64")
    df["answer"] = df["answer"].astype("int8")
    user_names = sorted(df["user_raw"].unique())
    user_codes = {v:i+1 for i,v in enumerate(user_names)}
    df["user"] = df["user_raw"].map(user_codes).astype("int64")
    df.drop(columns="user_raw", inplace=True)
    save(RUN / "user_key.json", user_codes)
    df.sort_values(["user","timestamp_ms","event"], kind="stable", inplace=True, ignore_index=True)
    keep = np.zeros(len(df), dtype=bool)
    starts, ends = df["timestamp_ms"].to_numpy(), df["end_ms"].to_numpy()
    for indices in df.groupby("user", sort=False).indices.values():
        last_end, last_start = -np.inf, -np.inf
        for pos in indices:
            if starts[pos] >= last_end and starts[pos] > last_start:
                keep[pos] = True
                last_start, last_end = starts[pos], ends[pos]
    excluded["overlapping_or_tied_start"] = int((~keep).sum())
    df = df.loc[keep].copy()
    previous_end = df.groupby("user", sort=False)["end_ms"].shift()
    assert (previous_end.isna() | (df.timestamp_ms >= previous_end)).all()
    df.drop(columns="end_ms", inplace=True)
    return df, dict(raw_sha256=sha(RAW), source_counts=source_counts, excluded=excluded,
                    event_key="Unique KDD Row across train and labelled master",
                    ordering="Recorded step start then Row; first of equal starts, no overlap with last retained step",
                    time_semantics="start to first transaction; full hint and answer-attempt counts only from completed steps",
                    item_definition="Ordered pair (Problem Name, Step Name)", skill_definition="Sorted unique KC(SubSkills) set; missing tags retained",
                    source_url="https://kdd.org/cupfiles/bridge_to_algebra_2006_2007.zip", users_before_minimum=len(user_names))


def audit_coverage(data, positions):
    from runtime import read
    splits = read(RUN / "split_students.json")
    sets = [set(users) for users in splits.values()]
    assert not any(a & b for i,a in enumerate(sets) for b in sets[i+1:])
    report = {}
    for split in ("train","val","test"):
        load = lambda k: np.load(data / f"{split}_{k}.npy", mmap_mode="r")
        idx = positions(data, split)
        event = load("event_ids")[idx]
        assert len(np.unique(event)) == len(idx)
        assert set(load("user_ids")) == set(splits[split])
        prefix = load("prefix_features")[idx]
        repeated = prefix[:,17].astype(bool)
        report[split] = dict(students=len(splits[split]), targets=len(idx), prior_identical_step=int(repeated.sum()),
                             prior_identical_step_fraction=float(repeated.mean()), missing_skill=int(load("skill_missing")[idx].sum()),
                             unseen_training_item=int(load("item_unseen")[idx].sum()))
    save(RUN / "coverage.json", report)
