"""Isolate the BD2006 run and reuse the previously published HAC-KT implementation."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GEN = ROOT / "src/mica_generalization_20260930"
OUT = ROOT / "reports/hackt_bd2006_20261123"
DATA = ROOT / "data/processed/hackt_bd2006_20261123"
RAW = ROOT / "data/raw/bd2006_20261123/bridge_to_algebra_2006_2007.zip"
RUN = OUT / "bd2006"
VENDOR = OUT / "adapter_runtime"
SEEDS = (42, 43, 44)
SINGLES = ("no_offset", "neutral_update", "main_effects", "drop_student", "drop_skill", "drop_item")
ARMS = ("full", *SINGLES, "drop_all", "direct_off", "both_off")
BASELINES = ("cognitive", "akt", "simplekt")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def create_runtime():
    # These two registration changes are made only in this run's private copies.
    # All feature construction, fitting and scoring implementations keep their original sources.
    replacements = {
        "common.py": ("('assist09', 'ednet', 'assist17_diagnostic')",
                      "('assist09', 'ednet', 'assist17_diagnostic', 'bd2006')"),
        "prepare.py": ("f,audit=assist09() if dataset=='assist09' else ednet()",
                       "from bd_data import load_events\n    f,audit=load_events()"),
    }
    VENDOR.mkdir(exist_ok=True, parents=True)
    for name, (old, new) in replacements.items():
        original = (GEN / name).read_text(encoding="utf8")
        if original.count(old) != 1:
            raise RuntimeError(f"Changed upstream adapter contract: {name}")
        changed = original.replace(old, new)
        dest = VENDOR / name
        if dest.exists():
            if dest.read_text(encoding="utf8") != changed:
                raise RuntimeError(f"Runtime changed after registration: {name}")
        else:
            dest.write_text(changed, encoding="utf8")
    importlib.invalidate_caches()


def setup():
    os.environ.update(MICA_ROOT=str(ROOT), MICA_OUT=str(OUT), MICA_DATA_ROOT=str(DATA),
                      MICA_SPLIT_SALT="student-split", PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    for location in (ROOT / "third_party/pykt-toolkit", GEN, HERE, VENDOR):
        if str(location) not in sys.path[:1]:
            sys.path.insert(0, str(location))


def model_dir(name, seed):
    return RUN / "models" / f"{name}_seed{seed}"
