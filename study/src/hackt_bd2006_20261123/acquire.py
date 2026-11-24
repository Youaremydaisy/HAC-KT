"""Fetch the official, labelled BD2006 release in resumable byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os
import time
import urllib.request
import zipfile

URL = "https://kdd.org/cupfiles/bridge_to_algebra_2006_2007.zip"
NAME = URL.rsplit("/", 1)[1]
TOTAL = 68480548
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/raw/bd2006_20261123"


def validate(path):
    if path.stat().st_size != TOTAL:
        raise ValueError("Unexpected length of official archive")
    with zipfile.ZipFile(path) as archive:
        stems = {Path(name).name for name in archive.namelist()}
        expected = {NAME[:-4] + suffix for suffix in ("_train.txt", "_master.txt", "_test.txt")}
        if not expected.issubset(stems) or archive.testzip() is not None:
            raise ValueError("Unexpected file list or ZIP CRC")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    print("Official ZIP verified", path, digest.hexdigest(), flush=True)


def main():
    DEST.mkdir(exist_ok=True, parents=True)
    target = DEST / NAME
    if target.exists():
        validate(target)
        return
    folder = DEST / "download_parts"
    folder.mkdir(exist_ok=True)
    width = 1 << 20
    ranges = [(n, min(n + width, TOTAL) - 1) for n in range(0, TOTAL, width)]

    def fetch(pair):
        left, right = pair
        dest = folder / str(left)
        if dest.exists() and dest.stat().st_size == right - left + 1:
            return dest
        request = urllib.request.Request(URL, headers={
            "Range": f"bytes={left}-{right}", "Accept-Encoding": "identity",
            "User-Agent": "HAC-KT-BD2006-research/1.0",
        })
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=65) as response:
                    expected = f"bytes {left}-{right}/{TOTAL}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Official endpoint did not return requested byte range")
                    payload = response.read(right - left + 2)
                if len(payload) != right - left + 1:
                    raise ValueError("Short range")
                tmp = dest.with_suffix(".partial")
                tmp.write_bytes(payload)
                os.replace(tmp, dest)
                return dest
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    combined = target.with_suffix(".verified-download")
    with ThreadPoolExecutor(max_workers=8) as pool, combined.open("wb") as out:
        for index, path in enumerate(pool.map(fetch, ranges), start=1):
            out.write(path.read_bytes())
            if index % 8 == 0:
                print("Downloaded parts", index, "of", len(ranges), flush=True)
    validate(combined)
    os.replace(combined, target)
    for start, _ in ranges:
        (folder / str(start)).unlink()
    folder.rmdir()


if __name__ == "__main__":
    main()
