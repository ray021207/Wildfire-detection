"""Phase 1a: choose which FIgLib sequences to keep, then download them one at a time.

For each chosen sequence we
  1. download its .tgz into data/raw/,
  2. read every JPEG straight out of the archive (no full-res copy on disk),
  3. save a 2x-smaller copy (width KEEP_WIDTH, JPEG q85) into data/frames/<seq>/,
  4. delete the .tgz.
This keeps peak disk use to one archive (~70-120 MB) above what we keep.

Sampling rules (whole sequences only, never single frames):
  * Monochrome twins ('-m' imager) are dropped when the colour twin of the same
    view exists: they show the same scene and would double-count the fire.
  * The 9 sequences with official HPWREN bounding boxes are always kept.
  * The rest are drawn at random (fixed seed) with at most MAX_PER_SITE
    sequences per camera site, so no single mountain top dominates.

Outputs: data/frames/<seq>/*.jpg + meta.json, data/labels/*.csv,
         results/p1_download.json (exact list of kept sequences).

Usage: python scripts/p1_download.py [--n 110] [--cap-gb 3.6]
"""
import argparse
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (DATA, FIGLIB_BB_CSV, FIGLIB_TAR, FRAMES, KEEP_WIDTH, RAW, SEED,  # noqa: E402
                        folder_size_bytes, parse_sequence_name, write_result)

MAX_PER_SITE = 6
JPEG_QUALITY = 85
UA = {"User-Agent": "research-download-script"}


def fetch(url: str, timeout=120) -> bytes:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup: retry with backoff
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def list_sequences() -> list[str]:
    html = fetch(f"{FIGLIB_TAR}/index.html").decode("utf-8", "replace")
    return sorted(set(re.findall(r"href=([^ >]+)\.tgz", html)))


def list_bb_sequences() -> list[str]:
    html = fetch(f"{FIGLIB_BB_CSV}/index.html").decode("utf-8", "replace")
    return sorted(set(re.findall(r"href=([^ >]+)\.csv", html)))


def choose(all_seqs, bb_seqs, n, rng):
    rows = [parse_sequence_name(s) for s in all_seqs]
    names = set(all_seqs)
    keep = []
    dropped_mono = []
    for r in rows:
        s = r["sequence"]
        if s.endswith("-m") and (s[:-1] + "c") in names:
            dropped_mono.append(s)
            continue
        keep.append(r)
    chosen = [r for r in keep if r["sequence"] in bb_seqs]
    per_site = {}
    for r in chosen:
        per_site[r["site"]] = per_site.get(r["site"], 0) + 1
    pool = [r for r in keep if r["sequence"] not in bb_seqs]
    rng.shuffle(pool)
    for r in pool:
        if len(chosen) >= n:
            break
        if per_site.get(r["site"], 0) >= MAX_PER_SITE:
            continue
        chosen.append(r)
        per_site[r["site"]] = per_site.get(r["site"], 0) + 1
    return chosen, dropped_mono


def process_sequence(seq: str) -> dict:
    out = FRAMES / seq
    meta_path = out / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / f"{seq}.tgz"
    data = fetch(f"{FIGLIB_TAR}/{seq}.tgz", timeout=600)
    tgz.write_bytes(data)
    tgz_bytes = len(data)
    del data
    out.mkdir(parents=True, exist_ok=True)
    sizes, n = {}, 0
    bad = []
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if not (m.isfile() and name.endswith(".jpg") and re.match(r"^\d+_[+-]\d+\.jpg$", name)):
                continue
            try:
                im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            except Exception as e:  # truncated JPEGs exist in the archive
                bad.append(f"{name}: {e}")
                continue
            sizes[f"{im.width}x{im.height}"] = sizes.get(f"{im.width}x{im.height}", 0) + 1
            h = round(im.height * KEEP_WIDTH / im.width)
            im.resize((KEEP_WIDTH, h), Image.LANCZOS).save(out / name, "JPEG", quality=JPEG_QUALITY)
            n += 1
    tgz.unlink()  # delete the full-resolution archive right away
    meta = {"sequence": seq, "n_frames": n, "original_sizes": sizes,
            "tgz_bytes": tgz_bytes, "unreadable": bad, "kept_width": KEEP_WIDTH}
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--cap-gb", type=float, default=3.6, help="stop if data/ grows past this")
    args = ap.parse_args()
    rng = random.Random(SEED)

    all_seqs = list_sequences()
    bb_seqs = list_bb_sequences()
    chosen, dropped_mono = choose(all_seqs, bb_seqs, args.n, rng)
    print(f"{len(all_seqs)} sequences listed, {len(dropped_mono)} mono twins dropped, {len(chosen)} chosen")

    labels = DATA / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for s in bb_seqs:
        p = labels / f"{s}.csv"
        if not p.exists():
            p.write_bytes(fetch(f"{FIGLIB_BB_CSV}/{s}.csv"))

    done = []
    stopped_early = False
    for i, r in enumerate(chosen):
        size_gb = folder_size_bytes(DATA) / 1e9
        if size_gb > args.cap_gb:
            print(f"data/ is {size_gb:.2f} GB > cap {args.cap_gb}; stopping")
            stopped_early = True
            break
        t0 = time.time()
        meta = process_sequence(r["sequence"])
        done.append({**r, **{k: meta[k] for k in ("n_frames", "original_sizes", "tgz_bytes", "unreadable")}})
        print(f"[{i + 1}/{len(chosen)}] {r['sequence']}: {meta['n_frames']} frames "
              f"({time.time() - t0:.0f}s, data/ {size_gb:.2f} GB)")

    write_result("p1_download", {
        "n_sequences_listed": len(all_seqs),
        "n_bb_label_files": len(bb_seqs),
        "bb_sequences": bb_seqs,
        "n_mono_twins_dropped": len(dropped_mono),
        "max_per_site": MAX_PER_SITE,
        "jpeg_quality": JPEG_QUALITY,
        "kept_width": KEEP_WIDTH,
        "n_requested": args.n,
        "n_kept": len(done),
        "stopped_early_for_budget": stopped_early,
        "total_tgz_downloaded_gb": round(sum(d["tgz_bytes"] for d in done) / 1e9, 2),
        "frames_dir_gb": round(folder_size_bytes(FRAMES) / 1e9, 2),
        "kept": done,
    })


if __name__ == "__main__":
    main()
