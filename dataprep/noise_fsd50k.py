"""FSD50K - restaurant, bar and party sounds plus about 4 h of music, CC0 and CC-BY clips only (plan section 2.3).

Source - Zenodo record 4060432, the official release. The audio is a split zip (dev = 5 parts of 3.2 GB plus a .zip,
eval = 1 part plus a .zip, 24.7 GB together). We never download those zips: noise_remote.RemoteZip reads the central
directory with HTTP range requests, then pulls out only the clips we keep.

Splits - FSD50K's own dev split gives "train" and "val" (the split column of dev.csv); its eval set becomes "test".
The dev and eval sets share no Freesound uploader (FSD50K README), so test clips are from people we never trained on.

Licenses - each clip has its own license in dev/eval_clips_info_FSD50K.json. We keep CC0 and CC-BY (3.0 and 4.0) and
drop CC-BY-NC and Sampling+. CC-BY clips need attribution in the app credits, so every manifest line keeps the uploader,
the title, the Freesound id and the license URL.
"""

from __future__ import annotations

import collections
import csv
import io
import json
import logging
import os
import re
import time
import zipfile

from . import common
from .common import DiskFull, entry, flac_ok, is_clipped, is_silent, write_flac
from .noise_remote import RateLimiter, RemoteZip, fetch_bytes, head_size, make_session
from .noise_util import NoiseManifests, decode_audio, license_tag, stable_unit, to_16k_mono

ZENODO = "https://zenodo.org/records/4060432/files/"
AUDIO_ZIPS = {"dev": ("FSD50K.dev_audio", 5), "eval": ("FSD50K.eval_audio", 1)}
META_ZIPS = ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip", "FSD50K.doc.zip")

# (FSD50K label, our category), highest priority first. A clip takes the category of the first label it carries,
# so a live recording labelled Applause + Music counts as music.
CLASSES = [
    ("Music", "music"),
    ("Chatter", "crowd"),
    ("Crowd", "crowd"),
    ("Laughter", "laughter"),
    ("Applause", "applause"),
    ("Dishes_and_pots_and_pans", "dishes"),
    ("Cutlery_and_silverware", "dishes"),
    ("Chink_and_clink", "dishes"),
    ("Glass", "dishes"),
    ("Walk_and_footsteps", "footsteps"),
    ("Door", "door"),
    ("Water_tap_and_faucet", "water"),
    ("Typing", "typing"),
]
MUSIC_MIN_S = 5.0   # shorter "Music" clips are mostly single notes or drum hits, not background music

# Hours kept per (FSD50K set, category). None = keep every licensed clip.
# dev (train + val) - every licensed clip of the plan's restaurant classes (about 7.4 h) plus 4 h of music.
# eval (test) - capped at about 3 h, weighted to crowd and dishes; the plan's restaurant test set mainly uses
# DEMAND SCAFE and WHAM!, so FSD50K test noise is extra variety and should not eat the 2.1 GB noise budget.
DEFAULT_CAPS = {
    ("dev", "music"): 4.0,
    ("dev", "crowd"): None, ("dev", "laughter"): None, ("dev", "applause"): None, ("dev", "dishes"): None,
    ("dev", "footsteps"): None, ("dev", "door"): None, ("dev", "water"): None, ("dev", "typing"): None,
    ("eval", "music"): 0.4,
    ("eval", "crowd"): 1.2, ("eval", "laughter"): 0.2, ("eval", "applause"): 0.15, ("eval", "dishes"): 0.5,
    ("eval", "footsteps"): 0.1, ("eval", "door"): 0.1, ("eval", "water"): 0.1, ("eval", "typing"): 0.1,
}
BUDGET_GB = 1.0          # hard stop for FSD50K bytes on disk
MAX_GAP = 2 << 20        # read two wanted clips in one request when less than 2 MB of unwanted bytes sits between
RATE_S = 0.55            # seconds between Zenodo requests (about 109 per minute, quota is 133 shared per address)
WORKERS = 5              # range requests in flight at once (one Zenodo connection gives only 0.5 to 1 MB/s)


def raw_dir():
    return common.DATA / "raw" / "fsd50k"


def rel_path(set_name: str, fname: str) -> str:
    return f"noise/fsd50k/{set_name}/{fname}.flac"


def fsd_license_ok(tag: str | None) -> bool:
    """CC0 or plain CC-BY only (the task rule for FSD50K). CC-BY-NC and Sampling+ are dropped."""
    return tag == "CC0-1.0" or bool(re.fullmatch(r"CC-BY-\d\.\d", tag or ""))


def category_for(labels: list[str]) -> tuple[str | None, str | None]:
    """(category, FSD50K label that decided it) or (None, None) if no wanted label is present."""
    for label, cat in CLASSES:
        if label in labels:
            return cat, label
    return None, None


def parse_rows(dev_csv: str, eval_csv: str, dev_info: dict, eval_info: dict) -> list[dict]:
    """One dict per clip - fname, set (dev/eval), split (train/val/test), labels, license tag/url, uploader, title."""
    rows = []
    for set_name, text, info in (("dev", dev_csv, dev_info), ("eval", eval_csv, eval_info)):
        for r in csv.DictReader(io.StringIO(text)):
            meta = info.get(r["fname"], {})
            split = r.get("split") if set_name == "dev" else "test"
            if split not in ("train", "val", "test"):
                continue
            rows.append({
                "fname": r["fname"], "set": set_name, "split": split, "labels": r["labels"].split(","),
                "license_url": meta.get("license"), "license": license_tag(meta.get("license")),
                "uploader": meta.get("uploader"), "title": meta.get("title"),
            })
    return rows


def select(rows: list[dict], dur_s: dict, caps: dict, music_min_s: float = MUSIC_MIN_S) -> list[dict]:
    """Pick clips deterministically. dur_s[(set, fname)] = seconds. Clips are visited in a fixed pseudo-random order
    (hash of the clip id), so the same caps always give the same picks on every machine."""
    used: collections.Counter = collections.Counter()
    out = []
    for r in sorted(rows, key=lambda r: stable_unit("fsd50k:" + r["fname"])):
        if not fsd_license_ok(r["license"]):
            continue
        cat, label = category_for(r["labels"])
        if cat is None:
            continue
        d = dur_s.get((r["set"], r["fname"]))
        if not d or d <= 0:
            continue
        if cat == "music" and d < music_min_s:
            continue
        key = (r["set"], cat)
        if key not in caps:
            continue
        cap = caps[key]
        if cap is not None and used[key] + d > cap * 3600:
            continue
        used[key] += d
        out.append({**r, "category": cat, "class": label, "src_seconds": d})
    return out


def summarize(picks: list[dict]) -> dict:
    h = collections.Counter()
    for p in picks:
        h[f"{p['split']}/{p['category']}"] += p["src_seconds"] / 3600
    return {k: round(v, 2) for k, v in sorted(h.items())}


def _load_meta(log: logging.Logger) -> list[dict]:
    """Download (once) and parse the small metadata zips. They are cached in data/raw/fsd50k/ (7 MB)."""
    d = raw_dir()
    d.mkdir(parents=True, exist_ok=True)
    sess = make_session()
    for name in META_ZIPS:
        p = d / name
        if not p.exists():
            data = fetch_bytes(sess, f"{ZENODO}{name}?download=1", log)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, p)
            log.info("fetched %s (%.1f MB)", name, len(data) / 1e6)
    with zipfile.ZipFile(d / "FSD50K.ground_truth.zip") as z:
        dev_csv = z.read("FSD50K.ground_truth/dev.csv").decode("utf-8")
        eval_csv = z.read("FSD50K.ground_truth/eval.csv").decode("utf-8")
    with zipfile.ZipFile(d / "FSD50K.metadata.zip") as z:
        dev_info = json.loads(z.read("FSD50K.metadata/dev_clips_info_FSD50K.json"))
        eval_info = json.loads(z.read("FSD50K.metadata/eval_clips_info_FSD50K.json"))
    return parse_rows(dev_csv, eval_csv, dev_info, eval_info)


def _open_zips(log: logging.Logger) -> dict[str, RemoteZip]:
    sess = make_session()
    lim = RateLimiter(RATE_S)
    zips = {}
    for set_name, (stem, nparts) in AUDIO_ZIPS.items():
        names = [f"{stem}.z{k:02d}" for k in range(1, nparts + 1)] + [f"{stem}.zip"]
        parts = [(f"{ZENODO}{n}?download=1", head_size(sess, f"{ZENODO}{n}?download=1")) for n in names]
        zips[set_name] = RemoteZip(parts, sess, lim, log)
        log.info("%s - %d members in the central directory, %.1f GB in %d parts", stem, len(zips[set_name].members),
                 sum(s for _, s in parts) / 1e9, len(parts))
    return zips


def _entry(p: dict, rel: str, seconds: float, clipped: bool) -> dict:
    return entry(rel, "fsd50k", seconds, p["split"], p["license"], ["music"] if p["category"] == "music" else ["noise"],
                 group=p["uploader"], category=p["category"], fsd50k_class=p["class"], labels=p["labels"],
                 fsd50k_id=int(p["fname"]), fsd50k_set=p["set"], uploader=p["uploader"], title=p["title"],
                 license_url=p["license_url"], url=f"https://freesound.org/s/{p['fname']}/", clipped=clipped)


def run(log: logging.Logger, caps: dict | None = None, budget_gb: float = BUDGET_GB, dry_run: bool = False,
        max_new: int | None = None) -> dict:
    caps = DEFAULT_CAPS if caps is None else caps
    rows = _load_meta(log)
    zips = _open_zips(log)
    dur, members = {}, {}
    for set_name, rz in zips.items():
        for name, m in rz.members.items():
            if name.endswith(".wav"):
                fname = name.rsplit("/", 1)[1][:-4]
                dur[(set_name, fname)] = (m.usize - 44) / 88200.0   # 44.1 kHz 16-bit mono PCM
                members[(set_name, fname)] = name
    picks = select(rows, dur, caps)
    log.info("selected %d clips, source hours per split/category %s", len(picks), json.dumps(summarize(picks)))
    if dry_run:
        for set_name, rz in zips.items():
            ms = [rz.members[members[(p["set"], p["fname"])]] for p in picks if p["set"] == set_name]
            from .noise_remote import plan_spans
            for gap in (256 << 10, 1 << 20, 4 << 20):
                spans = plan_spans(ms, gap, 48 << 20)
                log.info("dry run %s - %d clips, max_gap %d KB -> %d requests, %.2f GB to download", set_name,
                         len(ms), gap >> 10, len(spans), sum(s.end - s.start for s in spans) / 1e9)
        return {"picks": len(picks), "hours": summarize(picks)}

    stats = collections.Counter()
    budget = budget_gb * 1e9
    t0 = time.time()
    with NoiseManifests() as man:
        on_disk = 0
        todo = []
        for p in picks:
            rel = rel_path(p["set"], p["fname"])
            if rel in man:
                on_disk += (common.DATA / rel).stat().st_size if (common.DATA / rel).exists() else 0
                stats["already_done"] += 1
                continue
            secs = flac_ok(rel)
            if secs:  # FLAC written by an earlier run that died before its manifest line
                man.add(_entry(p, rel, secs, False))
                on_disk += (common.DATA / rel).stat().st_size
                stats["recovered"] += 1
                continue
            todo.append(p)
        log.info("%d clips already done (%.2f GB), %d to fetch", len(picks) - len(todo), on_disk / 1e9, len(todo))
        by_member = {members[(p["set"], p["fname"])]: p for p in todo}
        new = 0
        for set_name, rz in zips.items():
            names = [n for n, p in by_member.items() if p["set"] == set_name]
            if max_new is not None:
                names = names[:max(0, max_new - new)]
            for name, data in rz.iter_members(names, max_gap=MAX_GAP, workers=WORKERS):
                p = by_member[name]
                rel = rel_path(p["set"], p["fname"])
                if data is None:
                    stats["failed"] += 1
                    continue
                try:
                    x, sr, _ = decode_audio(data)
                except Exception as e:  # a broken WAV should not stop the run
                    log.warning("cannot decode %s (%s)", name, e)
                    stats["undecodable"] += 1
                    continue
                y = to_16k_mono(x, sr)
                if y.size < 1600 or is_silent(y, -60.0):
                    stats["silent_or_short"] += 1
                    continue
                write_flac(rel, y)
                man.add(_entry(p, rel, y.size / common.SR, is_clipped(x)))
                on_disk += (common.DATA / rel).stat().st_size
                new += 1
                stats["written"] += 1
                if new % 200 == 0:
                    el = time.time() - t0
                    log.info("%s - %d/%d new clips, %.2f GB on disk, %d requests, %.2f GB downloaded, %.0f s",
                             set_name, new, len(todo), on_disk / 1e9, rz.requests, rz.bytes / 1e9, el)
                if on_disk > budget:
                    log.warning("FSD50K reached its %.2f GB budget, stopping early", budget_gb)
                    break
            log.info("%s done - %d requests, %.2f GB downloaded", set_name, rz.requests, rz.bytes / 1e9)
            if on_disk > budget:
                break
    stats["gb_on_disk"] = round(on_disk / 1e9, 3)
    log.info("FSD50K finished %s in %.0f s", dict(stats), time.time() - t0)
    return dict(stats)


if __name__ == "__main__":  # python -m dataprep.noise_fsd50k --dry-run (selection stats only)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lg = common.get_logger("noise_fsd50k")
    try:
        run(lg, dry_run=a.dry_run)
    except DiskFull as e:
        lg.error("stopping cleanly - %s", e)
