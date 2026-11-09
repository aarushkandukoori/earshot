"""MUSAN (OpenSLR 17) - all of noise/ plus about 6 h of music, per-file licenses checked (plan sections 2.2 and 2.3).

Source - https://www.openslr.org/resources/17/musan.tar.gz (11 GB). We stream it through tarfile (never saved) and
keep only what we need. The tar is in directory order, so once music/ and noise/ have both gone by we stop early.

Why two phases - every MUSAN folder has a LICENSE file with the per-file licenses, but inside the tar it can come
after the audio it describes. So:
  1. stream - write every noise WAV and every music candidate as FLAC, and save the small text files (README,
     LICENSE, ANNOTATIONS) to data/noise/musan/meta/. When the stream is complete, write meta/STREAM_DONE.
  2. commit - read the saved LICENSE files, add every file whose license allows commercial use and adaptation to
     the noise manifests, pick 6 h of music from the allowed candidates, and delete the rest.
Phase 2 never needs the network, so it can be re-run (or fixed) without downloading 11 GB again.

Splits - MUSAN has no official split. Music is split by artist and noise by file (common.split_for, 3 percent val).
"""

from __future__ import annotations

import collections
import json
import logging
import re
import tarfile
import time
from pathlib import Path

from . import common
from .common import entry, flac_ok, split_for, write_flac
from .noise_remote import ResumableStream
from .noise_util import NoiseManifests, decode_audio, license_ok, license_tag, stable_unit, to_16k_mono

URL = "https://www.openslr.org/resources/17/musan.tar.gz"
CORPUS_LICENSE = "CC-BY-4.0"         # MUSAN as a whole (OpenSLR 17 page); each file also has its own license
MUSIC_HOURS = 6.0
MUSIC_CANDIDATE_FRAC = 0.2           # about 8.5 of the 42.5 music hours become candidates, 6 h are kept
EXPECTED = {"music": 600, "noise": 900}   # at least this many WAVs before we trust that a folder is complete
ID_RE = re.compile(r"\b((?:music|noise|speech)-[a-z0-9]+(?:-[a-z0-9]+)*-\d{4})(?:\.wav)?\b")
SAFE_NAME = re.compile(r"^musan(/[A-Za-z0-9_.\-]+)+$")

# keyword -> category for noise files that carry a text description
NOISE_CATEGORIES = [
    ("crowd", r"crowd|people|chatter|babble|cheer|applause|laugh|party|restaurant|cafe|bar\b|audience|talk"),
    ("dishes", r"dish|cutlery|glass|plate|kitchen|clink|cup|spoon|fork"),
    ("traffic", r"traffic|car\b|cars|truck|street|engine|motor|siren|horn|bus\b|train|vehicle|road"),
    ("hvac", r"fan\b|air.?con|hvac|ventilat|hum\b|refrigerator|fridge|compressor|ac unit"),
    ("tones", r"dtmf|dial|tone|beep|phone|fax|ring|alarm"),
    ("rain", r"rain|water|stream|river|shower|tap\b|drip|ocean|wave"),
    ("wind", r"wind"),
    ("footsteps", r"footstep|walk|step"),
    ("door", r"door"),
]


def meta_dir() -> Path:
    return common.DATA / "noise" / "musan" / "meta"


def rel_path(top: str, sub: str, fid: str) -> str:
    return f"noise/musan/{top}/{sub}/{fid}.flac"


def music_candidate(fid: str) -> bool:
    return stable_unit("musan:" + fid) < MUSIC_CANDIDATE_FRAC


SEPARATOR = re.compile(r"^\s*={10,}\s*$")


def parse_license(text: str) -> tuple[dict[str, str | None], str | None]:
    """Parse a MUSAN LICENSE file. Returns (per_file, default).

    The real files are blocks separated by lines of '=====': first the file ids of one album (often with song
    titles between them), then a license line ("CC BY-SA 4.0", "License: CC BY 3.0", "... is licensed under a
    Attribution 3.0 United States License."), then a URL. So every id gets the license of its block. If a block
    names a license that is not allowed (NC, ND) next to an allowed one, the whole block counts as not allowed.
    music/rfm/LICENSE instead starts with a header block (license and URL, no ids) and then lists every id, so a
    block of ids without a license takes the license of the last header block before it (None if there is none).
    A line that holds an id and a license together overrides its block. default is the license for the whole
    folder when the text names exactly one license and lists no ids ("all files in this folder are ...")."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if SEPARATOR.match(line):
            blocks.append([])
        else:
            blocks[-1].append(line)
    per_file: dict[str, str | None] = {}
    found = set()
    header_tag = None
    for block in blocks:
        ids, tags, own = [], [], {}
        for line in block:
            line_ids = ID_RE.findall(line)
            rest = ID_RE.sub(" ", line).strip()
            tag = license_tag(rest) if rest else None
            if tag:
                tags.append(tag)
                found.add(tag)
                for i in line_ids:
                    own[i] = tag
            ids += line_ids
        bad = [t for t in tags if not license_ok(t)]
        block_tag = bad[0] if bad else (tags[0] if tags else header_tag)
        if tags and not ids:
            header_tag = block_tag
        for i in ids:
            per_file[i] = own.get(i, block_tag)
    default = next(iter(found)) if (not per_file and len(found) == 1) else None
    return per_file, default


def parse_annotations(text: str, top: str) -> dict[str, dict]:
    """music lines look like 'music-fma-0026 blues Y Cullah' (id, genres, vocals Y/N, artist);
    other folders keep the rest of the line as free text."""
    out = {}
    for line in text.splitlines():
        tok = line.split()
        if not tok or not ID_RE.fullmatch(tok[0].removesuffix(".wav")):
            continue
        fid = tok[0].removesuffix(".wav")
        if top == "music":
            vocals = tok[2] if len(tok) > 2 and tok[2] in ("Y", "N") else None
            artist = " ".join(tok[3:]) if vocals and len(tok) > 3 else None
            out[fid] = {"genre": tok[1] if len(tok) > 1 else None, "vocals": vocals == "Y" if vocals else None,
                        "artist": artist}
        else:
            out[fid] = {"annotation": " ".join(tok[1:]) or None}
    return out


def noise_category(text: str | None) -> str:
    t = (text or "").lower()
    for cat, pat in NOISE_CATEGORIES:
        if re.search(pat, t):
            return cat
    return "misc"


def _save_meta(name: str, data: bytes) -> None:
    if not SAFE_NAME.match(name) or ".." in name:
        return
    p = meta_dir() / name.split("/", 1)[1]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(p)


def stream(log: logging.Logger, url: str = URL, fileobj=None) -> dict:
    """Phase 1. fileobj lets tests feed a small in-memory tar.gz instead of the 11 GB download."""
    s = fileobj if fileobj is not None else ResumableStream(url, log=log)
    seen: collections.Counter = collections.Counter()
    written, order, t0 = 0, [], time.time()
    stopped_early = False
    with tarfile.open(fileobj=s, mode="r|gz") as tf:
        for ti in tf:
            parts = ti.name.split("/")
            if len(parts) < 2 or parts[0] != "musan":
                continue
            top = parts[1] if len(parts) > 2 or ti.isdir() else ""
            if top and (not order or order[-1] != top):
                order.append(top)
                log.info("MUSAN stream entered %s/ (%.2f GB read, %.0f s)", top, getattr(s, "pos", 0) / 1e9,
                         time.time() - t0)
            if (top and top not in ("music", "noise") and "music" in order and "noise" in order
                    and all(seen[k] >= v for k, v in EXPECTED.items())):
                stopped_early = True
                log.info("music/ and noise/ are complete, stopping the stream early")
                break
            if not ti.isfile():
                continue
            base = parts[-1]
            if not base.endswith(".wav"):
                if top in ("", "music", "noise") and ti.size < 5_000_000:
                    _save_meta(ti.name, tf.extractfile(ti).read())
                continue
            if top not in ("music", "noise") or len(parts) != 4:
                continue
            seen[top] += 1
            sub, fid = parts[2], base[:-4]
            if top == "music" and not music_candidate(fid):
                continue
            rel = rel_path(top, sub, fid)
            if flac_ok(rel):
                continue
            x, sr, _ = decode_audio(tf.extractfile(ti).read())
            write_flac(rel, to_16k_mono(x, sr))
            written += 1
            if written % 100 == 0:
                log.info("MUSAN %d files written, %s WAVs seen, %.2f GB read", written, dict(seen),
                         getattr(s, "pos", 0) / 1e9)
    done = {"seen": dict(seen), "order": order, "written_this_run": written, "stopped_early": stopped_early,
            "bytes_read": getattr(s, "pos", None), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta_dir().mkdir(parents=True, exist_ok=True)
    (meta_dir() / "STREAM_DONE").write_text(json.dumps(done, indent=1))
    log.info("MUSAN stream complete %s", done)
    return done


def _load_folder_meta() -> tuple[dict, dict]:
    lic, ann = {}, {}
    for top in ("music", "noise"):
        d = meta_dir() / top
        if not d.exists():
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            if (sub / "LICENSE").exists():
                lic[(top, sub.name)] = parse_license((sub / "LICENSE").read_text("utf-8", "replace"))
            if (sub / "ANNOTATIONS").exists():
                ann[(top, sub.name)] = parse_annotations((sub / "ANNOTATIONS").read_text("utf-8", "replace"), top)
    return lic, ann


def commit(log: logging.Logger, music_hours: float = MUSIC_HOURS) -> dict:
    """Phase 2 - license check, music pick, manifest lines. Deletes FLACs that are not allowed or not picked."""
    lic, ann = _load_folder_meta()
    stats: collections.Counter = collections.Counter()
    root = common.DATA / "noise" / "musan"
    files = sorted(p for top in ("music", "noise") for p in (root / top).glob("*/*.flac"))
    music = []
    with NoiseManifests() as man:
        for p in files:
            top, sub, fid = p.parent.parent.name, p.parent.name, p.stem
            rel = rel_path(top, sub, fid)
            if (top, sub) not in lic:
                stats[f"no_license_file:{top}/{sub}"] += 1
                continue
            per_file, default = lic[(top, sub)]
            tag = per_file.get(fid, default)
            if not license_ok(tag):
                stats[f"dropped_license:{tag}"] += 1
                p.unlink()
                continue
            secs = flac_ok(rel)
            if not secs:
                stats["broken"] += 1
                p.unlink()
                continue
            a = ann.get((top, sub), {}).get(fid, {})
            if top == "noise":
                if rel not in man:
                    man.add(entry(rel, "musan", secs, split_for("musan:" + fid), tag, ["noise"], group=fid,
                                  category=noise_category(a.get("annotation")), musan_dir=f"{top}/{sub}",
                                  musan_id=fid, annotation=a.get("annotation"), corpus_license=CORPUS_LICENSE))
                stats["noise_kept"] += 1
                stats["noise_seconds"] += secs
            else:
                music.append((stable_unit("musan:" + fid), fid, sub, secs, tag, a))
        total = 0.0
        for _, fid, sub, secs, tag, a in sorted(music):
            rel = rel_path("music", sub, fid)
            if rel not in man and total >= music_hours * 3600:
                (common.DATA / rel).unlink()
                stats["music_extra_deleted"] += 1
                continue
            if rel not in man:
                group = a.get("artist") or fid
                man.add(entry(rel, "musan", secs, split_for("musan:" + group), tag, ["music"], group=group,
                              category="music", musan_dir=f"music/{sub}", musan_id=fid, genre=a.get("genre"),
                              vocals=a.get("vocals"), artist=a.get("artist"), corpus_license=CORPUS_LICENSE))
            total += secs
            stats["music_kept"] += 1
        stats["music_seconds"] = round(total, 1)
    stats["noise_seconds"] = round(stats["noise_seconds"], 1)
    log.info("MUSAN commit %s", dict(stats))
    return dict(stats)


def run(log: logging.Logger) -> dict:
    done = meta_dir() / "STREAM_DONE"
    if done.exists():
        log.info("MUSAN stream already complete (%s), only committing", done)
    else:
        stream(log)
    return commit(log)
