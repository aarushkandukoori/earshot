"""Noise pack entry point - FSD50K, DEMAND and MUSAN into data/noise/ and data/manifests/noise_{train,val,test}.jsonl.

Usage (from the project root):
    .venv/bin/python -m dataprep.noise_fetch --source {fsd50k,demand,musan,all}
    .venv/bin/python -m dataprep.noise_fetch --report          # files / hours / GB per source and split

Every source is resumable - re-running skips files that are already written and listed in the manifest. The three
sources can also run as three processes at the same time (manifest appends are locked). If free disk drops below the
12 GB floor, the fetcher stops cleanly with a "stopping cleanly" log line and exit code 3; free space and re-run.
Logs go to data/logs/noise_<source>.log. Plan - docs/research/data-training.md, sections 2.2, 2.3 and 7.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from . import common
from .common import DiskFull, get_logger, read_manifest

SOURCES = ("demand", "fsd50k", "musan")


def report() -> dict:
    """Files, hours and GB per source and split, read from the three noise manifests."""
    out: dict = collections.defaultdict(lambda: {"files": 0, "hours": 0.0, "gb": 0.0})
    for split in ("train", "val", "test"):
        p = common.MANIFESTS / f"noise_{split}.jsonl"
        if not p.exists():
            continue
        for e in read_manifest(p):
            for key in (f"{e['source']}/{split}", f"{e['source']}/all", "total"):
                r = out[key]
                r["files"] += 1
                r["hours"] += e["duration"] / 3600
                f = common.DATA / e["path"]
                r["gb"] += f.stat().st_size / 1e9 if f.exists() else 0.0
    return {k: {"files": v["files"], "hours": round(v["hours"], 2), "gb": round(v["gb"], 3)}
            for k, v in sorted(out.items())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES + ("all",))
    ap.add_argument("--report", action="store_true", help="print the per-source summary and exit")
    ap.add_argument("--fsd50k-dry-run", action="store_true", help="print the FSD50K selection without downloading")
    ap.add_argument("--musan-commit-only", action="store_true", help="redo the MUSAN license check from saved meta")
    a = ap.parse_args(argv)
    if a.report or not a.source:
        print(json.dumps(report(), indent=1))
        return 0
    from . import noise_demand, noise_fsd50k, noise_musan   # imported here so --report stays light
    for src in (SOURCES if a.source == "all" else (a.source,)):
        log = get_logger(f"noise_{src}")
        log.info("start %s, %.1f GB free", src, common.free_gb())
        try:
            if src == "demand":
                noise_demand.run(log)
            elif src == "fsd50k":
                noise_fsd50k.run(log, dry_run=a.fsd50k_dry_run)
            elif a.musan_commit_only:
                noise_musan.commit(log)
            else:
                noise_musan.run(log)
        except DiskFull as e:
            log.error("stopping cleanly - %s. Free some space and re-run the same command to resume.", e)
            return 3
        log.info("end %s, %.1f GB free", src, common.free_gb())
    print(json.dumps(report(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
