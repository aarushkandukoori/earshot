"""Earshot speech pack v1 - LibriTTS-R, VoxPopuli en, GLOBE V2 and VCTK as 16 kHz mono FLAC plus manifests.

Usage, from the project root:

    .venv/bin/python -m dataprep.speech_fetch --source {librittsr,voxpopuli,globe,vctk,all}
    .venv/bin/python -m dataprep.speech_fetch --source vctk globe voxpopuli     # several, in this order
    .venv/bin/python -m dataprep.speech_fetch --stats --source all              # only print what is on disk

Outputs - data/speech/<source>/<speaker>/<file>.flac and data/manifests/speech_{train,val,test}.jsonl (schema in
common.py). Logs - data/logs/speech_<source>.log. Every run is resumable: run the same command again after a crash,
a DiskFull stop or a reboot and it continues where it stopped (see each module's docstring for how).

Exit codes - 0 done, 1 a source failed (others still ran), 3 stopped at the disk floor, 4 a source hit its budget.
"""

from __future__ import annotations

import argparse
import sys

from .common import DiskFull, get_logger
from .speech_common import BUDGET_GB, SPEECH, TOTAL_BUDGET_GB, BudgetFull, format_stats, source_stats, tree_bytes

SOURCES = ["vctk", "globe", "voxpopuli", "librittsr"]  # order used by --source all: quick ones first


def run_source(name: str, args) -> None:
    if name == "librittsr":
        from . import speech_librittsr
        speech_librittsr.run(streams=args.streams, connections=args.connections, max_speakers=args.max_speakers)
    elif name == "voxpopuli":
        from . import speech_parquet
        speech_parquet.run_voxpopuli(max_shards=args.max_shards)
    elif name == "globe":
        from . import speech_parquet
        speech_parquet.run_globe(max_shards=args.max_shards)
    elif name == "vctk":
        from . import speech_vctk
        speech_vctk.run(max_speakers=args.max_speakers)
    else:
        raise ValueError(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", required=True, choices=SOURCES + ["all"])
    ap.add_argument("--streams", type=int, default=2, help="LibriTTS-R tarballs streamed at once (default 2)")
    ap.add_argument("--connections", type=int, default=3,
                    help="HTTP connections per LibriTTS-R tarball (default 3)")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="smoke test - stop each LibriTTS-R tarball (or VCTK) after this many speakers")
    ap.add_argument("--max-shards", type=int, default=None, help="smoke test - VoxPopuli / GLOBE shards to convert")
    ap.add_argument("--stats", action="store_true", help="only print speakers, hours and GB per source")
    args = ap.parse_args(argv)
    sources = SOURCES if "all" in args.source else list(dict.fromkeys(args.source))
    log = get_logger("speech_fetch")
    status = 0
    for name in sources:
        if not args.stats:
            log.info("=== %s ===", name)
            try:
                run_source(name, args)
            except DiskFull as e:
                log.error("STOP %s - disk floor reached: %s. Free some disk, then run the same command again.",
                          name, e)
                status = 3
            except BudgetFull as e:
                log.warning("%s stopped - %s", name, e)
                status = status or 4
            except Exception as e:
                log.exception("%s failed: %s", name, e)
                status = 1
        st = source_stats(name)
        line = format_stats(st) + f" (budget share {BUDGET_GB[name]:.2f} GB)"
        log.info(line)
        print(line, flush=True)
        if status == 3:
            break
    used = tree_bytes(SPEECH) / 1e9 if SPEECH.exists() else 0.0
    line = f"data/speech total {used:.3f} GB of the {TOTAL_BUDGET_GB:.1f} GB budget"
    log.info(line)
    print(line, flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
