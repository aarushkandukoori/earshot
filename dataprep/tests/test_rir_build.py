"""Offline test for dataprep.rir_build - a 3-room bank is built, finished, and resumed from a partial state."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import common, rir_build, rir_sim  # noqa: E402


def test_build_resume_and_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path / "data")
    monkeypatch.setitem(rir_sim.SPLITS, "val", (3, rir_sim.SPLITS["val"][1]))   # 3 rooms, same seed
    log = logging.getLogger("test_rir_build")
    rir_build.build_sim_split(log, "val", procs=2)
    d = tmp_path / "data" / "rir"
    first = np.load(d / "sim_val.npy")
    rows = [json.loads(x) for x in (d / "sim_val.jsonl").read_text().splitlines()]
    assert first.shape == (3, 8, rir_sim.L) and [r["room"] for r in rows] == [0, 1, 2]
    assert not (d / "sim_val.partial.npy").exists()

    # pretend the run died after room 0 - finals gone, partial holds only room 0
    part = np.lib.format.open_memmap(d / "sim_val.partial.npy", mode="w+", dtype=np.float16, shape=first.shape)
    part[0] = first[0]
    part.flush()
    del part
    (d / "sim_val.partial.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (d / "sim_val.npy").unlink()
    (d / "sim_val.jsonl").unlink()
    rir_build.build_sim_split(log, "val", procs=2)
    assert np.array_equal(np.load(d / "sim_val.npy"), first)                     # same rooms, deterministic
    assert [json.loads(x) for x in (d / "sim_val.jsonl").read_text().splitlines()] == rows

    rir_build.build_sim_split(log, "val", procs=2)                               # complete - nothing to do
    assert np.array_equal(np.load(d / "sim_val.npy"), first)
