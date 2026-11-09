"""Offline tests for dataprep.rir_sim - room sampling rules, the table mirror, cutting, direct-path index, simulation."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import rir_sim as rs  # noqa: E402


def _rooms(n=300, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        try:
            out.append(rs.sample_room(rng))
        except ValueError:
            pass
    return out


def test_sabine_clamps_instead_of_raising():
    e, order = rs.sabine(0.2, [20, 15, 5])           # far too dead for a hall - pyroomacoustics would raise
    assert e == 0.99 and 1 <= order <= rs.MAX_ORDER
    e, order = rs.sabine(1.2, [15, 10, 4])
    assert 0.01 < e < 0.99 and order == rs.MAX_ORDER   # Sabine wants about 110, capped at 40


def test_rt60_distribution():
    rng = np.random.default_rng(0)
    v = np.array([rs.sample_rt60(rng) for _ in range(5000)])
    assert v.min() >= 0.2 and v.max() <= 2.0
    assert 0.08 < np.mean(v > 1.3) < 0.12                  # about 10 % above 1.3 s


def test_room_geometry_rules():
    rooms = _rooms()
    share_table = np.mean([r["geometry"] == "on_table" for r in rooms])
    assert 0.3 < share_table < 0.5
    talkers_in_range = []
    for r in rooms:
        lx, ly, lz = r["dims"]
        assert 4.0 <= lx <= 20.0 and 3.0 <= ly <= 15.0 and 2.5 <= lz <= 5.0
        assert len(r["pos"]) == len(rs.ROLES) == 8
        for p, d in zip(r["pos"], r["dist"]):
            assert 0 < p[0] < lx and 0 < p[1] < ly and 0 < p[2] < lz
            assert abs(math.dist(p, r["mic"]) - d) < 1e-9
        d = r["dist"]
        lo = 0.5 if r["geometry"] == "in_hand" else 0.6
        assert lo - 1e-9 <= d[0] <= 2.5 + 1e-9                         # target
        assert all(x <= 5.0 + 1e-9 for x in d[1:4])                    # competing talkers
        talkers_in_range += [1.0 - 1e-9 <= x for x in d[1:4]]
        assert all(x <= 8.0 + 1e-9 for x in d[5:])                     # extras
        if r["geometry"] == "in_hand":
            assert 0.25 - 1e-9 <= d[4] <= 0.5 + 1e-9                     # own voice, phone in hand
            assert 0.15 - 1e-9 <= r["pos"][4][2] - r["mic"][2] <= 0.4 + 1e-9
            assert 1.0 <= r["mic"][2] <= 1.4
        else:
            h = r["table"]["height"]
            assert 0.72 <= h <= 0.78 and 0.6 <= r["table"]["reflection"] <= 0.9
            assert 0.4 - 1e-9 <= d[4] <= 0.8 + 1e-9                      # own voice, phone on table
            for p in r["pos"]:
                assert h + 0.1 <= p[2] <= 2 * h - 0.05 + 1e-9            # above the table, mirror inside room
    assert np.mean(talkers_in_range) > 0.95                             # only tiny rooms shrink below 1 m


def test_sampling_is_deterministic():
    a = rs.sample_room(np.random.default_rng([1001, 5]))
    b = rs.sample_room(np.random.default_rng([1001, 5]))
    assert a == b


def test_cut_and_fade():
    h = np.ones(rs.L + 500)
    c = rs.cut(h)
    assert len(c) == rs.L and c[0] == 1.0 and abs(c[-1]) < 1e-12 and c[rs.L - rs.FADE - 1] == 1.0
    s = rs.cut(np.ones(100))
    assert len(s) == rs.L and s[:100].sum() == 100 and s[100:].sum() == 0


def test_direct_index():
    h = np.zeros(2000)
    d = 1.7
    n0 = int(round(d / rs.C * rs.FS)) + rs.PRA_DELAY
    h[n0 + 2] = 0.5           # direct path, slightly off the nominal sample (random ISM jitter)
    h[n0 + 40] = 0.9          # a stronger reflection later must not win
    assert rs.direct_index(h, d) == n0 + 2


@pytest.mark.parametrize("idx", [0, 1])
def test_build_room_outputs(idx):
    room, rirs, meta = rs.build_room("test", idx)
    assert room == idx and rirs.shape == (8, rs.L) and rirs.dtype == np.float16
    assert np.isfinite(rirs.astype(np.float32)).all()
    assert meta["roles"] == list(rs.ROLES) and len(meta["dp"]) == 8 and meta["max_order"] <= rs.MAX_ORDER
    for k in range(8):
        dp, dist = meta["dp"][k], meta["dist"][k]
        assert abs(dp - (dist / rs.C * rs.FS + rs.PRA_DELAY)) <= 8
        assert np.argmax(np.abs(rirs[k].astype(np.float32))) >= dp - 8      # nothing big before the direct path
        amp = abs(float(rirs[k, dp]))
        assert 0.3 / dist < amp < 2.5 / dist                               # 1/r level (plus table mirror)
    assert 0.1 < meta["rt60"] < 3.0
    again = rs.build_room("test", idx)
    assert np.array_equal(again[1], rirs) and again[2] == meta                # same seed, same room
