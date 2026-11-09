"""Offline tests for dataprep.rir_real - direct-path detection, trimming and normalisation, list and name parsing."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import rir_real as rr  # noqa: E402


def _fake_rir(n=40000, dp=3000, sign=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    h = 1e-4 * rng.standard_normal(n)
    h[dp] = sign * 0.8
    t = np.arange(n - dp - 10)
    h[dp + 10:] += 0.2 * rng.standard_normal(len(t)) * np.exp(-t / 2000.0)
    return h


def test_find_direct():
    assert rr.find_direct(_fake_rir()) == 3000


def test_prepare_trims_normalises_and_cuts():
    out, info = rr.prepare(_fake_rir(), 16000)
    assert out.shape == (rr.L_REAL,) and out.dtype == np.float32
    assert info["dp"] == rr.PRE and info["trimmed"] == 3000 - rr.PRE
    assert out[rr.PRE] == pytest.approx(1.0)                    # sign flipped so the direct path is +1
    assert abs(out[-1]) < 1e-6                                  # faded at the cut
    assert 0.0 < info["rt60"] < 5.0


def test_prepare_resamples():
    out, info = rr.prepare(_fake_rir(n=64000, dp=6000), 32000)
    assert out.shape == (rr.L_REAL,)
    assert abs(info["trimmed"] + info["dp"] - 3000) <= 2       # 6000 samples at 32 kHz = 3000 at 16 kHz


def test_parse_rir_list():
    rows = rr.parse_rir_list("--rir-id 00001 --room-id RVB2014_largeroom1 RIRS_NOISES/real/RVB2014_type1_rir_a.wav\n"
                             "\n--rir-id 00002 --room-id air_x RIRS_NOISES/real/air_type1_air_x.wav\n")
    assert rows == [{"path": "RIRS_NOISES/real/RVB2014_type1_rir_a.wav", "rir_id": "00001", "room_id": "RVB2014_largeroom1"},
                    {"path": "RIRS_NOISES/real/air_type1_air_x.wav", "rir_id": "00002", "room_id": "air_x"}]
    assert rr.slr28_db(rows[0]["path"]) == "REVERB2014" and rr.slr28_db(rows[1]["path"]) == "AIR"
    assert rr.slr28_db("x/RWCP_type1_rir_circle_e1a_imp000.wav") == "RWCP"


@pytest.mark.parametrize("member,place,cat,bar,rest,indoor", [
    ("Audio/h035_Bar_LargeSportsBar_5txts.wav", "Bar_LargeSportsBar", "bar", True, False, True),
    ("Audio/h050_Bar_IrishPub_3txts.wav", "Bar_IrishPub", "bar", True, False, True),
    ("Audio/h123_WineBar_1txts.wav", "WineBar", "bar", True, True, True),
    ("Audio/h093_Restaurant_2txts.wav", "Restaurant", "restaurant", False, True, True),
    ("Audio/h114_Restaurant_txts.wav", "Restaurant", "restaurant", False, True, True),
    ("Audio/h063_Cafeteria_3txts.wav", "Cafeteria", "restaurant", False, True, True),
    ("Audio/h049_MallFoodCourt_BurlingtonMall_3txts.wav", "MallFoodCourt_BurlingtonMall", "restaurant", False, True, True),
    ("Audio/h136_SamdwichShop_1txts.wav", "SamdwichShop", "restaurant", False, True, True),
    ("Audio/h025_Diningroom_8txts.wav", "Diningroom", "indoor", False, False, True),
    ("Audio/h053_Office_ConferenceRoom_stxts.wav", "Office_ConferenceRoom", "indoor", False, False, True),
    ("Audio/h181_Hallway_MITInfiniteCorridor_1txs.wav", "Hallway_MITInfiniteCorridor", "indoor", False, False, True),
    ("Audio/h182_Hallway_MITInfiniteCorridor_1txt.wav", "Hallway_MITInfiniteCorridor", "indoor", False, False, True),
    ("Audio/h236_BostonPubliLibrary_1txts.wav", "BostonPubliLibrary", "indoor", False, False, True),
    ("Audio/h041_TrainStation_SouthStationBoston_4txts.wav", "TrainStation_SouthStationBoston", "indoor", False, False, True),
    ("Audio/h011_Car_29txts.wav", "Car", "vehicle", False, False, True),
    ("Audio/h126_Outside_Skatepark.wav", "Outside_Skatepark", "outdoor", False, False, False),
    ("Audio/h218_StreetsOfBoston_1txts.wav", "StreetsOfBoston", "outdoor", False, False, False),
    ("Audio/h192_PorchOfSuburbanHouse_1txts.wav", "PorchOfSuburbanHouse", "outdoor", False, False, False),
])
def test_mit_names(member, place, cat, bar, rest, indoor):
    hid, p = rr.mit_place(member)
    assert hid == member.split("/")[1][:4] and p == place
    f = rr.mit_flags(p)
    assert (f["category"], f["bar"], f["restaurant"], f["indoor"]) == (cat, bar, rest, indoor)


@pytest.mark.parametrize("rt", [0.3, 0.8, 1.5])
def test_rt60_ignores_the_noise_floor(rt):
    rng = np.random.default_rng(1)
    n = 3 * 16000
    t = np.arange(n) / 16000
    h = rng.standard_normal(n) * 10 ** (-3 * t / rt) + 10 ** (-50 / 20) * rng.standard_normal(n)
    assert rr.measure_rt60_t20(h) == pytest.approx(rt, rel=0.1)
