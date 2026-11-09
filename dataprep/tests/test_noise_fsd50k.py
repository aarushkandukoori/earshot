"""Offline tests for dataprep.noise_fsd50k - license rule, category priority, metadata parsing and clip selection."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import noise_fsd50k as fsd  # noqa: E402


def test_license_rule():
    assert fsd.fsd_license_ok("CC0-1.0") and fsd.fsd_license_ok("CC-BY-3.0") and fsd.fsd_license_ok("CC-BY-4.0")
    for bad in ("CC-BY-NC-3.0", "CC-Sampling+-1.0", "CC-BY-SA-3.0", "CC-BY-ND-4.0", None, ""):
        assert not fsd.fsd_license_ok(bad)


def test_category_priority():
    assert fsd.category_for(["Applause", "Music"]) == ("music", "Music")
    assert fsd.category_for(["Glass", "Chink_and_clink"]) == ("dishes", "Chink_and_clink")
    assert fsd.category_for(["Chatter", "Speech", "Human_voice"]) == ("crowd", "Chatter")
    assert fsd.category_for(["Speech"]) == (None, None)


def test_parse_rows():
    dev = 'fname,labels,mids,split\n1,"Chatter,Human_group_actions",/m/x,train\n2,Music,/m/y,val\n'
    ev = 'fname,labels,mids\n3,"Door,Domestic_sounds_and_home_sounds",/m/z\n'
    info = {"1": {"license": "http://creativecommons.org/licenses/by/3.0/", "uploader": "ann", "title": "t1"},
            "2": {"license": "http://creativecommons.org/publicdomain/zero/1.0/", "uploader": "bob", "title": "t2"}}
    rows = fsd.parse_rows(dev, ev, info, {"3": {"license": "http://creativecommons.org/licenses/by-nc/3.0/"}})
    assert [(r["fname"], r["set"], r["split"], r["license"]) for r in rows] == [
        ("1", "dev", "train", "CC-BY-3.0"), ("2", "dev", "val", "CC0-1.0"), ("3", "eval", "test", "CC-BY-NC-3.0")]
    assert rows[0]["labels"] == ["Chatter", "Human_group_actions"] and rows[0]["uploader"] == "ann"


def _rows():
    rows, dur = [], {}
    rnd = random.Random(0)
    for i in range(400):
        set_name = "dev" if i < 300 else "eval"
        split = ("train" if i % 10 else "val") if set_name == "dev" else "test"
        label = rnd.choice(["Music", "Chatter", "Door", "Dishes_and_pots_and_pans", "Speech"])
        lic = rnd.choice(["CC0-1.0", "CC-BY-3.0", "CC-BY-NC-3.0", "CC-Sampling+-1.0"])
        rows.append({"fname": str(i), "set": set_name, "split": split, "labels": [label], "license": lic,
                     "license_url": None, "uploader": f"u{i % 7}", "title": None})
        dur[(set_name, str(i))] = rnd.uniform(1.0, 20.0)
    return rows, dur


def test_select_respects_caps_licenses_and_splits():
    rows, dur = _rows()
    caps = {("dev", "music"): 0.02, ("dev", "crowd"): None, ("dev", "door"): 0.01, ("dev", "dishes"): None,
            ("eval", "crowd"): 0.005}
    picks = fsd.select(rows, dur, caps, music_min_s=5.0)
    assert picks
    for p in picks:
        assert fsd.fsd_license_ok(p["license"])
        assert (p["set"], p["category"]) in caps
        assert p["category"] != "music" or p["src_seconds"] >= 5.0
        assert (p["set"] == "eval") == (p["split"] == "test")
    hours = {}
    for p in picks:
        hours[(p["set"], p["category"])] = hours.get((p["set"], p["category"]), 0) + p["src_seconds"] / 3600
    for key, cap in caps.items():
        if cap is not None:
            assert hours.get(key, 0) <= cap + 1e-9
    # every licensed, uncapped dev crowd clip is kept
    want = {r["fname"] for r in rows if r["set"] == "dev" and r["labels"] == ["Chatter"]
            and fsd.fsd_license_ok(r["license"])}
    assert want == {p["fname"] for p in picks if p["set"] == "dev" and p["category"] == "crowd"}
    assert not any(p["set"] == "eval" and p["category"] in ("door", "dishes", "music") for p in picks)


def test_select_is_deterministic():
    rows, dur = _rows()
    caps = dict(fsd.DEFAULT_CAPS)
    a = [p["fname"] for p in fsd.select(rows, dur, caps)]
    rows2 = rows[:]
    random.Random(1).shuffle(rows2)
    assert sorted(a) == sorted(p["fname"] for p in fsd.select(rows2, dur, caps))


def test_paths():
    assert fsd.rel_path("eval", "123") == "noise/fsd50k/eval/123.flac"
