"""Offline tests for dataprep.noise_musan - LICENSE / ANNOTATIONS parsing and the stream-then-commit flow,
run on a tiny synthetic musan.tar.gz written to a temporary folder."""

import io
import json
import logging
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import common  # noqa: E402
from dataprep import noise_musan as nm  # noqa: E402


REAL_FORMAT = """music-fma-0027
“No Home” (by Cullah)
music-fma-0026
“Where You Do Belong” (by Cullah)
CC BY-SA 4.0
http://freemusicarchive.org/music/MC_Cullah_1450/Cullah_The_Wild/
==============================================================================
music-fma-0097
"Tiny Robot Armies” (by Quiet Music for Tiny Robots)
music-fma-0099
CC BY 4.0
http://freemusicarchive.org/music/Quiet_Music_for_Tiny_Robots/The_February_Album
=============================================================================
music-fma-wa-0003
music-fma-wa-0006
Selections from the November 2006 Concert by Advent Chamber Orchestra is licensed under a Attribution-Share Alike 3.0 United States License.
http://freemusicarchive.org/music/Advent_Chamber_Orchestra/Selections_from_the_November_2006_Concert/
==============================================================================
music-hd-0039
Gymnopedie No. 1
Composer: Erik Satie
License: CC BY 3.0
http://www.hdclassicalmusic.com/listen/?id_song=512
==================================================
music-hd-0034
Composer:Joseph Haydn
License:Public Domain
==================================================
music-jamendo-0213
Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
https://www.jamendo.com/en/list/a143840/x
=============================================================================
music-jamendo-0300
Some song
"""


def test_license_real_block_format():
    # copied from the shape of MUSAN's music/*/LICENSE files - ids first, then the license, then a URL
    per, default = nm.parse_license(REAL_FORMAT)
    assert per["music-fma-0027"] == per["music-fma-0026"] == "CC-BY-SA-4.0"
    assert per["music-fma-0097"] == per["music-fma-0099"] == "CC-BY-4.0"
    assert per["music-fma-wa-0003"] == per["music-fma-wa-0006"] == "CC-BY-SA-3.0"
    assert per["music-hd-0039"] == "CC-BY-3.0"
    assert per["music-hd-0034"] == "PD"
    assert per["music-jamendo-0213"] == "CC-BY-NC-SA-3.0"        # non-commercial, dropped by license_ok
    assert per["music-jamendo-0300"] is None                      # block without a license line
    assert default is None


def test_license_header_block_then_ids():
    # music/rfm/LICENSE - one license block first, then every id with its song title, no license after them
    text = ("CC BY 3.0\nhttp://incompetech.com/music/royalty-free/index.html?genre=Blues\n" + "=" * 78 + "\n"
            'music-rfm-0058\n"Hustle" (by Kevin MacLeod)\nmusic-rfm-0144\n"Whiskey on the Mississippi" (by Kevin MacLeod)\n')
    per, default = nm.parse_license(text)
    assert per == {"music-rfm-0058": "CC-BY-3.0", "music-rfm-0144": "CC-BY-3.0"} and default is None


def test_license_id_and_url_on_one_line():
    per, _ = nm.parse_license("music-fma-0000.wav http://creativecommons.org/licenses/by/3.0/\n"
                              "music-fma-0001.wav http://creativecommons.org/licenses/by-nc/3.0/\n")
    assert per == {"music-fma-0000": "CC-BY-3.0", "music-fma-0001": "CC-BY-NC-3.0"}


def test_license_whole_folder():
    per, default = nm.parse_license("All files in this directory are released under\n"
                                    "http://creativecommons.org/licenses/by-sa/3.0/\n")
    assert per == {} and default == "CC-BY-SA-3.0"


def test_annotations():
    a = nm.parse_annotations("music-fma-0026 blues Y Cullah\nmusic-fma-0098 pop,electronica N Quiet_Music\n", "music")
    assert a["music-fma-0026"] == {"genre": "blues", "vocals": True, "artist": "Cullah"}
    assert a["music-fma-0098"]["vocals"] is False
    b = nm.parse_annotations("noise-free-sound-0001 background crowd at a party\n", "noise")
    assert b["noise-free-sound-0001"]["annotation"] == "background crowd at a party"
    assert nm.noise_category(b["noise-free-sound-0001"]["annotation"]) == "crowd"
    assert nm.noise_category(None) == "misc"


def _wav(seconds=0.5, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    sf.write(buf, (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members:
            if data is None:
                ti = tarfile.TarInfo(name)
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    return buf


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path / "data")
    monkeypatch.setattr(common, "MANIFESTS", tmp_path / "data" / "manifests")
    monkeypatch.setattr(nm, "MUSIC_CANDIDATE_FRAC", 1.0)
    monkeypatch.setattr(nm, "EXPECTED", {"music": 4, "noise": 2})
    return tmp_path / "data"


def test_stream_then_commit(tmp_data):
    lic_music = ("music-fma-0000 http://creativecommons.org/licenses/by/3.0/\n"
                 "music-fma-0001 http://creativecommons.org/licenses/by-nc/3.0/\n"
                 "music-fma-0002 http://creativecommons.org/licenses/by-sa/3.0/\n"
                 "music-fma-0003 http://creativecommons.org/publicdomain/zero/1.0/\n"
                 "music-fma-0004 http://creativecommons.org/licenses/by/4.0/\n").encode()
    tar = _tar([
        ("musan", None), ("musan/README", b"MUSAN readme"),
        ("musan/music", None), ("musan/music/fma", None),
        *[(f"musan/music/fma/music-fma-000{i}.wav", _wav(seed=i)) for i in range(5)],
        ("musan/music/fma/ANNOTATIONS", b"music-fma-0000 rock Y Band_A\nmusic-fma-0002 jazz N Band_B\n"),
        ("musan/music/fma/LICENSE", lic_music),                      # LICENSE after the audio it describes
        ("musan/noise", None), ("musan/noise/free-sound", None),
        ("musan/noise/free-sound/LICENSE", b"All files: http://creativecommons.org/licenses/by/3.0/\n"),
        ("musan/noise/free-sound/noise-free-sound-0000.wav", _wav(seed=7)),
        ("musan/noise/free-sound/noise-free-sound-0001.wav", _wav(seed=8)),
        ("musan/speech", None), ("musan/speech/librivox", None),
        ("musan/speech/librivox/speech-librivox-0000.wav", _wav(seed=9)),
    ])
    log = logging.getLogger("test_musan")
    done = nm.stream(log, fileobj=tar)
    assert done["stopped_early"] is True                  # stopped when speech/ began
    assert not (tmp_data / "noise/musan/speech").exists()
    assert (tmp_data / "noise/musan/meta/music/fma/LICENSE").exists()
    assert (tmp_data / "noise/musan/meta/README").read_bytes() == b"MUSAN readme"

    stats = nm.commit(log, music_hours=1.2 / 3600)       # room for three 0.5 s music files
    assert stats["dropped_license:CC-BY-NC-3.0"] == 1
    assert not (tmp_data / "noise/musan/music/fma/music-fma-0001.flac").exists()
    assert stats["noise_kept"] == 2 and stats["music_kept"] == 3 and stats["music_extra_deleted"] == 1

    lines = [json.loads(x) for s in ("train", "val", "test")
             for x in (tmp_data / f"manifests/noise_{s}.jsonl").read_text().splitlines()]
    assert len(lines) == 5
    music = [e for e in lines if e["use"] == ["music"]]
    assert all(e["extra"]["category"] == "music" and e["license"] != "CC-BY-NC-3.0" for e in music)
    noise = [e for e in lines if e["use"] == ["noise"]]
    assert {e["license"] for e in noise} == {"CC-BY-3.0"}
    assert all(e["split"] in ("train", "val") and e["sr"] == 16000 for e in lines)

    # running the commit again changes nothing (resume safety)
    again = nm.commit(log, music_hours=1.2 / 3600)
    assert again["music_kept"] == 3 and again.get("music_extra_deleted", 0) == 0
    total = sum(len((tmp_data / f"manifests/noise_{s}.jsonl").read_text().splitlines()) for s in ("train", "val", "test"))
    assert total == 5
