"""Tests for dataprep.eval_dipco - tar member names, the keep/stop logic and transcript segments."""

from dataprep.eval_dipco import Keeper, member_kind, parse_time, segments


def test_parse_time():
    assert parse_time("00:02:12.79") == 132.79
    assert parse_time("01:00:00.00") == 3600.0


def test_member_kind():
    assert member_kind("Dipco/audio/eval/S08_U03.CH7.wav") == ("far", "S08", "U03", "CH7", "eval")
    assert member_kind("Dipco/audio/dev/S02_P05.wav") == ("close", "S02", "P05", "dev")
    assert member_kind("Dipco/transcriptions/eval/S01.json") == ("json", "eval", "S01")
    assert member_kind("Dipco/README.md") == ("doc", "README.md")
    assert member_kind("Dipco/audio/eval/._S08_U03.CH7.wav") is None      # macOS resource fork
    assert member_kind("Dipco/.DS_Store") is None


def test_keeper_one_far_channel_per_session_and_disk_cap():
    k = Keeper(disk_cap=100)
    a = ("far", "S08", "U03", "CH7", "eval")
    assert k.want(a)
    k.kept(a, 40)
    assert not k.want(("far", "S08", "U02", "CH1", "eval"))   # second channel of a kept session
    assert k.want(a)                                           # the same file again (resume) is fine
    assert k.want(("far", "S01", "U02", "CH1", "eval"))
    assert k.want(("close", "S08", "P25", "eval"))
    k.kept(("close", "S08", "P25", "eval"), 70)
    assert k.full() and not k.want(("far", "S07", "U01", "CH3", "eval"))
    # resume: sessions and bytes already kept are remembered
    r = Keeper(disk_cap=100, have_far={"S08": "S08_U03.CH7"}, written=10)
    assert not r.want(("far", "S08", "U01", "CH1", "eval")) and r.want(a)


def test_segments():
    utts = [{"session_id": "S08", "speaker_id": "P25", "words": "hello", "gender": "male",
             "nativeness": "non-native", "mother_tongue": "Hindi",
             "start_time": {"U03": "00:00:09.08", "close-talk": "00:00:09.10"},
             "end_time": {"U03": "00:00:09.96", "close-talk": "00:00:10.00"}}]
    seg = segments(utts, "U03", "eval/dipco/S08_U03.CH7.flac", {"P25": "eval/dipco/S08_P25.flac"})[0]
    assert seg["speaker"] == "dipco:P25" and seg["gender"] == "M"
    assert (seg["far_start"], seg["far_end"]) == (9.08, 9.96)
    assert (seg["close_start"], seg["close_end"]) == (9.1, 10.0)
    assert seg["close_path"] == "eval/dipco/S08_P25.flac"
    none = segments(utts, None, None, {})[0]
    assert none["far_path"] is None and none["far_start"] is None and none["close_path"] is None
