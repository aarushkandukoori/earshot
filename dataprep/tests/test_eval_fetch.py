"""Tests for dataprep.eval_fetch - zip read planning and the small metadata parsers."""

import zipfile

from dataprep.eval_fetch import librispeech_speakers, librispeech_transcripts, parse_wham_noise_meta, plan_spans


def _info(name, offset, csize):
    i = zipfile.ZipInfo(name)
    i.header_offset, i.compress_size = offset, csize
    return i


def test_plan_spans_groups_neighbours_and_respects_limits():
    a = _info("tt/a.wav", 0, 1000)
    b = _info("tt/b.wav", 2100, 1000)          # right after a (a ends near 2060)
    c = _info("tt/c.wav", 10_000_000, 1000)    # far away
    spans = plan_spans([c, b, a], max_gap=64 << 10, max_span=48 << 20)
    assert [[m.filename for m in s.members] for s in spans] == [["tt/a.wav", "tt/b.wav"], ["tt/c.wav"]]
    assert spans[0].start == 0 and spans[0].end >= 2100 + 30 + len("tt/b.wav") + 1000
    small = plan_spans([a, b], max_gap=64 << 10, max_span=3000)   # a span may not grow past max_span
    assert len(small) == 2


def test_librispeech_parsers(tmp_path):
    (tmp_path / "SPEAKERS.TXT").write_text(";ID  |SEX| SUBSET |MINUTES| NAME\n"
                                           "1089 | M | test-clean | 8.0 | Someone\n14 | F | train-clean-360 | 25 | X\n")
    assert librispeech_speakers(tmp_path / "SPEAKERS.TXT") == {"1089": ("M", "test-clean"), "14": ("F", "train-clean-360")}
    d = tmp_path / "test-clean" / "1089" / "134686"
    d.mkdir(parents=True)
    (d / "1089-134686.trans.txt").write_text("1089-134686-0000 HE HOPED\n1089-134686-0001 STUFF IT\n")
    assert librispeech_transcripts(tmp_path / "test-clean") == {"1089-134686-0000": "HE HOPED",
                                                                 "1089-134686-0001": "STUFF IT"}


def test_parse_wham_noise_meta():
    text = ("utterance_id,Noise Band,File ID,L to R Width (cm),Reverberation Level,Location ID,Location Day ID\n"
            "445c0206_0.60431_22gc0105_-0.60431.wav,3,file248,17 cm,h,loc24,day1\n")
    m = parse_wham_noise_meta(text)["445c0206_0.60431_22gc0105_-0.60431.wav"]
    assert m == {"recording": "file248", "location": "loc24", "day": "day1", "noise_band": "3", "reverb": "h",
                 "lr_width": "17 cm"}
