"""Tests for dataprep.eval_libri2mix - CSV parsing, enrollment choice, LibriMix noise extension and rendering."""

import numpy as np
import soundfile as sf

from dataprep.eval_libri2mix import (ENROLL_RULE, build_definition, extend_noise, parse_info_csv, parse_mixture_csv,
                                     pick_enrollment, render, utt_id)

MIX_CSV = (
    "mixture_ID,source_1_path,source_1_gain,source_2_path,source_2_gain,noise_path,noise_gain\n"
    "1-10-0001_2-20-0001,test-clean/1/10/1-10-0001.flac,0.5,test-clean/2/20/2-20-0001.flac,2.0,tt/n1.wav,1.5\n"
)
INFO_CSV = ("mixture_ID,speaker_1_ID,speaker_1_sex,speaker_2_ID,speaker_2_sex\n"
            "1-10-0001_2-20-0001,1.0,F,2.0,M\n")


def test_parse_csvs():
    rows = parse_mixture_csv(MIX_CSV)
    assert rows == [{"id": "1-10-0001_2-20-0001",
                     "sources": [("test-clean/1/10/1-10-0001.flac", 0.5), ("test-clean/2/20/2-20-0001.flac", 2.0)],
                     "noise": ("tt/n1.wav", 1.5)}]
    assert parse_info_csv(INFO_CSV) == {"1-10-0001_2-20-0001": [("1", "F"), ("2", "M")]}
    assert utt_id("test-clean/4077/13754/4077-13754-0001.flac") == "4077-13754-0001"


def test_pick_enrollment_rules():
    utts = {"1-10-0001": 5.0, "1-10-0002": 2.0, "1-10-0003": 4.0, "1-11-0001": 6.0}
    e = pick_enrollment("m", "1-10-0001", utts, {"1-10-0001", "2-20-0001"})
    assert e in ("1-10-0003", "1-11-0001")                      # never the target, never shorter than 3 s
    assert e == pick_enrollment("m", "1-10-0001", utts, {"1-10-0001"})   # deterministic
    # only short ones left -> falls back to a short one rather than nothing
    assert pick_enrollment("m", "1-10-0001", {"1-10-0001": 5.0, "1-10-0002": 2.0}, set()) == "1-10-0002"
    # nothing else -> None
    assert pick_enrollment("m", "1-10-0001", {"1-10-0001": 5.0}, set()) is None
    # different mixtures spread over the candidates
    picks = {pick_enrollment(f"mix{k}", "1-10-0001", utts, set()) for k in range(40)}
    assert picks == {"1-10-0003", "1-11-0001"}


def test_extend_noise_matches_librimix_crossfade():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(20000).astype(np.float32)
    out = extend_noise(noise, 30000)                     # one pass of LibriMix's loop: 11999 + 8001 + 11999 samples
    w = np.hanning(16001)
    i_w, d_w = w[:8001], w[8000::-1]
    assert len(out) == 30000
    assert np.allclose(out[:11999], noise[:11999])                                       # untouched head
    assert np.allclose(out[11999:20000], noise[11999:] * d_w + noise[:8001] * i_w)        # Hann cross-fade
    assert np.allclose(out[20000:], noise[8001:18001])                                   # the clip again
    short = noise[:10000]                                  # shorter than 2 windows: the loop runs several times
    assert len(extend_noise(short, 20000)) == 20000
    assert np.allclose(extend_noise(short, 20000)[:1999], short[:1999])
    assert np.allclose(extend_noise(noise, 5000), noise[:5000])


def _write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, 16000, format="FLAC", subtype="PCM_16")


def test_build_and_render(tmp_path):
    rng = np.random.default_rng(1)
    s1 = (0.1 * rng.standard_normal(32000)).astype(np.float32)
    s2 = (0.1 * rng.standard_normal(24000)).astype(np.float32)
    noise = (0.05 * rng.standard_normal(30000)).astype(np.float32)
    _write(tmp_path / "LibriSpeech/test-clean/1/10/1-10-0001.flac", s1)
    _write(tmp_path / "LibriSpeech/test-clean/2/20/2-20-0001.flac", s2)
    _write(tmp_path / "LibriSpeech/test-clean/1/10/1-10-0002.flac", s1[:16000])
    _write(tmp_path / "eval/wham_tt/n1.flac", noise * 2)        # stored at 2x, as if rescaled by 2
    s1, s2 = (sf.read(tmp_path / p, dtype="float32")[0] for p in
              ("LibriSpeech/test-clean/1/10/1-10-0001.flac", "LibriSpeech/test-clean/2/20/2-20-0001.flac"))
    stored = sf.read(tmp_path / "eval/wham_tt/n1.flac", dtype="float32")[0]
    row = parse_mixture_csv(MIX_CSV)[0]
    lengths = {"1-10-0001": 32000, "2-20-0001": 24000, "1-10-0002": 16000}
    by_spk = {"1": {"1-10-0001": 2.0, "1-10-0002": 1.0}, "2": {"2-20-0001": 1.5}}
    d = build_definition(row, parse_info_csv(INFO_CSV)["1-10-0001_2-20-0001"], lengths, by_spk, "eval/wham_tt/n1.flac",
                         noise_scale=2.0)
    assert d["length"] == 24000 and d["max_length"] == 32000 and d["mode"] == "min"
    assert [s["offset"] for s in d["sources"]] == [0, 0]
    assert d["sources"][0]["speaker"] == "librispeech:1" and d["sources"][1]["sex"] == "M"
    assert d["noise"]["gain"] == 0.75 and d["noise"]["gain_librimix"] == 1.5
    assert d["enrollment"][0]["utt"] == "1-10-0002" and d["enrollment"][0]["rule"] == ENROLL_RULE
    assert d["enrollment"][1] is None                           # speaker 2 has no other utterance
    assert d["use"] == ["eval"] and d["split"] == "test"
    mix, srcs, n = render(d, tmp_path)
    assert len(mix) == 24000 and all(len(s) == 24000 for s in srcs)
    want_noise = extend_noise(stored, 32000)[:24000] * 0.75    # clip shorter than max_length -> LibriMix extension
    assert np.allclose(n, want_noise, atol=1e-6)
    assert np.allclose(mix, s1[:24000] * 0.5 + s2[:24000] * 2.0 + want_noise, atol=1e-5)
    clean, _, none = render(d, tmp_path, noisy=False)
    assert none is None and np.allclose(clean, s1[:24000] * 0.5 + s2[:24000] * 2.0, atol=1e-6)
