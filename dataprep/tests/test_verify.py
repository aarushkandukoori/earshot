"""Tests for dataprep.verify - speaker id matching, license rules and every check on a tiny synthetic pack."""

import json

import numpy as np
import soundfile as sf

from dataprep.common import entry
from dataprep.verify import canonical_speaker, eval_only_license, license_ids, run

LICENSES = """# Licenses
| id | meaning |
|---|---|
| `CC-BY-4.0` | Attribution |
| `CC0-1.0` | public domain |
| `CC-BY-NC-4.0` | non-commercial, eval only |
"""


def test_canonical_speaker():
    assert canonical_speaker("librittsr:0103") == "libri:103"
    assert canonical_speaker("LibriSpeech:103") == "libri:103"
    assert canonical_speaker("libritts_r:103") == "libri:103"
    assert canonical_speaker("103", source="librittsr") == "libri:103"   # no prefix -> source decides
    assert canonical_speaker("vctk:p225") == "vctk:p225"
    assert canonical_speaker("globe:p225") != canonical_speaker("vctk:p225")
    assert canonical_speaker("voxpopuli:28221") == "voxpopuli:28221"
    assert canonical_speaker(None) is None


def test_license_rules():
    assert license_ids(LICENSES) == {"CC-BY-4.0", "CC0-1.0", "CC-BY-NC-4.0"}
    assert eval_only_license("CC-BY-NC-4.0") and eval_only_license("CC-BY-NC-SA-4.0")
    assert eval_only_license("BUT-NTT-EVAL")
    assert not eval_only_license("CC-BY-4.0") and not eval_only_license("CC0-1.0")


def _flac(root, rel, seconds=1.0, sr=16000, channels=1):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    x = 0.1 * np.sin(np.arange(int(seconds * sr)) * 0.05)
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    sf.write(p, x, sr, format="FLAC", subtype="PCM_16")


def _pack(tmp_path, train, test, extra_files=()):
    data, mdir = tmp_path / "data", tmp_path / "data" / "manifests"
    mdir.mkdir(parents=True)
    for name, rows in (("speech_train.jsonl", train), ("eval_x.jsonl", test)):
        (mdir / name).write_text("".join(json.dumps(e) + "\n" for e in rows))
    for e in train + test:
        if e["path"] not in extra_files:
            _flac(data, e["path"], e["duration"])
    lic = tmp_path / "LICENSES.md"
    lic.write_text(LICENSES)
    return data, mdir, lic


def _run(tmp_path, data, mdir, lic):
    return run(data=data, mdir=mdir, licenses_md=lic, full=True, workers=2, summary=tmp_path / "SUMMARY.md")[0]


def test_clean_pack_passes(tmp_path):
    train = [entry("speech/librittsr/14/a.flac", "librittsr", 1.0, "train", "CC-BY-4.0", ["target"],
                   speaker="librittsr:14")]
    test = [entry("LibriSpeech/test-clean/1089/1/b.flac", "librispeech", 1.0, "test", "CC-BY-4.0", ["eval"],
                  speaker="librispeech:1089"),
            entry("eval/wham_tt/n.flac", "wham", 1.0, "test", "CC-BY-NC-4.0", ["eval"])]
    data, mdir, lic = _pack(tmp_path, train, test)
    rep = _run(tmp_path, data, mdir, lic)
    assert rep.violations == []
    text = (tmp_path / "SUMMARY.md").read_text()
    assert "PASS" in text and "| librittsr | train | 1 | 1 |" in text


def test_every_violation_is_caught(tmp_path):
    train = [
        entry("speech/librittsr/1089/a.flac", "librittsr", 1.0, "train", "CC-BY-4.0", ["target"],
              speaker="librittsr:1089"),                                   # same person as the test speaker
        entry("speech/x/s1.flac", "x", 1.0, "train", "CC-BY-4.0", ["target"], speaker="x:1"),
        entry("speech/x/s2.flac", "x", 1.0, "val", "CC-BY-4.0", ["target"], speaker="x:1"),   # crosses splits
        entry("noise/nc.flac", "fsd", 1.0, "train", "CC-BY-NC-4.0", ["noise"]),             # NC in training
        entry("noise/weird.flac", "fsd", 1.0, "train", "Sampling+", ["noise"]),             # unknown license
        entry("speech/x/missing.flac", "x", 1.0, "train", "CC-BY-4.0", ["target"], speaker="x:2"),
        entry("speech/x/bad_use.flac", "x", 1.0, "train", "CC-BY-4.0", ["rir"], speaker="x:3"),
    ]
    test = [entry("LibriSpeech/test-clean/1089/1/b.flac", "librispeech", 2.0, "test", "CC-BY-4.0", ["eval"],
                  speaker="librispeech:1089")]
    data, mdir, lic = _pack(tmp_path, train, test, extra_files={"speech/x/missing.flac"})
    _flac(data, "LibriSpeech/test-clean/1089/1/b.flac", 2.0, sr=22050)     # wrong sample rate on disk
    rep = _run(tmp_path, data, mdir, lic)
    v = "\n".join(rep.violations)
    assert "eval/test speaker libri:1089" in v
    assert "speaker x:1 is in splits ['train', 'val']" in v
    assert "evaluation-only license CC-BY-NC-4.0" in v
    assert "license 'Sampling+'" in v
    assert "missing file speech/x/missing.flac" in v
    assert "use ['rir']" in v
    assert "22050 Hz" in v
    assert "FAIL" in (tmp_path / "SUMMARY.md").read_text()


def test_definitions_and_partial_lines(tmp_path):
    test = [entry("LibriSpeech/test-clean/1/1/a.flac", "librispeech", 1.0, "test", "CC-BY-4.0", ["eval"],
                  speaker="librispeech:1")]
    data, mdir, lic = _pack(tmp_path, [], test)
    (mdir / "speech_train.jsonl").write_text('{"path": "half')                  # writer still running
    (data / "eval").mkdir()
    d = {"sources": [{"path": "LibriSpeech/test-clean/1/1/a.flac"}], "noise": {"path": "eval/wham_tt/gone.flac"},
         "enrollment": [None]}
    (data / "eval" / "libri2mix_test.jsonl").write_text(json.dumps(d) + "\n")
    rep = _run(tmp_path, data, mdir, lic)
    assert any("incomplete last line" in w for w in rep.warnings)
    assert any("references to missing files under wham_tt/" in x for x in rep.violations)


def test_rir_banks(tmp_path):
    data, mdir, lic = _pack(tmp_path, [], [entry("LibriSpeech/test-clean/1/1/a.flac", "librispeech", 1.0, "test",
                                                 "CC-BY-4.0", ["eval"], speaker="librispeech:1")])
    rdir = data / "rir"
    rdir.mkdir()
    np.save(rdir / "sim_test.npy", np.zeros((3, 2, 100), dtype=np.float16))
    (rdir / "sim_test.jsonl").write_text("".join(json.dumps({"room": k, "split": "test"}) + "\n" for k in range(3)))
    np.save(rdir / "real_train.npy", np.zeros((2, 100), dtype=np.float16))
    (rdir / "real_train.jsonl").write_text(json.dumps({"license": "Weird-1.0"}) + "\n")     # 1 line for 2 rows
    bad = np.zeros((1, 10), dtype=np.float16)
    bad[0, 3] = np.nan
    np.save(rdir / "real_test.npy", bad)
    (rdir / "real_test.jsonl").write_text(json.dumps({"license": "CC-BY-4.0"}) + "\n")
    np.save(rdir / "sim_train.partial.npy", np.zeros((1, 10), dtype=np.float16))
    lic.write_text(LICENSES + "| `generated` | ours |\n")
    rep = _run(tmp_path, data, mdir, lic)
    v = "\n".join(rep.violations)
    assert "rir/real_train.npy: 2 rows but real_train.jsonl has 1 lines" in v
    assert "rir/real_train.npy: license 'Weird-1.0'" in v
    assert "rir/real_test.npy: NaN or Inf values" in v
    assert "sim_test" not in v
    assert any("sim_train.partial.npy" in w for w in rep.warnings)
    assert "| rir/sim_test.npy | test | 3 | 3x2x100 |" in (tmp_path / "SUMMARY.md").read_text()
