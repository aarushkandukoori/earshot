"""Offline test for the dataprep.speech_fetch command line (stats mode on an empty data folder)."""

from dataprep import common, speech_common as sc, speech_fetch


def test_stats_on_empty_data(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    for mod in (common, sc):
        monkeypatch.setattr(mod, "DATA", data)
        monkeypatch.setattr(mod, "MANIFESTS", data / "manifests")
    monkeypatch.setattr(common, "LOGS", tmp_path / "logs")  # keep test lines out of the real data/logs
    monkeypatch.setattr(sc, "SPEECH", data / "speech")
    monkeypatch.setattr(speech_fetch, "SPEECH", data / "speech")
    assert speech_fetch.main(["--stats", "--source", "all"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out[:4]] == ["vctk", "globe", "voxpopuli", "librittsr"]
    assert "0 speakers" in out[0] and out[-1].startswith("data/speech total 0.000 GB")


def test_budget_shares_fit_the_cap():
    assert abs(sum(sc.BUDGET_GB.values()) - sc.TOTAL_BUDGET_GB) < 1e-9
