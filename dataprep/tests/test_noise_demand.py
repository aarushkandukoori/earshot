"""Offline tests for dataprep.noise_demand - scene splits, channels, file names and manifest entries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import noise_demand as nd  # noqa: E402


def test_scene_splits():
    test = [s for s, (split, _, _) in nd.SCENES.items() if split == "test"]
    assert test == ["SCAFE"]
    train = {s for s, (split, _, _) in nd.SCENES.items() if split == "train"}
    assert train == {"PCAFETER", "PRESTO", "OMEETING", "OHALLWAY", "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TMETRO"}


def test_zip_names_and_channels():
    assert nd.zip_name("SCAFE") == "SCAFE_48k.zip"     # SCAFE exists only at 48 kHz
    assert nd.zip_name("PRESTO") == "PRESTO_16k.zip"
    assert nd.CHANNELS == (1, 4, 13, 16)                # the four corners of the 4 x 4 array
    assert nd.rel_path("PRESTO", 4) == "noise/demand/PRESTO/ch04.flac"


def test_entry_fields():
    e = nd.make_entry("SCAFE", 13, 300.0)
    assert e["split"] == "test" and e["license"] == "CC-BY-SA-3.0" and e["use"] == ["noise"]
    assert e["group"] == "SCAFE" and e["extra"]["channel"] == 13 and e["extra"]["category"] == "restaurant"
    assert e["speaker"] is None and e["sr"] == 16000
