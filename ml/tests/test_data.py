"""Tests for the data loader: sources.py, render.py, enroll.py, data.py (10-ml 8 and 14, contract 8.1).

They read the real data pack (data/manifests, data/rir) and skip if it is missing. The batch tests also need the
own-voice cache ml/cache/own_v1.npz (`python -m ml.tools.cache_own_embeddings`) and skip without it.
"""
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import multiprocessing

import numpy as np
import pytest

from ml.earshot import data, dsp, render, sources
from ml.earshot.config import HOP, MAX_O_TOK, MAX_T_TOK, SR

pytestmark = pytest.mark.skipif(not (sources.MANIFESTS / "speech_train.jsonl").exists(),
                                reason="data pack not present")
needs_cache = pytest.mark.skipif(not sources.OWN_CACHE_PATH.exists(),
                                 reason="ml/cache/own_v1.npz not built (python -m ml.tools.cache_own_embeddings)")

L4 = 4 * SR          # 4 s segment
LE6 = 6 * SR         # 6 s enrollment


@pytest.fixture(scope="module")
def src():
    return sources.get_sources("train")


def example(src, index, recipe="B", L=L4, Le=LE6, seed=11, **kwargs):
    """One example from its own generator (seed, index), as the loader makes them."""
    rng = np.random.default_rng([seed, index])
    return data.make_example(src, recipe, rng, L, Le, **kwargs)


def db(p):
    return 10.0 * math.log10(p)


# ---------------------------------------------------------------------------------------------------------------
# License gate, splits and pools
# ---------------------------------------------------------------------------------------------------------------
def row(license="CC-BY-4.0", split="train", use=("target",)):
    return {"path": "x.flac", "license": license, "split": split, "use": list(use)}


def test_license_gate():
    sources.check_row(row(), "train")                                   # a normal train row passes
    for lic in ["CC0-1.0", "PD", "CC-BY", "CC-BY-SA-3.0", "Apache-2.0", "cc-by-4.0"]:
        sources.check_row(row(lic), "train")                            # the pack's licenses, any case
    sources.check_row(row("CC-BY-NC-4.0", "test"), "test")              # NC is allowed in test
    sources.check_row(row("cc by nc 4.0", "test", ("eval",)), "test")   # eval rows are fine in test
    assert not sources.is_non_commercial("NCSA")                        # the token must be NC (or NC + digits)
    for bad in [row("CC-BY-NC-4.0"), row("cc-by-nc-sa-3.0"), row("CC BY NC"), row(""), row(None),
                row(use=("eval",)), row(split="val")]:
        with pytest.raises(sources.LicenseError):
            sources.check_row(bad, "train")
    with pytest.raises(sources.LicenseError):
        sources.check_row(row("CC-BY-NC-4.0", "val"), "val")
    # Other ways of writing a non-commercial license (found by the ML-1 review) are refused as non-commercial ...
    for lic in ["CC_BY_NC_4.0", "CC-BY-NC4.0", "CC BY-NC/4.0", "CC-BY-NC,4.0", "CC-BY-NonCommercial",
                "Creative Commons Attribution-NonCommercial 4.0", "CC-BY-4.0 (NC)"]:
        with pytest.raises(sources.LicenseError, match="non-commercial"):
            sources.check_row(row(lic), "train")
    # ... and licenses that are not on the allow-list are refused outside test, but fine in test.
    for lic in ["research-only", "unknown", "NCSA", "MIT"]:
        with pytest.raises(sources.LicenseError, match="TRAIN_LICENSES"):
            sources.check_row(row(lic), "train")
        sources.check_row(row(lic, "test"), "test")


def test_room_rows_go_through_the_gate():
    sources.check_rir_row({"name": "r", "license": "Apache-2.0", "row": 0}, "train")     # measured room
    sources.check_rir_row({"room": 3, "split": "val"}, "val")                           # simulated room
    with pytest.raises(sources.LicenseError, match="non-commercial"):
        sources.check_rir_row({"name": "r", "license": "CC-BY-NC-4.0"}, "train")
    with pytest.raises(sources.LicenseError, match="simulated room"):
        sources.check_rir_row({"room": 3, "split": "test"}, "train")
    # Every room row on disk passes for its own split.
    for split in ["train", "val", "test"]:
        assert len(sources.read_checked_rirs(sources.RIR_DIR / f"sim_{split}.jsonl", split)) > 0
    assert len(sources.read_checked_rirs(sources.RIR_DIR / "real_train.jsonl", "train")) == 325


def test_same_person_under_two_prefixes_is_caught(tmp_path):
    assert sources.person_id("librittsr:1089") == sources.person_id("librispeech:1089") == "libri:1089"
    assert sources.person_id("vctk:p225") == "vctk:p225"
    speech = {"path": "a.flac", "duration": 3.0, "license": "CC-BY-4.0", "use": ["target"]}
    for name, spk in [("speech_train", "librittsr:1089"), ("speech_val", "vctk:p1"),
                      ("eval_librispeech_test", "librispeech:1089")]:
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(dict(speech, speaker=spk)) + "\n")
    with pytest.raises(AssertionError, match="train and test"):
        sources.check_disjoint(tmp_path)


def test_sources_refuse_a_non_commercial_train_row(tmp_path):
    good = {"path": "a.flac", "source": "x", "speaker": "x:1", "duration": 3.0, "split": "train",
            "license": "CC-BY-4.0", "use": ["target"]}
    bad = dict(good, path="b.flac", license="CC-BY-NC-4.0")
    with open(tmp_path / "speech_train.jsonl", "w") as f:
        f.write(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(sources.LicenseError, match="non-commercial"):
        sources.Sources("train", manifests=tmp_path, check_speakers=False)


def test_splits_are_disjoint():
    sources.check_disjoint()          # raises AssertionError if any speaker is in two of train / val / test
    found = {}
    for name in ["speech_train", "speech_val"] + sources.TEST_SPEAKER_MANIFESTS:
        found[name] = {r["speaker"] for r in sources.read_manifest(sources.MANIFESTS / f"{name}.jsonl")}
    test = set().union(*(found[n] for n in sources.TEST_SPEAKER_MANIFESTS))
    assert not found["speech_train"] & found["speech_val"]
    assert not found["speech_train"] & test
    assert not found["speech_val"] & test


def test_pool_counts(src):
    # Contract 8.1 counts, measured 2026-11-07.
    assert len(src.speakers) == 3758 and src.speakers == sorted(src.speakers)
    assert len(src.target_pool) == 3118
    assert len(src.own_pool) == 3135
    for s in src.target_pool[:200]:
        assert len(src.rows_of[s]) >= 2 and src.dur[src.rows_of[s]].sum() >= 14.5 - 1e-4
    for s in src.own_pool[:200]:
        cache, mix = src.own_cache_rows[s], src.own_mix_rows[s]
        assert list(cache) + list(mix) == list(src.rows_of[s])          # manifest order, nothing dropped
        assert len(mix) >= 1 and src.dur[cache].sum() >= 8.0 - 1e-4 and src.dur[mix].sum() >= 4.0 - 1e-4
        assert src.dur[cache[:-1]].sum() < 8.0                            # the shortest such prefix


def test_recipes_table():
    assert data.RECIPE_VERSION == 2
    assert set(data.RECIPES) == {"SMOKE", "A", "B", "C", "FT"}
    a, b, c, ft = data.RECIPES["A"], data.RECIPES["B"], data.RECIPES["C"], data.RECIPES["FT"]
    assert a.talkers == (0.15, 0.45, 0.40, 0.0) and a.p_person == 1.0 and a.p_own_mix == 0.0
    assert b.talkers == (0.10, 0.35, 0.30, 0.25) and b.sir == (-8.0, 12.0) and b.p_quiet == 0.10
    assert (c.seconds, c.batch, c.rooms) == (8.0, 4, (0.60, 0.30, 0.10)) and ft.p_rig == 0.10
    for r in data.RECIPES.values():
        assert abs(sum(r.talkers) - 1) < 1e-9 and abs(sum(r.rooms) - 1) < 1e-9


def test_draw_proportions():
    # 4,000 scalar draws of recipe B (no audio) match the 8.2 table within about 4 standard deviations.
    draws = [render.draw_scalars(data.RECIPES["B"], np.random.default_rng([3, i])) for i in range(4000)]

    def near(values, p):
        values = np.asarray(values, dtype=float)
        assert abs(values.mean() - p) < 4 * math.sqrt(p * (1 - p) / len(values)) + 1e-9, (values.mean(), p)

    near([d.person for d in draws], 0.75)
    near([d.absent for d in draws if d.person], 0.10)
    near([d.absent for d in draws if not d.person], 0.10)
    near([d.own_given for d in draws], 0.90)
    near([d.own_in_mix for d in draws], 0.50)
    near([d.background == "music" for d in draws], 0.20)
    near([d.room == "real" for d in draws], 0.15)
    near([d.n_talkers == 3 for d in draws], 0.25)
    near([d.quiet for d in draws], 0.10)
    near([d.use_wes for d in draws], 0.80)
    loud = [d for d in draws if d.snr_db <= 5.0]
    near([d.lombard for d in loud], 0.30)
    assert not any(d.lombard for d in draws if d.snr_db > 5.0)
    snr = np.array([d.snr_db for d in draws if not d.quiet])
    assert snr.min() >= -8.0 and snr.max() <= 25.0 and abs(np.median(snr) - 5.0) < 0.5


def test_recipe_without_own_enrollment_draws_without_dividing_by_zero():
    never = dataclasses.replace(data.RECIPES["B"], p_own_given=0.0)
    for i in range(20):
        d = render.draw_scalars(never, np.random.default_rng([4, i]), {"own_given": True})   # forced own_given
        assert d.own_given and not d.own_in_mix


# ---------------------------------------------------------------------------------------------------------------
# Rendering: levels, references, labels, enrollments
# ---------------------------------------------------------------------------------------------------------------
def realized(scene):
    """(SIR, own level, SNR) in dB measured on the scaled signals before the mic chain (None where absent)."""
    p = scene["parts"]
    p_t = dsp.active_power(p["t"], p["vad_s"])
    sir = own = None
    if p["img"]:
        sir = db(p_t / max(dsp.active_power(x, v) for x, v in zip(p["img"], p["vad_i"])))
    if p["o"] is not None:
        own = db(dsp.active_power(p["o"], p["vad_o"]) / p_t)
    snr = db(p_t / np.mean(p["background"].astype(np.float64) ** 2)) if np.any(p["background"]) else None
    return sir, own, snr


def test_realized_levels_match_drawn_values(src):
    # Recipe B examples plus forced music-only, noise-only, music + babble and noise + babble backgrounds.
    cases = [{} for _ in range(16)] + [
        {"background": "music", "babble": False}, {"background": "noise", "babble": False},
        {"background": "music", "babble": True}, {"background": "noise", "babble": True, "n_noise": 2},
        {"own_given": True, "own_in_mix": True, "n_talkers": 3, "room": "real"}]
    for i, ov in enumerate(cases):
        ex = example(src, 100 + i, overrides=ov, return_parts=True)
        info = ex["info"]
        sir, own, snr = realized(ex["scene"])
        if info["n_talkers"]:
            assert abs(sir - info["sir_db"]) < 0.1, (i, sir, info["sir_db"])
        if info["own_in_mix"]:
            assert abs(own - info["own_db"]) < 0.1, (i, own, info["own_db"])
        assert abs(snr - info["snr_db"]) < 0.1, (i, snr, info["snr_db"])
        if info["lombard"]:
            assert info["snr_db"] > info["snr_drawn"] + 2.99
        else:
            assert info["snr_db"] == info["snr_drawn"]
        if ov.get("background") == "music":
            assert ex["scene"]["parts"]["music"] is not None and ex["scene"]["parts"]["noise"] is None
        if ov.get("background") == "noise":
            assert ex["scene"]["parts"]["noise"] is not None and ex["scene"]["parts"]["music"] is None
        # The logged SNR after the mic chain: target image vs background, both through the same filters.
        p, chain = ex["scene"]["parts"], ex["scene"]["chain"]
        if info["present"]:
            t_f, b_f = render.chain_filters(chain, p["t"]), render.chain_filters(chain, p["background"])
            want = db(dsp.active_power(t_f, p["vad_s"]) / np.mean(b_f.astype(np.float64) ** 2))
            assert abs(info["snr_chain_db"] - want) < 1e-3, (i, info["snr_chain_db"], want)
        else:
            assert info["snr_chain_db"] is None


def test_own_level_range_follows_geometry(src):
    # U(0, 18) dB in sim in_hand rooms, U(0, 12) dB otherwise (contract 8.1).
    for i in range(12):
        ov = {"own_given": True, "own_in_mix": True, "room": "sim", "geometry": "in_hand" if i % 2 else "on_table",
              "own_u": 0.999, "n_talkers": 0, "babble": False}
        ex = example(src, 200 + i, overrides=ov, return_parts=True)
        expected = 0.999 * (18.0 if i % 2 else 12.0)
        assert ex["info"]["geometry"] == ov["geometry"] and abs(ex["info"]["own_db"] - expected) < 1e-3


def test_absent_and_nobody_give_zero_reference(src):
    for ov in [{"person": True, "absent": True}, {"person": False, "absent": True}]:
        ex = example(src, 300, overrides=ov)
        assert not ex["present"] and not np.any(ex["ref"]) and not np.any(ex["vad_t"])
        assert np.any(ex["mix"])
    absent = example(src, 301, overrides={"person": True, "absent": True})
    assert absent["has_t"] and absent["spk_t"] >= 0 and np.any(absent["enr_t"])   # enrollment still given


def test_everyone_reference_is_sum_of_talkers(src):
    for i in range(4):
        ex = example(src, 400 + i, overrides={"person": False, "absent": False, "n_talkers": 1 + i % 3},
                     metric_signals=True)
        m = ex["scene"]["metric"]
        total = m["target"] + np.sum(m["talkers"], axis=0)
        scale = np.max(np.abs(ex["ref"]))
        assert np.max(np.abs(ex["ref"] - total)) <= 1e-6 * max(scale, 1.0)
        vad = m["vad_target"].copy()
        for v in m["vad_talkers"]:
            vad |= v
        assert np.array_equal(ex["vad_t"], vad.astype(np.float32))
        assert not ex["has_t"] and ex["spk_t"] == -1 and not np.any(ex["enr_t"]) and not ex["tok_val_t"].any()
        assert list(ex["mode"]) == [0.0, 1.0]


def test_dry_room_reference_is_the_target_part_and_own_voice_never_in_it(src):
    base = {"person": True, "absent": False, "room": "dry", "n_talkers": 0, "babble": False,
            "background": "none", "own_given": True}
    ex = example(src, 500, overrides=dict(base, own_in_mix=False), test_mode=True)
    assert np.max(np.abs(ex["mix"] - ex["ref"])) <= 1e-6
    ex = example(src, 501, overrides=dict(base, own_in_mix=True), test_mode=True, metric_signals=True)
    own = ex["scene"]["metric"]["own"]
    assert np.max(np.abs(own)) > 1e-3                                    # the own voice is in the mix ...
    assert np.max(np.abs(ex["mix"] - own - ex["ref"])) <= 1e-6           # ... and not in the reference


def test_labels_are_hop_vad_of_the_clean_references(src):
    for i in range(6):
        ex = example(src, 600 + i, overrides={"person": True, "absent": False, "own_given": True,
                                              "own_in_mix": i % 2 == 0}, return_parts=True)
        p = ex["scene"]["parts"]
        assert np.array_equal(ex["vad_t"], dsp.hop_vad(p["s"]).astype(np.float32))
        if p["o_e"] is not None:
            assert np.array_equal(ex["vad_o"], dsp.hop_vad(p["o_e"]).astype(np.float32))
            assert ex["vad_o"].any()
        else:
            assert not ex["vad_o"].any()
        assert ex["vad_t"].shape == (L4 // HOP,) and ex["vad_t"].dtype == np.float32


def test_enrollment_never_shares_audio_with_the_mixture(src):
    for i in range(20):
        ex = example(src, 700 + i, record_sources=True, overrides={"person": True, "own_given": True})
        sc = ex["scene"]
        enr = sc["enr_rows"]
        assert len(set(enr)) == len(enr)                                 # no row (so no sample) twice
        assert set(enr) <= set(src.rows_of[sc["target"]].tolist())
        stream = {r for r, _, _ in sc["rows"]["target"]}
        assert stream and stream <= set(sc["stream_rows"]) and not stream & set(enr)
        assert sum(src.dur[r] for r in enr) >= LE6 / SR + 0.5
        # The own talker uses only its mix rows; its enrollment clip only cache rows.
        own_mix = {r for r, _, _ in sc["rows"]["own_mix"]}
        assert own_mix <= set(src.own_mix_rows[sc["own"]].tolist())
        assert not own_mix & set(src.own_cache_rows[sc["own"]].tolist())


def test_own_clip_recipe(src):
    for s in src.own_pool[[0, 1000, 3134]]:
        for c in (0, 7):
            direct, segments, gain = sources.make_own_clip(int(s), c, src.cache_row_paths(int(s)))
            assert np.array_equal(src.build_clip(int(s), c), direct)
            assert {seg[0] for seg in segments} <= set(src.own_cache_rows[s].tolist())
            assert direct.shape == (sources.OWN_CLIP_LEN,)
            assert abs(db(dsp.active_power(direct, dsp.hop_vad(direct))) - sources.LEVEL_DBFS) < 0.01


@needs_cache
def test_own_cache_must_match_the_manifest(src):
    own = src.own_cache()                                   # the real cache passes the check on load
    assert own is not None
    s = int(src.own_pool[0])
    first = int(np.searchsorted(own["seg_clip"], s * sources.OWN_CLIPS))
    for row in [int(src.own_mix_rows[s][0]),                # the own talker's mix row: the waveform leak of 6.3
                int(src.own_cache_rows[int(src.own_pool[1])][0])]:   # another speaker's cache row
        bad = dict(own, seg_row=own["seg_row"].copy())
        bad["seg_row"][first] = row
        with pytest.raises(ValueError, match="not cache rows of their own speaker"):
            src.check_own_cache(bad)
    with pytest.raises(ValueError, match="own-voice pool"):
        src.check_own_cache(dict(own, in_pool=~own["in_pool"]))


def test_every_token_set_has_four_groups(src):
    for i in range(16):
        ex = example(src, 800 + i, Le=int(3.0 * SR))       # the shortest enrollment length
        if ex["has_t"]:
            assert ex["tok_val_t"].sum() >= 4
        if ex["has_o"]:
            assert ex["tok_val_o"].sum() >= 4
        for idx, val in [(ex["tok_idx_t"], ex["tok_val_t"]), (ex["tok_idx_o"], ex["tok_val_o"])]:
            assert not idx[~val].any()


def test_missing_decim_asset(src, tmp_path, monkeypatch):
    monkeypatch.setattr(dsp, "DECIM_FIR_PATH", tmp_path / "missing.npy")
    with pytest.raises(FileNotFoundError, match="make_resampler_tables"):
        example(src, 900)
    ex = example(src, 900, allow_missing_decim=True)
    assert ex["info"]["decim"] is False


def test_rig_recordings_are_a_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RIG_MANIFEST", tmp_path / "manifest.jsonl")
    assert sources.rig_rows() is None              # no file: the FT recipe has 0% rig rows
    (tmp_path / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(NotImplementedError, match="8.10"):
        sources.rig_rows()


# ---------------------------------------------------------------------------------------------------------------
# Batches, determinism and the loader
# ---------------------------------------------------------------------------------------------------------------
def batch_digest(batch):
    """SHA-256 over every array of a batch (key order of 8.7) and its info dicts."""
    h = hashlib.sha256()
    for key in data.BATCH_KEYS:
        value = np.asarray(batch[key])
        h.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
    h.update(json.dumps(batch["info"], sort_keys=True).encode())
    return h.hexdigest()


def render_batch_digest(args):
    """Module-level so a spawn process can run it: (split, recipe, seed, B, L, j) -> digest of batch j."""
    split, recipe, seed, B, L, j = args
    return batch_digest(data.MixDataset(split, recipe, seed, B, L)[j])


@needs_cache
def test_batch_schema():
    batch = data.MixDataset("train", "B", seed=0, B=8, L=L4)[3]
    Le = data.draw_le(0, 3)
    shapes = {"mix": (8, L4), "ref": (8, L4), "vad_t": (8, L4 // HOP), "vad_o": (8, L4 // HOP), "present": (8,),
              "own_in_mix": (8,), "person": (8,), "enr_t": (8, Le), "tok_idx_t": (8, MAX_T_TOK, 3),
              "tok_val_t": (8, MAX_T_TOK), "enr_o": (8, 128000), "tok_idx_o": (8, MAX_O_TOK, 3),
              "tok_val_o": (8, MAX_O_TOK), "e_o": (8, 256), "has_t": (8,), "has_o": (8,), "use_wes": (8,),
              "mode": (8, 2), "spk_t": (8,)}
    assert set(batch) == set(shapes) | {"info"}
    for key, shape in shapes.items():
        assert batch[key].shape == shape, key
        assert batch[key].dtype == np.dtype(data.BATCH_KEYS[key]), key
    assert len(batch["info"]) == 8 and Le in [int(round(s * SR)) for s in data.LE_CHOICES_S]
    assert np.all(np.isfinite(batch["mix"])) and np.all(np.isfinite(batch["enr_o"]))
    for i in range(8):
        assert batch["person"][i] == batch["has_t"][i] == (batch["mode"][i, 0] == 1.0)
        assert (batch["spk_t"][i] >= 0) == batch["person"][i]
        if batch["has_o"][i]:
            assert abs(np.linalg.norm(batch["e_o"][i]) - 1.0) < 1e-4    # the cached unit-length vector
            info = batch["info"][i]
            cache = sources.get_sources("train").own_cache()
            s = sources.get_sources("train").speakers.index(info["own"])
            assert np.array_equal(batch["e_o"][i], cache["emb"][s, info["own_clip"]])
        else:
            assert not batch["e_o"][i].any() and not batch["enr_o"][i].any() and not batch["own_in_mix"][i]
        assert batch["present"][i] == bool(np.any(batch["ref"][i]))


@needs_cache
def test_same_index_same_batch_in_fresh_spawn_processes():
    args = ("train", "B", 5, 4, 2 * SR, 2)
    digests = []
    for _ in range(2):         # two separate fresh processes
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            digests.append(pool.submit(render_batch_digest, args).result())
    here = render_batch_digest(args)
    assert digests[0] == digests[1] == here
    assert render_batch_digest(("train", "B", 5, 4, 2 * SR, 3)) != here      # another index differs
    assert render_batch_digest(("train", "B", 6, 4, 2 * SR, 2)) != here      # another seed differs


@needs_cache
def test_loader_gives_the_same_batches_with_any_worker_count():
    runs = []
    for workers in (0, 2):
        loader = data.make_loader("train", "B", 9, 4, 7, B=2, L=SR, workers=workers)
        runs.append([batch_digest(b) for b in loader])           # the loader yields numpy batches
        del loader
    assert runs[0] == runs[1] and len(runs[0]) == 3 and len(set(runs[0])) == 3
    direct = [render_batch_digest(("train", "B", 9, 2, SR, j)) for j in (4, 5, 6)]
    assert direct == runs[0]


@needs_cache
def test_fine_tune_recipe_refuses_rig_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RIG_MANIFEST", tmp_path / "manifest.jsonl")
    data.MixDataset("train", "FT", 0, 4, 8 * SR)          # no rig file: fine, 0% rig rows
    (tmp_path / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(NotImplementedError):
        data.MixDataset("train", "FT", 0, 4, 8 * SR)


def test_testvec_scene_rendered_in_test_mode():
    # 10-ml 7.4: the long test-vector mixture from render_example in test mode (ml.export --scene render).
    from ml.earshot import testaudio
    tv = testaudio.testvec_inputs_render(7)
    assert tv["in_long"].shape == (160000,) and tv["enr_t"].shape == (96000,) and tv["enr_o"].shape == (128000,)
    rms_db = 10 * math.log10(np.mean(tv["in_long"].astype(np.float64) ** 2))
    assert abs(rms_db - (-26.0)) < 1e-3 and np.max(np.abs(tv["in_long"])) < 1.0
    assert {s["license"] for s in tv["sources"] if s["role"] != "room"} <= {"CC-BY-4.0", "CC0-1.0"}
    assert any(s["role"] == "noise" and s["path"].startswith("noise/fsd50k/") for s in tv["sources"])
    again = testaudio.testvec_inputs_render(7)
    assert all(np.array_equal(tv[k], again[k]) for k in ("in_long", "enr_t", "enr_o"))


def test_mixtures_and_enrollments_stay_under_full_scale(src):
    """Headroom rule (DECISIONS L 2026-11-08): loud scenes lower the gain instead of peaking past 1.0."""
    lowered = 0
    for i in range(24):
        ex = example(src, 4242 + i)
        assert float(np.max(np.abs(ex["mix"]))) <= 1.0
        assert float(np.max(np.abs(ex["enr_t"]))) <= 1.0
        info = ex["info"]
        if info["level_real_dbfs"] < info["level_dbfs"] - 0.01:
            lowered += 1
    assert lowered >= 1          # seed 4242 has loud scenes; the rule really ran
