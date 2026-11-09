"""Tests for ml/earshot/fileio.py and ml/export.py (10-ml 7 and 14, contract 3.1 to 3.3 and 6).

The module fixture `export_dir` runs the whole random_v1 export (seed 0) once into a temporary folder (about 30 s):
earshot.esw, both test-vector files, thresholds.json, wespk_selftest.est and quality_vectors.json.
"""
import hashlib
import json
import shutil
import struct

import numpy as np
import pytest
import torch

from ml import export
from ml.earshot import fileio
from ml.earshot.config import CONFIG, CONFIG_HASH, config_hash
from ml.earshot.dsp import stft
from ml.earshot.model import ZERO_INIT_NAMES, features, init_random_for_parity
from ml.earshot.vad import es_vad, num_frames
from ml.earshot.wespeaker import DEFAULT_WEIGHTS

from .helpers import CASES, make_cond_inputs, noise, speech
from .test_model import contract_shapes

NEEDS = [export.SELFTEST_NPZ, DEFAULT_WEIGHTS, export.testaudio.SPEECH_TEST, export.testaudio.SIM_TEST]


# ---------------------------------------------------------------------------------------------------------------
# Container (fileio)
# ---------------------------------------------------------------------------------------------------------------
def sample_arrays():
    rng = np.random.default_rng(5)
    return [("a.f32", rng.standard_normal((3, 5)).astype(np.float32)),
            ("b.u8", rng.integers(0, 256, 7).astype(np.uint8)),
            ("c.i32", rng.integers(-1000, 1000, (2, 2, 2)).astype(np.int32)),
            ("d.empty", np.zeros((0, 4), dtype=np.float32)),
            ("e.one", np.array([42], dtype=np.int32))]


def write_sample(path, magic=b"EST1"):
    header = {"format": "earshot-testvec", "format_version": 1, "model": "esb16-df", "model_version": "1.0.0",
              "config": CONFIG, "config_hash": config_hash(), "meta": {"note": "test"}}
    return fileio.write_container(path, magic, header, sample_arrays())


def rewrite_header(path, change):
    """Rewrite a container with change(header) applied, keeping the data bytes (header_len and padding redone)."""
    buf = open(path, "rb").read()
    header_len = struct.unpack("<I", buf[8:12])[0]
    header = json.loads(buf[16:16 + header_len])
    data = buf[fileio.round_up(16 + header_len):]
    change(header)
    text = json.dumps(header, separators=(",", ":")).encode()
    out = buf[:4] + struct.pack("<III", 1, len(text), 0) + text
    out += b"\x00" * (fileio.round_up(len(out)) - len(out)) + data
    open(path, "wb").write(out)


def test_container_round_trip(tmp_path):
    path = tmp_path / "x.est"
    written = write_sample(path)
    magic, header, arrays = fileio.read_container(path, expect_magic="EST1")
    assert magic == "EST1"
    assert list(arrays) == [name for name, _ in sample_arrays()]
    for name, value in sample_arrays():
        assert arrays[name].dtype == value.dtype and arrays[name].shape == value.shape
        assert np.array_equal(arrays[name], value)
    # Preamble, 64-byte alignment and the data hash, checked from the raw bytes.
    buf = path.read_bytes()
    assert buf[:4] == b"EST1"
    version, header_len, reserved = struct.unpack("<III", buf[4:16])
    assert (version, reserved) == (1, 0)
    data_start = fileio.round_up(16 + header_len)
    assert data_start % 64 == 0 and set(buf[16 + header_len:data_start]) <= {0}
    assert header["data_bytes"] == len(buf) - data_start
    assert header["data_sha256"] == hashlib.sha256(buf[data_start:]).hexdigest() == written["data_sha256"]
    for entry in header["arrays"]:
        assert entry["offset"] % 64 == 0
    last = header["arrays"][-1]
    assert last["offset"] + last["nbytes"] == header["data_bytes"]      # no padding after the last array
    assert buf[16:16 + header_len].decode("ascii")                       # plain ASCII header


def test_container_rejects_damage(tmp_path):
    path = tmp_path / "x.est"
    write_sample(path)
    good = path.read_bytes()

    def expect(fragment, data):
        path.write_bytes(data)
        with pytest.raises(ValueError, match=fragment):
            fileio.read_container(path)

    flipped = bytearray(good)
    flipped[-3] ^= 0x01
    expect("data_sha256", bytes(flipped))
    expect("magic", b"XYZ1" + good[4:])
    expect("format_version", good[:4] + struct.pack("<I", 2) + good[8:])
    expect("reserved", good[:12] + struct.pack("<I", 7) + good[16:])
    expect("header_len", good[:200])
    expect("preamble", good[:10])
    expect("data_bytes", good + b"\x00")
    # A file with the right magic but read as the other kind.
    path.write_bytes(good)
    with pytest.raises(ValueError, match="magic"):
        fileio.read_container(path, expect_magic="ESW1")

    # Header edits (the data hash stays valid, so the array checks are reached).
    def edit(change, fragment):
        path.write_bytes(good)
        rewrite_header(path, change)
        with pytest.raises(ValueError, match=fragment):
            fileio.read_container(path)

    edit(lambda h: h["arrays"][1].update(offset=h["arrays"][1]["offset"] + 4), "offset")
    edit(lambda h: h["arrays"][0].update(dtype="f64"), "dtype")
    edit(lambda h: h["arrays"][0].update(shape=[3, 6]), "nbytes")
    edit(lambda h: h["arrays"][2].update(offset=0), "overlaps")
    edit(lambda h: h.update(format="earshot-weights"), "format")
    edit(lambda h: h["arrays"][-1].update(offset=fileio.round_up(h["data_bytes"])), "past the end")


def test_writer_rejects_bad_input(tmp_path):
    header = {"format": "earshot-testvec", "format_version": 1}
    with pytest.raises(ValueError, match="dtype"):
        fileio.write_container(tmp_path / "a.est", b"EST1", header, [("x", np.zeros(3))])          # float64
    with pytest.raises(ValueError, match="name"):
        fileio.write_container(tmp_path / "a.est", b"EST1", header,
                               [("x", np.zeros(3, np.float32)), ("x", np.zeros(3, np.float32))])
    with pytest.raises(ValueError, match="ASCII"):
        fileio.write_container(tmp_path / "a.est", b"EST1", dict(header, meta={"name": "café"}), [])
    with pytest.raises(ValueError, match="magic"):
        fileio.write_container(tmp_path / "a.est", b"ABCD", header, [])


def test_config_hash():
    assert config_hash() == CONFIG_HASH == "089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec"
    assert len(CONFIG) == 29


# ---------------------------------------------------------------------------------------------------------------
# Weight layout
# ---------------------------------------------------------------------------------------------------------------
def test_export_order_matches_contract(net):
    order = export.EXPORT_ORDER()
    assert len(order) == 270 and len(set(order)) == 270
    tensors = export.to_export(net)
    assert [name for name, _ in tensors] == order
    shapes = {name: tuple(value.shape) for name, value in tensors}
    assert shapes == contract_shapes()
    assert sum(value.size for _, value in tensors) == 2570500
    assert all(value.dtype == np.float32 and value.flags["C_CONTIGUOUS"] for _, value in tensors)


def test_export_values_follow_layout_rules(net):
    """Linear weights are transposed, LSTM biases summed, forget-gate bias exported as exactly +1."""
    tensors = dict(export.to_export(net))
    state = net.state_dict()
    assert np.array_equal(tensors["att.q.weight"], state["att.q.weight"].numpy().T)
    assert np.array_equal(tensors["blocks.1.t.w_hh"], state["blocks.1.t.weight_hh_l0"].numpy().T)
    assert np.array_equal(tensors["tok.blk.b.w_ih_rev"], state["tok.blk.b.weight_ih_l0_reverse"].numpy().T)
    assert np.array_equal(tensors["blocks.3.b.bias_rev"],
                          (state["blocks.3.b.bias_ih_l0_reverse"] + state["blocks.3.b.bias_hh_l0_reverse"]).numpy())
    for name, value in tensors.items():
        if name.endswith(".bias") or name.endswith(".bias_rev"):
            if ".t.bias" in name or ".b.bias" in name:          # LSTM biases: [4H], gates i, f, g, o
                H = value.shape[0] // 4
                assert np.all(value[H:2 * H] == 1.0), name


def test_random_init_fills_zero_init_tensors(net):
    params = dict(net.named_parameters())
    for name in ZERO_INIT_NAMES:
        std = float(params[name].std())
        tolerance = 0.005 if params[name].numel() > 10000 else 0.03
        assert abs(std - 0.1) < tolerance, (name, std)
    # Seed 0 is reproducible.
    again = dict(init_random_for_parity(0).named_parameters())
    assert all(torch.equal(params[name], again[name]) for name in params)


def test_esw_round_trip(net, tmp_path):
    """model -> .esw -> model gives the same outputs (1e-6) and a byte-identical re-export."""
    meta = {"run": "test", "note": "fixed meta so the files can be compared byte for byte"}
    header = export.write_esw(tmp_path / "a.esw", net, meta)
    assert header["data_sha256"] == fileio.model_hash(tmp_path / "a.esw")
    loaded, loaded_header = export.load_esw(tmp_path / "a.esw")
    assert loaded_header["meta"] == meta and loaded_header["config"] == CONFIG
    export.write_esw(tmp_path / "b.esw", loaded, meta)
    assert (tmp_path / "a.esw").read_bytes() == (tmp_path / "b.esw").read_bytes()

    # Same outputs for all 4 cases on 2 s of speech (enrollment, conditioning and separation).
    n = 32000
    mix = (speech(n, 0, 3) + 0.5 * speech(n, 1, 3) + noise(n, 9)).astype(np.float32)
    enr_t, enr_o = speech(96000, 0, 20), speech(128000, 2, 0)
    with torch.no_grad():
        results = []
        for model in (net, loaded):
            ci = make_cond_inputs(model, enr_t, enr_o, CASES)
            cond = model.build_cond(ci)
            y, logits = model(torch.from_numpy(mix)[None].expand(len(CASES), -1).contiguous(), cond)
            results.append((ci.tok_t, ci.g_t, cond.film, cond.k, y, logits))
    for a, b in zip(*results):
        assert float((a - b).abs().max()) <= 1e-6


def test_load_esw_rejects_wrong_files(net, tmp_path):
    tensors = export.to_export(net)
    header = export.base_header("earshot-weights", {})
    fileio.write_container(tmp_path / "missing.esw", b"ESW1", header, tensors[:-1])
    with pytest.raises(ValueError, match="missing"):
        export.load_esw(tmp_path / "missing.esw")
    extra = tensors + [("aam.weight", np.zeros((3, 3), np.float32))]
    fileio.write_container(tmp_path / "extra.esw", b"ESW1", header, extra)
    with pytest.raises(ValueError, match="extra"):
        export.load_esw(tmp_path / "extra.esw")
    bad = [(n, v.T.copy() if n == "blocks.0.t.w_hh" else v) for n, v in tensors]
    fileio.write_container(tmp_path / "shape.esw", b"ESW1", header, bad)
    with pytest.raises(ValueError, match="blocks.0.t.w_hh"):
        export.load_esw(tmp_path / "shape.esw")
    wrong = dict(header, config=dict(CONFIG, hop=128))
    fileio.write_container(tmp_path / "config.esw", b"ESW1", wrong, tensors)
    with pytest.raises(ValueError, match="config.hop"):
        export.load_esw(tmp_path / "config.esw")


# ---------------------------------------------------------------------------------------------------------------
# The whole random_v1 folder
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    """Run `python -m ml.export --random --seed 0` into a temporary folder (about 30 s)."""
    missing = [str(p) for p in NEEDS if not p.exists()]
    if missing:
        pytest.skip(f"export needs {missing}")
    out = tmp_path_factory.mktemp("random_v1")
    export.main(["--random", "--seed", "0", "--out", str(out)])
    return out


def test_release_folder_holds_only_the_six_files(export_dir):
    # Contract 11: a release folder (random_v1/, v1/) holds exactly the six contract files, nothing else.
    assert sorted(p.name for p in export_dir.iterdir()) == sorted(export.FILES)


def expected_testvec_arrays(n, short, n_t, n_o):
    """Every array of contract 3.3 / 10-ml 7.4 in file order: (name, dtype, shape)."""
    out = [("in", "f32", [n]), ("enr_t", "f32", [96000]), ("mask_t", "u8", [num_frames(96000)]),
           ("enr_o", "f32", [128000]), ("mask_o", "u8", [num_frames(128000)]), ("vad_t", "u8", [num_frames(96000)]),
           ("e_t", "f32", [256]), ("e_o", "f32", [256]), ("tok_t", "f32", [n_t, 64]), ("g_t", "f32", [128]),
           ("tok_o", "f32", [n_o, 64]), ("g_o", "f32", [128])]
    for case in CASES:
        out += [(f"cond.{case}.film", "f32", [4096]), (f"cond.{case}.k", "f32", [4, 68, 16]),
                (f"cond.{case}.v", "f32", [4, 68, 16]), (f"cond.{case}.n", "i32", [1]),
                (f"out.{case}", "f32", [n]), (f"heads.{case}", "f32", [n // 64, 2])]
    if short:
        out += [("dbg.feat", "f32", [8, 771]), ("dbg.z_enc", "f32", [8, 16, 64])]
        out += [(f"dbg.z_blk{b}", "f32", [8, 16, 64]) for b in range(4)]
        out += [("dbg.coef", "f32", [8, 1026])]
    return out


def test_export_folder_has_six_files(export_dir):
    for name in export.FILES:
        assert (export_dir / name).exists(), name
    _, header, arrays = fileio.read_container(export_dir / "earshot.esw", expect_magic="ESW1")
    assert len(arrays) == 270 and [e["name"] for e in header["arrays"]] == export.EXPORT_ORDER()
    meta = header["meta"]
    for key in ["run", "step", "ema", "git_commit", "date", "init", "data_licenses", "wespeaker_note", "credits",
                "eval_only_data"]:
        assert key in meta, key
    assert meta["init"] == "random-seed-0" and meta["wespeaker_note"] == export.WESPEAKER_NOTE
    assert len(header["data_sha256"]) == 64 and header["data_sha256"] == header["data_sha256"].lower()


@pytest.mark.parametrize("name", ["testvec_short.est", "testvec_long.est"])
def test_testvec_contents(export_dir, name):
    short = name == "testvec_short.est"
    _, header, arrays = fileio.read_container(export_dir / name, expect_magic="EST1")
    export.check_model_header(header)
    assert header["format"] == "earshot-testvec"
    n = 32000 if short else 160000
    n_t, n_o = arrays["tok_t"].shape[0], arrays["tok_o"].shape[0]
    assert 4 <= n_t <= 48 and 4 <= n_o <= 16
    expected = expected_testvec_arrays(n, short, n_t, n_o)
    assert [(e["name"], e["dtype"], e["shape"]) for e in header["arrays"]] == expected

    meta = header["meta"]
    # meta.tolerances: exactly the contract 3.3 key set and values.
    assert set(meta["tolerances"]) == {"vad", "tokens", "g", "film", "k", "v", "dbg", "out", "out_snr_db", "heads"}
    assert meta["tolerances"] == (export.TOLERANCES_SHORT if short else export.TOLERANCES_LONG)
    assert meta["tolerances"]["out"] == (2e-4 if short else 1e-3)
    assert meta["tolerances"]["out_snr_db"] == (60.0 if short else 50.0)
    assert meta["weights_sha256"] == fileio.model_hash(export_dir / "earshot.esw")
    assert meta["sources"] and all(s["license"] for s in meta["sources"])

    # Masks and token counts.
    assert np.array_equal(arrays["vad_t"], es_vad(arrays["enr_t"]))
    zeroed = arrays["vad_t"].copy()
    zeroed[40:50] = 0
    assert np.array_equal(arrays["mask_t"], zeroed) and arrays["vad_t"][40:50].any()
    assert np.array_equal(arrays["mask_o"], es_vad(arrays["enr_o"]))
    counts = {"person": n_t + n_o + 4, "everyone": n_o + 4, "noown": n_t + 4, "nowespk": n_t + n_o + 4}
    for case in CASES:
        k = arrays[f"cond.{case}.n"][0]
        assert k == counts[case]
        assert not arrays[f"cond.{case}.k"][:, k:].any() and not arrays[f"cond.{case}.v"][:, k:].any()
        heads = arrays[f"heads.{case}"]
        assert np.all((heads > 0) & (heads < 1)) and np.all(np.isfinite(arrays[f"out.{case}"]))
    for vec in ["e_t", "e_o", "g_t", "g_o"]:
        assert abs(np.linalg.norm(arrays[vec]) - 1.0) < 1e-5


def test_testvec_short_is_start_of_long(export_dir):
    _, _, short = fileio.read_container(export_dir / "testvec_short.est")
    _, _, long = fileio.read_container(export_dir / "testvec_long.est")
    assert np.array_equal(short["in"], long["in"][:32000])
    for case in CASES:
        assert np.array_equal(short[f"cond.{case}.film"], long[f"cond.{case}.film"])
        # Both runs start from a reset state and the stream is causal, so the first 2 s agree.
        assert float(np.abs(short[f"out.{case}"] - long[f"out.{case}"][:32000]).max()) <= 1e-6


def test_testvec_matches_model(export_dir):
    """The stored vectors agree with the weights read from earshot.esw: tokens, packed conditioning, the offline
    model one hop earlier (contract 2.1 streaming identity) and the debug taps."""
    model, _ = export.load_esw(export_dir / "earshot.esw")
    _, _, a = fileio.read_container(export_dir / "testvec_long.est")
    ci, tok_t, g_t, tok_o, g_o = export.case_cond_inputs(model, a["enr_t"], a["mask_t"], a["enr_o"], a["mask_o"],
                                                         a["e_t"], a["e_o"])
    for mine, stored in [(tok_t, "tok_t"), (g_t, "g_t"), (tok_o, "tok_o"), (g_o, "g_o")]:
        assert float(np.abs(mine - a[stored]).max()) <= 1e-6
    with torch.no_grad():
        cond = model.build_cond(ci)
        y, logits = model(torch.from_numpy(a["in"])[None].expand(len(CASES), -1).contiguous(), cond)
    for i, case in enumerate(CASES):
        pc = export.pack_cond(cond, i)
        assert pc.n == a[f"cond.{case}.n"][0]
        assert float(np.abs(pc.film.numpy() - a[f"cond.{case}.film"]).max()) <= 1e-6
        assert float(np.abs(pc.k.numpy() - a[f"cond.{case}.k"]).max()) <= 1e-6
        out = a[f"out.{case}"]
        assert float(np.abs(out[64:] - y[i, :-64].numpy()).max()) <= 1e-5
        assert float(np.abs(a[f"heads.{case}"] - torch.sigmoid(logits[i]).numpy()).max()) <= 1e-5

    # Debug taps of the short file, recomputed with the offline model on the first 8 frames (contract 3.3 tap
    # points): streaming hop j sees exactly offline STFT frame j, so the taps must agree to float rounding.
    _, _, s = fileio.read_container(export_dir / "testvec_short.est")
    film, k, v, valid = cond.film[:1], cond.k[:1], cond.v[:1], cond.valid[:1]      # case person = row 0
    with torch.no_grad():
        X = stft(torch.from_numpy(s["in"][:8 * 64])[None])                         # [1, 8, 257]
        z = model.encode(X)
        taps = {"feat": features(X).reshape(1, 8, 771), "z_enc": z}
        for b in range(4):
            z = model.blocks[b](z)
            if b == 0:
                z = z * (1 + film[:, 0, 0, None]) + film[:, 0, 1, None]
                z = model.att(z, k, v, valid)
            if b == 2:
                z = z * (1 + film[:, 1, 0, None]) + film[:, 1, 1, None]
            taps[f"z_blk{b}"] = z
        taps["coef"] = torch.cat(model.decode(z), dim=-1)
    for tap, value in taps.items():
        assert float(np.abs(value[0].numpy() - s[f"dbg.{tap}"]).max()) <= 1e-5, tap


def test_testvec_command_reproduces(export_dir, tmp_path):
    """`export --testvec` on a folder holding only the .esw re-creates the three .est files of 10-ml 7.3 with the
    same arrays (1e-6), and nothing else."""
    shutil.copy(export_dir / "earshot.esw", tmp_path / "earshot.esw")
    export.main(["--testvec", "--out", str(tmp_path)])
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["earshot.esw", "testvec_long.est", "testvec_short.est", "wespk_selftest.est"]
    for name in ["testvec_short.est", "testvec_long.est", "wespk_selftest.est"]:
        _, _, first = fileio.read_container(export_dir / name)
        _, _, again = fileio.read_container(tmp_path / name)
        assert list(first) == list(again)
        for key in first:
            diff = np.abs(first[key].astype(np.float64) - again[key].astype(np.float64))
            assert diff.size == 0 or float(diff.max()) <= 1e-6, (name, key)


def test_wespk_selftest_file(export_dir):
    magic, header, arrays = fileio.read_container(export_dir / "wespk_selftest.est")
    assert magic == "EST1" and header["format"] == "earshot-testvec"
    export.check_model_header(header)
    assert header["meta"]["kind"] == "wespeaker-selftest"
    assert header["meta"]["tolerances"] == {"cosine_min": 0.999}
    assert [(e["name"], e["dtype"], e["shape"]) for e in header["arrays"]] == [
        ("clip", "f32", [80000]), ("embedding", "f32", [256])]
    saved = np.load(export.SELFTEST_NPZ)
    assert np.array_equal(arrays["clip"], saved["clip"])
    assert float(np.abs(arrays["embedding"] - saved["embedding"]).max()) <= 1e-5
    assert abs(np.linalg.norm(arrays["embedding"]) - 1.0) < 1e-5


def test_thresholds_file(export_dir, tmp_path):
    with open(export_dir / "thresholds.json", encoding="utf-8") as f:
        obj = json.load(f)
    model_hash = fileio.model_hash(export_dir / "earshot.esw")
    # The contract 6 object with the initial values.
    assert obj == {"version": 1, "model_hash": model_hash,
                   "target": {"speech_s": [4, 5], "sbr_db": [0, 6], "one_voice": [0.45, 0.60],
                              "own_cos": [0.50, 0.60], "dup_cos": 0.75, "clip_pct": [0.1, 1.0]},
                   "own": {"speech_s": 8, "snr_db": [15, 20]}, "own_window_cos": 0.60, "locked_cos": 0.0}
    with pytest.raises(ValueError, match="calibrated for"):
        bad = tmp_path / "other_thresholds.json"        # kept out of export_dir, which must hold only six files
        bad.write_text(json.dumps(dict(obj, model_hash="0" * 64)))
        export.write_thresholds(tmp_path / "unused.json", model_hash, source=bad)


# ---------------------------------------------------------------------------------------------------------------
# Trained checkpoints: --ckpt and --thresholds (ML-2b)
# ---------------------------------------------------------------------------------------------------------------
def fake_checkpoint(path, model_seed=1, ema_seed=2, **extra):
    """A small checkpoint shaped like ml/train.py's (10-ml 10.4): model and EMA weights from two different random
    inits (so the test can tell which one was exported), plus the bookkeeping entries. extra overrides entries."""
    ckpt = {"model": init_random_for_parity(model_seed).state_dict(),
            "ema": None if ema_seed is None else init_random_for_parity(ema_seed).state_dict(),
            "step": 1234, "seed": 1, "config": {"run": "export_test"}, "recipe_version": 2, "best": [],
            "torch_version": str(torch.__version__)}
    ckpt.update(extra)
    torch.save(ckpt, path)
    return ckpt


def outputs(model, n=32000):
    """Token encoder, conditioning and separation outputs of a model for the 4 cases on 2 s of speech."""
    mix = (speech(n, 0, 3) + 0.5 * speech(n, 1, 3) + noise(n, 9)).astype(np.float32)
    enr_t, enr_o = speech(96000, 0, 20), speech(128000, 2, 0)
    with torch.no_grad():
        ci = make_cond_inputs(model, enr_t, enr_o, CASES)
        cond = model.build_cond(ci)
        y, logits = model(torch.from_numpy(mix)[None].expand(len(CASES), -1).contiguous(), cond)
    return [ci.tok_t, ci.g_t, cond.film, cond.k, y, logits]


def test_ckpt_loader_picks_ema_and_checks_architecture(tmp_path):
    # EMA present -> the EMA weights; --raw or no EMA -> the model weights.
    ckpt = fake_checkpoint(tmp_path / "ema.pt")
    model, step, used_ema = export.load_checkpoint_model(tmp_path / "ema.pt")
    assert used_ema and step == 1234
    assert all(torch.equal(v, ckpt["ema"][k]) for k, v in model.state_dict().items())
    model, _, used_ema = export.load_checkpoint_model(tmp_path / "ema.pt", raw=True)
    assert not used_ema
    assert all(torch.equal(v, ckpt["model"][k]) for k, v in model.state_dict().items())
    ckpt = fake_checkpoint(tmp_path / "noema.pt", ema_seed=None)
    model, _, used_ema = export.load_checkpoint_model(tmp_path / "noema.pt")
    assert not used_ema
    assert all(torch.equal(v, ckpt["model"][k]) for k, v in model.state_dict().items())

    # A declared config hash must be contract 3.1's (top level or inside "config"); the right one loads.
    fake_checkpoint(tmp_path / "hash_ok.pt", config_hash=CONFIG_HASH)
    assert export.load_checkpoint_model(tmp_path / "hash_ok.pt")[2]
    fake_checkpoint(tmp_path / "hash_top.pt", config_hash="0" * 64)
    with pytest.raises(ValueError, match="config_hash"):
        export.load_checkpoint_model(tmp_path / "hash_top.pt")
    fake_checkpoint(tmp_path / "hash_cfg.pt", config={"run": "x", "config_hash": "1" * 64})
    with pytest.raises(ValueError, match="config_hash"):
        export.load_checkpoint_model(tmp_path / "hash_cfg.pt")

    # Weights of another shape (another architecture) and a file that is not a checkpoint are refused.
    state = init_random_for_parity(2).state_dict()
    state["heads.weight"] = torch.zeros(3, state["heads.weight"].shape[1])
    fake_checkpoint(tmp_path / "shape.pt", ema=state)
    with pytest.raises(ValueError, match="do not match"):
        export.load_checkpoint_model(tmp_path / "shape.pt")
    torch.save({"weights": state}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a training checkpoint"):
        export.load_checkpoint_model(tmp_path / "other.pt")


@pytest.fixture(scope="module")
def ckpt_export_dir(tmp_path_factory):
    """`python -m ml.export --ckpt fake.pt --out DIR` (EMA weights, rendered test-vector scene), about 1 min."""
    missing = [str(p) for p in NEEDS if not p.exists()]
    if missing:
        pytest.skip(f"export needs {missing}")
    base = tmp_path_factory.mktemp("ckpt_export")
    ckpt = fake_checkpoint(base / "fake.pt")
    out = base / "v1"
    export.main(["--ckpt", str(base / "fake.pt"), "--out", str(out)])
    return out, ckpt


def test_ckpt_export_round_trip(ckpt_export_dir):
    out, ckpt = ckpt_export_dir
    assert sorted(p.name for p in out.iterdir()) == sorted(export.FILES)
    loaded, header = export.load_esw(out / "earshot.esw")
    meta = header["meta"]
    assert meta["ema"] is True and meta["step"] == 1234 and meta["init"] == "trained"
    assert meta["run"] == out.parent.name                    # the checkpoint's folder name
    assert meta["data_licenses"] == export.training_licenses() and len(meta["data_licenses"]) > 0
    # The exported weights reproduce the checkpoint's EMA model (1e-6), not its raw model.
    ema_model = export.ESB16DF()
    ema_model.load_state_dict(ckpt["ema"])
    ema_model.eval()
    for a, b in zip(outputs(ema_model), outputs(loaded)):
        assert float((a - b).abs().max()) <= 1e-6
    raw_model = export.ESB16DF()
    raw_model.load_state_dict(ckpt["model"])
    assert float((outputs(raw_model.eval())[4] - outputs(loaded)[4]).abs().max()) > 1e-3
    # Rendered test-vector scene by default with --ckpt; parity files point at this .esw.
    for name in ["testvec_short.est", "testvec_long.est"]:
        _, tv, _ = fileio.read_container(out / name, expect_magic="EST1")
        assert "render.py" in tv["meta"]["scene"]["builder"]
        assert tv["meta"]["weights_sha256"] == header["data_sha256"]
    with open(out / "thresholds.json", encoding="utf-8") as f:
        assert json.load(f) == export.initial_thresholds(header["data_sha256"])


def test_thresholds_copy_checks_model_hash(ckpt_export_dir, tmp_path):
    out = tmp_path / "v1"
    shutil.copytree(ckpt_export_dir[0], out)               # keep the shared folder unchanged
    model_hash = fileio.model_hash(out / "earshot.esw")
    calibrated = export.initial_thresholds(model_hash)
    calibrated["target"]["sbr_db"] = [2.5, 9.75]
    calibrated["locked_cos"] = 0.731
    good = tmp_path / "calibrated.json"
    good.write_text(json.dumps(calibrated))
    export.main(["--thresholds", str(good), "--out", str(out)])
    with open(out / "thresholds.json", encoding="utf-8") as f:
        assert json.load(f) == calibrated
    assert export.verify_folder(out)                         # the folder still checks out

    # Another model's file is refused and leaves the installed one alone.
    bad = tmp_path / "other_model.json"
    bad.write_text(json.dumps(dict(calibrated, model_hash="f" * 64)))
    with pytest.raises(ValueError, match="calibrated for"):
        export.main(["--thresholds", str(bad), "--out", str(out)])
    with open(out / "thresholds.json", encoding="utf-8") as f:
        assert json.load(f) == calibrated
    # A file missing a contract 6 key is refused.
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({k: v for k, v in calibrated.items() if k != "locked_cos"}))
    with pytest.raises(ValueError, match="keys"):
        export.main(["--thresholds", str(broken), "--out", str(out)])
    # --thresholds only goes alone or with --ckpt, and needs an exported folder.
    with pytest.raises(SystemExit):
        export.main(["--random", "--thresholds", str(good), "--out", str(tmp_path / "x")])
    with pytest.raises(FileNotFoundError):
        export.main(["--thresholds", str(good), "--out", str(tmp_path / "empty")])
