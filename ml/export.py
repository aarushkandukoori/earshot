"""Export weights, test vectors, thresholds and check data for the C engine and the app (10-ml 7, contract 3 and 6).

Commands (run from the repo root):
    .venv/bin/python -m ml.export --random --seed 0 --out ml/exports/random_v1/
    .venv/bin/python -m ml.export --ckpt ml/runs/main/best_1.pt --out ml/exports/v1/ [--raw] [--thresholds FILE]
    .venv/bin/python -m ml.export --thresholds ml/runs/main/thresholds.json --out ml/exports/v1/
    .venv/bin/python -m ml.export --testvec --out DIR [--scene render]

--random and --ckpt write all six files of a release folder:
    earshot.esw          the 270 weight tensors (ESW1 container, contract 3.2)
    testvec_short.est    2 s parity vectors (EST1, contract 3.3), with the debug taps of the first 8 hops
    testvec_long.est     10 s parity vectors
    thresholds.json      enrollment check thresholds (contract 6; initial values unless --thresholds is given)
    wespk_selftest.est   the WeSpeaker launch self-test (contract 6)
    quality_vectors.json 5 captures with the expected check-formula values (contract 6)
--ckpt exports the EMA weights of a training checkpoint (the raw weights with --raw, or when the checkpoint has no
EMA yet), refuses a checkpoint built for another architecture (a declared config hash other than contract 3.1's, or
tensors that do not load strictly), writes to ml/exports/v1/ unless --out is given, and makes the test vectors from
the rendered scene (--scene render) unless --scene says otherwise.
--thresholds FILE alone copies a calibrated thresholds.json (ml.tools.calibrate_thresholds) into the existing export
--out DIR, after checking that its model_hash equals the data_sha256 of DIR/earshot.esw (10-ml 7.3 and 13 order:
export --ckpt, calibrate, export --thresholds).
--testvec re-creates the three .est files (testvec_short, testvec_long, wespk_selftest) in DIR, as 10-ml 7.3 says.
The two parity files are made from DIR/earshot.esw; wespk_selftest.est does not depend on the weights. The test
vectors are always made from the weights read back from earshot.esw (load_esw), so the reference really uses the
exported numbers.

Weight layout (10-ml 7.2): every Linear weight is stored [in, out] (PyTorch's weight transposed); each LSTM
direction is stored as w_ih [in, 4H], w_hh [H, 4H] and bias [4H] = bias_ih + bias_hh, with the suffix _rev for the
reverse direction; LayerNorms and plain parameters are unchanged. Gate order i, f, g, o is PyTorch's own.
"""
import argparse
import base64
import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.earshot import quality, testaudio
from ml.earshot.config import (CONFIG, CONFIG_HASH, EXPORTS, MAX_O_TOK, MAX_T_TOK, ROOT, WESPK_SELFTEST_NPZ,
                               config_hash)
from ml.earshot.fileio import read_container, write_container
from ml.earshot.model import ESB16DF, CondInputs, init_random_for_parity
from ml.earshot.stream_ref import StreamRef, pack_cond
from ml.earshot.vad import es_vad

MODEL_NAME = "esb16-df"
MODEL_VERSION = "1.0.0"
CASES = ["person", "everyone", "noown", "nowespk"]            # contract 3.3 test cases, in file order
WESPEAKER_NOTE = "WeSpeaker conditioning trained with VoxCeleb research-terms weights - personal use only"
CREDITS = "interaction idea from Look Once to Hear (UW, CHI 2024); no LOTH code used"

# meta.tolerances of the two test-vector files: exactly these keys (contract 3.3; a missing or extra key fails the
# parity test). The long file keeps "dbg" although it has no dbg arrays.
TOLERANCE_KEYS = ["vad", "tokens", "g", "film", "k", "v", "dbg", "out", "out_snr_db", "heads"]
TOLERANCES_SHORT = {"vad": 0, "tokens": 1e-4, "g": 1e-4, "film": 1e-4, "k": 1e-4, "v": 1e-4, "dbg": 1e-4,
                    "out": 2e-4, "out_snr_db": 60.0, "heads": 1e-3}
TOLERANCES_LONG = {"vad": 0, "tokens": 1e-4, "g": 1e-4, "film": 1e-4, "k": 1e-4, "v": 1e-4, "dbg": 1e-4,
                   "out": 1e-3, "out_snr_db": 50.0, "heads": 1e-3}

SHORT_N = 32000          # testvec_short.est: 2 s
LONG_N = 160000          # testvec_long.est: 10 s
DEBUG_HOPS = 8           # dbg.* hold the first 8 hops of case person (short file only)
MASK_T_ZEROED = (40, 50)  # mask_t = vad_t with frames 40..49 forced to 0 (proves the engine uses the given mask)
TESTVEC_SEED = 7
SELFTEST_NPZ = WESPK_SELFTEST_NPZ   # ml/exports/wespeaker_selftest_inputs.npz, written by ml.tools.convert_wespeaker

FILES = ["earshot.esw", "testvec_short.est", "testvec_long.est", "thresholds.json", "wespk_selftest.est",
         "quality_vectors.json"]
DEFAULT_CKPT_OUT = EXPORTS / "v1"        # where `--ckpt` writes when --out is not given (10-ml 7.3)


# ---------------------------------------------------------------------------------------------------------------
# Tensor order and the PyTorch <-> export mapping (10-ml 7.2)
# ---------------------------------------------------------------------------------------------------------------
def EXPORT_ORDER():
    """The 270 export names in file order: the contract 3.2 table read row by row, index loops outermost."""
    names = []
    for k in range(16):
        names += [f"enc.lin.{k}.weight", f"enc.lin.{k}.bias", f"enc.ln.{k}.weight", f"enc.ln.{k}.bias"]
    for b in range(4):
        for part in ["tn.weight", "tn.bias", "t.w_ih", "t.w_hh", "t.bias", "tf.weight", "tf.bias",
                     "bn.weight", "bn.bias", "b.w_ih", "b.w_hh", "b.bias", "b.w_ih_rev", "b.w_hh_rev", "b.bias_rev",
                     "bf.weight", "bf.bias"]:
            names.append(f"blocks.{b}.{part}")
    names += ["att.q.weight", "att.q.bias", "att.o.weight", "att.o.bias", "att.kv.weight", "att.kv.bias",
              "att.type_emb", "att.null_tok"]
    names += ["film.0.weight", "film.0.bias", "film.2.weight", "film.2.bias"]
    names += ["null.e_t", "null.g_t", "null.e_o", "null.g_o"]
    for k in range(16):
        names += [f"dec.{k}.ln.weight", f"dec.{k}.ln.bias", f"dec.{k}.fc1.weight", f"dec.{k}.fc1.bias",
                  f"dec.{k}.fc2.weight", f"dec.{k}.fc2.bias"]
    names += ["heads.weight", "heads.bias"]
    names += ["tok.blk.tn.weight", "tok.blk.tn.bias", "tok.blk.bn.weight", "tok.blk.bn.bias"]
    for path in ["tok.blk.t", "tok.blk.b"]:
        names += [f"{path}.w_ih", f"{path}.w_hh", f"{path}.bias", f"{path}.w_ih_rev", f"{path}.w_hh_rev",
                  f"{path}.bias_rev"]
    names += ["tok.blk.tf.weight", "tok.blk.tf.bias", "tok.blk.bf.weight", "tok.blk.bf.bias"]
    names += ["tok.proj.weight", "tok.proj.bias", "tok.g.weight", "tok.g.bias"]
    return names


def param_map(model):
    """For every export name: (rule, PyTorch state_dict keys). Rules:
    "copy"      - stored unchanged (LayerNorm weight/bias, Linear bias, plain parameters)
    "transpose" - stored transposed (Linear weight -> [in, out]; LSTM weight_ih / weight_hh -> [in, 4H] / [H, 4H])
    "bias_sum"  - an LSTM direction's bias_ih + bias_hh
    """
    rules = {}
    for path, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            for torch_suffix, tag in [("", ""), ("_reverse", "_rev")]:
                if not hasattr(module, "weight_ih_l0" + torch_suffix):
                    continue                     # one-direction LSTM: no reverse weights
                rules[f"{path}.w_ih{tag}"] = ("transpose", [f"{path}.weight_ih_l0{torch_suffix}"])
                rules[f"{path}.w_hh{tag}"] = ("transpose", [f"{path}.weight_hh_l0{torch_suffix}"])
                rules[f"{path}.bias{tag}"] = ("bias_sum", [f"{path}.bias_ih_l0{torch_suffix}",
                                                           f"{path}.bias_hh_l0{torch_suffix}"])
        elif isinstance(module, nn.Linear):
            rules[f"{path}.weight"] = ("transpose", [f"{path}.weight"])
            rules[f"{path}.bias"] = ("copy", [f"{path}.bias"])
        elif isinstance(module, nn.LayerNorm):
            rules[f"{path}.weight"] = ("copy", [f"{path}.weight"])
            rules[f"{path}.bias"] = ("copy", [f"{path}.bias"])
    for name in ["att.type_emb", "att.null_tok", "null.e_t", "null.g_t", "null.e_o", "null.g_o"]:
        rules[name] = ("copy", [name])

    # Every export name has a rule and every state_dict entry is used exactly once.
    if set(rules) != set(EXPORT_ORDER()):
        raise ValueError("model tensors do not match EXPORT_ORDER: "
                         f"extra {sorted(set(rules) - set(EXPORT_ORDER()))}, "
                         f"missing {sorted(set(EXPORT_ORDER()) - set(rules))}")
    used = [key for _, keys in rules.values() for key in keys]
    if sorted(used) != sorted(model.state_dict().keys()):
        raise ValueError("export rules do not cover the state_dict exactly once")
    return rules


def to_export(model):
    """Model -> list of (export name, float32 numpy array) in EXPORT_ORDER, laid out as contract 3.2."""
    state = model.state_dict()
    rules = param_map(model)
    out = []
    for name in EXPORT_ORDER():
        rule, keys = rules[name]
        if rule == "copy":
            value = state[keys[0]]
        elif rule == "transpose":
            value = state[keys[0]].t()
        else:                                          # bias_sum
            value = state[keys[0]] + state[keys[1]]
        out.append((name, np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)))
    return out


def from_export(tensors):
    """Inverse of to_export: {export name: array} (or a list of pairs) -> a new ESB16DF in eval mode on the CPU.

    The summed LSTM bias goes into bias_ih and bias_hh is set to 0 (the LSTM output is the same).
    Raises ValueError for a missing or extra tensor or a wrong shape.
    """
    tensors = dict(tensors)
    model = ESB16DF()
    rules = param_map(model)
    if set(tensors) != set(rules):
        raise ValueError(f"tensors: missing {sorted(set(rules) - set(tensors))}, "
                         f"extra {sorted(set(tensors) - set(rules))}")
    shapes = {key: value.shape for key, value in model.state_dict().items()}
    state = {}
    for name in EXPORT_ORDER():
        rule, keys = rules[name]
        value = torch.from_numpy(np.array(tensors[name], dtype=np.float32))
        if rule == "transpose":
            value = value.t().contiguous()
        if tuple(value.shape) != tuple(shapes[keys[0]]):
            raise ValueError(f"tensor {name}: shape {list(tensors[name].shape)} does not fit {keys[0]}")
        state[keys[0]] = value
        if rule == "bias_sum":
            state[keys[1]] = torch.zeros_like(value)
    model.load_state_dict(state, strict=True)
    return model.eval()


# ---------------------------------------------------------------------------------------------------------------
# Headers and the weights file
# ---------------------------------------------------------------------------------------------------------------
def git_commit():
    """Current commit of the repo (read only), or a note when there is none yet."""
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "none (no commit yet)"


def base_header(fmt, meta):
    """The common header fields of contract 3.1 (the writer adds arrays, data_bytes and data_sha256)."""
    return {"format": fmt, "format_version": 1, "model": MODEL_NAME, "model_version": MODEL_VERSION,
            "config": CONFIG, "config_hash": config_hash(), "meta": meta}


def weights_meta(run, step, ema, init, data_licenses):
    """meta of earshot.esw (10-ml 7.2)."""
    return {"run": run, "step": step, "ema": ema, "git_commit": git_commit(),
            "date": datetime.date.today().isoformat(), "init": init, "data_licenses": data_licenses,
            "wespeaker_note": WESPEAKER_NOTE, "credits": CREDITS, "eval_only_data": ["WHAM!"]}


def write_esw(path, model, meta):
    """Write earshot.esw -> the written header (its data_sha256 is the model_hash)."""
    header = base_header("earshot-weights", meta)
    return write_container(path, b"ESW1", header, to_export(model))


def check_model_header(header):
    """The engine's loader checks 4 to 7 (20-engine 4.1) in Python; raises ValueError naming the field."""
    if header.get("model") != MODEL_NAME:
        raise ValueError(f"model: {header.get('model')!r}, expected {MODEL_NAME!r}")
    version = str(header.get("model_version", ""))
    if version.split(".")[0] != MODEL_VERSION.split(".")[0]:
        raise ValueError(f"model_version: {version!r}, expected major {MODEL_VERSION.split('.')[0]}")
    cfg = header.get("config")
    if not isinstance(cfg, dict) or set(cfg) != set(CONFIG):
        raise ValueError("config: keys differ from contract 3.1")
    for key in CONFIG:
        if cfg[key] != CONFIG[key]:
            raise ValueError(f"config.{key}: {cfg[key]!r}, expected {CONFIG[key]!r}")
    if header.get("config_hash") != CONFIG_HASH:
        raise ValueError(f"config_hash: {header.get('config_hash')!r}, expected {CONFIG_HASH}")


def load_esw(path):
    """Read earshot.esw -> (model in eval mode on the CPU, header). Checks the container (fileio), the model header,
    the 270 tensor names and dtypes; from_export checks the shapes."""
    _, header, arrays = read_container(path, expect_magic="ESW1")
    check_model_header(header)
    for entry in header["arrays"]:
        if entry["dtype"] != "f32":
            raise ValueError(f"array {entry['name']}: dtype {entry['dtype']}, expected f32")
    return from_export(arrays), header


# ---------------------------------------------------------------------------------------------------------------
# Test vectors (10-ml 7.4, contract 3.3)
# ---------------------------------------------------------------------------------------------------------------
def wespeaker_embed(clips):
    """WeSpeaker vectors of a list of 1-D float32 clips with the PyTorch module on the CPU (contract 6)."""
    from ml.earshot.wespeaker import WeSpeaker   # imported here: only the commands that need WeSpeaker load it
    encoder = WeSpeaker.load()
    return [encoder.embed_numpy(clip).astype(np.float32) for clip in clips]


def case_cond_inputs(model, enr_t, mask_t, enr_o, mask_o, e_t, e_o):
    """CondInputs for the 4 cases (one row each, in CASES order) plus the expected token-encoder outputs.

    person   = PERSON(target, own, use_wespk 1)    everyone = EVERYONE(own, 1): has_t False, mode [0, 1]
    noown    = PERSON(target, no own, 1)           nowespk  = PERSON(target, own, 0): use_wes False
    Returns (ci, tok_t [n_t, 64], g_t [128], tok_o [n_o, 64], g_o [128]) with the tokens as numpy float32.
    """
    with torch.no_grad():
        tok_t, val_t, g_t, n_avail_t = model.enroll_clip(enr_t, mask_t, MAX_T_TOK)
        tok_o, val_o, g_o, n_avail_o = model.enroll_clip(enr_o, mask_o, MAX_O_TOK)
    if n_avail_t < 4 or n_avail_o < 4:
        raise ValueError(f"test enrollments give too few tokens ({n_avail_t}, {n_avail_o}); need at least 4")
    B = len(CASES)

    def rows(tensor):
        """The same tensor repeated for the 4 cases."""
        return tensor[None].expand(B, *tensor.shape).clone()

    ci = CondInputs(
        e_t=rows(torch.from_numpy(e_t)), tok_t=rows(tok_t), val_t=rows(val_t), g_t=rows(g_t),
        e_o=rows(torch.from_numpy(e_o)), tok_o=rows(tok_o), val_o=rows(val_o), g_o=rows(g_o),
        has_t=torch.tensor([case != "everyone" for case in CASES]),
        has_o=torch.tensor([case != "noown" for case in CASES]),
        use_wes=torch.tensor([case != "nowespk" for case in CASES]),
        mode=torch.tensor([[0.0, 1.0] if case == "everyone" else [1.0, 0.0] for case in CASES]))
    # Expected encoder outputs = the valid rows only ([n_t, 64] and [n_o, 64]).
    return (ci, tok_t[val_t].numpy().astype(np.float32), g_t.numpy().astype(np.float32),
            tok_o[val_o].numpy().astype(np.float32), g_o.numpy().astype(np.float32))


def make_testvecs(out_dir, seed=TESTVEC_SEED, scene="handbuilt"):
    """Write testvec_short.est and testvec_long.est into out_dir from out_dir/earshot.esw -> {file name: header}.

    scene picks the audio builder: "handbuilt" (testaudio.testvec_inputs, the random_v1 default) or "render"
    (testaudio.testvec_inputs_render: render.render_example in test mode, 10-ml 7.4).

    Array order (10-ml 7.4): in, enr_t, mask_t, enr_o, mask_o, vad_t, e_t, e_o, tok_t, g_t, tok_o, g_o, then per case
    cond.{case}.film, .k, .v, .n, out.{case}, heads.{case}, then (short file only) the 7 dbg arrays.
    """
    out_dir = Path(out_dir)
    model, weights_header = load_esw(out_dir / "earshot.esw")
    weights_sha256 = weights_header["data_sha256"]
    torch.set_num_threads(4)

    # Inputs (fixed, seed 7), masks and WeSpeaker vectors.
    if scene == "render":
        inputs = testaudio.testvec_inputs_render(seed)
    elif scene == "handbuilt":
        inputs = testaudio.testvec_inputs(seed)
    else:
        raise ValueError(f"unknown test-vector scene {scene!r} (handbuilt or render)")
    enr_t, enr_o = inputs["enr_t"], inputs["enr_o"]
    vad_t = es_vad(enr_t)
    mask_t = vad_t.copy()
    mask_t[MASK_T_ZEROED[0]:MASK_T_ZEROED[1]] = 0
    mask_o = es_vad(enr_o)
    e_t, e_o = wespeaker_embed([enr_t, enr_o])

    # Conditioning for the 4 cases, packed the way es_cond_build lays it out.
    ci, tok_t, g_t, tok_o, g_o = case_cond_inputs(model, enr_t, mask_t, enr_o, mask_o, e_t, e_o)
    with torch.no_grad():
        cond = model.build_cond(ci)
    packed = [pack_cond(cond, i) for i in range(len(CASES))]

    reference = StreamRef(model)
    headers = {}
    for name, n, tolerances in [("testvec_short.est", SHORT_N, TOLERANCES_SHORT),
                                ("testvec_long.est", LONG_N, TOLERANCES_LONG)]:
        x = inputs["in_long"][:n]
        arrays = [("in", x), ("enr_t", enr_t), ("mask_t", mask_t), ("enr_o", enr_o), ("mask_o", mask_o),
                  ("vad_t", vad_t), ("e_t", e_t), ("e_o", e_o), ("tok_t", tok_t), ("g_t", g_t),
                  ("tok_o", tok_o), ("g_o", g_o)]
        dbg = None
        for i, case in enumerate(CASES):
            pc = packed[i]
            if case == "person" and n == SHORT_N:
                y, heads, dbg = reference.run(pc, x, debug=DEBUG_HOPS)
            else:
                y, heads = reference.run(pc, x)
            arrays += [(f"cond.{case}.film", pc.film.numpy().astype(np.float32)),
                       (f"cond.{case}.k", pc.k.numpy().astype(np.float32)),
                       (f"cond.{case}.v", pc.v.numpy().astype(np.float32)),
                       (f"cond.{case}.n", np.array([pc.n], dtype=np.int32)),
                       (f"out.{case}", y), (f"heads.{case}", heads)]
        if n == SHORT_N:
            for tap in ["feat", "z_enc", "z_blk0", "z_blk1", "z_blk2", "z_blk3", "coef"]:
                arrays.append((f"dbg.{tap}", dbg[tap]))

        meta = {
            "kind": "parity",
            "length": "short" if n == SHORT_N else "long",
            "weights_sha256": weights_sha256,
            "tolerances": dict(tolerances),
            "cases": {"person": "PERSON(target from enr_t, own from enr_o, use_wespk 1)",
                      "everyone": "EVERYONE(own from enr_o, use_wespk 1)",
                      "noown": "PERSON(target from enr_t, no own, use_wespk 1)",
                      "nowespk": "PERSON(target from enr_t, own from enr_o, use_wespk 0)"},
            "notes": ["out.{case}: hop j emits samples 64j .. 64j+63, from a reset state",
                      "mask_t = vad_t with frames 40 to 49 forced to 0; mask_o = es_vad(enr_o)",
                      "tok_t, tok_o hold only the valid token rows; cond k, v are zero beyond n",
                      "dbg.*: first 8 hops of case person (short file only)"],
            "made_by": "python -m ml.export, CPU fp32, ml/earshot/stream_ref.py on weights read back from "
                       "earshot.esw",
            "wespeaker": "e_t, e_o from ml/earshot/wespeaker.py (PyTorch CPU fp32); " + WESPEAKER_NOTE,
            "scene": inputs["scene"],
            "sources": inputs["sources"],
            "weights_init": weights_header["meta"].get("init"),
            "weights_run": weights_header["meta"].get("run"),
            "git_commit": git_commit(),
            "date": datetime.date.today().isoformat(),
        }
        headers[name] = write_container(out_dir / name, b"EST1", base_header("earshot-testvec", meta), arrays)
    return headers


# ---------------------------------------------------------------------------------------------------------------
# WeSpeaker self-test, thresholds, quality vectors (contract 6)
# ---------------------------------------------------------------------------------------------------------------
def write_wespk_selftest(path, npz_path=SELFTEST_NPZ):
    """wespk_selftest.est from the clip saved by ml.tools.convert_wespeaker -> the written header.

    The embedding is recomputed here with the PyTorch module on the CPU and must equal the saved one within 1e-5.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} is missing; make it with `.venv/bin/python -m ml.tools.convert_wespeaker`")
    saved = np.load(npz_path)
    clip = np.asarray(saved["clip"], dtype=np.float32)
    info = json.loads(str(saved["meta_json"]))
    embedding = wespeaker_embed([clip])[0]
    diff = float(np.max(np.abs(embedding - saved["embedding"])))
    if diff > 1e-5:
        raise ValueError(f"WeSpeaker embedding of the self-test clip moved by {diff} since it was saved")
    if clip.shape != (80000,) or embedding.shape != (256,):
        raise ValueError(f"self-test shapes {clip.shape}, {embedding.shape}, expected (80000,), (256,)")
    meta = {"kind": "wespeaker-selftest", "tolerances": {"cosine_min": 0.999},
            "sources": info.get("sources", []),
            "embedding_from": "ml/earshot/wespeaker.py, PyTorch CPU fp32, eval mode",
            "wespeaker_weights_sha256": info.get("wespeaker_weights_sha256"),
            "note": "WeSpeaker VoxCeleb research-terms weights - personal use only",
            "git_commit": git_commit(), "date": datetime.date.today().isoformat()}
    return write_container(path, b"EST1", base_header("earshot-testvec", meta),
                           [("clip", clip), ("embedding", embedding)])


def initial_thresholds(model_hash):
    """The contract 6 thresholds.json object with the initial (pre-calibration) values."""
    return {"version": 1, "model_hash": model_hash,
            "target": {"speech_s": [4, 5], "sbr_db": [0, 6], "one_voice": [0.45, 0.60], "own_cos": [0.50, 0.60],
                       "dup_cos": 0.75, "clip_pct": [0.1, 1.0]},
            "own": {"speech_s": 8, "snr_db": [15, 20]},
            "own_window_cos": 0.60, "locked_cos": 0.0}


def check_thresholds_shape(obj):
    """Raise ValueError unless obj has exactly the keys (and nesting) of the contract 6 thresholds.json."""
    template = initial_thresholds("x")
    if set(obj) != set(template):
        raise ValueError(f"thresholds.json keys {sorted(obj)}, expected {sorted(template)}")
    for group in ["target", "own"]:
        if not isinstance(obj[group], dict) or set(obj[group]) != set(template[group]):
            raise ValueError(f"thresholds.json {group} keys differ from contract 6")
    if obj["version"] != 1:
        raise ValueError(f"thresholds.json version {obj['version']}, expected 1")


def write_thresholds(path, model_hash, source=None):
    """thresholds.json: the initial values, or the calibrated file `source` (its model_hash must match)."""
    if source is None:
        obj = initial_thresholds(model_hash)
    else:
        with open(source, encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("model_hash") != model_hash:
            raise ValueError(f"{source} was calibrated for model_hash {obj.get('model_hash')}, "
                             f"but the exported weights have {model_hash}")
    check_thresholds_shape(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    return obj


def write_quality_vectors(path, seed=TESTVEC_SEED):
    """quality_vectors.json: 5 fixed captures (base64 little-endian float32) with the values quality.py computes.
    The values do not depend on the weights."""
    vectors = []
    for capture in testaudio.quality_captures(seed):
        pcm = capture["pcm"]
        vectors.append({"name": capture["name"], "description": capture["description"], "n": len(pcm),
                        "sources": capture["sources"],
                        "expected": quality.vector_values(pcm),
                        "pcm_f32le_base64": base64.b64encode(pcm.astype("<f4").tobytes()).decode("ascii")})
    obj = {"version": 1, "kind": "earshot-quality-vectors", "seed": seed,
           "made_by": "python -m ml.export, ml/earshot/quality.py (contract 6 check formulas)",
           "sample_rate": 16000,
           "constants": {"frame_hop": quality.FRAME_HOP, "pad": quality.PAD, "clip_max": quality.CLIP_MAX,
                         "round_step": quality.ROUND_STEP, "min_clip": quality.MIN_CLIP,
                         "win_len": quality.WIN_LEN, "win_hop": quality.WIN_HOP, "clip_level": quality.CLIP_LEVEL,
                         "frame_seconds": quality.FRAME_SECONDS, "power_floor": quality.POWER_FLOOR},
           "definitions": quality.DEFINITIONS,
           "vectors": vectors}
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return obj


def decode_pcm(vector):
    """The float32 samples of one quality vector entry."""
    return np.frombuffer(base64.b64decode(vector["pcm_f32le_base64"]), dtype="<f4").astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Whole release folder
# ---------------------------------------------------------------------------------------------------------------
def training_licenses():
    """Sorted unique license strings of the training manifests (meta.data_licenses of a trained export)."""
    licenses = set()
    for name in ["speech_train.jsonl", "noise_train.jsonl"]:
        for row in testaudio.read_jsonl(testaudio.DATA / "manifests" / name):
            licenses.add(row["license"])
    return sorted(licenses)


def checkpoint_config_hash(ckpt):
    """The architecture config hash a checkpoint declares, or None when it declares none.

    Looked up as ckpt["config_hash"] or ckpt["config"]["config_hash"]. The checkpoints ml/train.py writes (10-ml
    10.4) hold the training YAML under "config" and no architecture hash, because model.py always builds the
    contract 3.1 CONFIG; for those the strict state_dict load in load_checkpoint_model (every tensor name and shape)
    is the architecture check.
    """
    declared = ckpt.get("config_hash")
    if declared is None and isinstance(ckpt.get("config"), dict):
        declared = ckpt["config"].get("config_hash")
    return declared


def load_checkpoint_model(path, raw=False):
    """A training checkpoint (10-ml 10.4) -> (model, step, used EMA). Uses the EMA weights unless raw=True or the
    checkpoint has none (ema is None before step 2,000 and in the smoke run). Both `model` and `ema` must be
    state_dicts with the ESB16DF keys.

    Refuses (ValueError) a checkpoint that declares a config hash other than contract 3.1's CONFIG_HASH, or whose
    tensors do not load strictly into ESB16DF (another architecture).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{path}: not a training checkpoint (no 'model' entry)")
    declared = checkpoint_config_hash(ckpt)
    if declared is not None and declared != CONFIG_HASH:
        raise ValueError(f"{path}: config_hash {declared} differs from contract 3.1 ({CONFIG_HASH}); refusing to "
                         f"export weights of another architecture")
    use_ema = (not raw) and ckpt.get("ema") is not None
    state = ckpt["ema"] if use_ema else ckpt["model"]
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a state_dict under {'ema' if use_ema else 'model'}")
    model = ESB16DF()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as err:          # missing, unexpected or reshaped tensors: another architecture
        raise ValueError(f"{path}: weights do not match the contract 3.1 architecture (config hash "
                         f"{CONFIG_HASH}): {str(err).splitlines()[0]}") from err
    return model.eval(), int(ckpt.get("step", -1)), use_ema


def install_thresholds(out_dir, source):
    """`export --thresholds FILE --out DIR` on an existing export: copy the calibrated thresholds.json into DIR after
    checking its shape and that its model_hash equals the data_sha256 of DIR/earshot.esw -> the copied object."""
    out_dir = Path(out_dir)
    esw = out_dir / "earshot.esw"
    if not esw.exists():
        raise FileNotFoundError(f"{esw} is missing; export the weights first (--ckpt or --random)")
    _, header, _ = read_container(esw, expect_magic="ESW1")      # verifies data_sha256 too
    return write_thresholds(out_dir / "thresholds.json", header["data_sha256"], source)


def export_all(out_dir, model, meta, thresholds_source=None, scene="handbuilt"):
    """Write the six files of a release folder and return the model_hash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_header = write_esw(out_dir / "earshot.esw", model, meta)
    model_hash = weights_header["data_sha256"]
    print(f"earshot.esw written, model_hash {model_hash}")
    write_thresholds(out_dir / "thresholds.json", model_hash, thresholds_source)
    write_wespk_selftest(out_dir / "wespk_selftest.est")
    write_quality_vectors(out_dir / "quality_vectors.json")
    print("thresholds.json, wespk_selftest.est, quality_vectors.json written; making test vectors (about 30 s)")
    make_testvecs(out_dir, scene=scene)
    return model_hash


def verify_folder(out_dir):
    """Read every file back and check it against contract 3 and 6 -> list of summary lines. Raises on a problem."""
    out_dir = Path(out_dir)
    lines = []
    _, weights_header, tensors = read_container(out_dir / "earshot.esw", expect_magic="ESW1")
    check_model_header(weights_header)
    names = [entry["name"] for entry in weights_header["arrays"]]
    floats = sum(int(np.prod(entry["shape"])) for entry in weights_header["arrays"])
    if names != EXPORT_ORDER() or len(names) != 270 or floats != 2570500:
        raise ValueError(f"earshot.esw: {len(names)} tensors and {floats} floats, expected 270 and 2,570,500 in order")
    model_hash = weights_header["data_sha256"]
    lines.append(f"earshot.esw  {os.path.getsize(out_dir / 'earshot.esw'):,} bytes, 270 tensors, {floats:,} floats, "
                 f"model_hash {model_hash}")

    for name in ["testvec_short.est", "testvec_long.est"]:
        _, header, arrays = read_container(out_dir / name, expect_magic="EST1")
        check_model_header(header)
        meta = header["meta"]
        if list(meta["tolerances"]) != TOLERANCE_KEYS or meta["weights_sha256"] != model_hash:
            raise ValueError(f"{name}: tolerances keys or weights_sha256 wrong")
        lines.append(f"{name}  {os.path.getsize(out_dir / name):,} bytes, {len(arrays)} arrays, "
                     f"in {arrays['in'].shape[0]} samples, n_t {arrays['tok_t'].shape[0]}, "
                     f"n_o {arrays['tok_o'].shape[0]}, n per case "
                     f"{[int(arrays[f'cond.{c}.n'][0]) for c in CASES]}, weights_sha256 matches")

    _, header, arrays = read_container(out_dir / "wespk_selftest.est", expect_magic="EST1")
    check_model_header(header)
    if header["meta"]["kind"] != "wespeaker-selftest" or header["meta"]["tolerances"] != {"cosine_min": 0.999}:
        raise ValueError("wespk_selftest.est: meta.kind or meta.tolerances wrong")
    lines.append(f"wespk_selftest.est  {os.path.getsize(out_dir / 'wespk_selftest.est'):,} bytes, "
                 f"clip {arrays['clip'].shape}, embedding norm {np.linalg.norm(arrays['embedding']):.6f}")

    with open(out_dir / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    check_thresholds_shape(thresholds)
    if thresholds["model_hash"] != model_hash:
        raise ValueError("thresholds.json: model_hash differs from earshot.esw")
    lines.append("thresholds.json  model_hash matches")

    with open(out_dir / "quality_vectors.json", encoding="utf-8") as f:
        vectors = json.load(f)["vectors"]
    for v in vectors:
        if quality.vector_values(decode_pcm(v)) != v["expected"]:
            raise ValueError(f"quality_vectors.json: {v['name']} does not reproduce")
    lines.append(f"quality_vectors.json  {os.path.getsize(out_dir / 'quality_vectors.json'):,} bytes, "
                 f"{len(vectors)} captures reproduce")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export Earshot weights, test vectors and check data.")
    parser.add_argument("--random", action="store_true", help="random-init weights (init_random_for_parity)")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random (default 0)")
    parser.add_argument("--ckpt", help="training checkpoint to export (EMA weights unless --raw)")
    parser.add_argument("--raw", action="store_true", help="with --ckpt: export the raw weights, not the EMA")
    parser.add_argument("--thresholds",
                        help="calibrated thresholds.json: with --ckpt it is written into the new export; alone "
                             "(with --out DIR) it is copied into the existing export DIR; its model_hash must equal "
                             "the earshot.esw data_sha256")
    parser.add_argument("--testvec", action="store_true",
                        help="only re-create the three .est files (parity vectors from OUT/earshot.esw)")
    parser.add_argument("--out", help=f"output folder (default {DEFAULT_CKPT_OUT} with --ckpt; required otherwise)")
    parser.add_argument("--scene", choices=["handbuilt", "render"], default=None,
                        help="test-vector audio: handbuilt (reproduces random_v1; default for --random and "
                             "--testvec) or render (render.render_example in test mode, 10-ml 7.4; default for "
                             "--ckpt)")
    args = parser.parse_args(argv)
    thresholds_only = args.thresholds is not None and not (args.random or args.ckpt or args.testvec)
    if sum([args.random, args.ckpt is not None, args.testvec, thresholds_only]) != 1:
        parser.error("give exactly one of --random, --ckpt PATH, --testvec, --thresholds FILE")
    if args.thresholds is not None and (args.random or args.testvec):
        parser.error("--thresholds goes with --ckpt or alone (to copy it into an existing export)")
    if args.out is None and not args.ckpt:
        parser.error("--out is required unless --ckpt is given")
    out_dir = Path(args.out) if args.out is not None else DEFAULT_CKPT_OUT
    if args.scene is None:
        args.scene = "render" if args.ckpt else "handbuilt"

    if thresholds_only:
        obj = install_thresholds(out_dir, args.thresholds)
        print(f"thresholds.json copied into {out_dir} from {args.thresholds} (model_hash {obj['model_hash']} matches "
              f"earshot.esw)")
        if all((out_dir / name).exists() for name in FILES):
            for line in verify_folder(out_dir):
                print(line)
        return

    if args.testvec:
        # 10-ml 7.3: re-create all three .est files (the parity pair from DIR/earshot.esw, plus the self-test).
        write_wespk_selftest(out_dir / "wespk_selftest.est")
        make_testvecs(out_dir, scene=args.scene)
        print(f"testvec_short.est, testvec_long.est and wespk_selftest.est re-created in {out_dir}")
        return

    if args.random:
        model = init_random_for_parity(args.seed).eval()
        # Random weights were never trained on any data, so data_licenses is empty.
        meta = weights_meta(run=out_dir.name, step=0, ema=False, init=f"random-seed-{args.seed}", data_licenses=[])
        export_all(out_dir, model, meta, scene=args.scene)
    else:
        model, step, used_ema = load_checkpoint_model(args.ckpt, raw=args.raw)
        which = "EMA" if used_ema else ("raw (--raw)" if args.raw else "raw (the checkpoint has no EMA yet)")
        print(f"{args.ckpt}: step {step}, exporting the {which} weights to {out_dir}; test vectors from the "
              f"{args.scene} scene")
        meta = weights_meta(run=Path(args.ckpt).parent.name, step=step, ema=used_ema, init="trained",
                            data_licenses=training_licenses())
        export_all(out_dir, model, meta, thresholds_source=args.thresholds, scene=args.scene)

    # Read everything back and check it against contract 3 and 6.
    for line in verify_folder(out_dir):
        print(line)


if __name__ == "__main__":
    main()
