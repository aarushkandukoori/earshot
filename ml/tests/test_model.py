"""Tests for ml/earshot/model.py (10-ml 3 and 14, contract 2.2, 2.3, 2.5 and 3.2)."""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.earshot.config import COEF_LEN, OFFSETS, TAPS, WIDTHS
from ml.earshot.model import (ESB16DF, Cond, CondInputs, ZERO_INIT_NAMES, features, init_random_for_parity,
                              init_train, parameter_counts)

from .helpers import CASES


def rows(ci, i):
    """CondInputs holding only example i (as a batch of 1)."""
    return CondInputs(**{name: getattr(ci, name)[i:i + 1] for name in ci.__dataclass_fields__})


# ---------------------------------------------------------------------------------------------------------------
# Size and export layout
# ---------------------------------------------------------------------------------------------------------------
def test_parameter_counts(net):
    counts = parameter_counts(net)
    assert counts == {
        "enc": 52416, "blocks": 730624, "att_qo": 132160, "dec": 267522, "heads": 130, "film": 1250048,
        "kv": 8320, "type_emb": 192, "null_tok": 256, "nulls": 768, "tok": 132672, "total": 2575108,
    }
    # Contract 2.5: used every hop = enc + blocks + att q/o + dec + heads.
    assert counts["enc"] + counts["blocks"] + counts["att_qo"] + counts["dec"] + counts["heads"] == 1182852


def export_shapes(model):
    """Names and shapes after the 10-ml 7.2 rename: Linear weights transposed to [in, out], each LSTM direction as
    w_ih [in, 4H], w_hh [H, 4H], bias [4H] (suffix _rev for the reverse direction), everything else unchanged."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.LSTM):
            for suffix, tag in [("", ""), ("_reverse", "_rev")]:
                if hasattr(m, "weight_ih_l0" + suffix):
                    out[f"{name}.w_ih{tag}"] = tuple(getattr(m, "weight_ih_l0" + suffix).T.shape)
                    out[f"{name}.w_hh{tag}"] = tuple(getattr(m, "weight_hh_l0" + suffix).T.shape)
                    out[f"{name}.bias{tag}"] = tuple(getattr(m, "bias_ih_l0" + suffix).shape)
        elif isinstance(m, nn.Linear):
            out[f"{name}.weight"] = (m.in_features, m.out_features)
            out[f"{name}.bias"] = (m.out_features,)
        elif isinstance(m, nn.LayerNorm):
            out[f"{name}.weight"] = tuple(m.weight.shape)
            out[f"{name}.bias"] = tuple(m.bias.shape)
    for name, p in model.named_parameters():
        if name in ("att.type_emb", "att.null_tok") or name.startswith("null."):
            out[name] = tuple(p.shape)
    return out


def contract_shapes():
    """The contract 3.2 weight table, written out."""
    exp = {}
    for k in range(16):
        exp.update({f"enc.lin.{k}.weight": (3 * WIDTHS[k], 64), f"enc.lin.{k}.bias": (64,),
                    f"enc.ln.{k}.weight": (64,), f"enc.ln.{k}.bias": (64,)})
    for b in range(4):
        p = f"blocks.{b}."
        exp.update({p + "tn.weight": (64,), p + "tn.bias": (64,),
                    p + "t.w_ih": (64, 512), p + "t.w_hh": (128, 512), p + "t.bias": (512,),
                    p + "tf.weight": (128, 64), p + "tf.bias": (64,),
                    p + "bn.weight": (64,), p + "bn.bias": (64,),
                    p + "b.w_ih": (64, 256), p + "b.w_hh": (64, 256), p + "b.bias": (256,),
                    p + "b.w_ih_rev": (64, 256), p + "b.w_hh_rev": (64, 256), p + "b.bias_rev": (256,),
                    p + "bf.weight": (128, 64), p + "bf.bias": (64,)})
    exp.update({"att.q.weight": (1024, 64), "att.q.bias": (64,), "att.o.weight": (64, 1024), "att.o.bias": (1024,),
                "att.kv.weight": (64, 128), "att.kv.bias": (128,), "att.type_emb": (3, 64), "att.null_tok": (4, 64),
                "film.0.weight": (770, 256), "film.0.bias": (256,), "film.2.weight": (256, 4096),
                "film.2.bias": (4096,),
                "null.e_t": (256,), "null.g_t": (128,), "null.e_o": (256,), "null.g_o": (128,)})
    for k in range(16):
        p = f"dec.{k}."
        exp.update({p + "ln.weight": (64,), p + "ln.bias": (64,), p + "fc1.weight": (64, 128), p + "fc1.bias": (128,),
                    p + "fc2.weight": (128, COEF_LEN[k]), p + "fc2.bias": (COEF_LEN[k],)})
    exp.update({"heads.weight": (64, 2), "heads.bias": (2,),
                "tok.blk.tn.weight": (64,), "tok.blk.tn.bias": (64,),
                "tok.blk.bn.weight": (64,), "tok.blk.bn.bias": (64,)})
    for path in ["tok.blk.t", "tok.blk.b"]:
        for tag in ["", "_rev"]:
            exp.update({f"{path}.w_ih{tag}": (64, 128), f"{path}.w_hh{tag}": (32, 128), f"{path}.bias{tag}": (128,)})
    exp.update({"tok.blk.tf.weight": (64, 64), "tok.blk.tf.bias": (64,),
                "tok.blk.bf.weight": (64, 64), "tok.blk.bf.bias": (64,),
                "tok.proj.weight": (1024, 64), "tok.proj.bias": (64,),
                "tok.g.weight": (64, 128), "tok.g.bias": (128,)})
    return exp


def test_export_layout_matches_contract(net):
    got = export_shapes(net)
    expected = contract_shapes()
    assert len(expected) == 270
    assert got == expected
    assert sum(int(np.prod(shape)) for shape in got.values()) == 2570500
    # 270 export tensors come from 286 state_dict entries (each of the 16 LSTM directions has 2 bias vectors).
    assert len(net.state_dict()) == 286


# ---------------------------------------------------------------------------------------------------------------
# Shapes and layouts
# ---------------------------------------------------------------------------------------------------------------
def test_output_shapes(net, signals, cond_all):
    ci, cond = cond_all
    B = len(CASES)
    assert cond.film.shape == (B, 2, 2, 16, 64)
    assert cond.k.shape == (B, 4, 68, 16) and cond.v.shape == (B, 4, 68, 16)
    assert cond.valid.shape == (B, 68) and cond.valid.dtype == torch.bool
    mix = torch.from_numpy(signals["mix"][:6400])[None].expand(B, -1).contiguous()
    with torch.no_grad():
        y, head_logits = net(mix, cond)
    assert y.shape == (B, 6400) and y.dtype == torch.float32
    assert head_logits.shape == (B, 100, 2)
    assert torch.isfinite(y).all() and torch.isfinite(head_logits).all()


@pytest.mark.parametrize("T", [1, 2, 5])
def test_deep_filter_matches_plain_sum(net, T):
    # Contract 2.2 row 10, written as a plain loop: Y(t) = sum over tau < T_k with t - tau >= 0 of C_tau(t) X(t - tau).
    # T = 1 and 2 are fewer frames than the 3 taps of bands 0-11 (the offline forward used to crash there).
    gen = torch.Generator().manual_seed(T)
    X = torch.complex(torch.randn(1, T, 257, generator=gen), torch.randn(1, T, 257, generator=gen))
    coefs = [torch.randn(1, T, n, generator=gen) for n in COEF_LEN]
    Y = net.deep_filter(coefs, X)
    assert Y.shape == (1, T, 257)
    for k in range(16):
        C = net.coef_complex(coefs[k], k)                               # [1, T, taps, W_k]
        Xk = X[:, :, OFFSETS[k]:OFFSETS[k] + WIDTHS[k]]
        for t in range(T):
            expected = torch.zeros(WIDTHS[k], dtype=torch.complex64)
            for tau in range(TAPS[k]):
                if t - tau >= 0:
                    expected = expected + C[0, t, tau] * Xk[0, t - tau]
            assert torch.allclose(Y[0, t, OFFSETS[k]:OFFSETS[k] + WIDTHS[k]], expected, atol=1e-5), (k, t)


@pytest.mark.parametrize("L", [64, 128])
def test_one_and_two_hop_clips_match_stream(net, signals, cond_all, L):
    # The offline forward runs on 1 and 2 hops and agrees with the streaming reference: heads at every hop, and
    # y_stream[n + 64] == y_offline[n] where both exist (a 64-sample clip has no such pair).
    from ml.earshot.stream_ref import StreamRef, pack_cond
    _, cond = cond_all
    one = Cond(film=cond.film[:1], k=cond.k[:1], v=cond.v[:1], valid=cond.valid[:1])
    x = signals["mix"][:L]
    with torch.no_grad():
        y, head_logits = net(torch.from_numpy(x)[None], one)
    assert y.shape == (1, L) and head_logits.shape == (1, L // 64, 2)
    assert torch.isfinite(y).all()
    y_stream, heads = StreamRef(net).run(pack_cond(cond, 0), x)
    assert np.abs(heads - torch.sigmoid(head_logits[0]).numpy()).max() <= 1e-5
    if L > 64:
        assert np.abs(y_stream[64:] - y[0, :-64].numpy()).max() <= 1e-5


def test_feature_layout_is_bin_major(net):
    # Band k's input vector must be [bin o: f0 f1 f2, bin o+1: f0 f1 f2, ...] (index 3 * j + f).
    X = torch.randn(1, 2, 257, dtype=torch.complex64)
    f = features(X)
    assert f.shape == (1, 2, 257, 3)
    k = 13
    o, w = OFFSETS[k], WIDTHS[k]
    by_hand = torch.zeros(1, 2, 3 * w)
    for j in range(w):
        for c in range(3):
            by_hand[:, :, 3 * j + c] = f[:, :, o + j, c]
    with torch.no_grad():
        expected = net.enc.ln[k](net.enc.lin[k](by_hand))
        z = net.encode(X)
    assert torch.equal(z[:, :, k], expected)
    # Feature formulas at one bin.
    x = X[0, 0, 5]
    mag = x.abs()
    c = x * (mag + 1e-8) ** -0.7
    assert torch.allclose(f[0, 0, 5], torch.stack([c.real, c.imag, 0.1 * torch.log(mag ** 2 + 1e-8)]), atol=1e-6)


def test_encode_enrollment(net, signals):
    from ml.earshot.vad import es_vad, token_groups
    enr = signals["enr_t"]
    idx, valid, n_avail = token_groups(es_vad(enr), 48)
    wav = torch.from_numpy(enr)[None]
    with torch.no_grad():
        tok, g = net.encode_enrollment(wav, torch.from_numpy(idx)[None], torch.from_numpy(valid)[None])
    assert tok.shape == (1, 48, 64) and g.shape == (1, 128)
    n_valid = int(valid.sum())
    assert n_valid == min(n_avail, 48)
    assert torch.all(tok[0, n_valid:] == 0)                      # invalid rows are exactly zero
    assert abs(g.norm().item() - 1.0) < 1e-5
    # No valid rows at all (Everyone target, no own voice): g must still be finite, never 0 / 0.
    none_valid = torch.zeros(1, 48, dtype=torch.bool)
    with torch.no_grad():
        tok0, g0 = net.encode_enrollment(wav, torch.from_numpy(idx)[None], none_valid)
    assert torch.all(tok0 == 0) and torch.isfinite(g0).all()


def test_embed_g(net, signals):
    g = net.embed_g(signals["enr_t"])
    assert g.shape == (128,) and abs(g.norm().item() - 1.0) < 1e-5
    with pytest.raises(ValueError, match="ES_ERR_ENROLL_SHORT"):
        net.embed_g(np.zeros(96000, dtype=np.float32))          # silence: no speech frames
    with pytest.raises(ValueError, match="ES_ERR_ENROLL_SHORT"):
        net.embed_g(signals["enr_t"][:575])                     # too short: F = 0


# ---------------------------------------------------------------------------------------------------------------
# Null rules (contract 2.3)
# ---------------------------------------------------------------------------------------------------------------
def test_null_rules_change_only_intended_inputs(net, cond_all):
    ci, _ = cond_all
    with torch.no_grad():
        u = net.film_input(ci)                   # rows: person, everyone, noown, nowespk
        tokens, valid = net.token_list(ci)
    n = net.null
    E_T, G_T, E_O, G_O, MODE = slice(0, 256), slice(256, 384), slice(384, 640), slice(640, 768), slice(768, 770)
    person, everyone, noown, nowespk = 0, 1, 2, 3

    # Person: the given vectors, untouched.
    assert torch.equal(u[person, E_T], ci.e_t[0]) and torch.equal(u[person, G_T], ci.g_t[0])
    assert torch.equal(u[person, E_O], ci.e_o[0]) and torch.equal(u[person, G_O], ci.g_o[0])
    assert torch.equal(u[person, MODE], torch.tensor([1.0, 0.0]))
    # Everyone: target vectors become nulls, own vectors unchanged, mode [0, 1].
    assert torch.equal(u[everyone, E_T], n.e_t) and torch.equal(u[everyone, G_T], n.g_t)
    assert torch.equal(u[everyone, 384:768], u[person, 384:768])
    assert torch.equal(u[everyone, MODE], torch.tensor([0.0, 1.0]))
    # No own voice: own vectors become nulls, target unchanged.
    assert torch.equal(u[noown, 0:384], u[person, 0:384])
    assert torch.equal(u[noown, E_O], n.e_o) and torch.equal(u[noown, G_O], n.g_o)
    # No WeSpeaker: both e vectors become nulls, both g vectors unchanged.
    assert torch.equal(u[nowespk, E_T], n.e_t) and torch.equal(u[nowespk, E_O], n.e_o)
    assert torch.equal(u[nowespk, G_T], u[person, G_T]) and torch.equal(u[nowespk, G_O], u[person, G_O])

    # Token list: the tokens themselves never depend on the flags; only validity does.
    for i in range(4):
        assert torch.equal(tokens[i], tokens[person])
    assert torch.all(valid[:, 64:])                                       # null tokens always valid
    assert not valid[everyone, :48].any() and torch.equal(valid[everyone, 48:], valid[person, 48:])
    assert not valid[noown, 48:64].any() and torch.equal(valid[noown, :48], valid[person, :48])
    assert torch.equal(valid[nowespk], valid[person])
    assert valid[person, :48].sum() == ci.val_t[0].sum() and valid[person, 48:64].sum() == ci.val_o[0].sum()


@pytest.mark.parametrize("case,used", [
    ("person", []),
    ("everyone", ["e_t", "g_t"]),
    ("noown", ["e_o", "g_o"]),
    ("nowespk", ["e_t", "e_o"]),
])
def test_gradients_reach_used_nulls_only(net, cond_all, case, used):
    model = copy.deepcopy(net)
    ci, _ = cond_all
    one = rows(ci, CASES.index(case))
    cond = model.build_cond(one)
    loss = (cond.film ** 2).sum() + (cond.k ** 2).sum() + (cond.v ** 2).sum()
    loss.backward()
    for name in ["e_t", "g_t", "e_o", "g_o"]:
        grad = getattr(model.null, name).grad
        if name in used:
            assert grad is not None and grad.abs().max() > 0, f"null.{name} should get a gradient in {case}"
        else:
            assert grad is None or torch.all(grad == 0), f"null.{name} must get no gradient in {case}"
    # Type embeddings and null tokens are always part of the token list.
    assert model.att.type_emb.grad.abs().max() > 0 and model.att.null_tok.grad.abs().max() > 0


# ---------------------------------------------------------------------------------------------------------------
# Attention masking
# ---------------------------------------------------------------------------------------------------------------
def test_attention_ignores_invalid_tokens(net, signals, cond_all):
    _, cond = cond_all
    i = CASES.index("noown")                     # own slots and unused target slots are invalid
    mix = torch.from_numpy(signals["mix"][:16000])[None]
    one = Cond(film=cond.film[i:i + 1], k=cond.k[i:i + 1], v=cond.v[i:i + 1], valid=cond.valid[i:i + 1])
    invalid = ~one.valid[0]
    assert invalid.sum() > 0
    gen = torch.Generator().manual_seed(0)
    k2, v2 = one.k.clone(), one.v.clone()
    k2[0][:, invalid] = 100.0 * torch.randn(k2[0][:, invalid].shape, generator=gen)
    v2[0][:, invalid] = 100.0 * torch.randn(v2[0][:, invalid].shape, generator=gen)
    with torch.no_grad():
        y_ref, h_ref = net(mix, one)
        y_new, h_new = net(mix, Cond(film=one.film, k=k2, v=v2, valid=one.valid))
    assert (y_new - y_ref).abs().max().item() == 0.0
    assert (h_new - h_ref).abs().max().item() == 0.0
    # Sanity: changing a valid token (null token slot 64) does change the output.
    v3 = one.v.clone()
    v3[0][:, 64] += 1.0
    with torch.no_grad():
        y3, _ = net(mix, Cond(film=one.film, k=one.k, v=v3, valid=one.valid))
    assert (y3 - y_ref).abs().max().item() > 1e-4


# ---------------------------------------------------------------------------------------------------------------
# Initialization (10-ml 3.4)
# ---------------------------------------------------------------------------------------------------------------
def lstm_directions(model):
    """(name, H, bias_ih, bias_hh) for every LSTM direction in the model."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.LSTM):
            for suffix in ["", "_reverse"]:
                if hasattr(m, "bias_ih_l0" + suffix):
                    out.append((name + suffix, m.hidden_size, getattr(m, "bias_ih_l0" + suffix),
                                getattr(m, "bias_hh_l0" + suffix)))
    return out


def test_forget_gate_bias_sums_to_one(net):
    trained_init = init_train(ESB16DF())
    for model in [net, trained_init]:
        directions = lstm_directions(model)
        assert len(directions) == 16
        for name, H, b_ih, b_hh in directions:
            total = (b_ih + b_hh).detach()
            assert torch.all(total[H:2 * H] == 1.0), name


def test_init_train_zeros():
    model = init_train(ESB16DF())
    params = dict(model.named_parameters())
    for name in ZERO_INIT_NAMES:
        assert torch.all(params[name] == 0), name


def test_init_random_for_parity_fills_and_is_deterministic(net):
    params = dict(net.named_parameters())
    for name in ZERO_INIT_NAMES:
        std = params[name].detach().std().item()
        assert 0.06 < std < 0.14, f"{name} std {std}"
    again = init_random_for_parity(0)
    for (name_a, a), (name_b, b) in zip(net.state_dict().items(), again.state_dict().items()):
        assert name_a == name_b and torch.equal(a, b), name_a
    other = init_random_for_parity(1)
    assert not torch.equal(other.state_dict()["att.o.weight"], net.state_dict()["att.o.weight"])
