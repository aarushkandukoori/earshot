"""Tests for ml/earshot/stream_ref.py (10-ml 4 and 14, contract 2.1, 2.6 and 11)."""
import copy

import numpy as np
import pytest
import torch

from ml.earshot.dsp import windows
from ml.earshot.model import features
from ml.earshot.stream_ref import StreamRef, pack_cond

from .helpers import CASES, cond_inputs_to


@pytest.fixture(scope="module")
def offline_10s(net, signals, cond_all):
    """Offline forward of the 10 s mixture for all 4 cases at once: (y [4, L], head_logits [4, T, 2])."""
    _, cond = cond_all
    mix = torch.from_numpy(signals["mix"])[None].expand(len(CASES), -1).contiguous()
    with torch.no_grad():
        y, head_logits = net(mix, cond)
    return y.numpy(), head_logits


@pytest.mark.parametrize("case", CASES)
def test_streaming_identity_10s(net, signals, cond_all, offline_10s, case):
    _, cond = cond_all
    i = CASES.index(case)
    y_off, head_logits = offline_10s
    x = signals["mix"]
    assert len(x) == 160000
    y_stream, heads = StreamRef(net).run(pack_cond(cond, i), x)
    assert y_stream.shape == (160000,) and heads.shape == (2500, 2)
    # y_stream[n + 64] == y_offline[n]
    err = np.abs(y_stream[64:] - y_off[i, :-64]).max()
    assert err <= 1e-5, f"{case}: streaming identity error {err}"
    heads_off = torch.sigmoid(head_logits[i]).numpy()
    assert np.abs(heads - heads_off).max() <= 1e-5
    assert np.abs(y_off[i]).max() > 1e-3                 # the output is not trivially silent


def test_allpass_is_a_64_sample_delay(net, signals):
    x = signals["mix"][:32000]
    y, heads = StreamRef(net, allpass=True).run(None, x)
    assert np.abs(y[64:] - x[:-64]).max() <= 1e-5
    assert np.abs(y[:64]).max() <= 1e-5


def test_pack_cond_order_and_padding(cond_all):
    ci, cond = cond_all
    n_t = int(ci.val_t[0].sum())
    n_o = int(ci.val_o[0].sum())
    for case in CASES:
        i = CASES.index(case)
        pc = pack_cond(cond, i)
        assert pc.film.shape == (4096,) and pc.k.shape == (4, 68, 16) and pc.v.shape == (4, 68, 16)
        assert torch.equal(pc.film, cond.film[i].reshape(-1))
        # Which slots are valid, in order: target slots 0..n_t-1, own slots 48..48+n_o-1, null slots 64..67.
        slots = []
        if case != "everyone":
            slots += list(range(n_t))
        if case != "noown":
            slots += list(range(48, 48 + n_o))
        slots += [64, 65, 66, 67]
        assert pc.n == len(slots)
        for row, slot in enumerate(slots):
            assert torch.equal(pc.k[:, row], cond.k[i][:, slot])
            assert torch.equal(pc.v[:, row], cond.v[i][:, slot])
        assert torch.all(pc.k[:, pc.n:] == 0) and torch.all(pc.v[:, pc.n:] == 0)


def test_debug_taps(net, signals, cond_all):
    _, cond = cond_all
    x = signals["mix"][:6400]
    sref = StreamRef(net)
    pc = pack_cond(cond, 0)
    y, heads, dbg = sref.run(pc, x, debug=8)
    shapes = {"feat": (8, 771), "z_enc": (8, 16, 64), "z_blk0": (8, 16, 64), "z_blk1": (8, 16, 64),
              "z_blk2": (8, 16, 64), "z_blk3": (8, 16, 64), "coef": (8, 1026)}
    for name, shape in shapes.items():
        assert dbg[name].shape == shape and dbg[name].dtype == np.float32, name
    assert np.abs(dbg["coef"]).max() < 1.5
    # Hop 3's features equal features() of the windowed frame that ends at sample 3 * 64 + 63.
    wa, _ = windows()
    frame = torch.cat([torch.zeros(448), torch.from_numpy(x)])[3 * 64: 3 * 64 + 512]
    expected = features(torch.fft.rfft(frame * wa)).reshape(-1).numpy()
    assert np.abs(dbg["feat"][3] - expected).max() <= 1e-6
    # Debug taps do not change the output.
    y_plain, _ = sref.run(pc, x)
    assert np.array_equal(y, y_plain)


def test_mps_matches_cpu_offline(net, signals, cond_all):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this machine")
    ci, _ = cond_all
    L = 4 * 16000
    mix = torch.from_numpy(signals["mix"][:L])[None].expand(len(CASES), -1).contiguous()
    with torch.no_grad():
        cond_cpu = net.build_cond(ci)
        y_cpu, h_cpu = net(mix, cond_cpu)
        net_mps = copy.deepcopy(net).to("mps")
        cond_mps = net_mps.build_cond(cond_inputs_to(ci, "mps"))
        y_mps, h_mps = net_mps(mix.to("mps"), cond_mps)
    y_mps, h_mps = y_mps.cpu(), h_mps.cpu()
    film_err = (cond_mps.film.cpu() - cond_cpu.film).abs().max().item()
    y_err = (y_mps - y_cpu).abs().max().item()
    h_err = (h_mps - h_cpu).abs().max().item()
    assert film_err <= 1e-4, f"film MPS vs CPU {film_err}"
    assert y_err <= 1e-4, f"output MPS vs CPU {y_err}"
    assert h_err <= 1e-4, f"head logits MPS vs CPU {h_err}"


def test_streamref_leaves_the_callers_model_alone(net):
    # StreamRef works on its own CPU copy: the caller's model keeps its mode and device (a live training model on
    # MPS must not be moved to the CPU or switched to eval mode behind the caller's back).
    model = copy.deepcopy(net).train()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    sref = StreamRef(model)
    assert model.training
    assert next(model.parameters()).device.type == device
    assert not sref.model.training
    assert next(sref.model.parameters()).device.type == "cpu"
    assert sref.model is not model
