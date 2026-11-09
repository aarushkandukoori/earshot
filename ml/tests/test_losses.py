"""Tests for the training losses (ml/earshot/losses.py, contract 8.2, 10-ml 9 and the test_losses row of 10-ml 14).

Part 1 checks every term against a value worked out by hand (or by a plain numpy loop) on tiny tensors.
Part 2 checks how total_loss switches terms on and off per example, the weights of the 10-ml 9 table (mrstft at the
spec weight 1, DECISIONS L 2026-11-08 "Loss A/B outcome"), and the weight-0 path the A/B harness's tsnr_only variant
uses: with the weight set to 0 in the test, mrstft is logged but sends no gradient.
Part 3 runs the real ESB16DF on a batch with one Everyone, one no-own and one absent example and checks that the
loss and every gradient are finite. Part 4 compares MPS with CPU (skipped without MPS).
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.earshot import dsp, losses
from ml.earshot.losses import AAMHead, total_loss
from ml.earshot.model import CondInputs, ESB16DF, init_random_for_parity, init_train
from ml.earshot.vad import es_vad, token_groups

from .helpers import noise, synthetic_speech, unit_vector

HAS_MPS = torch.backends.mps.is_available()


def randn(shape, seed, scale=1.0):
    """Seeded float32 Gaussian tensor."""
    rng = np.random.default_rng([seed, 5])
    return torch.from_numpy((scale * rng.standard_normal(shape)).astype(np.float32))


# ---------------------------------------------------------------------------------------------------------------
# Part 1 - each term by hand
# ---------------------------------------------------------------------------------------------------------------
def test_tsnr_perfect_output_hits_the_cap():
    s = randn((2, 640), 1)
    value = losses.tsnr(s.clone(), s)
    S = (s.double() ** 2).sum(-1)
    expected = -10 * torch.log10(S / (1e-3 * S + 1e-8) + 1e-8)
    assert torch.allclose(value.double(), expected, atol=1e-4)
    assert torch.allclose(value, torch.full((2,), -30.0), atol=1e-3)     # the -30 dB cap


def test_tsnr_known_error():
    # y = 0.9 s: |s - y|^2 = 0.01 |s|^2, so the value is -10 log10(1 / (0.01 + 0.001)) = 10 log10(0.011) dB.
    s = randn((1, 640), 2)
    value = losses.tsnr(0.9 * s, s)
    assert abs(value.item() - 10 * math.log10(0.011)) < 1e-3              # -19.586 dB


def test_tsnr_of_silent_reference_is_finite_with_zero_gradient():
    y = randn((1, 640), 3).requires_grad_(True)
    value = losses.tsnr(y, torch.zeros(1, 640))
    assert abs(value.item() - 80.0) < 1e-4                                 # -10 log10(1e-8)
    value.sum().backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().max() == 0


def test_absent_by_hand():
    x = randn((2, 640), 4)
    X = (x.double() ** 2).sum(-1)
    zero_out = losses.absent(torch.zeros_like(x), x)
    assert torch.allclose(zero_out.double(), 10 * torch.log10(10 ** -2.5 * X + 1e-8), atol=1e-4)
    half_out = losses.absent(0.5 * x, x)
    assert torch.allclose(half_out.double(), 10 * torch.log10((0.25 + 10 ** -2.5) * X + 1e-8), atol=1e-4)


def test_masked_absent_uses_masked_means():
    # 6 hops of constant level: y hop j = a[j], x hop j = b[j]; mask = hops 1, 2 and 4.
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    b = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    y = torch.tensor([v for v in a for _ in range(64)])[None]
    x = torch.tensor([v for v in b for _ in range(64)])[None]
    mask = torch.tensor([[False, True, True, False, True, False]])
    value, count = losses.masked_absent(y, x, mask)
    mean_y = (0.2 ** 2 + 0.3 ** 2 + 0.5 ** 2) / 3          # each masked hop has 64 equal samples
    mean_x = (2.0 ** 2 + 1.0 ** 2 + 1.0 ** 2) / 3
    assert count.item() == 3 * 64
    assert abs(value.item() - 10 * math.log10(mean_y + 10 ** -2.5 * mean_x + 1e-8)) < 1e-4
    # An empty mask divides by max(count, 1): finite (log10 of eps), not 0 / 0.
    value, count = losses.masked_absent(y, x, torch.zeros_like(mask))
    assert count.item() == 0 and abs(value.item() - (-80.0)) < 1e-3


def stft_mag_reference(x, n_fft, hop):
    """Plain numpy loop version of losses.stft_mag for one signal (float64)."""
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad), np.asarray(x, np.float64), np.zeros(pad)])
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = []
    start = 0
    while start + n_fft <= len(xp):
        frames.append(np.abs(np.fft.rfft(xp[start:start + n_fft] * window)))
        start += hop
    return np.array(frames)


def test_mrstft_matches_numpy_and_is_zero_for_identical_signals():
    y = randn((2, 2048), 5, 0.1)
    s = randn((2, 2048), 6, 0.1)
    assert torch.all(losses.mrstft(s, s) == 0)
    value = losses.mrstft(y, s)
    for i in range(2):
        expected = 0.0
        for n_fft in (256, 512, 1024):
            A = stft_mag_reference(y[i].numpy(), n_fft, n_fft // 4)
            R = stft_mag_reference(s[i].numpy(), n_fft, n_fft // 4)
            expected += np.mean(np.abs(A - R)) + np.mean(np.abs(np.log(A + 1e-5) - np.log(R + 1e-5)))
        assert abs(value[i].item() - expected) < 1e-4 * expected


def test_oversup_by_hand():
    s = randn((2, 2048), 7, 0.1)
    assert torch.all(losses.oversup(s, s) == 0)                 # identical signals
    assert torch.all(losses.oversup(2.0 * s, s) == 0)           # a louder output is never over-suppression
    value = losses.oversup(0.5 * s, s)
    for i in range(2):
        S = stft_mag_reference(s[i].numpy(), 512, 128)
        short = np.maximum((S + 1e-8) ** 0.3 - (0.5 * S + 1e-8) ** 0.3, 0.0)
        assert abs(value[i].item() - np.mean(short ** 2)) < 1e-4 * np.mean(short ** 2)
        # Away from |S| = 0 this is (1 - 0.5^0.3)^2 mean(|S|^0.6).
        assert abs(value[i].item() - (1 - 0.5 ** 0.3) ** 2 * np.mean(S ** 0.6)) < 1e-3 * value[i].item()


def test_heads_bce_by_hand():
    logits = torch.tensor([[[2.0, -1.0], [0.0, 0.0]]])          # B = 1, T = 2 hops
    vad_t = torch.tensor([[1.0, 0.0]])
    vad_o = torch.tensor([[0.0, 1.0]])
    bce_t, bce_o = losses.heads_bce(logits, vad_t, vad_o)

    def sig(z):
        return 1 / (1 + math.exp(-z))

    expected_t = (-math.log(sig(2.0)) - math.log(1 - sig(0.0))) / 2
    expected_o = (-math.log(1 - sig(-1.0)) - math.log(sig(0.0))) / 2
    assert abs(bce_t.item() - expected_t) < 1e-6 and abs(bce_o.item() - expected_o) < 1e-6


def test_aam_true_class_logit_is_scaled_cos_theta_plus_margin():
    head = AAMHead(n_classes=5, dim=4)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 2] = 1.0                                  # class 0 is orthogonal to g: cos = 0
        head.weight[2, 0] = math.cos(1.0)                        # class 2 at angle 1.0 rad from g
        head.weight[2, 1] = math.sin(1.0)
        head.weight[3, 0] = 2 * math.cos(3.0)                    # class 3 at angle 3.0 rad (row length 2)
        head.weight[3, 1] = 2 * math.sin(3.0)
        head.weight[1, 0] = head.weight[4, 3] = 1.0
    g = torch.tensor([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])   # not unit length: the head normalizes
    logits = head.logits(g, torch.tensor([2, 3]))
    assert abs(logits[0, 2].item() - 30 * math.cos(1.0 + 0.2)) < 1e-4
    assert abs(logits[0, 3].item() - 30 * math.cos(3.0)) < 1e-4     # not the label in row 0: plain 30 cos
    assert abs(logits[0, 0].item()) < 1e-6 and abs(logits[0, 1].item() - 30.0) < 1e-4
    # Row 1: theta = 3.0 > pi - 0.2, so the true class uses cos - sin(pi - m) m (standard ArcFace).
    expected = 30 * (math.cos(3.0) - math.sin(math.pi - 0.2) * 0.2)
    assert abs(logits[1, 3].item() - expected) < 1e-4
    ce = head(g, torch.tensor([2, 3]))
    assert torch.allclose(ce, F.cross_entropy(logits, torch.tensor([2, 3]), reduction="none"))


def test_aam_head_size_and_empty_batch():
    head = AAMHead()
    assert tuple(head.weight.shape) == (3758, 128) and head.weight.numel() == 481024
    out = head(torch.zeros(0, 128), torch.zeros(0, dtype=torch.int64))
    assert out.shape == (0,) and out.sum().item() == 0.0


# ---------------------------------------------------------------------------------------------------------------
# Part 2 - total_loss switches terms per example (section 9 "Applies when")
# ---------------------------------------------------------------------------------------------------------------
HOPS = 50                       # tiny batches: 50 hops = 3,200 samples


def hop_mask(on_hops):
    """float 0/1 [HOPS] with the listed hops set to 1."""
    m = np.zeros(HOPS, dtype=np.float32)
    m[list(on_hops)] = 1.0
    return m


def tiny_batch(seed=0):
    """Six hand-made rows covering every switch (only the batch keys total_loss reads, numpy as from the loader).

    row 0 Person present, 30 target-silent hops, own voice alone in 30 of them  -> gap on, own on
    row 1 Everyone with talkers                                                  -> no AAM (spk_t = -1)
    row 2 Person, target absent, own voice in the mix                            -> absent instead of tsnr
    row 3 Person present, 24 silent hops (1,536 samples, under 0.1 s), own alone in those 24 -> gap off, own off
    row 4 Person present, exactly 25 silent hops (1,600 samples)                 -> gap on
    row 5 Everyone with nobody talking (ref = 0)                                 -> absent, no AAM
    """
    B, L = 6, HOPS * 64
    rng = np.random.default_rng([seed, 9])
    mix = (0.1 * rng.standard_normal((B, L))).astype(np.float32)
    ref = (0.05 * rng.standard_normal((B, L))).astype(np.float32)
    ref[2] = 0.0
    ref[5] = 0.0
    vad_t = np.stack([hop_mask(range(30, 50)), hop_mask(range(50)), hop_mask([]), hop_mask(range(24, 50)),
                      hop_mask(range(25, 50)), hop_mask([])])
    vad_o = np.stack([hop_mask(range(0, 30)), hop_mask([]), hop_mask(range(10, 40)), hop_mask(range(0, 24)),
                      hop_mask([]), hop_mask([])])
    return {
        "mix": mix, "ref": ref, "vad_t": vad_t, "vad_o": vad_o,
        "present": np.array([True, True, False, True, True, False]),
        "own_in_mix": np.array([True, False, True, True, False, False]),
        "person": np.array([True, False, True, True, True, False]),
        "spk_t": np.array([7, -1, 3757, 0, 12, -1], dtype=np.int64),
    }


def model_outputs(seed=1, B=6, requires_grad=False):
    """Random stand-ins for the model outputs: y [B, L], head_logits [B, HOPS, 2], g_t [B, 128]."""
    y = randn((B, HOPS * 64), seed, 0.05).requires_grad_(requires_grad)
    head_logits = randn((B, HOPS, 2), seed + 1).requires_grad_(requires_grad)
    g_t = F.normalize(randn((B, 128), seed + 2), dim=-1).requires_grad_(requires_grad)
    return y, head_logits, g_t


def expected_total(y, head_logits, g_t, batch, aam, oversup_on):
    """The section 9 total written with plain Python ifs, one example at a time (independent of total_loss)."""
    B = y.shape[0]
    total = 0.0
    for i in range(B):
        yi, xi, si = y[i:i + 1], torch.from_numpy(batch["mix"][i:i + 1]), torch.from_numpy(batch["ref"][i:i + 1])
        vt, vo = torch.from_numpy(batch["vad_t"][i:i + 1]), torch.from_numpy(batch["vad_o"][i:i + 1])
        row = 0.0
        if batch["present"][i]:
            row += losses.tsnr(yi, si).item() + losses.mrstft(yi, si).item()
            if oversup_on:
                row += 0.3 * losses.oversup(yi, si).item()
            silent = vt < 0.5
            if silent.sum().item() * 64 >= 1600:
                row += 0.1 * losses.masked_absent(yi, xi, silent)[0].item()
        else:
            row += losses.absent(yi, xi).item()
        own_only = (vo > 0.5) & (vt < 0.5)
        if batch["own_in_mix"][i] and own_only.sum().item() * 64 >= 1600:
            row += 0.3 * losses.masked_absent(yi, xi, own_only)[0].item()
        bce_t, bce_o = losses.heads_bce(head_logits[i:i + 1], vt, vo)
        row += 0.1 * bce_t.item() + 0.1 * bce_o.item()
        if batch["person"][i]:
            row += 0.05 * aam(g_t[i:i + 1], torch.tensor([int(batch["spk_t"][i])])).item()
        total += row
    return total / B


@pytest.mark.parametrize("oversup_on", [True, False])
def test_total_matches_the_per_example_rules(oversup_on):
    torch.manual_seed(0)
    aam = AAMHead()
    batch = tiny_batch()
    y, head_logits, g_t = model_outputs()
    with torch.no_grad():
        total, parts = total_loss(y, head_logits, g_t, batch, aam, oversup_on)
        expected = expected_total(y, head_logits, g_t, batch, aam, oversup_on)
    assert abs(total.item() - expected) < 1e-4 * max(1.0, abs(expected))
    assert set(parts) == set(losses.TERMS)
    assert all(not p.requires_grad and p.dim() == 0 for p in parts.values())
    with torch.no_grad():
        # parts = unweighted mean over the rows where the term applies.
        tsnr_rows = losses.tsnr(y, torch.from_numpy(batch["ref"]))[[0, 1, 3, 4]]
        absent_rows = losses.absent(y, torch.from_numpy(batch["mix"]))[[2, 5]]
        gap_rows = losses.masked_absent(y, torch.from_numpy(batch["mix"]),
                                        torch.from_numpy(batch["vad_t"]) < 0.5)[0][[0, 4]]
    assert abs(parts["tsnr"].item() - tsnr_rows.mean().item()) < 1e-4
    assert abs(parts["absent"].item() - absent_rows.mean().item()) < 1e-4
    assert abs(parts["gap"].item() - gap_rows.mean().item()) < 1e-4      # row 3 (24 hops) is not in the mean
    if not oversup_on:
        assert parts["oversup"].item() == 0.0
    # The optional rows dict: how many examples each term applied to (train.py weights its log means by it).
    rows = {}
    with torch.no_grad():
        total_again, _ = total_loss(y, head_logits, g_t, batch, aam, oversup_on, rows)
    assert total_again.item() == total.item() and set(rows) == set(losses.TERMS)
    assert [int(rows[n]) for n in ("tsnr", "absent", "gap", "bce_t")] == [4, 2, 2, len(batch["mix"])]
    assert int(rows["oversup"]) == (4 if oversup_on else 0)
    assert int(rows["aam"]) == int(np.sum(batch["person"]))


def test_weights_match_the_spec_table():
    # 10-ml 9 / contract 8.2. mrstft keeps the spec weight 1 (DECISIONS L 2026-11-08, "Loss A/B outcome"); the order
    # is the train.csv order.
    assert losses.WEIGHTS == {"tsnr": 1.0, "mrstft": 1.0, "absent": 1.0, "gap": 0.1, "own": 0.3, "oversup": 0.3,
                              "bce_t": 0.1, "bce_o": 0.1, "aam": 0.05}
    assert losses.TERMS == ("tsnr", "mrstft", "absent", "gap", "own", "oversup", "bce_t", "bce_o", "aam")


def loss_and_grads(batch, aam):
    """total_loss on tiny_batch-style inputs with gradients -> (total, parts, rows, grads of y / head_logits / g_t)."""
    y, head_logits, g_t = model_outputs(requires_grad=True)
    rows = {}
    total, parts = total_loss(y, head_logits, g_t, batch, aam, True, rows)
    total.backward()
    return total, parts, rows, [t.grad.clone() for t in (y, head_logits, g_t)]


def test_mrstft_weight_zero_is_logged_but_sends_no_gradient(monkeypatch):
    # The spec weight is 1; weight 0 is the A/B harness's tsnr_only variant. Set it here only (monkeypatch undoes it).
    monkeypatch.setitem(losses.WEIGHTS, "mrstft", 0.0)
    torch.manual_seed(0)
    aam = AAMHead()
    batch = tiny_batch()
    real_mrstft = losses.mrstft
    total, parts, rows, grads = loss_and_grads(batch, aam)
    # The logged value: the plain mrstft mean over the 4 present rows, finite and not 0, with 4 rows counted.
    y, _, _ = model_outputs()
    with torch.no_grad():
        expected = real_mrstft(y, torch.from_numpy(batch["ref"]))[[0, 1, 3, 4]].mean().item()
    assert math.isfinite(parts["mrstft"].item()) and parts["mrstft"].item() > 0
    assert parts["mrstft"].item() == pytest.approx(expected, rel=1e-6) and int(rows["mrstft"]) == 4
    # Swap mrstft for a function 1,000 times larger: the logged column follows it, while the total and every
    # gradient stay bit-identical, and the function runs with autograd off (no graph is built for it).
    grad_on = []

    def big_mrstft(y_, s_):
        grad_on.append(torch.is_grad_enabled())
        return 1000.0 * real_mrstft(y_, s_)

    monkeypatch.setattr(losses, "mrstft", big_mrstft)
    total_big, parts_big, _, grads_big = loss_and_grads(batch, aam)
    assert grad_on == [False]
    assert parts_big["mrstft"].item() == pytest.approx(1000.0 * parts["mrstft"].item(), rel=1e-6)
    assert torch.equal(total_big, total) and all(torch.equal(a, b) for a, b in zip(grads_big, grads))
    # Same against a total with no spectral term at all (mrstft returning 0): again bit-identical.
    monkeypatch.setattr(losses, "mrstft", lambda y_, s_: torch.zeros(y_.shape[0]))
    total_none, parts_none, _, grads_none = loss_and_grads(batch, aam)
    assert parts_none["mrstft"].item() == 0.0
    assert torch.equal(total_none, total) and all(torch.equal(a, b) for a, b in zip(grads_none, grads))
    for name in losses.TERMS:
        if name != "mrstft":
            assert torch.equal(parts_none[name], parts[name]), name


def test_mrstft_with_a_nonzero_weight_is_trained_on(monkeypatch):
    # The spec weight 1 (the default) keeps the term in the graph; against weight 0 (set here only) it changes the
    # gradient of y.
    assert losses.WEIGHTS["mrstft"] == 1.0
    torch.manual_seed(0)
    aam = AAMHead()
    batch = tiny_batch()
    real_mrstft = losses.mrstft
    grad_on = []

    def watched(y_, s_):
        grad_on.append(torch.is_grad_enabled())
        return real_mrstft(y_, s_)

    monkeypatch.setattr(losses, "mrstft", watched)
    total_one, parts_one, _, grads_one = loss_and_grads(batch, aam)
    monkeypatch.setitem(losses.WEIGHTS, "mrstft", 0.0)
    _, _, _, grads_zero = loss_and_grads(batch, aam)
    assert grad_on == [True, False] and torch.isfinite(total_one)
    assert all(torch.isfinite(g).all() for g in grads_one)
    assert not torch.equal(grads_one[0], grads_zero[0])         # y gets the spectral gradient
    assert torch.equal(grads_one[1], grads_zero[1])             # the head gradient does not depend on mrstft


def test_gap_and_own_need_one_tenth_of_a_second():
    aam = AAMHead()
    for n_silent, should_apply in [(24, False), (25, True)]:
        batch = {
            "mix": randn((1, HOPS * 64), 3).numpy(), "ref": randn((1, HOPS * 64), 4).numpy(),
            "vad_t": hop_mask(range(n_silent, HOPS))[None], "vad_o": hop_mask(range(n_silent))[None],
            "present": np.array([True]), "own_in_mix": np.array([True]), "person": np.array([True]),
            "spk_t": np.array([0], dtype=np.int64),
        }
        y, head_logits, g_t = model_outputs(B=1)
        with torch.no_grad():
            _, parts = total_loss(y, head_logits, g_t, batch, aam, True)
        assert (parts["gap"].item() != 0.0) == should_apply
        assert (parts["own"].item() != 0.0) == should_apply
    # own is off when the own talker is not in the mix, whatever vad_o says.
    batch["own_in_mix"] = np.array([False])
    with torch.no_grad():
        _, parts = total_loss(y, head_logits, g_t, batch, aam, True)
    assert parts["own"].item() == 0.0 and parts["gap"].item() != 0.0


def test_aam_ignores_everyone_rows():
    aam = AAMHead()
    batch = tiny_batch()
    y, head_logits, g_t = model_outputs(requires_grad=True)
    total, parts = total_loss(y, head_logits, g_t, batch, aam, True)
    total.backward()
    # Rows 1 and 5 are Everyone (spk_t = -1): their g_t gets no gradient, Person rows do.
    assert torch.all(g_t.grad[[1, 5]] == 0)
    assert torch.all(g_t.grad[[0, 2, 3, 4]].abs().sum(-1) > 0)
    # Changing an Everyone row's g_t changes nothing.
    g_other = g_t.detach().clone()
    g_other[1] = F.normalize(randn((128,), 99), dim=-1)
    with torch.no_grad():
        total_other, _ = total_loss(y, head_logits, g_other, batch, aam, True)
    assert total_other.item() == pytest.approx(total.item(), abs=1e-6)
    # aam part = mean of the 4 Person rows' cross-entropy.
    with torch.no_grad():
        ce = aam(g_t[[0, 2, 3, 4]], torch.tensor([7, 3757, 0, 12]))
    assert abs(parts["aam"].item() - ce.mean().item()) < 1e-4


def test_batch_without_person_rows_has_zero_aam():
    aam = AAMHead()
    batch = tiny_batch()
    batch["person"][:] = False
    batch["spk_t"][:] = -1
    y, head_logits, g_t = model_outputs(requires_grad=True)
    total, parts = total_loss(y, head_logits, g_t, batch, aam, True)
    total.backward()
    assert torch.isfinite(total) and parts["aam"].item() == 0.0
    assert torch.all(g_t.grad == 0) and torch.all(aam.weight.grad == 0)


# ---------------------------------------------------------------------------------------------------------------
# Part 3 - the real ESB16DF on a mixed batch: finite loss and finite gradients for every parameter
# ---------------------------------------------------------------------------------------------------------------
L_REAL = 16000                  # 1 s segments (250 hops)
LE_T = 48000                    # 3 s target enrollment (the shortest the recipe draws)
LE_O = 128000                   # 8 s own enrollment (10-ml 8.7)


def groups(clip, max_tokens):
    """Token groups of a clip with the es_vad mask, as the loader builds them."""
    idx, valid, n_avail = token_groups(es_vad(clip), max_tokens)
    assert n_avail >= 4
    return idx, valid


def real_batch():
    """A hand-built batch with every key of 10-ml 8.7, as numpy, B = 4:

    row 0 Person, target present, own voice enrolled and in the mix
    row 1 Everyone (both talkers are the reference), own voice enrolled and in the mix
    row 2 Person, target present, no own voice enrolled (zero arrays, has_o False)
    row 3 Person, target absent (ref 0), own voice in the mix, WeSpeaker path dropped
    The target stops after hop 150 so the gap term applies; the own voice talks in hops 160..239 so the own term does.
    """
    B = 4
    target = synthetic_speech(L_REAL, 0)
    target[150 * 64:] = 0.0
    other = 0.7 * synthetic_speech(L_REAL, 1)
    own = np.zeros(L_REAL, dtype=np.float32)
    own[160 * 64:240 * 64] = 1.5 * synthetic_speech(80 * 64, 2)
    n = noise(L_REAL, seed=5, level=0.005)

    refs = [target, target + other, target, np.zeros(L_REAL, np.float32)]
    own_in_mix = np.array([True, True, False, True])
    mix = np.zeros((B, L_REAL), np.float32)
    for i in range(B):
        talkers = other if i != 1 else 0.0                     # Everyone: other is already in the reference
        mix[i] = refs[i] + talkers + n + (own if own_in_mix[i] else 0.0)
        if i == 3:
            mix[i] = other + n + own                           # absent: the target is not in the mix
    ref = np.stack(refs).astype(np.float32)
    vad_t = np.stack([dsp.hop_vad(r) for r in ref]).astype(np.float32)
    vad_o = np.stack([dsp.hop_vad(own) if own_in_mix[i] else np.zeros(L_REAL // 64, np.uint8)
                      for i in range(B)]).astype(np.float32)

    has_t = np.array([True, False, True, True])
    has_o = np.array([True, True, False, True])
    enr_t = np.zeros((B, LE_T), np.float32)
    idx_t = np.zeros((B, 48, 3), np.int64)
    val_t = np.zeros((B, 48), bool)
    enr_o = np.zeros((B, LE_O), np.float32)
    idx_o = np.zeros((B, 16, 3), np.int64)
    val_o = np.zeros((B, 16), bool)
    e_o = np.zeros((B, 256), np.float32)
    own_clip = synthetic_speech(LE_O, 9)
    for i in range(B):
        if has_t[i]:
            enr_t[i] = synthetic_speech(LE_T, 20 + i) + noise(LE_T, seed=30 + i, level=0.003)
            idx_t[i], val_t[i] = groups(enr_t[i], 48)
        if has_o[i]:
            enr_o[i] = own_clip
            idx_o[i], val_o[i] = groups(own_clip, 16)
            e_o[i] = unit_vector(256, 2).numpy()
    return {
        "mix": mix, "ref": ref, "vad_t": vad_t, "vad_o": vad_o,
        "present": np.array([True, True, True, False]), "own_in_mix": own_in_mix,
        "person": np.array([True, False, True, True]),
        "enr_t": enr_t, "tok_idx_t": idx_t, "tok_val_t": val_t,
        "enr_o": enr_o, "tok_idx_o": idx_o, "tok_val_o": val_o, "e_o": e_o,
        "has_t": has_t, "has_o": has_o, "use_wes": np.array([True, True, True, False]),
        "mode": np.array([[1, 0], [0, 1], [1, 0], [1, 0]], np.float32),
        "spk_t": np.array([11, -1, 2024, 3757], np.int64),
        "info": [{} for _ in range(B)],
    }


def run_step(model, aam, batch, device):
    """The forward part of 10-ml 10.2 step_batch (WeSpeaker replaced by fixed unit vectors) -> (loss, parts)."""
    b = {k: torch.as_tensor(v).to(device) for k, v in batch.items() if k != "info"}
    e_t = torch.stack([unit_vector(256, 10 + i) for i in range(b["mix"].shape[0])]).to(device)
    tok_t, g_t = model.encode_enrollment(b["enr_t"], b["tok_idx_t"], b["tok_val_t"])
    tok_o, g_o = model.encode_enrollment(b["enr_o"], b["tok_idx_o"], b["tok_val_o"])
    cond = model.build_cond(CondInputs(e_t, tok_t, b["tok_val_t"], g_t, b["e_o"], tok_o, b["tok_val_o"], g_o,
                                       b["has_t"], b["has_o"], b["use_wes"], b["mode"]))
    y, head_logits = model(b["mix"], cond)
    return total_loss(y, head_logits, g_t, batch, aam, oversup_on=True)


def check_finite_gradients(model, aam, loss, parts, require_nonzero):
    assert torch.isfinite(loss), parts
    for name, p in list(model.named_parameters()) + [("aam.weight", aam.weight)]:
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
    if require_nonzero:
        for prefix in ("enc.", "tok.", "blocks.", "att.", "film.", "null.", "dec.", "heads."):
            grads = [p.grad.abs().max().item() for n, p in model.named_parameters() if n.startswith(prefix)]
            assert max(grads) > 0, f"no gradient reaches {prefix}*"


@pytest.mark.parametrize("init", ["train", "parity"])
def test_mixed_batch_gives_finite_loss_and_gradients(init):
    """One Everyone, one no-own and one absent example (plus a normal one): every term that applies is used, and
    the loss and the gradient of every model parameter (tok.* and enc.* included) and of the AAM head are finite.
    "train" is the real starting point (zero-initialized attention output); "parity" fills those tensors, so
    every path also gets a non-zero gradient."""
    torch.manual_seed(0)
    model = init_train(ESB16DF()) if init == "train" else init_random_for_parity(0)
    model.train()
    aam = AAMHead()
    batch = real_batch()
    loss, parts = run_step(model, aam, batch, "cpu")
    loss.backward()
    check_finite_gradients(model, aam, loss, parts, require_nonzero=(init == "parity"))
    for name in ("tsnr", "mrstft", "absent", "gap", "own", "oversup", "bce_t", "bce_o", "aam"):
        assert parts[name].item() != 0.0, f"{name} should apply to some row of this batch"
        assert math.isfinite(parts[name].item()), name


# ---------------------------------------------------------------------------------------------------------------
# Part 4 - MPS agrees with CPU (training runs on MPS)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_MPS, reason="no MPS device")
def test_loss_mps_matches_cpu():
    torch.manual_seed(0)
    aam_cpu = AAMHead()
    aam_mps = AAMHead().to("mps")
    aam_mps.load_state_dict(aam_cpu.state_dict())
    batch = tiny_batch()
    outs = model_outputs()
    results = {}
    for device, aam in (("cpu", aam_cpu), ("mps", aam_mps)):
        y, head_logits, g_t = [t.detach().clone().to(device).requires_grad_(True) for t in outs]
        total, parts = total_loss(y, head_logits, g_t, batch, aam, True)
        total.backward()
        results[device] = (total.item(), {k: v.item() for k, v in parts.items()},
                           [t.grad.cpu() for t in (y, head_logits, g_t)])
    total_cpu, parts_cpu, grads_cpu = results["cpu"]
    total_mps, parts_mps, grads_mps = results["mps"]
    assert abs(total_mps - total_cpu) < 1e-4 * max(1.0, abs(total_cpu))
    for name in losses.TERMS:
        assert abs(parts_mps[name] - parts_cpu[name]) < 1e-4 * max(1.0, abs(parts_cpu[name])), name
    for g_cpu, g_mps in zip(grads_cpu, grads_mps):
        assert torch.isfinite(g_mps).all()
        assert (g_mps - g_cpu).abs().max().item() <= 1e-3 * max(1e-6, g_cpu.abs().max().item())


@pytest.mark.skipif(not HAS_MPS, reason="no MPS device")
def test_mixed_batch_on_mps():
    """The real-model mixed batch on MPS: finite loss and gradients, and the same loss as on the CPU."""
    model_cpu = init_random_for_parity(0).train()
    model_mps = init_random_for_parity(0).train().to("mps")
    torch.manual_seed(0)
    aam_cpu = AAMHead()
    aam_mps = AAMHead().to("mps")
    aam_mps.load_state_dict(aam_cpu.state_dict())
    batch = real_batch()
    loss_cpu, _ = run_step(model_cpu, aam_cpu, batch, "cpu")
    loss_mps, parts_mps = run_step(model_mps, aam_mps, batch, "mps")
    loss_mps.backward()
    check_finite_gradients(model_mps, aam_mps, loss_mps, parts_mps, require_nonzero=True)
    assert abs(loss_mps.item() - loss_cpu.item()) < 1e-3 * max(1.0, abs(loss_cpu.item()))
