"""Causality test in its exact form (10-ml 14, contract 11).

"Input after t never changes output before t", stated precisely: for t in {6400, 6437, 32000}, add N(0, 1) noise to
x[t:]. Let j = floor(t / 64), the first frame that sees the noise.
- Offline output (aligned with the input, so it may use up to 127 samples ahead = the 8 ms algorithmic latency) is
  unchanged before sample 64 j - 64, because frame j only writes output samples 64 j - 64 .. 64 j + 63.
- Streamed output is causal at hop granularity: unchanged before sample 64 j (hop j emits 64 j .. 64 j + 63).
"Unchanged" means max diff <= 1e-6; the samples right after must change by more than 1e-3.
"""
import numpy as np
import pytest
import torch

from ml.earshot.model import Cond
from ml.earshot.stream_ref import StreamRef, pack_cond

from .helpers import CASES

L = 40000          # 2.5 s, a multiple of 64, long enough for t = 32000 plus a few hops


@pytest.fixture(scope="module")
def person_cond(cond_all):
    _, cond = cond_all
    i = CASES.index("person")
    return Cond(film=cond.film[i:i + 1], k=cond.k[i:i + 1], v=cond.v[i:i + 1], valid=cond.valid[i:i + 1])


@pytest.fixture(scope="module")
def clean_outputs(net, signals, person_cond):
    """Offline and streamed outputs of the unperturbed signal."""
    x = signals["mix"][:L]
    with torch.no_grad():
        y_off, _ = net(torch.from_numpy(x)[None], person_cond)
    y_st, _ = StreamRef(net).run(pack_cond(person_cond, 0), x)
    return x, y_off[0].numpy(), y_st


@pytest.mark.parametrize("t", [6400, 6437, 32000])
def test_input_after_t_never_changes_output_before_t(net, person_cond, clean_outputs, t):
    x, y_off, y_st = clean_outputs
    rng = np.random.default_rng(t)
    x2 = x.copy()
    x2[t:] += rng.standard_normal(L - t).astype(np.float32)
    with torch.no_grad():
        y_off2, _ = net(torch.from_numpy(x2)[None], person_cond)
    y_off2 = y_off2[0].numpy()
    y_st2, _ = StreamRef(net).run(pack_cond(person_cond, 0), x2)

    j = t // 64
    off_edge = 64 * j - 64         # first offline sample allowed to change
    st_edge = 64 * j               # first streamed sample allowed to change

    assert np.abs(y_off2[:off_edge] - y_off[:off_edge]).max() <= 1e-6
    assert np.abs(y_st2[:st_edge] - y_st[:st_edge]).max() <= 1e-6
    # ... and the samples right after the edge do change.
    assert np.abs(y_off2[off_edge:off_edge + 128] - y_off[off_edge:off_edge + 128]).max() > 1e-3
    assert np.abs(y_st2[st_edge:st_edge + 64] - y_st[st_edge:st_edge + 64]).max() > 1e-3
