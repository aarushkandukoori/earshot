"""Session fixtures for the ML-0 tests (spec 10-ml section 14).

- All tests run on the CPU in float32 with 4 threads (10-ml 1.1). The MPS test moves its own copy of the model.
- Parity tests use init_random_for_parity(seed 0), so the paths training starts at zero are exercised.
- Signal and conditioning helpers live in helpers.py.
"""
import numpy as np
import pytest
import torch

from ml.earshot import config
from ml.earshot.model import init_random_for_parity

from .helpers import CASES, make_cond_inputs, noise, speech

torch.set_num_threads(4)


# ---------------------------------------------------------------------------------------------------------------
# Fixtures (built once per test session)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def net():
    """The random_v1 model: init_random_for_parity(seed 0), eval mode, CPU."""
    return init_random_for_parity(0).eval()


@pytest.fixture(scope="session")
def signals():
    """10 s mixture (target + interferer + noise) and the two enrollment clips of contract 3.3 lengths."""
    n = 10 * config.SR
    target = speech(n, speaker_index=0, first_row=0)
    other = speech(n, speaker_index=1, first_row=0)
    mix = (0.6 * target + 0.4 * other + noise(n, seed=3)).astype(np.float32)
    enr_t = (speech(96000, speaker_index=0, first_row=20) + noise(96000, seed=4, level=0.005)).astype(np.float32)
    enr_o = speech(128000, speaker_index=2, first_row=0)
    return {"mix": mix, "enr_t": enr_t, "enr_o": enr_o}


@pytest.fixture(scope="session")
def cond_all(net, signals):
    """(CondInputs, Cond) with one row per case in CASES order, on the CPU."""
    ci = make_cond_inputs(net, signals["enr_t"], signals["enr_o"], CASES)
    with torch.no_grad():
        cond = net.build_cond(ci)
    return ci, cond
