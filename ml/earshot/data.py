"""Recipes, the batch Dataset and the DataLoader (10-ml 8.2, 8.7, 8.8; contract 8.1 and 8.3).

- `RECIPES` is the one table of training recipes (SMOKE, A, B, C, FT). A YAML config only names a recipe.
- `make_example` renders one complete example (mixture, reference, labels, enrollments) as a dict of numpy arrays.
- `MixDataset[j]` returns the whole batch j as numpy (8.7). Example i of batch j uses its own generator
  default_rng([seed, 8 j + i]); batch-level values (the enrollment length Le) use default_rng([seed, j, 7]).
  So batch j is the same in any process, with any number of workers, and after any resume (contract 8.4).
- `make_loader` wraps it in a DataLoader with macOS spawn workers (8.8).

`Draws` (every per-example scalar, in draw order) is defined in render.py next to the code that draws it, and is
re-exported here so `data.Draws` works too.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import enroll, render
from .config import HOP, MAX_O_TOK, MAX_T_TOK, SR, WESPK_DIM
from .render import Draws  # noqa: F401  (re-exported: data.Draws)
from .sources import OWN_CLIP_LEN, OWN_CACHE_PATH, get_sources, rig_rows

RECIPE_VERSION = 2             # saved in checkpoints; bump when any number in RECIPES changes
                               # (2 = headroom rule: mixture and target enrollment peaks <= 0.99, 2026-11-08)
EXAMPLE_STRIDE = 8             # example i of batch j has index 8 j + i, even when B = 4 (no collisions across stages)
LE_CHOICES_S = np.arange(3.0, 10.01, 0.5)   # target enrollment lengths U{3.0, 3.5, ..., 10.0} s, one per batch


@dataclass(frozen=True)
class Recipe:
    """One column of the 10-ml 8.2 table. Probabilities are fractions; ranges are (low, high)."""
    name: str
    p_person: float            # Person vs Everyone
    p_absent: float            # target absent, of Person examples
    talkers: tuple             # P(0, 1, 2, 3 competing talkers)
    sir: tuple                 # SIR range, dB (uniform)
    p_own_given: float         # own enrollment given
    p_own_mix: float           # own talker in the mix, of ALL examples
    p_babble: float            # babble present
    p_music: float             # music instead of noise
    snr: tuple                 # ("uniform", lo, hi) or ("normal", mean, std, lo, hi)
    p_quiet: float             # quiet room U(20, 40) dB
    rooms: tuple               # P(sim, real, dry)
    p_lombard: float           # Lombard, of loud scenes (SNR <= 5 dB)
    clipping: bool             # 3% hard + 3% soft clipping on the mixture
    seconds: float             # segment length
    batch: int                 # examples per batch
    p_rig: float               # phone-rig rows (fine-tune only, 8.10)


RECIPES = {
    "SMOKE": Recipe("SMOKE", p_person=1.0, p_absent=0.0, talkers=(0.0, 1.0, 0.0, 0.0), sir=(0.0, 6.0),
                    p_own_given=0.9, p_own_mix=0.0, p_babble=0.0, p_music=0.0, snr=("uniform", 5.0, 10.0),
                    p_quiet=0.0, rooms=(1.0, 0.0, 0.0), p_lombard=0.0, clipping=False, seconds=4.0, batch=8,
                    p_rig=0.0),
    "A": Recipe("A", p_person=1.0, p_absent=0.0, talkers=(0.15, 0.45, 0.40, 0.0), sir=(-3.0, 12.0),
                p_own_given=0.9, p_own_mix=0.0, p_babble=0.7, p_music=0.2, snr=("normal", 8.0, 4.0, 0.0, 25.0),
                p_quiet=0.0, rooms=(0.75, 0.15, 0.10), p_lombard=0.3, clipping=True, seconds=4.0, batch=8,
                p_rig=0.0),
    "B": Recipe("B", p_person=0.75, p_absent=0.10, talkers=(0.10, 0.35, 0.30, 0.25), sir=(-8.0, 12.0),
                p_own_given=0.9, p_own_mix=0.5, p_babble=0.7, p_music=0.2, snr=("normal", 5.0, 5.0, -8.0, 25.0),
                p_quiet=0.10, rooms=(0.75, 0.15, 0.10), p_lombard=0.3, clipping=True, seconds=4.0, batch=8,
                p_rig=0.0),
}
RECIPES["C"] = dataclasses.replace(RECIPES["B"], name="C", rooms=(0.60, 0.30, 0.10), seconds=8.0, batch=4)
RECIPES["FT"] = dataclasses.replace(RECIPES["C"], name="FT", p_rig=0.10)


def get_recipe(recipe):
    """A Recipe from a name ("B") or a Recipe."""
    return RECIPES[recipe] if isinstance(recipe, str) else recipe


def draw_le(seed, j):
    """Target enrollment length of batch j in samples: U{3.0, 3.5, ..., 10.0} s from default_rng([seed, j, 7])."""
    rng = np.random.default_rng([seed, j, 7])
    return int(round(LE_CHOICES_S[int(rng.integers(len(LE_CHOICES_S)))] * SR))


# ---------------------------------------------------------------------------------------------------------------
# One complete example
# ---------------------------------------------------------------------------------------------------------------
def make_example(src, recipe, rng, L, Le, overrides=None, test_mode=False, metric_signals=False,
                 record_sources=False, return_parts=False, allow_missing_decim=False):
    """Mixture + reference + labels (render.py) + enrollments (enroll.py) -> dict with the 8.7 keys for one example
    (no batch dimension), plus "scene" extras when metric_signals / record_sources / return_parts are set.

    e_o is the cached WeSpeaker vector; without a cache (val, test) it is zeros and info["e_o_cached"] is False, so
    the caller (render_val, the test vectors) computes it from "own_clean".
    """
    recipe = get_recipe(recipe)
    scene = render.render_example(src, recipe, rng, L, Le, overrides, test_mode, metric_signals, record_sources,
                                  return_parts, allow_missing_decim)
    d = scene["draws"]
    info = dict(scene["info"])
    ex = {"mix": scene["mix"], "ref": scene["ref"],
          "vad_t": scene["vad_t"].astype(np.float32), "vad_o": scene["vad_o"].astype(np.float32),
          "present": bool(scene["present"]), "own_in_mix": bool(scene["own_in_mix"]), "person": bool(d.person)}

    # Target enrollment: Person examples only (Everyone has none: zeros and has_t False).
    if d.person:
        t = enroll.render_enrollment(src, scene, rng, Le, test_mode)
        ex.update(enr_t=t["enr_t"], tok_idx_t=t["tok_idx_t"], tok_val_t=t["tok_val_t"])
        info.update(t["info"])
    else:
        ex.update(enr_t=np.zeros(Le, dtype=np.float32), tok_idx_t=np.zeros((MAX_T_TOK, 3), dtype=np.int64),
                  tok_val_t=np.zeros(MAX_T_TOK, dtype=bool))

    # Own enrollment: given in 90%; otherwise zeros, has_o False (and render.py put no own talker in the mix).
    own_clean = None
    if d.own_given:
        o = enroll.render_own_enrollment(src, scene, rng, test_mode)
        e_o = o["e_o"]
        info.update(o["info"])
        info["e_o_cached"] = e_o is not None
        own_clean = o["clean"]
        ex.update(enr_o=o["enr_o"], tok_idx_o=o["tok_idx_o"], tok_val_o=o["tok_val_o"],
                  e_o=e_o if e_o is not None else np.zeros(WESPK_DIM, dtype=np.float32))
    else:
        ex.update(enr_o=np.zeros(OWN_CLIP_LEN, dtype=np.float32), tok_idx_o=np.zeros((MAX_O_TOK, 3), dtype=np.int64),
                  tok_val_o=np.zeros(MAX_O_TOK, dtype=bool), e_o=np.zeros(WESPK_DIM, dtype=np.float32))

    info["use_wes"] = bool(d.use_wes)
    ex.update(has_t=bool(d.person), has_o=bool(d.own_given), use_wes=bool(d.use_wes),
              mode=np.array([1.0, 0.0] if d.person else [0.0, 1.0], dtype=np.float32),
              spk_t=int(scene["target"]) if d.person else -1, info=info)
    ex["scene"] = scene                 # kept for tests and tools; stack_batch drops it
    ex["own_clean"] = own_clean
    return ex


# The 8.7 batch schema: key -> numpy dtype (shapes are [B, ...] of the per-example arrays).
BATCH_KEYS = {"mix": np.float32, "ref": np.float32, "vad_t": np.float32, "vad_o": np.float32,
              "present": bool, "own_in_mix": bool, "person": bool, "enr_t": np.float32,
              "tok_idx_t": np.int64, "tok_val_t": bool, "enr_o": np.float32, "tok_idx_o": np.int64,
              "tok_val_o": bool, "e_o": np.float32, "has_t": bool, "has_o": bool, "use_wes": bool,
              "mode": np.float32, "spk_t": np.int64}


def stack_batch(examples):
    """List of make_example dicts -> one batch dict of numpy arrays [B, ...] plus info (list of B dicts)."""
    batch = {key: np.stack([np.asarray(ex[key]) for ex in examples]).astype(dtype)
             for key, dtype in BATCH_KEYS.items()}
    batch["info"] = [ex["info"] for ex in examples]
    return batch


# ---------------------------------------------------------------------------------------------------------------
# Dataset and loader (8.7, 8.8)
# ---------------------------------------------------------------------------------------------------------------
class MixDataset(Dataset):
    """Batch j of a recipe -> one whole batch dict (numpy), see BATCH_KEYS and 10-ml 8.7.

    Holds only small values (split name, recipe, seed, B, L), so it pickles cheaply to spawn workers; each process
    builds its own Sources on first use (sources.get_sources).
    """

    def __init__(self, split, recipe, seed, B, L, overrides=None, allow_missing_decim=False):
        self.split = split
        self.recipe = get_recipe(recipe)
        self.seed = int(seed)
        self.B = int(B)
        self.L = int(L)
        self.overrides = dict(overrides or {})
        self.allow_missing_decim = bool(allow_missing_decim)
        if self.B > EXAMPLE_STRIDE:
            raise ValueError(f"B = {B} is larger than the example stride {EXAMPLE_STRIDE}")
        if self.L % HOP:
            raise ValueError(f"L = {L} is not a multiple of {HOP}")
        if split == "train" and not OWN_CACHE_PATH.exists():
            raise FileNotFoundError(f"{OWN_CACHE_PATH} is missing; build it with "
                                    "`.venv/bin/python -m ml.tools.cache_own_embeddings` (about 15 min on MPS)")
        if self.recipe.p_rig > 0:
            rig_rows()        # None without data/rig/manifest.jsonl (then 0% rig); raises if the file exists

    def __len__(self):
        return 10 ** 9        # batches are addressed by index; the sampler decides which ones are used

    def __getitem__(self, j):
        src = get_sources(self.split)
        Le = draw_le(self.seed, j)
        examples = []
        for i in range(self.B):
            rng = np.random.default_rng([self.seed, EXAMPLE_STRIDE * int(j) + i])
            ex = make_example(src, self.recipe, rng, self.L, Le, self.overrides,
                              allow_missing_decim=self.allow_missing_decim)
            ex["info"]["index"] = EXAMPLE_STRIDE * int(j) + i
            examples.append(ex)
        return stack_batch(examples)


def worker_init(worker_id):
    """DataLoader worker start: one torch thread, and open this split's RIR memory maps once."""
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    get_sources(info.dataset.split).open_banks()


def as_numpy(batch):
    """Collate for batch_size=None: keep the batch exactly as MixDataset returned it (numpy arrays + info).

    The DataLoader's default would turn the arrays into torch tensors inside the worker. Tensors sent from a worker
    go through torch's shared-memory manager; on macOS that starts torch_shm_manager helper processes that inherit
    Python's multiprocessing resource-tracker pipe and outlive the run, and Python 3.12 then hangs at exit waiting
    for the tracker (seen 2026-11-07). Plain numpy arrays are pickled through the worker pipe instead, which costs a
    few ms per batch. The training loop converts with to_torch.
    """
    return batch


def to_torch(batch):
    """numpy batch -> torch tensors (no copy, CPU); info stays a list of dicts. Called in the training process."""
    return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


def make_loader(src_split, recipe, seed, start_batch, end_batch, B, L, workers=6, overrides=None,
                allow_missing_decim=False):
    """DataLoader over batches start_batch .. end_batch - 1, in that order (8.8).

    workers > 0 uses macOS spawn workers with persistent workers and prefetch 2; workers = 0 renders in the calling
    process (tests). Batch j is identical either way. Batches come out as numpy dicts (see as_numpy); call
    to_torch(batch) in the training loop.
    """
    dataset = MixDataset(src_split, recipe, seed, B, L, overrides, allow_missing_decim)
    kwargs = {"batch_size": None, "sampler": range(start_batch, end_batch), "num_workers": workers,
              "pin_memory": False, "collate_fn": as_numpy}
    if workers > 0:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2,
                      worker_init_fn=worker_init)
    return DataLoader(dataset, **kwargs)
