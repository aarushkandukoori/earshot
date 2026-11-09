"""Test material and the per-mixture builders of the evaluation suites (10-ml 12.3, contract 9).

Nothing here is used by training. The training modules (sources.py, render.py, enroll.py, data.py) are imported
and reused unchanged; this file only adds the test-split hooks they lack:

- `EvalSources` - the held-out test material in the same shape as `sources.Sources`, so `render.render_example`
  and `enroll.py` run on it unchanged:
    speech  - the 48 held-out speakers: VCTK test (8, speech_test.jsonl) + LibriSpeech test-clean (40,
              eval_librispeech_test.jsonl). Targets, competing talkers, babble and own voice all come from them.
    noise   - noise_test.jsonl (DEMAND SCAFE, FSD50K test) and WHAM! test (eval_wham_tt.jsonl, CC BY-NC, test only).
    rooms   - sim_test (100 simulated rooms, in_hand / on_table) and real_test rows of category bar, restaurant or
              indoor (MIT IR Survey, 193 rows; vehicle and outdoor rows are left out).
- `AccentSources` - the same, plus the EdAcc test and VoxPopuli accented speakers as targets (accents suite).
- The restaurant grid (1,080 cells) and the builders of the restaurant, absent, accents and libri2mix items.

How WHAM! fits into render.py without editing it. render.noise_clip knows the kinds demand (an ambient recording:
random crop, no room), fsd50k (a sound event placed in the room), musan and synthetic. WHAM! is an ambient
recording like DEMAND, so EvalSources puts WHAM! rows and DEMAND SCAFE rows into one "demand" (ambient) pool. A
uniform pick from that pool lands on WHAM! 2 times in 3: the 4 DEMAND channel files are each listed 266 times next
to the 2,127 WHAM! files. The restaurant builder then asks for kind "demand" 75% of the time and "fsd50k" 25%, so
each noise clip is WHAM! 50%, DEMAND SCAFE 25% and FSD50K test 25% = "noise_test 50% and WHAM! test 50%" (12.3).

Every builder is deterministic per (suite seed, item index): the render generator is default_rng([seed, i]) as in
render_val, and the per-item plan (grid cell, own level, noise kinds, accent speaker) uses its own generator
default_rng([seed, i, PLAN_STREAM]), so a plan choice never shifts the render draws.
"""
import copy
import itertools
import time

import numpy as np

from dataprep.common import read_manifest   # frozen data-pack code, imported read-only
from dataprep import eval_libri2mix         # LibriMix rebuild (MIT), imported read-only

from . import data, dsp, enroll, render
from .config import DATA, HOP, MAX_O_TOK, MAX_T_TOK, SR
from .sources import (MANIFESTS, OWN_CLIP_LEN, RIR_DIR, TARGET_MIN_ROWS, TARGET_MIN_SECONDS, Sources, check_disjoint,
                      own_split, read_checked, read_checked_rirs, read_part)
from .vad import es_vad

# ---------------------------------------------------------------------------------------------------------------
# Constants of the suites (10-ml 12.3)
# ---------------------------------------------------------------------------------------------------------------
L_EVAL = 8 * SR                 # restaurant, absent and accents mixtures are 8 s
LE_EVAL = 6 * SR                # their target enrollments are 6 s
LIBRI_ENR_MAX = 10 * SR         # libri2mix enrollment = first 10 s of the enrollment utterance
PLAN_STREAM = 1                 # plan generator default_rng([seed, i, 1]); the render generator is ([seed, i])

SEEDS = {"restaurant": 2027, "absent": 2028, "stability": 2029,
         "accents_edacc": 2031, "accents_voxpopuli": 2032}   # 2030 is taken by threshold calibration (10-ml 13)
COUNTS = {"restaurant": 2000, "absent": 500, "accents": 500, "libri2mix": 6000}   # libri2mix = 3,000 x 2 targets

HELD_OUT_MANIFESTS = ["speech_test", "eval_librispeech_test"]      # the 48 held-out speakers
ACCENT_MANIFESTS = {"edacc": "eval_edacc_test", "voxpopuli": "eval_voxpopuli_accented"}
REAL_CATEGORIES = ("bar", "restaurant", "indoor")                  # real_test rows used as "real" rooms
P_AMBIENT = 0.75                # noise kind "demand" (ambient pool: WHAM! 2/3, DEMAND 1/3), else "fsd50k"

# The restaurant grid (contract 9): 4 x 5 x 3 x 2 x 3 x 3 = 1,080 cells. Cell c is the c-th entry of
# itertools.product over these lists, in this order, so the LAST factor (enrollment) changes fastest.
GRID = [
    ("snr_db", (-5.0, 0.0, 5.0, 10.0)),
    ("sir_db", (-8.0, -5.0, 0.0, 5.0, 10.0)),
    ("n_talkers", (1, 2, 3)),
    ("own", ("off", "on")),
    ("room", ("sim_in_hand", "sim_on_table", "real")),
    ("enroll", ("clean", "noisy_10", "noisy_0")),
]
GRID_NAMES = [name for name, _ in GRID]
CELLS = list(itertools.product(*[levels for _, levels in GRID]))   # 1,080 tuples
N_CELLS = len(CELLS)
OWN_DB_MAX = 12.0               # own voice "on" = U(0, 12) dB above the target at the mic

# Base recipe for every value a cell does not fix (recipe B: babble 70%, partial target 25%, ...). Test mode
# (8.9) then fixes the level (-26 dBFS), high-pass 100 Hz, no EQ, no clipping, no Lombard, fir_decim on.
EVAL_RECIPE = data.RECIPES["B"]


def cell_factors(c):
    """Grid cell index -> {factor name: level}, e.g. cell 0 -> snr -5, sir -8, 1 talker, own off, ..."""
    return dict(zip(GRID_NAMES, CELLS[c]))


# ---------------------------------------------------------------------------------------------------------------
# Test sources
# ---------------------------------------------------------------------------------------------------------------
def read_rows(names):
    """Rows of several manifests (data/manifests/<name>.jsonl), each through the license gate with split "test"."""
    rows = []
    for name in names:
        rows += read_checked(MANIFESTS / f"{name}.jsonl", "test")
    return rows


class EvalSources(Sources):
    """The held-out test material with the attribute names of sources.Sources (see that class for their meaning).

    Sources.__init__ is not called, because it reads exactly one speech manifest and has real rooms only for
    train; this __init__ fills the same attributes from the test manifests instead. The inherited methods work
    unchanged: `sim` opens data/rir/sim_test.npy, `own_cache()` is None for the test split, so `build_clip` makes
    the own-voice clip with the 10-ml 6.3 recipe and the caller computes e_o with WeSpeaker.
    """

    def __init__(self):
        self.split = "test"
        self.manifests = MANIFESTS
        self.rir_dir = RIR_DIR
        check_disjoint(MANIFESTS)          # train / val / test speakers never meet (10-ml 8.1)
        self._sim = None
        self._real = None
        self._own = None
        self.set_speech(read_rows(HELD_OUT_MANIFESTS))
        self.set_noise()
        self.set_rooms()

    # -- speech ----------------------------------------------------------------------------------------------
    def set_speech(self, rows):
        """Per-row arrays, speaker list and pools from a list of speech rows (same rules as Sources)."""
        self.paths = [r["path"] for r in rows]
        self.dur = np.array([r["duration"] for r in rows], dtype=np.float32)
        self.lic = [r["license"] for r in rows]
        self.speakers = sorted({r["speaker"] for r in rows})
        index_of = {s: i for i, s in enumerate(self.speakers)}
        self.spk_of = np.array([index_of[r["speaker"]] for r in rows], dtype=np.int32)
        groups = [[] for _ in self.speakers]
        for r_index, s in enumerate(self.spk_of):
            groups[s].append(r_index)
        self.rows_of = [np.array(g, dtype=np.int32) for g in groups]
        self.spk_seconds = np.array([float(self.dur[g].sum()) for g in self.rows_of])
        target = [s for s, g in enumerate(self.rows_of)
                  if len(g) >= TARGET_MIN_ROWS and self.spk_seconds[s] >= TARGET_MIN_SECONDS]
        self.target_pool = np.array(target, dtype=np.int32)
        self.target_fallback = int(self.target_pool[np.argmax(self.spk_seconds[self.target_pool])])
        self.own_cache_rows, self.own_mix_rows, own = [], [], []
        for s, g in enumerate(self.rows_of):
            cache, mix, in_pool = own_split(g, self.dur)
            self.own_cache_rows.append(np.array(cache, dtype=np.int32))
            self.own_mix_rows.append(np.array(mix, dtype=np.int32))
            if in_pool:
                own.append(s)
        self.own_pool = np.array(own, dtype=np.int32)
        self.in_own_pool = np.zeros(len(self.speakers), dtype=bool)
        self.in_own_pool[self.own_pool] = True

    # -- noise -----------------------------------------------------------------------------------------------
    def set_noise(self):
        """noise_test rows plus WHAM! test rows; the "demand" kind is the ambient pool (module docstring)."""
        noise = read_rows(["noise_test"])
        wham = read_rows(["eval_wham_tt"])
        rows = noise + wham
        self.noise_paths = [r["path"] for r in rows]
        self.noise_dur = np.array([r["duration"] for r in rows], dtype=np.float32)
        self.noise_lic = [r["license"] for r in rows]
        demand = [i for i, r in enumerate(noise) if "noise" in r["use"] and r["source"] == "demand"]
        fsd = [i for i, r in enumerate(noise) if "noise" in r["use"] and r["source"] == "fsd50k"]
        music = [i for i, r in enumerate(noise) if "music" in r["use"]]
        wham_rows = list(range(len(noise), len(rows)))
        # Each DEMAND file is listed `repeat` times so DEMAND gets 1/3 of the ambient pool and WHAM! 2/3.
        repeat = max(1, round(len(wham_rows) / (2 * max(1, len(demand)))))
        self.ambient_repeat = repeat
        self.noise_rows = {"demand": np.array(wham_rows + demand * repeat, dtype=np.int32),
                           "fsd50k": np.array(fsd, dtype=np.int32),
                           "musan": np.zeros(0, dtype=np.int32),          # no MUSAN test rows (falls back to fsd50k)
                           "music": np.array(music, dtype=np.int32)}
        self.noise_counts = {"demand_scafe": len(demand), "fsd50k": len(fsd), "wham": len(wham_rows)}

    # -- rooms -----------------------------------------------------------------------------------------------
    def set_rooms(self):
        """sim_test (as Sources does) and the real_test rows of category bar / restaurant / indoor.

        real_test has no room_id or db field. render.real_rows gives the target row r0 and fills positions 1-7
        with other rows of the same room_id; here the category plays that part, so the other sources of a real
        scene are measured in places of the same kind (bar, restaurant or indoor).
        """
        sim_rows = read_checked_rirs(self.rir_dir / "sim_test.jsonl", "test")
        self.n_sim = len(sim_rows)
        self.sim_dp = np.array([r["dp"] for r in sim_rows], dtype=np.int32)
        self.sim_in_hand = np.array([r["geometry"] == "in_hand" for r in sim_rows])
        self.sim_rt60 = np.array([r["rt60"] for r in sim_rows], dtype=np.float32)
        real_rows = read_checked_rirs(self.rir_dir / "real_test.jsonl", "test")
        keep = [i for i, r in enumerate(real_rows) if r["category"] in REAL_CATEGORIES]
        self.real_keep = np.array(keep, dtype=np.int64)          # rows of real_test.npy, in order
        self.has_real = True
        self.n_real = len(keep)
        self.real_dp = np.array([real_rows[i]["dp"] for i in keep], dtype=np.int32)
        self.real_rt60 = np.array([real_rows[i]["rt60"] for i in keep], dtype=np.float32)   # NaN for a few rows
        self.real_room = [real_rows[i]["category"] for i in keep]
        self.real_db = list(self.real_room)
        self.real_name = [real_rows[i]["name"] for i in keep]
        self.real_room_rows, self.real_db_rows = {}, {}
        for local, cat in enumerate(self.real_room):
            self.real_room_rows.setdefault(cat, []).append(local)
            self.real_db_rows.setdefault(cat, []).append(local)

    @property
    def real(self):
        """float16 [193, 24000]: the kept rows of data/rir/real_test.npy, read once (about 9 MB)."""
        if self._real is None:
            bank = np.load(self.rir_dir / "real_test.npy", mmap_mode="r")
            self._real = np.asarray(bank[self.real_keep])
        return self._real


class AccentSources(EvalSources):
    """EvalSources plus the EdAcc test and VoxPopuli accented speakers as targets (accents suite).

    The accent rows are appended after the held-out rows in paths / dur / lic, but the accent speakers are NOT
    added to `speakers`: render.py draws competing talkers and babble from range(len(speakers)), so those keep
    coming from the 48 held-out speakers ("everything else as restaurant"). `view(set_name, k)` returns a shallow
    copy in which exactly one accent speaker is appended as speaker 48 and is the only target.

    Eligible accent targets = speakers with at least 2 utterances and at least 14.5 s in total (the target-pool
    rule, so render.pick_target can cut a 6.5 s enrollment and an 8 s stream from different utterances; render.py
    never cycles an enrollment). `excluded[set_name]` lists the speakers this leaves out.
    """

    def __init__(self):
        super().__init__()
        self.accent, self.excluded = {}, {}
        paths, dur, lic = list(self.paths), list(self.dur), list(self.lic)
        for set_name, manifest in ACCENT_MANIFESTS.items():
            rows = read_rows([manifest])
            by_speaker = {}
            for r in rows:
                by_speaker.setdefault(r["speaker"], []).append(len(paths))
                paths.append(r["path"])
                dur.append(r["duration"])
                lic.append(r["license"])
            self.accent[set_name], self.excluded[set_name] = [], []
            for spk in sorted(by_speaker):
                idx = by_speaker[spk]
                seconds = float(sum(dur[k] for k in idx))
                if len(idx) >= TARGET_MIN_ROWS and seconds >= TARGET_MIN_SECONDS:
                    self.accent[set_name].append((spk, np.array(idx, dtype=np.int32)))
                else:
                    self.excluded[set_name].append((spk, len(idx), round(seconds, 1)))
        self.paths, self.dur, self.lic = paths, np.array(dur, dtype=np.float32), lic

    def view(self, set_name, k):
        """A shallow copy whose only target is accent speaker k of the set (speaker index 48)."""
        name, rows = self.accent[set_name][k]
        v = copy.copy(self)
        t = len(self.speakers)
        v.speakers = self.speakers + [name]
        v.rows_of = self.rows_of + [rows]
        v.target_pool = np.array([t], dtype=np.int32)
        v.target_fallback = t
        v.in_own_pool = np.append(self.in_own_pool, False)        # never the own voice
        v.own_cache_rows = self.own_cache_rows + [np.zeros(0, dtype=np.int32)]
        v.own_mix_rows = self.own_mix_rows + [np.zeros(0, dtype=np.int32)]
        return v


# One object per kind per process (each spawn worker builds its own, like sources.get_sources).
_CACHE = {}


def get_eval_sources():
    """The process-wide EvalSources (built on first call, about 0.2 s)."""
    if "eval" not in _CACHE:
        _CACHE["eval"] = EvalSources()
    return _CACHE["eval"]


def get_accent_sources():
    """The process-wide AccentSources."""
    if "accent" not in _CACHE:
        _CACHE["accent"] = AccentSources()
    return _CACHE["accent"]


def libri2mix_definitions():
    """The 3,000 lines of data/eval/libri2mix_test.jsonl (cached). Each must be a test / eval definition."""
    if "libri2mix" not in _CACHE:
        defs = list(read_manifest(DATA / "eval" / "libri2mix_test.jsonl"))
        for d in defs:
            if d.get("split") != "test" or "eval" not in (d.get("use") or []):
                raise ValueError(f"libri2mix definition {d.get('id')} is not marked split test / use eval")
        _CACHE["libri2mix"] = defs
    return _CACHE["libri2mix"]


# ---------------------------------------------------------------------------------------------------------------
# Per-item plan and grid overrides
# ---------------------------------------------------------------------------------------------------------------
def item_plan(seed, i, grid_first=True, n_accent=0):
    """The per-item choices that are not render.py draws -> dict.

    Draw order (fixed): a random cell, the own level U(0, 12) dB, two noise kinds (ambient 75% / fsd50k 25%), and,
    for the accents suite, an order in which to try the accent speakers. cell = i for the first 1,080 items of a
    grid-first suite (restaurant), else the random cell (absent, accents, restaurant items 1,080 and up).
    """
    rng = np.random.default_rng([seed, i, PLAN_STREAM])
    random_cell = int(rng.integers(N_CELLS))
    own_db = float(rng.uniform(0.0, OWN_DB_MAX))
    kinds = tuple("demand" if rng.random() < P_AMBIENT else "fsd50k" for _ in range(2))
    order = [int(k) for k in rng.permutation(n_accent)] if n_accent else []
    cell = i if (grid_first and i < N_CELLS) else random_cell
    return {"cell": cell, "own_db": own_db, "noise_kinds": kinds, "accent_order": order}


def cell_overrides(cell, own_db, noise_kinds, absent=False):
    """render.py overrides (8.9 test mode) for one grid cell.

    Fixed by the cell: SNR, SIR, competing talkers, own voice off / on (at own_db), room kind (and sim geometry),
    enrollment clean / noisy 10 dB / noisy 0 dB. Fixed for every evaluation mixture: Person mode, own voice always
    enrolled (contract 9), noise as the background (no music), no quiet room, no weaker talker in the enrollment,
    es_vad enrollment mask (what the phone uses), WeSpeaker on, clean own enrollment.
    Left to render.py's draws: babble (70%, 6-20 talkers), 1 or 2 noise clips, partial target activity (25%),
    activity windows, speakers, rooms of the chosen kind, the enrollment room and noise source.
    """
    f = cell_factors(cell)
    ov = {"person": True, "absent": bool(absent), "n_talkers": int(f["n_talkers"]), "sir_db": float(f["sir_db"]),
          "own_given": True, "own_in_mix": f["own"] == "on", "own_db": float(own_db),
          "background": "noise", "noise_kinds": tuple(noise_kinds), "quiet": False, "snr_db": float(f["snr_db"]),
          "enr_weak": False, "enr_noise": f["enroll"] != "clean",
          "enr_snr_db": 10.0 if f["enroll"] == "noisy_10" else 0.0,
          "enr_mask": "vad", "use_wes": True, "own_noise": False}
    if f["room"] == "real":
        ov["room"] = "real"
    else:
        ov["room"] = "sim"
        ov["geometry"] = "in_hand" if f["room"] == "sim_in_hand" else "on_table"
    return ov


def noise_class(path):
    """'eval/wham_tt/...' -> 'wham', 'noise/demand/...' -> 'demand', anything else -> 'fsd50k'."""
    if path.startswith("eval/wham_tt/"):
        return "wham"
    if path.startswith("noise/demand/"):
        return "demand"
    return "fsd50k"


# ---------------------------------------------------------------------------------------------------------------
# Items: what one mixture of a suite hands to the evaluator (numpy only, so spawn workers can send it)
# ---------------------------------------------------------------------------------------------------------------
# Model inputs (the train.VAL_INPUT_KEYS without e_t / e_o, which the evaluator computes with WeSpeaker) + ref.
MODEL_KEYS = ("mix", "ref", "enr_t", "tok_idx_t", "tok_val_t", "enr_o", "tok_idx_o", "tok_val_o", "has_t", "has_o",
              "use_wes", "mode")


def make_item(ex, row):
    """data.make_example output -> item: model inputs, metric signals (8.9) and the CSV fields in `row`.

    Metric signals: talkers [n, L] (competing talkers' early references after the mic chain), vad_talkers [n, T],
    own [L] (the own voice image at the mic after the chain), vad_t / vad_o [T] (hop VADs of the reference and of
    the own voice). own_clean is the clean own clip that the evaluator embeds as e_o (as render_val does).
    """
    m = ex["scene"]["metric"]
    L = len(ex["mix"])
    T = L // HOP
    talkers = np.stack(m["talkers"]).astype(np.float32) if m["talkers"] else np.zeros((0, L), np.float32)
    vad_talkers = (np.stack(m["vad_talkers"]).astype(np.uint8) if m["vad_talkers"]
                   else np.zeros((0, T), np.uint8))
    item = {key: ex[key] for key in MODEL_KEYS}
    item.update(own_clean=ex["own_clean"], vad_t=np.asarray(ex["vad_t"]).astype(np.uint8),
                vad_o=np.asarray(ex["vad_o"]).astype(np.uint8), talkers=talkers, vad_talkers=vad_talkers,
                own=np.asarray(m["own"], dtype=np.float32), row=row)
    return item


def scene_row(src, ex):
    """CSV fields that describe one rendered scene (speakers, room, backgrounds, enrollment)."""
    info, scene = ex["info"], ex["scene"]
    noise = [noise_class(s["path"]) for s in scene["sources"] if s["role"] == "noise"]
    return {"target": info["target"], "talkers": "|".join(src.speakers[t] for t in scene["talkers"]),
            "own_speaker": info.get("own", ""), "room_index": info["room_index"], "geometry": info["geometry"],
            "rt60": info["rt60"], "babble": info["babble"], "noise": "+".join(noise) or "none",
            "noise_kinds": "+".join(info["noise_kinds"]), "target_frac": info["target_frac"],
            "enr_room": info.get("enr_room"), "enr_noise": info.get("enr_noise"), "short_t": info.get("short_t"),
            "snr_chain_db": info.get("snr_chain_db")}


def grid_row(i, plan):
    """CSV fields of the plan: item index, grid cell and its factor levels, own level when the own voice is on."""
    f = cell_factors(plan["cell"])
    return {"index": i, "cell": plan["cell"], **f, "own_db": round(plan["own_db"], 3) if f["own"] == "on" else ""}


def render_eval(src, seed, i, overrides, attempt=0):
    """One evaluation mixture with make_example in test mode, with metric signals and source records."""
    rng = np.random.default_rng([seed, i] if attempt == 0 else [seed, i, 2, attempt])
    return data.make_example(src, EVAL_RECIPE, rng, L_EVAL, LE_EVAL, overrides, test_mode=True,
                             metric_signals=True, record_sources=True)


def build_restaurant(i, absent=False):
    """Restaurant item i (seed 2027; item i < 1,080 takes cell i) or absent item i (seed 2028, random cell,
    target removed)."""
    suite = "absent" if absent else "restaurant"
    seed = SEEDS[suite]
    t0 = time.perf_counter()
    src = get_eval_sources()
    plan = item_plan(seed, i, grid_first=not absent)
    ex = render_eval(src, seed, i, cell_overrides(plan["cell"], plan["own_db"], plan["noise_kinds"], absent))
    row = {**grid_row(i, plan), **scene_row(src, ex), "render_s": round(time.perf_counter() - t0, 4)}
    return make_item(ex, row)


def build_accent(set_name, i):
    """Accents item i of one set (edacc seed 2031, voxpopuli seed 2032): a random restaurant cell with an accent
    speaker as the target. Speakers are tried in the plan's order; one whose rows cannot give a 6.5 s enrollment
    plus an 8 s stream (render.pick_target raises after 21 tries) is skipped for the next (column `attempt`)."""
    seed = SEEDS[f"accents_{set_name}"]
    t0 = time.perf_counter()
    src = get_accent_sources()
    plan = item_plan(seed, i, grid_first=False, n_accent=len(src.accent[set_name]))
    overrides = cell_overrides(plan["cell"], plan["own_db"], plan["noise_kinds"])
    for attempt, k in enumerate(plan["accent_order"]):
        view = src.view(set_name, k)
        try:
            ex = render_eval(view, seed, i, overrides, attempt)
        except ValueError as err:
            if "no target can give" not in str(err):
                raise
            continue
        row = {"set": set_name, **grid_row(i, plan), **scene_row(view, ex), "attempt": attempt,
               "render_s": round(time.perf_counter() - t0, 4)}
        return make_item(ex, row)
    raise RuntimeError(f"accents {set_name} item {i}: no accent speaker gave a valid scene")


def eval_chain():
    """The test-mode mic chain (8.9): high-pass 100 Hz, no EQ, -26 dBFS, no self-noise, no clipping, fir_decim."""
    chain = render.MicChain(render.TEST_HIGHPASS_HZ, (), render.TEST_LEVEL_DBFS, None, False, "none", 1.0)
    chain.decim = render.decim_available(False)
    return chain


def through_chain(chain, x, others=()):
    """Mic chain on x at the chain's level (with the headroom rule) and the same filters and gain on `others`."""
    x_f = render.chain_filters(chain, x)
    gain = render.headroom_gain(render.level_gain(chain.level_dbfs, x_f), x_f)
    y = np.clip(x_f * gain, -1.0, 1.0).astype(np.float32)
    return y, [(render.chain_filters(chain, o) * gain).astype(np.float32) for o in others]


def build_libri2mix(k):
    """Libri2Mix item k = definition k // 2, target k % 2 (contract 9 row, 12.3).

    Sources with the definition's gains and offsets cut to `length` (min mode) by dataprep.eval_libri2mix.render,
    no noise. Test mode (12.3 table header) then puts the mixture through the 8.9 chain at -26 dBFS, with the same
    filters and gain on both sources, so the SI-SNRi is still measured against the definition's target source.
    The length is cut down to a multiple of 64 samples (the model's hop; at most 3.9 ms dropped). Enrollment = the
    first 10 s of the definition's enrollment utterance for this target (3 to 10 s), through the same chain at
    -26 dBFS, es_vad mask. No own voice (has_o False).
    """
    t0 = time.perf_counter()
    defn = libri2mix_definitions()[k // 2]
    t = k % 2
    mix, srcs, _ = eval_libri2mix.render(defn, DATA, noisy=False)
    L = (len(mix) // HOP) * HOP
    chain = eval_chain()
    x, (s, other) = through_chain(chain, mix[:L], [srcs[t][:L], srcs[1 - t][:L]])
    e_info = defn["enrollment"][t]
    enr_raw = read_part(e_info["path"], 0, LIBRI_ENR_MAX)
    enr, _ = through_chain(chain, enr_raw)
    idx, valid, short = enroll.groups_or_ones(es_vad(enr), MAX_T_TOK)
    T = L // HOP
    item = {"mix": x, "ref": s, "enr_t": enr, "tok_idx_t": idx, "tok_val_t": valid,
            "enr_o": np.zeros(OWN_CLIP_LEN, np.float32), "tok_idx_o": np.zeros((MAX_O_TOK, 3), np.int64),
            "tok_val_o": np.zeros(MAX_O_TOK, bool), "has_t": True, "has_o": False, "use_wes": True,
            "mode": np.array([1.0, 0.0], np.float32), "own_clean": None,
            "vad_t": dsp.hop_vad(s), "vad_o": np.zeros(T, np.uint8),
            "talkers": other[None], "vad_talkers": dsp.hop_vad(other)[None], "own": np.zeros(L, np.float32)}
    item["row"] = {"index": k, "id": defn["id"], "target": t, "target_speaker": defn["sources"][t]["speaker"],
                   "other_speaker": defn["sources"][1 - t]["speaker"], "L": L, "Le": len(enr),
                   "enr_utt": e_info["utt"], "short_t": short, "render_s": round(time.perf_counter() - t0, 4)}
    return item


def suite_jobs(suite, limit=None):
    """The (suite, key) jobs of a batch suite in item order; `limit` keeps the first N (accents: N per set)."""
    def first(n):
        return range(n if limit is None else min(n, int(limit)))
    if suite in ("restaurant", "absent", "libri2mix"):
        return [(suite, i) for i in first(COUNTS[suite])]
    if suite == "accents":
        return [(suite, (set_name, i)) for set_name in ACCENT_MANIFESTS for i in first(COUNTS["accents"])]
    raise ValueError(f"{suite} is not a batch suite")


def build_item(job):
    """(suite, key) -> item. Top-level, so a spawn worker pool can call it (evaluate --workers)."""
    suite, key = job
    if suite == "restaurant":
        return build_restaurant(key)
    if suite == "absent":
        return build_restaurant(key, absent=True)
    if suite == "accents":
        return build_accent(*key)
    if suite == "libri2mix":
        return build_libri2mix(key)
    raise ValueError(f"unknown batch suite {suite!r}")

