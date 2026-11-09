"""Data sources for the training loader: manifests, the license gate, speaker pools and audio reads (10-ml 8.1).

What lives here (and nothing else):
- `check_row(row, split)` - the license gate. The loader refuses (raises LicenseError on) any row whose license is
  empty or non-commercial (token NC) unless the split is test, any "eval" row outside test, and any row whose own
  split field is not the split being loaded. Outside test, the license must also be on TRAIN_LICENSES (the
  licenses the data pack actually uses), so a license written in an unexpected way stops the loader instead of
  slipping through. `check_rir_row` does the same for room rows.
- `Sources(split)` - everything the renderer needs about one split, loaded once per process: speech rows grouped by
  speaker, the noise and music rows, the RIR banks (opened lazily as memory maps, never pickled), the speaker pools
  and the own-voice split of every speaker.
- `own_split(...)` and `make_own_clip(...)` / `Sources.build_clip(...)` - the own-voice clip recipe of 10-ml 6.3.
- `read_row`, `read_crop` - the only two functions that open audio files.

Speaker pools (contract 8.1, counts measured 2026-11-07 on speech_train.jsonl and checked by test_data.py):
- target pool = speakers with at least 2 utterances and at least 14.5 s in total            -> 3,118 speakers
- own pool    = speakers whose rows, in manifest order, split into cache rows (the shortest prefix reaching 8 s,
                never the last row) and mix rows (the rest) holding at least 4 s              -> 3,135 speakers
  A speaker whose cache rows cannot reach 8 s without the last row is not in the pool (its 8 s own clip could not
  be cut from cache rows only). This reading reproduces the measured 3,135 exactly.
- interferer and babble pools = all 3,758 speakers. AAM class = index in the sorted speaker ids.
"""
import re

import numpy as np
import soundfile as sf

from dataprep.common import read_manifest   # frozen data-pack code, imported read-only

from . import dsp
from .config import CACHE, DATA, SR

MANIFESTS = DATA / "manifests"
RIR_DIR = DATA / "rir"
OWN_CACHE_PATH = CACHE / "own_v1.npz"           # written by `python -m ml.tools.cache_own_embeddings`
RIG_MANIFEST = DATA / "rig" / "manifest.jsonl"  # optional phone-rig recordings (10-ml 8.10)

# Pool rules (contract 8.1)
TARGET_MIN_ROWS = 2
TARGET_MIN_SECONDS = 14.5        # = 10 s longest enrollment + 0.5 s margin + 4 s segment
OWN_CACHE_SECONDS = 8.0          # cache rows must reach 8 s
OWN_MIX_MIN_SECONDS = 4.0        # mix rows must hold at least 4 s

# Own-voice clip recipe (10-ml 6.3)
OWN_CLIPS = 8                    # 8 clips per own-pool speaker
OWN_CLIP_LEN = 128000            # 8.0 s at 16 kHz
OWN_SEED = 2026                  # rng = default_rng([2026, speaker index, clip index])
OWN_GAP_S = (0.10, 0.30)         # zeros between cache rows
LEVEL_DBFS = -26.0               # active power of every own clip
FADE = 80                        # 5 ms half-cosine fades on every row piece

# Manifests whose speakers must never meet (10-ml 8.1 disjointness check)
TEST_SPEAKER_MANIFESTS = ["speech_test", "eval_librispeech_test", "eval_edacc_test", "eval_voxpopuli_accented"]
# Corpora that share people under different speaker prefixes: LibriTTS-R is cut from LibriSpeech recordings and
# keeps its speaker numbers, so "librittsr:1089" and "librispeech:1089" are the same person.
SAME_PEOPLE_PREFIX = {"librittsr": "libri", "libritts": "libri", "librispeech": "libri"}

# Licenses a train or val row may carry (every license in the data pack's train and val manifests and in
# data/rir/real_train.jsonl on 2026-11-07, all commercial-use friendly). Compared in upper case. A new license
# needs a deliberate edit here; until then the loader refuses the row.
TRAIN_LICENSES = {"CC0-1.0", "PD", "CC-BY", "CC-BY-3.0", "CC-BY-4.0", "CC-BY-SA", "CC-BY-SA-1.0", "CC-BY-SA-3.0",
                  "CC-BY-SA-4.0", "APACHE-2.0"}


class LicenseError(ValueError):
    """A manifest row that the requested split may not use (license gate, 10-ml 8.1)."""


# ---------------------------------------------------------------------------------------------------------------
# License gate
# ---------------------------------------------------------------------------------------------------------------
def license_tokens(text):
    """'CC-BY-NC-4.0' -> ['CC', 'BY', 'NC', '4', '0']: upper case, split on anything that is not a letter or digit
    (so 'CC_BY_NC_4.0' and 'CC BY-NC/4.0' give the NC token too)."""
    return [t for t in re.split(r"[^A-Z0-9]+", text.upper()) if t]


def is_non_commercial(text):
    """True if a license reads as non-commercial: a token NC or NC plus digits ('NC4'), or the word
    NONCOMMERCIAL anywhere (spaces and dashes ignored). 'NCSA' (a permissive license) is not NC."""
    tokens = license_tokens(text)
    if any(t == "NC" or re.fullmatch(r"NC\d+", t) for t in tokens):
        return True
    return "NONCOMMERCIAL" in "".join(tokens)


def check_license(lic, split, where):
    """The license part of the gate: refuse an empty license; outside the test split refuse a non-commercial one
    and any license that is not on TRAIN_LICENSES."""
    lic = (lic or "").strip()
    if not lic:
        raise LicenseError(f"{where}: empty license")
    if split == "test":
        return
    if is_non_commercial(lic):
        raise LicenseError(f"{where}: non-commercial license {lic!r} is allowed only in the test split")
    if lic.upper() not in TRAIN_LICENSES:
        raise LicenseError(f"{where}: license {lic!r} is not on sources.TRAIN_LICENSES; check it is commercial-use "
                           "friendly and add it there before training on it")


def check_row(row, split):
    """Raise LicenseError unless `row` may be used when loading `split` (10-ml 8.1). Returns None when it may.

    Refused: an empty license; outside the test split a non-commercial license (token NC) or one not on
    TRAIN_LICENSES; a row whose `use` list contains "eval" unless split is "test"; a row whose own `split` field
    differs from `split`. Evaluation code calls this with split "test".
    """
    where = row.get("path", "?")
    check_license(row.get("license"), split, where)
    if "eval" in (row.get("use") or []) and split != "test":
        raise LicenseError(f"{where}: evaluation-only row in split {split!r}")
    if row.get("split") != split:
        raise LicenseError(f"{where}: row split {row.get('split')!r} loaded as {split!r}")


def check_rir_row(row, split):
    """The gate for room rows (data/rir/*.jsonl), whose fields differ from the audio manifests.

    A measured room row carries a license and no split field (real_train.jsonl is train only by its file name):
    its license goes through check_license. A simulated room row (made by dataprep, no license) must carry the
    split being loaded.
    """
    where = f"room {row.get('name') or row.get('room') or row.get('row', '?')}"
    if "license" in row:
        check_license(row["license"], split, where)
    elif row.get("split") != split:
        raise LicenseError(f"{where}: simulated room of split {row.get('split')!r} loaded as {split!r}")


def read_checked_rirs(manifest_path, split):
    """All rows of a room manifest, each passed through check_rir_row (the first bad row raises)."""
    rows = list(read_manifest(manifest_path))
    for row in rows:
        check_rir_row(row, split)
    return rows


def read_checked(manifest_path, split):
    """All rows of a manifest, each passed through check_row (the first bad row raises)."""
    rows = []
    for row in read_manifest(manifest_path):
        check_row(row, split)
        rows.append(row)
    return rows


_DISJOINT_CHECKED = set()


def person_id(speaker):
    """Speaker id -> one id per person: 'librittsr:1089' and 'librispeech:1089' both give 'libri:1089'
    (SAME_PEOPLE_PREFIX); every other id is returned unchanged."""
    prefix, sep, rest = str(speaker).partition(":")
    return f"{SAME_PEOPLE_PREFIX[prefix]}:{rest}" if sep and prefix in SAME_PEOPLE_PREFIX else str(speaker)


def check_disjoint(manifests=MANIFESTS):
    """Assert that the train, val and test speaker sets do not intersect (10-ml 8.1). Done once per process.
    Speakers are compared by person_id, so the same person under two corpus prefixes counts as one."""
    if str(manifests) in _DISJOINT_CHECKED:
        return
    groups = {"train": ["speech_train"], "val": ["speech_val"], "test": TEST_SPEAKER_MANIFESTS}
    speakers = {}
    for name, files in groups.items():
        found = set()
        for f in files:
            path = manifests / f"{f}.jsonl"
            if path.exists():
                found |= {person_id(r["speaker"]) for r in read_manifest(path) if r.get("speaker")}
        speakers[name] = found
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        both = speakers[a] & speakers[b]
        assert not both, f"{len(both)} speakers are in both {a} and {b}, e.g. {sorted(both)[:3]}"
    _DISJOINT_CHECKED.add(str(manifests))


# ---------------------------------------------------------------------------------------------------------------
# Audio reads (the only place that opens audio files)
# ---------------------------------------------------------------------------------------------------------------
def read_row(path):
    """Whole file at DATA/path -> mono float32 at 16 kHz."""
    audio, sr = sf.read(str(DATA / path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {SR}")
    return np.ascontiguousarray(audio[:, 0])


def read_part(path, start, n):
    """Up to n samples from sample `start` of DATA/path (fewer at the end of the file), mono float32."""
    audio, sr = sf.read(str(DATA / path), start=int(start), frames=int(n), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {SR}")
    return np.ascontiguousarray(audio[:, 0])


def read_crop(path, start, n):
    """Exactly n samples from sample `start` of DATA/path, zero-padded at the end if the file is shorter."""
    part = read_part(path, start, n)
    if len(part) < n:
        part = np.concatenate([part, np.zeros(n - len(part), dtype=np.float32)])
    return part


# ---------------------------------------------------------------------------------------------------------------
# Own-voice split and clip recipe (10-ml 6.3)
# ---------------------------------------------------------------------------------------------------------------
def own_split(rows, durations):
    """Split one speaker's rows (manifest order) into (cache_rows, mix_rows, in_pool).

    cache_rows = the shortest prefix whose durations reach 8 s, never including the last row; mix_rows = the rest.
    in_pool is True when the cache rows reach 8 s and the mix rows hold at least 4 s.
    """
    rows = list(rows)
    n_cache = 0
    cache_total = 0.0
    while cache_total < OWN_CACHE_SECONDS and n_cache < len(rows) - 1:
        cache_total += float(durations[rows[n_cache]])
        n_cache += 1
    cache, mix = rows[:n_cache], rows[n_cache:]
    mix_total = sum(float(durations[r]) for r in mix)
    in_pool = cache_total >= OWN_CACHE_SECONDS and mix_total >= OWN_MIX_MIN_SECONDS
    return cache, mix, in_pool


def make_own_clip(spk, c, row_paths, audios=None):
    """Own clip c of speaker index spk from its cache rows -> (clip f32 [128000], segments, gain).

    Recipe (10-ml 6.3), with rng = default_rng([2026, spk, c]): fade every cache row 5 ms at both ends, join them in
    order with U(0.10, 0.30) s of zeros between rows, take one 8.0 s window at a random start (the 8 clips may
    overlap), then scale so the active power (dsp.active_power with dsp.hop_vad) is -26 dBFS.
    row_paths = [(row index, relative path)] of the cache rows; audios = the same rows already read (optional).
    segments = [(row, src_start, length, dst_start)]: clip[dst:dst+len] = faded row[src:src+len] before the gain.
    """
    rng = np.random.default_rng([OWN_SEED, int(spk), int(c)])
    if audios is None:
        audios = [read_row(path) for _, path in row_paths]
    faded = [dsp.fade(a, FADE) for a in audios]
    # Lay the rows out one after another with a random gap before every row but the first.
    starts = []
    pos = 0
    for k, a in enumerate(faded):
        if k > 0:
            pos += int(round(rng.uniform(OWN_GAP_S[0], OWN_GAP_S[1]) * SR))
        starts.append(pos)
        pos += len(a)
    total = pos
    win = int(rng.integers(0, max(0, total - OWN_CLIP_LEN) + 1))   # window [win, win + 128000)
    clip = np.zeros(OWN_CLIP_LEN, dtype=np.float32)
    segments = []
    for (row, _), a, p in zip(row_paths, faded, starts):
        lo = max(p, win)
        hi = min(p + len(a), win + OWN_CLIP_LEN)
        if hi > lo:
            segments.append((int(row), lo - p, hi - lo, lo - win))
            clip[lo - win:hi - win] = a[lo - p:hi - p]
    power = dsp.active_power(clip, dsp.hop_vad(clip))
    gain = np.float32(np.sqrt(10.0 ** (LEVEL_DBFS / 10.0) / power)) if power > 0 else np.float32(1.0)
    return clip * gain, segments, gain


def clip_from_segments(segments, gain, row_paths):
    """Rebuild an own clip from its saved recipe. row_paths maps row index -> relative path."""
    clip = np.zeros(OWN_CLIP_LEN, dtype=np.float32)
    for row, src, length, dst in segments:
        a = dsp.fade(read_row(row_paths[row]), FADE)          # the fade depends on the whole row, so read it all
        clip[dst:dst + length] = a[src:src + length]
    return clip * np.float32(gain)


# ---------------------------------------------------------------------------------------------------------------
# Sources: one split, loaded once per process
# ---------------------------------------------------------------------------------------------------------------
class Sources:
    """Everything the renderer reads for one split ("train", "val" or "test").

    Speech (per row, small arrays so the object stays cheap):
      paths[r] (relative path), dur[r] (seconds, float32), spk_of[r] (speaker index, int32);
      speakers (sorted speaker ids; the index is also the AAM class), rows_of[s] (int32 rows in manifest order).
    Pools: target_pool, own_pool (int32 speaker indices), own_cache_rows[s], own_mix_rows[s], target_fallback.
    Noise: noise_paths, noise_dur, noise_lic and noise_rows[kind] for kind in demand, fsd50k, musan (use "noise")
      and music (use "music").
    Rooms: sim_dp [rooms, 8], sim_in_hand, sim_rt60, n_sim; for train also real_dp, real_rt60, real_room_rows,
      real_db_rows (rows per room_id and per db). The .npy banks open lazily (self.sim, self.real) as memory maps.
    noise_licenses (optional set) keeps only noise rows with those licenses (the test vectors use CC0 noise).
    """

    def __init__(self, split, manifests=MANIFESTS, rir_dir=RIR_DIR, noise_licenses=None, check_speakers=True):
        self.split = split
        self.manifests = manifests
        self.rir_dir = rir_dir
        if check_speakers:
            check_disjoint(manifests)

        # Speech rows (every row passes the license gate, or loading stops with LicenseError).
        rows = read_checked(manifests / f"speech_{split}.jsonl", split)
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

        # Pools (contract 8.1).
        target = [s for s, g in enumerate(self.rows_of)
                  if len(g) >= TARGET_MIN_ROWS and self.spk_seconds[s] >= TARGET_MIN_SECONDS]
        self.target_pool = np.array(target, dtype=np.int32)
        # Fallback after 20 failed target draws: the pool speaker with the most audio (10-ml 8.3 step 4).
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

        # Noise and music rows.
        noise = read_checked(manifests / f"noise_{split}.jsonl", split)
        if noise_licenses is not None:
            noise = [r for r in noise if r["license"] in noise_licenses]
        self.noise_paths = [r["path"] for r in noise]
        self.noise_dur = np.array([r["duration"] for r in noise], dtype=np.float32)
        self.noise_lic = [r["license"] for r in noise]
        self.noise_rows = {"demand": [], "fsd50k": [], "musan": [], "music": []}
        for i, r in enumerate(noise):
            if "music" in r["use"]:
                self.noise_rows["music"].append(i)
            elif "noise" in r["use"] and r["source"] in self.noise_rows:
                self.noise_rows[r["source"]].append(i)
        self.noise_rows = {k: np.array(v, dtype=np.int32) for k, v in self.noise_rows.items()}

        # Room tables (the big .npy banks open lazily, see the sim and real properties).
        sim_rows = read_checked_rirs(rir_dir / f"sim_{split}.jsonl", split)
        self.n_sim = len(sim_rows)
        self.sim_dp = np.array([r["dp"] for r in sim_rows], dtype=np.int32)          # [rooms, 8]
        self.sim_in_hand = np.array([r["geometry"] == "in_hand" for r in sim_rows])
        self.sim_rt60 = np.array([r["rt60"] for r in sim_rows], dtype=np.float32)
        self.has_real = split == "train" and (rir_dir / "real_train.jsonl").exists()
        if self.has_real:
            real_rows = read_checked_rirs(rir_dir / "real_train.jsonl", split)
            self.n_real = len(real_rows)
            self.real_dp = np.array([r["dp"] for r in real_rows], dtype=np.int32)
            self.real_rt60 = np.array([r["rt60"] for r in real_rows], dtype=np.float32)
            self.real_room = [r["room_id"] for r in real_rows]
            self.real_db = [r["db"] for r in real_rows]
            self.real_room_rows, self.real_db_rows = {}, {}
            for i, r in enumerate(real_rows):
                self.real_room_rows.setdefault(r["room_id"], []).append(i)
                self.real_db_rows.setdefault(r["db"], []).append(i)
        else:
            self.n_real = 0
        self._sim = None
        self._real = None
        self._own = None

    # -- memory-mapped banks (opened on first use inside each worker; dropped when pickled) ----------------------
    @property
    def sim(self):
        """float16 [rooms, 8, 17600] memory map of data/rir/sim_<split>.npy."""
        if self._sim is None:
            self._sim = np.load(self.rir_dir / f"sim_{self.split}.npy", mmap_mode="r")
        return self._sim

    @property
    def real(self):
        """float16 [325, 24000] memory map of data/rir/real_train.npy (train only)."""
        if not self.has_real:
            raise ValueError(f"split {self.split!r} has no real RIR bank (real rooms are train only)")
        if self._real is None:
            self._real = np.load(self.rir_dir / "real_train.npy", mmap_mode="r")
        return self._real

    def open_banks(self):
        """Open the memory maps now (called by the DataLoader worker_init)."""
        _ = self.sim
        if self.has_real:
            _ = self.real

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_sim"] = None      # memory maps are reopened in the new process, never pickled
        state["_real"] = None
        state["_own"] = None
        return state

    # -- own-voice cache (10-ml 6.3) ----------------------------------------------------------------------------
    def own_cache(self):
        """The arrays of ml/cache/own_v1.npz as a dict, or None if the file does not exist (val, test, or not built).

        Only the train split has a cache, and it must still match the manifest (check_own_cache).
        """
        if self._own is None:
            if self.split != "train" or not OWN_CACHE_PATH.exists():
                self._own = False
            else:
                with np.load(OWN_CACHE_PATH) as z:
                    own = {k: z[k] for k in z.files}
                self.check_own_cache(own)
                self._own = own
        return self._own if self._own is not False else None

    def check_own_cache(self, own):
        """Raise ValueError unless the cache was built from this manifest (10-ml 6.3 leak guard, about 3 ms).

        The clips are rebuilt from saved row indices (seg_row), so a speech_train.jsonl regenerated with the same
        speakers but another row order could silently build own enrollments from the wrong rows, even from the own
        talker's mix rows. Checked: the same speaker list and own pool, and every segment of clip c of speaker s
        (seg_clip = 8 s + c) comes from one of speaker s's cache rows.
        """
        fix = "; rebuild it with `.venv/bin/python -m ml.tools.cache_own_embeddings`"
        if list(own["speakers"]) != self.speakers:
            raise ValueError(f"{OWN_CACHE_PATH} was built for another speaker list{fix}")
        if not np.array_equal(own["in_pool"], self.in_own_pool):
            raise ValueError(f"{OWN_CACHE_PATH} has another own-voice pool than the manifest{fix}")
        is_cache_row = np.zeros(len(self.paths), dtype=bool)
        for rows in self.own_cache_rows:
            is_cache_row[rows] = True
        seg_row = own["seg_row"].astype(np.int64)
        if seg_row.min(initial=0) < 0 or seg_row.max(initial=0) >= len(self.paths):
            raise ValueError(f"{OWN_CACHE_PATH} points at rows outside speech_train.jsonl{fix}")
        right_speaker = self.spk_of[seg_row] == own["seg_clip"] // OWN_CLIPS
        if not (right_speaker.all() and is_cache_row[seg_row].all()):
            bad = int(np.sum(~(right_speaker & is_cache_row[seg_row])))
            raise ValueError(f"{OWN_CACHE_PATH}: {bad} clip segments are not cache rows of their own speaker (the "
                             f"manifest changed since the cache was built){fix}")

    def own_embedding(self, spk, c):
        """Cached WeSpeaker vector e_o [256] of own clip c of speaker spk, or None without a cache."""
        own = self.own_cache()
        if own is None:
            return None
        if not own["in_pool"][spk]:
            raise ValueError(f"speaker {self.speakers[spk]} is not in the own-voice pool")
        return own["emb"][spk, c].astype(np.float32)

    def cache_row_paths(self, spk):
        """[(row index, relative path)] of the cache rows of speaker spk, in manifest order."""
        return [(int(r), self.paths[r]) for r in self.own_cache_rows[spk]]

    def build_clip(self, spk, c):
        """The exact own clip c of speaker spk (f32 [128000], -26 dBFS active power), as used for e_o.

        With the train cache it is rebuilt from the saved recipe (seg_* arrays and gain); otherwise (val, test, or
        before the cache exists) the recipe is recomputed with make_own_clip. Both give the same samples.
        """
        if not self.in_own_pool[spk]:
            raise ValueError(f"speaker {self.speakers[spk]} is not in the own-voice pool")
        own = self.own_cache()
        if own is None:
            clip, _, _ = make_own_clip(spk, c, self.cache_row_paths(spk))
            return clip
        clip_id = spk * OWN_CLIPS + c
        lo, hi = np.searchsorted(own["seg_clip"], [clip_id, clip_id + 1])
        segments = zip(own["seg_row"][lo:hi], own["seg_src"][lo:hi], own["seg_len"][lo:hi], own["seg_dst"][lo:hi])
        return clip_from_segments(list(segments), own["gain"][spk, c], self.paths)


# One Sources object per (split, noise license filter) per process: the DataLoader workers each build their own
# copy on first use instead of receiving it pickled from the main process.
_SOURCES = {}


def get_sources(split, noise_licenses=None):
    """The process-wide Sources for this split (built on first call)."""
    key = (split, None if noise_licenses is None else tuple(sorted(noise_licenses)))
    if key not in _SOURCES:
        _SOURCES[key] = Sources(split, noise_licenses=noise_licenses)
    return _SOURCES[key]


def rig_rows():
    """Phone-rig recordings for the fine-tune recipe (10-ml 8.10). Not built in this version.

    Without data/rig/manifest.jsonl the FT recipe simply has 0% rig rows (returns None). If the file exists, this
    raises, because using rig rows is not implemented yet.
    """
    if not RIG_MANIFEST.exists():
        return None
    raise NotImplementedError(f"{RIG_MANIFEST} exists, but rig recordings (10-ml 8.10) are not implemented in "
                              "ml/earshot/sources.py yet; move the file away or implement 8.10 before fine-tuning")
