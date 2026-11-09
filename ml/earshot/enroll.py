"""Target and own-voice enrollments for one example (10-ml 8.5, contract 8.1 rows "Target enrollment" and
"Own enrollment").

Both functions take the scene that render.render_example returned and keep using the same example generator, so
the enrollments are as deterministic as the mixture.

Target enrollment (Person examples, present or absent): other utterances of the target (the rows E of 8.3 step 4,
never used by the mixture, never cycled), in the same room (50%), another room (30%) or dry (20%), with a weaker
talker (30%), noise (80%, half of those with babble), the mixture's phone filters, its own level, and a mask that is
oracle (40%), es_vad (40%) or all ones (20%).

Own enrollment (own enrollment given, 90%): one of the speaker's 8 cached clean 8 s clips (sources.build_clip; its
WeSpeaker vector e_o comes from the cache). Only the token path is augmented: noise at 15-30 dB in 30%, the phone
filters, a level U(-40, -15) dBFS and self-noise; the mask is es_vad, as on the phone.
"""
import numpy as np

from . import dsp
from .config import MAX_O_TOK, MAX_T_TOK, SR
from .render import (FADE, NOISE_KINDS, NOISE_PROBS, TEST_LEVEL_DBFS, babble_mix, chain_filters, choose,
                     convolve, headroom_gain, level_gain, noise_clip, pick_distinct, pick_room, self_noise, set_power,
                     speech_stream)
from .sources import OWN_CLIP_LEN, read_row
from .vad import es_vad, num_frames, token_groups

ENR_GAP_S = (0.1, 0.4)        # gaps between the enrollment rows


def join_rows(src, rows, rng):
    """The rows read whole, faded 5 ms at both ends, joined in order with U(0.1, 0.4) s of zeros between them."""
    pieces = []
    for k, row in enumerate(rows):
        if k > 0:
            pieces.append(np.zeros(int(round(rng.uniform(ENR_GAP_S[0], ENR_GAP_S[1]) * SR)), dtype=np.float32))
        pieces.append(dsp.fade(read_row(src.paths[row]), FADE))
    return np.concatenate(pieces).astype(np.float32)


def groups_or_ones(mask, max_tokens):
    """token_groups(mask, max_tokens); if fewer than 4 groups are available, the all-ones mask is used instead.

    Returns (idx int64 [max_tokens, 3], valid bool [max_tokens], short) where short says the fallback was used
    (the loader's stats.short_enroll count).
    """
    idx, valid, n_avail = token_groups(mask, max_tokens)
    if n_avail >= 4:
        return idx, valid, False
    idx, valid, _ = token_groups(np.ones(len(mask), dtype=np.uint8), max_tokens)
    return idx, valid, True


def render_enrollment(src, scene, rng, Le, test_mode=False):
    """Target enrollment of a Person example -> dict enr_t f32 [Le], tok_idx_t, tok_val_t, short_t, info."""
    d = scene["draws"]
    chain = scene["chain"]
    rec = [] if "sources" in scene else None      # record_sources mode: list every file used

    # Dry: the enrollment rows E joined with gaps, one Le window (E holds >= Le + 0.5 s, so nothing repeats).
    joined = join_rows(src, scene["enr_rows"], rng)
    start = int(rng.integers(0, max(0, len(joined) - Le) + 1))
    dry = np.zeros(Le, dtype=np.float32)
    piece = joined[start:start + Le]
    dry[:len(piece)] = piece

    # Room: the mixture's room, another room (sim 80% / real 20%, position 0), or dry.
    if d.enr_room == "same":
        room = scene["room"]
    elif d.enr_room == "other":
        room = pick_room(src, "real" if (d.enr_other_real and src.has_real) else "sim", rng)
    else:
        room = pick_room(src, "dry", rng)
    clean = convolve(dry, room.h[0])
    p_clean = dsp.active_power(clean, dsp.hop_vad(clean))
    x = clean.copy()

    # Weaker talker: one of this example's competing talkers (50%, when one exists) or a fresh speaker, position 1
    # of the enrollment room, 5-15 dB below the clean enrollment.
    used = set(scene["used"])
    if d.enr_weak:
        if d.enr_weak_from_mix and scene["talkers"]:
            spk = scene["talkers"][int(rng.integers(len(scene["talkers"])))]
        else:
            spk = pick_distinct(len(src.speakers), used, rng)
        weak_rows = [] if rec is not None else None
        weak = convolve(speech_stream(src, src.rows_of[spk], Le, rng, weak_rows), room.h[1])
        p_weak = dsp.active_power(weak, dsp.hop_vad(weak))
        if p_weak > 0:
            weak = weak * np.float32(np.sqrt(p_clean * 10.0 ** (-d.enr_weak_db / 10.0) / p_weak))
        x += weak

    # Noise: one clip drawn as in 8.3, plus babble of 6-12 talkers in half of the noisy enrollments, at
    # SNR U(0, 25) dB against the clean enrollment's active power.
    kind = None
    if d.enr_noise:
        noise, kind = noise_clip(src, choose(rng, NOISE_KINDS, NOISE_PROBS), Le, room, rng, rec)
        p_b = p_clean / 10.0 ** (d.enr_snr_db / 10.0)
        if d.enr_noise_babble:
            speakers = [pick_distinct(len(src.speakers), used, rng) for _ in range(d.enr_n_babble)]
            babble = babble_mix(src, speakers, Le, room, rng, rec)
            w = float(rng.uniform(0.2, 0.8))
            noise = set_power(noise, w * p_b) + set_power(babble, (1.0 - w) * p_b)
        x += set_power(noise, p_b)

    # Mic chain: the mixture's filters (same phone), its own level, self-noise, no clipping.
    level = float(rng.uniform(-40.0, -15.0))
    if test_mode:
        level = TEST_LEVEL_DBFS
    x_f = chain_filters(chain, x)
    gain = headroom_gain(level_gain(level, x_f), x_f)   # same headroom rule as the mixture (DECISIONS L 2026-11-08)
    enr = np.clip(x_f * gain + self_noise(chain, Le, rng), -1.0, 1.0).astype(np.float32)

    # Mask: oracle (es_vad of the clean part after the same chain), es_vad of the final clip, or all ones.
    if d.enr_mask == "oracle":
        mask = es_vad(chain_filters(chain, clean) * gain)
    elif d.enr_mask == "vad":
        mask = es_vad(enr)
    else:
        mask = np.ones(num_frames(Le), dtype=np.uint8)
    idx, valid, short = groups_or_ones(mask, MAX_T_TOK)
    if rec is not None:
        scene["sources"] += [{"role": "target enrollment", "path": src.paths[r], "license": src.lic[r], "row": r}
                             for r in scene["enr_rows"]]
        if d.enr_weak:
            scene["sources"] += [{"role": "enrollment weaker talker", "path": src.paths[r], "license": src.lic[r],
                                  "row": r} for r in sorted({r for r, _, _ in weak_rows})]
        scene["sources"] += [{"role": "enrollment " + role, "path": path, "license": lic} for role, path, lic in rec]
    info = {"enr_room": room.kind if d.enr_room != "same" else "same", "enr_weak": d.enr_weak,
            "enr_noise": kind, "enr_snr_db": round(d.enr_snr_db, 3) if d.enr_noise else None,
            "enr_mask": d.enr_mask, "enr_level_dbfs": round(level, 3), "short_t": short}
    return {"enr_t": enr.astype(np.float32), "tok_idx_t": idx, "tok_val_t": valid, "short_t": short, "info": info}


def render_own_enrollment(src, scene, rng, test_mode=False):
    """Own enrollment -> dict enr_o f32 [128000], tok_idx_o, tok_val_o, e_o (f32 [256] or None without a cache),
    clean (the cached clean clip), short_o, info. Call only when draws.own_given."""
    d = scene["draws"]
    chain = scene["chain"]
    spk, c = scene["own"], d.own_clip
    rec = [] if "sources" in scene else None
    clean = src.build_clip(spk, c)
    e_o = src.own_embedding(spk, c)
    x = clean
    kind = None
    if d.own_noise:
        dry_room = pick_room(src, "dry", rng)
        noise, kind = noise_clip(src, choose(rng, NOISE_KINDS, NOISE_PROBS), OWN_CLIP_LEN, dry_room, rng, rec)
        p_clean = dsp.active_power(clean, dsp.hop_vad(clean))
        x = clean + set_power(noise, p_clean / 10.0 ** (d.own_snr_db / 10.0))
    level = float(rng.uniform(-40.0, -15.0))
    if test_mode:
        level = TEST_LEVEL_DBFS
    x_f = chain_filters(chain, x)
    enr = x_f * level_gain(level, x_f) + self_noise(chain, OWN_CLIP_LEN, rng)
    idx, valid, short = groups_or_ones(es_vad(enr), MAX_O_TOK)
    if rec is not None:
        scene["sources"] += [{"role": "own enrollment", "path": path, "license": src.lic[r], "row": r}
                             for r, path in src.cache_row_paths(spk)]
        scene["sources"] += [{"role": "own enrollment " + role, "path": path, "license": lic}
                             for role, path, lic in rec]
    info = {"own": src.speakers[spk], "own_clip": c, "own_noise": kind, "short_o": short}
    return {"enr_o": enr.astype(np.float32), "tok_idx_o": idx, "tok_val_o": valid, "e_o": e_o, "clean": clean,
            "short_o": short, "info": info}
