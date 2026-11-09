"""Energy VAD for enrollment clips and token grouping (spec 10-ml section 5, contract 2.4).

The C engine's `es_vad` follows the exact same rule, and the test vectors check the two agree with zero mismatches,
so every step below is written the plain way the C code does it (float64 sums in index order, a strict `>`).
"""
import numpy as np

TOK_HOP = 512     # token frames are 512 samples long at hop 512 (32 ms, no overlap)
PAD = 448         # same 448 zeros of left padding as the streaming STFT


def num_frames(n):
    """Number of token frames for a clip of n samples (same as the engine's es_enroll_num_frames).

    0 below 576 samples on purpose (so a non-empty clip always has at least 2 frames), else (n - 64) // 512 + 1.
    Frame f covers padded samples 512f .. 512f + 511, and the last one ends inside the padded clip of n + 448 samples.
    """
    if n < 576:
        return 0
    return (n - 64) // TOK_HOP + 1


def frame_energies_db(pcm):
    """Step 1 and 2 of es_vad: energy in dB of each 512-sample token frame of the padded clip -> float64 [F].

    E_f = 10 log10(sum(seg^2) / 512 + 1e-10), where the sum is a sequential float64 sum in ascending index order
    (np.cumsum adds one element at a time, exactly like the C `double` loop; np.sum would add pairwise).
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    n_frames = num_frames(len(pcm))
    xp = np.concatenate([np.zeros(PAD, dtype=np.float64), pcm.astype(np.float64)])
    energies = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        seg = xp[TOK_HOP * f: TOK_HOP * f + TOK_HOP]
        ss = np.cumsum(seg * seg)[-1]
        energies[f] = 10.0 * np.log10(ss / 512.0 + 1e-10)
    return energies


def threshold_mask(energies):
    """Step 3 of es_vad: frames clearly above the quiet floor -> bool [F] (before dilation).

    P10 = the sorted energy at index (F - 1) // 10 (equal to floor(0.1 (F - 1)) for every F);
    thr = max(P10 + 6, max(E) - 40); a frame is speech if E_f > thr (strict).
    """
    energies = np.asarray(energies, dtype=np.float64)
    n_frames = len(energies)
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    e_sorted = np.sort(energies)
    p10 = e_sorted[(n_frames - 1) // 10]
    thr = max(p10 + 6.0, energies.max() - 40.0)
    return energies > thr


def dilate1(mask):
    """Step 4 of es_vad: m2[f] = m[f-1] | m[f] | m[f+1], with neighbours outside the clip counted as false."""
    mask = np.asarray(mask, dtype=bool)
    n_frames = len(mask)
    out = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        left = mask[f - 1] if f - 1 >= 0 else False
        right = mask[f + 1] if f + 1 < n_frames else False
        out[f] = left or mask[f] or right
    return out


def es_vad(pcm):
    """Speech mask of an enrollment clip: float32 [n] -> uint8 [F], F = num_frames(n) (contract 2.4).

    Returns an empty array at once when F = 0 (below 576 samples).
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    if num_frames(len(pcm)) == 0:
        return np.zeros(0, dtype=np.uint8)
    energies = frame_energies_db(pcm)
    mask = dilate1(threshold_mask(energies))
    return mask.astype(np.uint8)


def token_groups(mask, max_tokens):
    """Which frames make up each voice token (contract 2.4 steps 4 and 5).

    The frames with mask = 1 form one ordered list (not one list per run). That list is cut into groups of 3 from its
    start; a final group of 1 or 2 frames is dropped. If there are more groups than max_tokens (48 target, 16 own),
    groups floor(i * n_avail / max_tokens) for i = 0 .. max_tokens - 1 are kept.

    Returns (idx int64 [max_tokens, 3], valid bool [max_tokens], n_avail). Rows past the kept groups are zero and
    invalid. The caller treats n_avail < 4 as ES_ERR_ENROLL_SHORT.
    """
    kept = np.flatnonzero(np.asarray(mask))          # frame indices with m = 1, in order
    n_avail = len(kept) // 3                         # drop a remainder under 3
    groups = kept[:3 * n_avail].reshape(n_avail, 3)
    if n_avail > max_tokens:
        groups = groups[(np.arange(max_tokens) * n_avail) // max_tokens]   # floor(i * n_avail / max)
    idx = np.zeros((max_tokens, 3), dtype=np.int64)
    valid = np.zeros(max_tokens, dtype=bool)
    idx[:len(groups)] = groups
    valid[:len(groups)] = True
    return idx, valid, n_avail
