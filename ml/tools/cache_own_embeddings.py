"""Build the own-voice cache ml/cache/own_v1.npz (10-ml 6.3, contract 6 "Training match" and 8.1).

    .venv/bin/python -m ml.tools.cache_own_embeddings            # all 3,135 own-pool speakers, 10-20 min on MPS
    .venv/bin/python -m ml.tools.cache_own_embeddings --limit 20 --out /tmp/own_test.npz   # quick check

For every own-pool speaker (sorted speaker ids; this index over all 3,758 speakers is also the AAM class) and clip
c in 0..7, sources.make_own_clip builds one clean 8 s clip from the speaker's cache rows (rows joined with
U(0.10, 0.30) s gaps and 5 ms fades, one 8 s window at a random start, -26 dBFS active power, generator
default_rng([2026, speaker index, c])). WeSpeaker embeds the clips in batches of 32 on MPS.

Saved arrays (np.savez, no pickle needed to read):
    emb      f32 [3758, 8, 256]  unit-length WeSpeaker vectors (zeros for speakers outside the pool)
    in_pool  bool [3758]
    speakers str [3758]          sorted speaker ids of speech_train.jsonl
    seg_clip, seg_row, seg_src, seg_len, seg_dst  int32 [S]  the clip recipes: clip id (spk * 8 + c), row in
             speech_train.jsonl, start sample in that file, length, start sample in the clip (sorted by clip id)
    gain     f32 [3758, 8]       the gain that brought each clip to -26 dBFS active power
sources.Sources.build_clip(spk, c) rebuilds the exact samples from these arrays (the loader's token path).
"""
import os

# Same MPS settings as training (10-ml 1.1), set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")

import argparse      # noqa: E402
import time          # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

from ml.earshot.sources import (OWN_CACHE_PATH, OWN_CLIPS, Sources, clip_from_segments,  # noqa: E402
                                make_own_clip, read_row)
from ml.earshot.wespeaker import WeSpeaker, cosine   # noqa: E402

EXPECTED_POOL = 3135        # contract 8.1, measured 2026-11-07


def embed(encoder, clips, device):
    """WeSpeaker vectors of a list of equal-length clips -> numpy float32 [n, 256]."""
    with torch.no_grad():
        x = torch.from_numpy(np.stack(clips)).to(device)
        return encoder(x).cpu().numpy().astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the own-voice WeSpeaker cache (10-ml 6.3).")
    parser.add_argument("--out", default=str(OWN_CACHE_PATH), help="output .npz (default ml/cache/own_v1.npz)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32, help="clips per WeSpeaker batch (default 32)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N pool speakers (quick check)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.limit and out.resolve() == OWN_CACHE_PATH.resolve():
        parser.error("--limit writes an incomplete cache; give --out somewhere else")

    t0 = time.time()
    src = Sources("train")
    n_spk = len(src.speakers)
    pool = [int(s) for s in src.own_pool]
    print(f"{n_spk} speakers, own pool {len(pool)} (expected {EXPECTED_POOL})", flush=True)
    if args.limit:
        pool = pool[:args.limit]
    encoder = WeSpeaker.load().to(args.device)

    emb = np.zeros((n_spk, OWN_CLIPS, 256), dtype=np.float32)
    gain = np.zeros((n_spk, OWN_CLIPS), dtype=np.float32)
    in_pool = np.zeros(n_spk, dtype=bool)
    in_pool[pool] = True
    segs = []                      # (clip id, row, src, len, dst)
    pending, pending_ids = [], []
    for done, s in enumerate(pool, start=1):
        row_paths = src.cache_row_paths(s)
        audios = [read_row(path) for _, path in row_paths]      # read once, shared by the 8 clips
        for c in range(OWN_CLIPS):
            clip, segments, g = make_own_clip(s, c, row_paths, audios)
            gain[s, c] = g
            segs.extend((s * OWN_CLIPS + c, row, a, n, b) for row, a, n, b in segments)
            pending.append(clip)
            pending_ids.append((s, c))
            if len(pending) == args.batch:
                for (si, ci), vec in zip(pending_ids, embed(encoder, pending, args.device)):
                    emb[si, ci] = vec
                pending, pending_ids = [], []
        if done % 200 == 0 or done == len(pool):
            elapsed = time.time() - t0
            print(f"  {done}/{len(pool)} speakers, {elapsed:.0f} s, about "
                  f"{elapsed / done * (len(pool) - done):.0f} s left", flush=True)
    if pending:
        for (si, ci), vec in zip(pending_ids, embed(encoder, pending, args.device)):
            emb[si, ci] = vec

    seg = np.array(segs, dtype=np.int64)
    arrays = {"emb": emb, "in_pool": in_pool, "speakers": np.array(src.speakers),
              "seg_clip": seg[:, 0].astype(np.int32), "seg_row": seg[:, 1].astype(np.int32),
              "seg_src": seg[:, 2].astype(np.int32), "seg_len": seg[:, 3].astype(np.int32),
              "seg_dst": seg[:, 4].astype(np.int32), "gain": gain}
    assert np.all(np.diff(arrays["seg_clip"]) >= 0), "segments must be sorted by clip id"

    # Checks: unit vectors inside the pool, zeros outside; three clips rebuilt from the saved recipe are
    # bit-identical to the embedded ones, and their CPU embedding matches the cached MPS vector.
    norms = np.linalg.norm(emb[in_pool], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms {norms.min()}..{norms.max()}"
    assert not np.any(emb[~in_pool]), "speakers outside the pool must have zero vectors"
    cpu = WeSpeaker.load()
    rng = np.random.default_rng(0)
    for s in rng.choice(pool, size=min(3, len(pool)), replace=False):
        c = int(rng.integers(OWN_CLIPS))
        clip_id = int(s) * OWN_CLIPS + c
        pick = arrays["seg_clip"] == clip_id
        rebuilt = clip_from_segments(list(zip(arrays["seg_row"][pick], arrays["seg_src"][pick],
                                              arrays["seg_len"][pick], arrays["seg_dst"][pick])),
                                     gain[s, c], src.paths)
        direct, _, _ = make_own_clip(int(s), c, src.cache_row_paths(int(s)))
        assert np.array_equal(rebuilt, direct), "recipe rebuild differs from the embedded clip"
        cos = cosine(cpu.embed_numpy(rebuilt), emb[s, c])
        print(f"  check {src.speakers[s]} clip {c}: rebuild identical, CPU vs {args.device} cosine {cos:.6f}")
        assert cos > 0.9999

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): in_pool.sum() = {int(in_pool.sum())}, emb "
          f"{emb.shape}, {len(seg)} segments, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
