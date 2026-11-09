"""Training losses for ESB-16-DF (spec 10-ml section 9, contract 8.2). Everything runs in float32 on MPS or CPU.

Notation per example i of a batch (sums run over the samples of the 4 s or 8 s segment):
  x = mixture (batch key `mix`), s = reference (`ref`, zeros when the target is absent), y = model output.

| Term    | What it rewards                                                    | Weight | Applies when            |
|---------|--------------------------------------------------------------------|--------|-------------------------|
| tsnr    | y close to s, capped at -30 dB so easy examples stop pulling       | 1      | present                 |
| mrstft  | spectra of y and s match at 3 resolutions (linear and log)         | 1 (*)  | present                 |
| absent  | y quiet when nobody we want is talking                             | 1      | not present             |
| gap     | y quiet in the pauses of the target (target VAD 0)                 | 0.1    | present, >= 0.1 s pause |
| own     | y quiet while only the user's own voice talks                      | 0.3    | own voice in the mix    |
| oversup | y never weaker than s in any time-frequency bin (no eaten speech)  | 0.3    | present, stages B/C/FT  |
| bce_t/o | the two heads predict the target VAD and the own-voice VAD         | 0.1    | all                     |
| aam     | the enrollment voiceprint g_t identifies the speaker (ArcFace)     | 0.05   | Person rows             |

Two rules keep a mixed batch (Everyone rows, rows without own voice, absent rows) free of NaN:
1. A term is switched on or off per example with torch.where(applies, term, 0), never by multiplying with a 0/1 flag.
   0 * NaN is NaN, but torch.where simply ignores the unused value. Its gradient is a different story: torch.where
   sends a zero gradient into the unused branch, and zero times an infinite local derivative is NaN again. So every
   term below is written to stay finite with a finite derivative for every input, including all-zero signals
   (that is what the small epsilons are for).
2. Masked means divide by max(count, 1), so an empty mask gives 0 instead of 0 / 0.

(*) mrstft keeps the spec weight 1 (DECISIONS L 2026-11-08, "Loss A/B outcome"). After the spec-loss pilot failed
its old gate, an A/B from scratch on the same data and seed (paired held-out set) tried two other losses: tsnr_only
(mrstft at weight 0) and comp03 (a compressed spectral loss). At step 4,000 both were about 0.2 dB ahead of the spec
loss, but the acceptance test at step 6,000 tied (tsnr_only vs the spec-loss pilot, +0.06 dB, 95% bootstrap CI -0.06
to +0.13), so the spec loss stays. The weight-0 path is kept for the A/B harness's tsnr_only variant: when the weight
is 0, total_loss computes mrstft under torch.no_grad(), so the value still goes to train.csv but no graph is built,
which saves the backward pass through its three STFTs. That update is bit-identical to keeping the term in the graph
at weight 0 (that only adds exact zeros to the gradient).

total = mean over the B examples of (sum over terms of weight * torch.where(applies, term, 0)). The AAM term is
computed only on the Person rows (picked out by index), so an Everyone row and its spk_t = -1 never reach
cross_entropy; with no Person row in the batch the AAM term is exactly 0.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import G_DIM, HOP
from .dsp import hann_np

# ---------------------------------------------------------------------------------------------------------------
# Constants (contract 8.2 and 10-ml 9)
# ---------------------------------------------------------------------------------------------------------------
EPS = 1e-8                    # keeps every log and division finite
TSNR_CAP = 1e-3               # the 1e-3 |s|^2 floor in the tsnr denominator: a perfect output scores -30 dB
ABSENT_FLOOR = 10 ** -2.5     # absent form 10 log10(|y|^2 + 10^-2.5 |x|^2): no reward beyond 25 dB below the mix
MIN_MASKED_SAMPLES = 1600     # gap and own need at least 0.1 s (1,600 samples) of masked audio
MRSTFT_SIZES = (256, 512, 1024)   # FFT sizes of the multi-resolution STFT loss, hop = size / 4
LOG_EPS = 1e-5                # log(|STFT| + 1e-5) in the log-magnitude part of mrstft
OVERSUP_FFT = 512             # over-suppression STFT size ...
OVERSUP_HOP = 128             # ... and hop
OVERSUP_POW = 0.3             # |S|^0.3 - |Y|^0.3 compression
MAG_EPS = 1e-8                # (|X| + 1e-8)^0.3: the derivative of |X|^0.3 is infinite at |X| = 0 (see oversup)
N_CLASSES = 3758              # AAM speaker classes = every speech_train speaker (contract 8.1)
AAM_MARGIN = 0.2              # additive angular margin m (radians)
AAM_SCALE = 30.0              # logit scale
SIN_FLOOR = 1e-7              # sqrt(clamp(1 - cos^2, 1e-7, 1)): sqrt has an infinite derivative at exactly 0

# mrstft weight 1, the spec value (DECISIONS L 2026-11-08, "Loss A/B outcome"; contract 8.2). The A/B's tsnr_only arm
# (weight 0) led at step 4,000 but tied at the step-6,000 acceptance test, so the spec loss stays.
WEIGHTS = {"tsnr": 1.0, "mrstft": 1.0, "absent": 1.0, "gap": 0.1, "own": 0.3, "oversup": 0.3,
           "bce_t": 0.1, "bce_o": 0.1, "aam": 0.05}
TERMS = tuple(WEIGHTS)        # the logging order of 10-ml 10.3 (train.csv columns)


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------
def select(applies, term):
    """Per-example switch: term where applies is True, exactly 0 elsewhere (torch.where, never term * flag)."""
    return torch.where(applies, term, torch.zeros_like(term))


def energy(x):
    """Sum of squares over the last axis: [B, L] -> [B]."""
    return (x * x).sum(dim=-1)


def hop_energy(x):
    """Sum of squares of every 64-sample hop: [B, L] -> [B, L / 64]. Hop j holds samples 64j .. 64j + 63, the same
    hop the VAD label j describes (10-ml 8.4)."""
    B, L = x.shape
    if L % HOP != 0:
        raise ValueError(f"signal length {L} is not a multiple of the {HOP}-sample hop")
    return (x * x).reshape(B, L // HOP, HOP).sum(dim=-1)


_HANN_CACHE = {}


def hann(n, device):
    """Periodic Hann window of length n as float32 on `device` (dsp.hann_np, cached per size and device)."""
    key = (n, str(device))
    if key not in _HANN_CACHE:
        _HANN_CACHE[key] = torch.from_numpy(hann_np(n).astype(np.float32)).to(device)
    return _HANN_CACHE[key]


def stft_mag(x, n_fft, hop):
    """Magnitude STFT used by mrstft and oversup: [B, L] -> [B, frames, n_fft / 2 + 1].

    The signal gets n_fft / 2 zeros on both sides (so the first and last samples sit in the middle of a frame), is cut
    into frames of n_fft samples every `hop` samples, multiplied by a periodic Hann window and sent through rfft.
    unfold + rfft is used instead of torch.stft so the framing is visible here and runs the same on MPS and CPU.
    abs() of a complex number has a zero gradient at exactly 0 in PyTorch (checked on MPS and CPU), so all-zero
    frames are safe.
    """
    pad = n_fft // 2
    frames = F.pad(x, (pad, pad)).unfold(-1, n_fft, hop)          # [B, frames, n_fft]
    return torch.fft.rfft(frames * hann(n_fft, x.device), dim=-1).abs()


# ---------------------------------------------------------------------------------------------------------------
# The loss terms, one value per example ([B] tensors)
# ---------------------------------------------------------------------------------------------------------------
def tsnr(y, s):
    """Thresholded SNR: -10 log10(|s|^2 / (|s - y|^2 + 1e-3 |s|^2 + eps) + eps).

    Lower is better. A perfect output gives -10 log10(1 / 1e-3) = -30 dB, the cap: once the error is 30 dB below the
    reference, the example stops pulling on the gradient. With s = 0 (absent rows, where the term is switched off)
    the value is -10 log10(eps) = 80 and the derivative is 0, so torch.where stays clean.
    """
    ref_energy = energy(s)
    err_energy = energy(s - y)
    ratio = ref_energy / (err_energy + TSNR_CAP * ref_energy + EPS)
    return -10.0 * torch.log10(ratio + EPS)


def absent(y, x):
    """Absent form: 10 log10(|y|^2 + 10^-2.5 |x|^2 + eps). Pushes the output energy down, but only until it is 25 dB
    below the mixture (the 10^-2.5 |x|^2 floor), so silence is not chased forever."""
    return 10.0 * torch.log10(energy(y) + ABSENT_FLOOR * energy(x) + EPS)


def masked_absent(y, x, hop_mask):
    """The absent form on only the hops where hop_mask [B, T] (bool) is True, using masked means.

    value = 10 log10(mean_masked(y^2) + 10^-2.5 mean_masked(x^2) + eps), where mean_masked sums y^2 over the masked
    hops' samples and divides by max(count, 1). Returns (value [B], count [B]) with count = number of masked samples,
    so the caller can require the 0.1 s minimum. Means instead of sums only shift the value by 10 log10(count), which
    has no gradient, and make examples with long and short pauses comparable in the logs.
    """
    n_samples = hop_mask.sum(dim=-1).to(y.dtype) * HOP
    count = n_samples.clamp(min=1.0)
    zero = torch.zeros((), dtype=y.dtype, device=y.device)
    mean_y = torch.where(hop_mask, hop_energy(y), zero).sum(dim=-1) / count
    mean_x = torch.where(hop_mask, hop_energy(x), zero).sum(dim=-1) / count
    value = 10.0 * torch.log10(mean_y + ABSENT_FLOOR * mean_x + EPS)
    return value, n_samples


def mrstft(y, s):
    """Multi-resolution STFT loss: for n in (256, 512, 1024), hop n / 4, Hann, with A = |STFT_n(y)| and
    R = |STFT_n(s)|: mean(|A - R|) + mean(|log(A + 1e-5) - log(R + 1e-5)|), summed over the three sizes.
    Means are over the frames and bins of each example. Identical signals give exactly 0."""
    total = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    for n_fft in MRSTFT_SIZES:
        A = stft_mag(y, n_fft, n_fft // 4)
        R = stft_mag(s, n_fft, n_fft // 4)
        linear = (A - R).abs().mean(dim=(1, 2))
        log = (torch.log(A + LOG_EPS) - torch.log(R + LOG_EPS)).abs().mean(dim=(1, 2))
        total = total + linear + log
    return total


def oversup(y, s):
    """Over-suppression: mean(ReLU(|S|^0.3 - |Y|^0.3)^2) with S, Y = STFT 512 / 128 (Hann) of s and y.

    Only bins where the output is weaker than the reference count (ReLU), so an output that is louder costs nothing
    here. Magnitudes are compressed as (|X| + 1e-8)^0.3: the plain |X|^0.3 has the derivative 0.3 |X|^-0.7, which is
    infinite at |X| = 0 and would turn into NaN through torch.where. The 1e-8 changes values by less than 1e-8^0.3
    (0.004 on magnitudes that are typically 1e-3 to 10) and identical signals still give exactly 0.
    """
    S = stft_mag(s, OVERSUP_FFT, OVERSUP_HOP)
    Y = stft_mag(y, OVERSUP_FFT, OVERSUP_HOP)
    short = torch.relu((S + MAG_EPS) ** OVERSUP_POW - (Y + MAG_EPS) ** OVERSUP_POW)
    return (short * short).mean(dim=(1, 2))


def heads_bce(head_logits, vad_t, vad_o):
    """Binary cross-entropy of the two heads against the 4 ms labels, mean over hops: -> (bce_t [B], bce_o [B]).

    head_logits [B, T, 2] are before the sigmoid ([..., 0] = p_target, [..., 1] = p_self); vad_t and vad_o are float
    0/1 [B, T]. binary_cross_entropy_with_logits applies the sigmoid itself in a numerically safe way.
    """
    bce_t = F.binary_cross_entropy_with_logits(head_logits[..., 0], vad_t, reduction="none").mean(dim=-1)
    bce_o = F.binary_cross_entropy_with_logits(head_logits[..., 1], vad_o, reduction="none").mean(dim=-1)
    return bce_t, bce_o


# ---------------------------------------------------------------------------------------------------------------
# Speaker-ID head (training only, not exported)
# ---------------------------------------------------------------------------------------------------------------
class AAMHead(nn.Module):
    """Additive angular margin softmax (ArcFace) on the token-encoder voiceprint g_t (contract 8.2).

    Each of the 3,758 training speakers has a learned direction (a row of `weight`, 128 numbers). The logit of class
    c is 30 cos(theta_c), the cosine between g and that row, except that the true class uses cos(theta + 0.2): the
    voiceprint must beat every other speaker by an angle of 0.2 rad, which packs each speaker's voiceprints tightly.
    Standard ArcFace guard: when theta + m would pass pi (cos < cos(pi - m)), cos(theta + m) stops falling, so that
    region uses cos - sin(pi - m) m instead, which keeps falling.
    """

    def __init__(self, n_classes=N_CLASSES, dim=G_DIM, margin=AAM_MARGIN, scale=AAM_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, dim))     # [3758, 128] = 481,024 parameters
        nn.init.xavier_uniform_(self.weight)
        self.n_classes = n_classes
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)                 # below this, theta + m > pi
        self.fallback = math.sin(math.pi - margin) * margin

    def logits(self, g, labels):
        """g [N, 128], labels [N] int64 in 0..n_classes-1 -> scaled logits [N, n_classes]."""
        cos = F.normalize(g, dim=-1) @ F.normalize(self.weight, dim=-1).T     # [N, C], cosines
        cos = cos.clamp(-1.0, 1.0)                                            # rounding can give 1.0000001
        cos_y = cos.gather(1, labels[:, None]).squeeze(1)                     # cosine of the true class, [N]
        sin_y = torch.sqrt(torch.clamp(1.0 - cos_y * cos_y, SIN_FLOOR, 1.0))
        phi = cos_y * self.cos_m - sin_y * self.sin_m                         # cos(theta + m)
        phi = torch.where(cos_y < self.threshold, cos_y - self.fallback, phi)
        classes = torch.arange(self.n_classes, device=g.device)
        is_label = classes[None, :] == labels[:, None]                        # [N, C] bool one-hot
        return self.scale * torch.where(is_label, phi[:, None], cos)

    def forward(self, g, labels):
        """Cross-entropy per row, [N] (reduction 'none', so the caller decides how to average). N may be 0."""
        return F.cross_entropy(self.logits(g, labels), labels, reduction="none")


# ---------------------------------------------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------------------------------------------
def batch_tensor(batch, name, device, dtype):
    """One batch field as a torch tensor on `device` with `dtype`.

    Works for the numpy batch dict of 10-ml 8.7, a dict of tensors, or an object with attributes (train.py's b.mix).
    """
    value = batch[name] if isinstance(batch, dict) else getattr(batch, name)
    return torch.as_tensor(value).to(device=device, dtype=dtype)


def total_loss(y, head_logits, g_t, batch, aam, oversup_on, rows=None):
    """The training loss of contract 8.2 for one batch.

    y [B, L] model output, head_logits [B, T, 2] (before the sigmoid, T = L / 64), g_t [B, 128] target voiceprint
    from encode_enrollment, batch = the batch of 10-ml 8.7 (keys used: mix, ref, vad_t, vad_o, present, own_in_mix,
    person, spk_t), aam = an AAMHead on the same device, oversup_on = True from stage B on (recipes B, C, FT).

    Returns (total, parts). total is a scalar tensor with gradients. parts maps every name in TERMS to a detached
    0-dim tensor: the term's mean over the examples where it applies (0 when it applies to none), unweighted, so the
    log reads in the term's own units (tsnr in dB, and so on). They stay on the device; train.py syncs them every
    50 steps.
    rows (optional dict): if given, it receives for every name in TERMS the number of examples the term applied to
    (a detached 0-dim tensor on the device). train.py uses it to average a sparse term (absent, gap, own) over the
    examples where it applied in its 50-step log window; a plain mean over steps would count the batches where the
    term applied to no row as 0 and log it too low.
    """
    y = y.float()
    head_logits = head_logits.float()
    g_t = g_t.float()
    device = y.device
    x = batch_tensor(batch, "mix", device, torch.float32)
    s = batch_tensor(batch, "ref", device, torch.float32)
    vad_t = batch_tensor(batch, "vad_t", device, torch.float32)
    vad_o = batch_tensor(batch, "vad_o", device, torch.float32)
    present = batch_tensor(batch, "present", device, torch.bool)
    own_in_mix = batch_tensor(batch, "own_in_mix", device, torch.bool)
    person = batch_tensor(batch, "person", device, torch.bool)
    spk_t = batch_tensor(batch, "spk_t", device, torch.int64)
    B = y.shape[0]
    if y.shape != x.shape or head_logits.shape[:2] != vad_t.shape:
        raise ValueError(f"shape mismatch: y {tuple(y.shape)}, mix {tuple(x.shape)}, "
                         f"head_logits {tuple(head_logits.shape)}, vad_t {tuple(vad_t.shape)}")

    # Masks at hop resolution (the VAD labels are float 0/1, one per 64-sample hop).
    target_silent = vad_t < 0.5                       # pauses of the target (or everything, if absent)
    own_only = (vad_o > 0.5) & target_silent          # the user talks and the target does not
    every_row = torch.ones(B, dtype=torch.bool, device=device)

    # Each entry: name -> (value per example [B], applies per example [B] bool).
    terms = {}
    terms["tsnr"] = (tsnr(y, s), present)
    if WEIGHTS["mrstft"] == 0.0:
        # Weight 0 (the A/B harness's tsnr_only variant; the spec weight is 1): logged only. no_grad builds no graph
        # for the three STFTs. The forward value is the same, and the gradient loses only the exact zeros that weight 0
        # would have sent through mrstft.
        with torch.no_grad():
            terms["mrstft"] = (mrstft(y, s), present)
    else:
        terms["mrstft"] = (mrstft(y, s), present)
    terms["absent"] = (absent(y, x), ~present)
    gap_value, gap_samples = masked_absent(y, x, target_silent)
    terms["gap"] = (gap_value, present & (gap_samples >= MIN_MASKED_SAMPLES))
    own_value, own_samples = masked_absent(y, x, own_only)
    terms["own"] = (own_value, own_in_mix & (own_samples >= MIN_MASKED_SAMPLES))
    if oversup_on:
        terms["oversup"] = (oversup(y, s), present)
    else:
        # Stage A: the term is off for every example (a fixed setting, not data), so it is not even computed.
        terms["oversup"] = (torch.zeros(B, device=device), torch.zeros(B, dtype=torch.bool, device=device))
    bce_t, bce_o = heads_bce(head_logits, vad_t, vad_o)
    terms["bce_t"] = (bce_t, every_row)
    terms["bce_o"] = (bce_o, every_row)

    per_example = torch.zeros(B, device=device)
    parts = {}
    for name, (value, applies) in terms.items():
        chosen = select(applies, value)
        per_example = per_example + WEIGHTS[name] * chosen
        n_rows = applies.sum().clamp(min=1)
        parts[name] = (chosen.sum() / n_rows).detach()
        if rows is not None:
            rows[name] = applies.sum().detach()
    total = per_example.mean()

    # Speaker ID on the Person rows only. Indexing picks those rows, so an Everyone row (spk_t = -1) never reaches
    # cross_entropy. Its share of the batch mean is weight * (sum over Person rows) / B, the same as averaging
    # torch.where(person, ce, 0) over all B rows. With no Person row the sum of an empty tensor is exactly 0.
    ce = aam(g_t[person], spk_t[person])
    total = total + WEIGHTS["aam"] * ce.sum() / B
    parts["aam"] = (ce.sum() / person.sum().clamp(min=1)).detach()
    if rows is not None:
        rows["aam"] = person.sum().detach()
    return total, parts
