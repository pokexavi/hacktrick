"""
Find referee whistles in a match's audio with plain signal processing, no learning.

A referee's whistle is a loud, nearly pure tone at about 2.5-4.5 kHz that holds its pitch. Each
spectrogram frame is scored on how tonal and loud its peak is, runs of good frames become blows, and
close blows are merged into one whistle.

    python -m dead_alive_detection.audio.whistle --video data/broadcast/BAR-ATH_broadcast.mp4 --preset broadcast --source broadcast --out cache/audio/BAR-ATH_broadcast_whistles.csv
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

SR = 16000
WIN = 512                   # 32 ms
HOP = 160                   # 10 ms: 100 frames a second
BAND = (1800.0, 5000.0)     # Hz: where the peak is looked for
PEAK_BINS = 3               # +-3 bins = +-94 Hz around the peak: a pea's trill, a pealess whistle's
                            # three close tones
BLOCK = 10                  # frames per background block (0.1 s)
BACKGROUND_S = 5.0          # background: median over this many seconds of blocks
CHUNK = 20000               # frames per FFT chunk (bounds the memory)
PROM_BINS = 16              # local prominence: the peak against the median of +-16 bins (+-500 Hz)
                            # around it, the peak's own +-PEAK_BINS left out

FEATURES = ("conc", "snr", "prom", "lconc")
DEFAULTS = {"min_conc": 0.35, "min_snr": 10.0, "min_prom": -np.inf, "min_lconc": -np.inf, "min_dur_s": 0.08, "max_dur_s": 4.0, "max_gap_s": 0.05,
            "max_spread_hz": 350.0, "group_s": 0.5, "pitch_lo_hz": -np.inf, "pitch_hi_hz": np.inf,
            **{f"core_{f}": -np.inf for f in FEATURES}}
# Settings for a real track. "broadcast": chosen on BAR-ATH's broadcast audio without any ground
# truth: of every pitch band, only 3.3-4.0 kHz gathers whistles at the starts of the stoppages a
# dead-ball model predicts (2-4x chance; the other bands are voice, music and crowd, ~1x), so the
# peak is looked for there; the thresholds trade how many whistles are found against how many fall
# within 3 s of a predicted start (0.4 a minute of play, ~45% of them, ~4.5x chance).
PRESETS = {"broadcast": {"band": BAND, "search": (3200.0, 4100.0),
                         "params": {"min_conc": 0.35, "min_prom": 20.0, "min_snr": 20.0}}}
COLUMNS = ["start_s", "end_s", "freq_hz", "spread_hz", "conc", "snr_db", "blows", "duration_s"]


def _band_power(frames, lo, hi):
    """
    Power of the band's bins for a chunk of frames.
    """
    spec = np.fft.rfft(frames * np.hanning(WIN).astype(np.float32), axis=1)[:, lo:hi]
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32) + 1e-12


def frame_features(x, sr=SR, band=BAND, search=None):
    """
    Per-frame features: t, peak, conc, snr, prom and lconc. The peak is looked for in search
    (default the whole band), while conc is measured over the whole band.
    """
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    x = np.asarray(x, np.float32)
    if len(x) < WIN:
        return {k: np.zeros(0) for k in ("t", "peak") + FEATURES}
    frames = sliding_window_view(x, WIN)[::HOP]
    n = len(frames)
    freqs = np.fft.rfftfreq(WIN, 1 / sr)
    lo, hi = np.searchsorted(freqs, band)
    width = hi - lo
    s0, s1 = (0, width) if search is None else tuple(np.searchsorted(freqs, search) - lo)
    if not 0 <= s0 < s1 <= width:
        raise ValueError(f"search band {search} is not inside band {band}")
    keep = np.r_[np.arange(PROM_BINS - PEAK_BINS), np.arange(PROM_BINS + PEAK_BINS + 1, 2 * PROM_BINS + 1)]

    # Pass 1: block means of each band bin's log power, then their running median = background.
    n_blocks = -(-n // BLOCK)
    blocks = np.zeros((n_blocks, width), np.float32)
    for c in range(0, n, CHUNK):
        db = 10 * np.log10(_band_power(frames[c:c + CHUNK], lo, hi))
        b0 = c // BLOCK
        m = len(db) // BLOCK * BLOCK
        blocks[b0:b0 + m // BLOCK] = db[:m].reshape(-1, BLOCK, width).mean(1)
        if m < len(db):
            blocks[b0 + m // BLOCK] = db[m:].mean(0)
    size = max(1, int(BACKGROUND_S * sr / HOP / BLOCK) | 1)
    background = median_filter(blocks, size=(size, 1), mode="nearest")

    # Pass 2: peak, concentration and level above background, frame by frame.
    peak = np.empty(n, np.float32)
    conc = np.empty(n, np.float32)
    snr = np.empty(n, np.float32)
    prom = np.empty(n, np.float32)
    lconc = np.empty(n, np.float32)
    rows = np.arange(n)
    for c in range(0, n, CHUNK):
        p = _band_power(frames[c:c + CHUNK], lo, hi)
        r = np.arange(len(p))
        k = s0 + p[:, s0:s1].argmax(1)
        padded = np.pad(p, ((0, 0), (PEAK_BINS, PEAK_BINS)))
        near = sliding_window_view(padded, 2 * PEAK_BINS + 1, axis=1)[r, k].sum(1)
        conc[c:c + len(p)] = near / p.sum(1)
        db = 10 * np.log10(p)
        snr[c:c + len(p)] = db[r, k] - background[rows[c:c + len(p)] // BLOCK, k]
        around = sliding_window_view(np.pad(db, ((0, 0), (PROM_BINS, PROM_BINS)), mode="edge"),
                                     2 * PROM_BINS + 1, axis=1)[r, k][:, keep]
        prom[c:c + len(p)] = db[r, k] - np.median(around, axis=1)
        local = sliding_window_view(np.pad(p, ((0, 0), (PROM_BINS, PROM_BINS))),
                                    2 * PROM_BINS + 1, axis=1)[r, k].sum(1)
        lconc[c:c + len(p)] = near / local
        # Parabola through the log power of the peak and its neighbours: sub-bin frequency.
        km, kp = np.clip(k - 1, 0, width - 1), np.clip(k + 1, 0, width - 1)
        a, b, g = db[r, km], db[r, k], db[r, kp]
        denom = a - 2 * b + g
        inner = (k > 0) & (k < width - 1) & (denom < 0)
        shift = np.where(inner, 0.5 * (a - g) / np.where(inner, denom, -1.0), 0.0)
        peak[c:c + len(p)] = freqs[lo + k] + shift * (freqs[1] - freqs[0])
    t = (rows * HOP + WIN / 2) / sr
    return {"t": t, "peak": peak, "conc": conc, "snr": snr, "prom": prom, "lconc": lconc}


def _runs(mask):
    """
    (start, end) index pairs of the True runs of a bool array.
    """
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def blows(feat, params=None):
    """
    Single blows: one row per kept run of candidate frames.
    """
    p = {**DEFAULTS, **(params or {})}
    unknown = set(p) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown whistle parameters {sorted(unknown)}")
    cand = np.ones(len(feat["t"]), bool)
    core = np.ones(len(feat["t"]), bool)
    for f in FEATURES:
        if np.isfinite(p[f"min_{f}"]):
            cand &= feat[f] >= p[f"min_{f}"]
        if np.isfinite(p[f"core_{f}"]):
            core &= feat[f] >= p[f"core_{f}"]
    gap = int(round(p["max_gap_s"] * SR / HOP))
    # Bridge short gaps: a frame lost to a spike of noise does not split a blow.
    filled = cand.copy()
    for a, b in _runs(~cand):
        if 0 < a and b < len(cand) and b - a <= gap:
            filled[a:b] = True
    rows = []
    step = HOP / SR
    for a, b in _runs(filled):
        dur = (b - a) * step
        if not p["min_dur_s"] <= dur <= p["max_dur_s"]:
            continue
        sel = np.arange(a, b)[cand[a:b]]
        if not core[sel].any():
            continue
        f = feat["peak"][sel]
        spread = float(np.percentile(f, 90) - np.percentile(f, 10))
        if spread > p["max_spread_hz"]:
            continue
        if not p["pitch_lo_hz"] <= float(np.median(f)) <= p["pitch_hi_hz"]:
            continue
        rows.append((feat["t"][a] - step / 2, feat["t"][b - 1] + step / 2, float(np.median(f)),
                     spread, float(feat["conc"][sel].mean()), float(feat["snr"][sel].mean())))
    return pd.DataFrame(rows, columns=["start_s", "end_s", "freq_hz", "spread_hz", "conc", "snr_db"])


def whistles(feat, params=None):
    """
    Whistles: blows closer than group_s merged into one.
    """
    p = {**DEFAULTS, **(params or {})}
    b = blows(feat, p)
    if b.empty:
        return pd.DataFrame(columns=COLUMNS)
    group = np.r_[0, np.cumsum(b["start_s"].values[1:] - b["end_s"].values[:-1] > p["group_s"])]
    out = b.groupby(group).agg(start_s=("start_s", "min"), end_s=("end_s", "max"),
                               freq_hz=("freq_hz", "median"), spread_hz=("spread_hz", "max"),
                               conc=("conc", "mean"), snr_db=("snr_db", "max"),
                               blows=("start_s", "size"))
    out["duration_s"] = out["end_s"] - out["start_s"]
    return out.reset_index(drop=True)[COLUMNS]


def referee_pitch(found, half_width_hz=200.0):
    """
    Pitch window around the median pitch of a strict pass's whistles, so a second pass keeps only
    the referee's whistle.
    """
    if not len(found):
        return -np.inf, np.inf
    f0 = float(np.median(found["freq_hz"]))
    return f0 - half_width_hz, f0 + half_width_hz


def detect(x, sr=SR, params=None, band=BAND, search=None):
    """
    Whistles of an audio signal.
    """
    return whistles(frame_features(x, sr, band, search), params)


def match(found, truth, tolerance_s=0.3):
    """
    Match found whistles to true ones by start time, within tolerance_s. Returns (tp, fp, fn, onset
    errors).
    """
    used, errors = set(), []
    starts = found["start_s"].to_numpy() if len(found) else np.zeros(0)
    for s in truth["start_s"]:
        d = np.abs(starts - s)
        for i in np.argsort(d):
            if d[i] > tolerance_s:
                break
            if i not in used:
                used.add(i)
                errors.append(starts[i] - s)
                break
    tp = len(used)
    return tp, len(found) - tp, len(truth) - tp, np.array(errors)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract(video, wav):
    """
    Write a video's first audio track as 16 kHz mono WAV, unless it already exists.
    """
    if not os.path.exists(wav):
        os.makedirs(os.path.dirname(wav) or ".", exist_ok=True)
        subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", video,
                        "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
                        wav], check=True)
    return wav


def read_wav(path):
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.ndim == 2:
        x = x.mean(1)
    return sr, (x / 32768.0).astype(np.float32)


def summary(found, seconds):
    """
    Printable summary of a whistle table: counts, rates, pitch and duration.
    """
    lines = [f"{len(found)} whistles in {seconds / 60:.1f} min ({len(found) / seconds * 60:.2f} a min)"]
    if len(found):
        q = lambda c: " / ".join(f"{v:.2f}" for v in np.percentile(found[c], [10, 50, 90]))  # noqa: E731
        lines += [f"  pitch Hz p10/50/90: {q('freq_hz')}", f"  duration s: {q('duration_s')}",
                  f"  snr dB: {q('snr_db')}   conc: {q('conc')}",
                  f"  blows: {found['blows'].value_counts().sort_index().to_dict()}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, help="video whose first audio track is read")
    ap.add_argument("--wav", default=None, help="extracted audio (default cache/audio/<stem>.wav)")
    ap.add_argument("--out", required=True, help="whistle table (CSV)")
    ap.add_argument("--frames", default=None, help="also save per-frame features (npz)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="band, search band and thresholds of a real track (overrides --band, --search)")
    ap.add_argument("--source", choices=("tactical", "broadcast"), default=None,
                    help="BAR-ATH audio track: adds each whistle's tactical video frame (audio/align.py)")
    ap.add_argument("--band", type=float, nargs=2, default=BAND, metavar=("LO", "HI"),
                    help=f"analysis band, Hz (default {BAND[0]:.0f} {BAND[1]:.0f})")
    ap.add_argument("--search", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="where the peak is looked for, inside --band (default: all of it)")
    args = ap.parse_args(argv)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    wav = extract(args.video, args.wav or os.path.join("cache", "audio", f"{stem}_mono16k.wav"))
    sr, x = read_wav(wav)
    preset = PRESETS.get(args.preset, {"band": tuple(args.band),
                                       "search": args.search and tuple(args.search), "params": {}})
    feat = frame_features(x, sr, preset["band"], preset["search"])
    found = whistles(feat, preset["params"])
    if args.source:
        from dead_alive_detection.audio.align import to_tactical
        found["frame"] = to_tactical(found["start_s"], args.source)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    found.to_csv(args.out, index=False, float_format="%.3f")
    if args.frames:
        np.savez_compressed(args.frames, **feat)
    print(summary(found, len(x) / sr))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
