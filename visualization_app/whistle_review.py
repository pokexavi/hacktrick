"""Helpers of the Whistles page: referee-whistle candidates on BAR-ATH's broadcast audio, where they
fall against a dead-ball model's stoppages, and short clips with sound to listen to them.

Times: the master clock is the tactical video's frame (the submission's index). The detector runs on
the broadcast's audio; dead_alive_detection/audio/align.py maps between the two.
"""
import base64
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from dead_alive_detection.audio import align, whistle

FPS = 25
PLAYED = ((11185, 90651), (113742, 190197))    # graded play: kickoff to final whistle, per half
CATEGORIES = ("near start", "near stop", "inside dead", "inside alive", "not graded")


def load_features(wav_path, out_path, band, search):
    """Per-frame whistle features of the broadcast audio (whistle.frame_features), computed once per
    (band, search) and kept in out_path."""
    if out_path.exists():
        feat = dict(np.load(out_path))
        if all(k in feat for k in ("t", "peak") + whistle.FEATURES):
            return feat
    sr, x = whistle.read_wav(str(wav_path))
    feat = whistle.frame_features(x, sr, tuple(band), tuple(search))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **feat)
    return feat


def detections(feat, params):
    """Whistle table (whistle.COLUMNS) plus each whistle's tactical frame (of its start)."""
    found = whistle.whistles(feat, params)
    found["frame"] = align.to_tactical(found["start_s"].to_numpy(), "broadcast")
    found["end_frame"] = align.to_tactical(found["end_s"].to_numpy(), "broadcast")
    return found


def played(frames):
    frames = np.asarray(frames)
    return np.any([(frames >= a) & (frames <= b) for a, b in PLAYED], axis=0)


def categorise(frames, intervals, window_s=3.0):
    """(categories, seconds to the nearest predicted start) of whistle frames against a model's
    stoppages (intervals: start, stop frames). A whistle within window_s of a start is 'near start',
    else of a stop 'near stop', else inside a stoppage or in play."""
    frames = np.asarray(frames, np.int64)
    starts, stops = intervals["start"].to_numpy(), intervals["stop"].to_numpy()
    if len(starts) == 0:
        to_start = np.full(len(frames), np.inf)
        to_stop = to_start
        inside = np.zeros(len(frames), bool)
    else:
        to_start = (frames - starts[np.abs(frames[:, None] - starts[None]).argmin(1)]) / FPS
        to_stop = (frames - stops[np.abs(frames[:, None] - stops[None]).argmin(1)]) / FPS
        inside = ((frames[:, None] >= starts[None]) & (frames[:, None] < stops[None])).any(1)
    cat = np.where(np.abs(to_start) <= window_s, "near start",
                   np.where(np.abs(to_stop) <= window_s, "near stop",
                            np.where(inside, "inside dead", "inside alive")))
    cat = np.where(played(frames), cat, "not graded")
    return cat, to_start


def summary(frames, categories, intervals, window_s=3.0):
    """How a threshold setting does against a model, without any ground truth: whistles per minute
    of play, the share near a predicted start against what chance would give, and how many
    predicted starts have a whistle within window_s."""
    in_play = categories != "not graded"
    minutes = sum(b - a + 1 for a, b in PLAYED) / FPS / 60
    starts = intervals["start"].to_numpy()
    covered = np.zeros(PLAYED[-1][1] + 1, bool)
    for s in starts:
        covered[max(s - int(window_s * FPS), 0):s + int(window_s * FPS)] = True
    total = sum(b - a + 1 for a, b in PLAYED)
    chance = sum(covered[a:b + 1].sum() for a, b in PLAYED) / total
    near = float((categories[in_play] == "near start").mean()) if in_play.any() else 0.0
    frames = np.asarray(frames)
    hit = sum(bool((np.abs(frames - s) <= window_s * FPS).any()) for s in starts)
    return {"whistles": int(in_play.sum()), "per_min": in_play.sum() / minutes, "near_start": near,
            "chance": chance, "enrichment": near / chance if chance else np.nan,
            "starts_hit": hit, "starts": len(starts)}


def starts_without_whistle(intervals, frames, window_s=3.0):
    """The model's stoppages with no whistle within window_s of their start (to check what the
    threshold misses: fouls should have one, throw-ins and goal kicks usually not)."""
    frames = np.asarray(frames)
    keep = [not (np.abs(frames - s) <= window_s * FPS).any() for s in intervals["start"]]
    return intervals[np.array(keep, bool)] if len(intervals) else intervals


def near_misses(feat, params, passing, relax=0.75):
    """Whistles found with every threshold relaxed (x relax) that the real setting drops."""
    loose = {k: (v * relax if k in ("min_snr", "min_prom", "min_conc") else v) for k, v in params.items()}
    found = detections(feat, loose)
    if len(passing) == 0:
        return found
    taken = np.abs(found["start_s"].to_numpy()[:, None] - passing["start_s"].to_numpy()[None]).min(1) < 0.3
    return found[~taken].reset_index(drop=True)


def unmatched(table, other, tolerance_s=0.5):
    """Rows of table (start_s) with no row of other starting within tolerance_s: whistles one
    detector finds and the other does not."""
    if len(table) == 0 or len(other) == 0:
        return table.reset_index(drop=True)
    gap = np.abs(table["start_s"].to_numpy()[:, None] - other["start_s"].to_numpy()[None]).min(1)
    return table[gap > tolerance_s].reset_index(drop=True)


def variant_setting(name, preset_found):
    """(search band, params) of a compare_whistles variant, with its pitch window (pitch-locked
    variants) taken from the preset's whistles, as compare_whistles.detect_variant does."""
    from dead_alive_detection.audio import compare_whistles
    spec = compare_whistles.VARIANTS[name]
    params = dict(spec["params"])
    if "lock" in spec:
        params["pitch_lo_hz"], params["pitch_hi_hz"] = whistle.referee_pitch(preset_found, spec["lock"][0])
    return tuple(int(f) for f in spec["search"]), params


def spectrogram_png(wav_path, start_s, seconds, fmin=1500.0, fmax=5500.0, max_cols=None):
    """(base64 PNG, fmin, fmax) of the broadcast audio's spectrogram over [start_s, start_s +
    seconds): time left to right, high pitch at the top, magma colours, contrast per clip.
    max_cols: at most this many columns (10 ms each), max-pooled in time (long clips)."""
    from matplotlib import colormaps
    from PIL import Image
    from scipy.io import wavfile
    sr, x = wavfile.read(str(wav_path), mmap=True)
    a = max(int(start_s * sr), 0)
    seg = np.asarray(x[a:a + int(seconds * sr)], np.float32) / 32768.0
    if seg.ndim == 2:
        seg = seg.mean(1)
    if len(seg) < whistle.WIN:
        return None, fmin, fmax
    frames = np.lib.stride_tricks.sliding_window_view(seg, whistle.WIN)[::whistle.HOP]
    spec = np.abs(np.fft.rfft(frames * np.hanning(whistle.WIN), axis=1)) ** 2
    freqs = np.fft.rfftfreq(whistle.WIN, 1 / sr)
    lo, hi = np.searchsorted(freqs, (fmin, fmax))
    spec = spec[:, lo:hi]
    if max_cols and len(spec) > max_cols:
        k = -(-len(spec) // max_cols)
        spec = spec[:len(spec) // k * k].reshape(-1, k, spec.shape[1]).max(1)
    db = 10 * np.log10(spec + 1e-12)
    low, high = np.percentile(db, [5, 99.7])
    img = colormaps["magma"](np.clip((db - low) / max(high - low, 1e-6), 0, 1).T[::-1])
    buf = io.BytesIO()
    Image.fromarray((img[:, :, :3] * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), float(freqs[lo]), float(freqs[hi - 1])


def av_clip(image_path, image_start_s, audio_path, audio_start_s, seconds, highpass_hz=None):
    """Base64 MP4 of `seconds` of one video's picture with another's (or the same) sound track, in
    memory only. highpass_hz: keep only the sound above it (makes a whistle stand out)."""
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
               "-ss", f"{image_start_s:.3f}", "-i", str(image_path),
               "-ss", f"{audio_start_s:.3f}", "-i", str(audio_path),
               "-t", f"{seconds:.3f}", "-map", "0:v:0", "-map", "1:a:0",
               "-vf", "scale=1280:-2", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "25", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
    if highpass_hz:
        command += ["-af", f"highpass=f={highpass_hz}:poles=2,highpass=f={highpass_hz}:poles=2,volume=4"]
    command += ["-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    if not result.stdout:
        raise ValueError("No video decoded. Check the video paths and the time window.")
    return base64.b64encode(result.stdout).decode("ascii")


def curves(feat, start_s, seconds, step=4):
    """Per-frame features over a broadcast time window, max-pooled over `step` frames (10 ms each):
    t (broadcast s), snr, prom, conc, peak."""
    t = feat["t"]
    a, b = np.searchsorted(t, (start_s, start_s + seconds))
    m = (b - a) // step * step
    if m <= 0:
        return {k: [] for k in ("t", "snr", "prom", "conc", "peak")}
    pool = lambda v: v[a:a + m].reshape(-1, step).max(1)  # noqa: E731
    out = {"t": t[a:a + m].reshape(-1, step).mean(1)}
    for k in ("snr", "prom", "conc"):
        out[k] = pool(feat[k])
    out["peak"] = feat["peak"][a:a + m].reshape(-1, step)[np.arange(m // step),
                                                          feat["snr"][a:a + m].reshape(-1, step).argmax(1)]
    return {k: np.round(v.astype(float), 3).tolist() for k, v in out.items()}


def mmss(frame):
    """'m:ss' of a tactical frame."""
    s = int(frame) // FPS
    return f"{s // 60}:{s % 60:02d}"


# --- a whistle model against its counterfactual (dead_alive_detection/audio/whistle_influence.py) --

def whistle_info(folder):
    """whistle.json of a model folder ({counterfactual, kind, whistles}), or None."""
    path = Path(folder) / "whistle.json"
    return json.loads(path.read_text()) if path.exists() else None


def influence(folder):
    """whistle_influence.csv of a model folder, or None."""
    path = Path(folder) / "whistle_influence.csv"
    return pd.read_csv(path) if path.exists() else None


def match_rows(frames, table, tolerance=12):
    """Index into table (its 'frame' column) of the row within tolerance frames of each frame,
    else -1: joins the page's detections to the influence table of the preset whistles."""
    frames = np.asarray(frames, np.int64)
    if table is None or not len(table) or not len(frames):
        return np.full(len(frames), -1)
    other = table["frame"].to_numpy(np.int64)
    gap = np.abs(frames[:, None] - other[None])
    nearest = gap.argmin(1)
    return np.where(gap[np.arange(len(frames)), nearest] <= tolerance, nearest, -1)


def verdict(row):
    """One line about what a whistle does to the model, from a whistle_influence.csv row."""
    text = (f"Without the whistle: **{row['category_no_whistle']}** → with it: **{row['category']}**. "
            f"p_dead over the next 3 s {row['p_dead_after_no_whistle']:.2f} → {row['p_dead_after']:.2f} "
            f"(Δ {row['d_p_dead']:+.2f}); p_start up to {row['d_p_start_max']:+.2f}, "
            f"p_stop up to {row['d_p_stop_max']:+.2f}. Stoppages: **{row['effect']}**.")
    if pd.notna(row.get("decision_whistle")):
        text += (f" Decoder: the whistle gives {row['decision_whistle']:+.1f} of the margin "
                 f"{row['decision_margin']:.1f} for keeping the {'stoppage' if row['decision_state'] == 'D' else 'alive gap'}"
                 f" (without it {row['decision_margin_no_whistle']:.1f}"
                 + (": **the whistle decides it**)." if row["decisive"] else ")."))
    return text


def window_frames(frames_table, start, end, step=5):
    """{t (s from start), p_dead, p_start, p_stop} of frames.csv over [start, end), every step
    frames (the heads are 5 Hz)."""
    part = frames_table.iloc[start:end:step]
    out = {"t": [round((f - start) / FPS, 3) for f in part["frame"]]}
    for k in ("p_dead", "p_start", "p_stop"):
        if k in part:
            out[k] = [None if not np.isfinite(v) else round(float(v), 3) for v in part[k].to_numpy(float)]
    return out
