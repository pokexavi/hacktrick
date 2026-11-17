"""
Compare whistle-detector variants on BAR-ATH without any ground truth.

    python -m dead_alive_detection.audio.compare_whistles --insert

Each variant runs on the broadcast audio and is scored by how its whistles line up with a reference
model's stoppages, and by how many inserted synthetic whistles it finds. The reference model must
not read whistles, or the comparison is circular.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dead_alive_detection.audio import align, synthetic, whistle

FPS = 25
PLAYED = ((11185, 90651), (113742, 190197))
WAV = Path("cache/audio/BAR-ATH_broadcast_mono16k.wav")
VARIANT_DIR = Path("cache/audio/variants")
PRESET = whistle.PRESETS["broadcast"]
WIDE = (3200.0, 4100.0)
LOCKED = (3530.0, 4030.0)   # the referee's pitch (3780 Hz, median of the preset's whistles) +- 250 Hz

# name -> {"search": band, "params": thresholds, "lock": second pass at the referee's pitch}.
# "lock": (half width Hz, search band of the pitch-locked pass), the window centred on the median
# pitch of the preset's whistles (whistle.referee_pitch): the search band then is f0 +- 250 Hz.
_P = PRESET["params"]
VARIANTS = {
    "preset": {"search": WIDE, "params": dict(_P)},
    # A: lower thresholds
    "A_prom15": {"search": WIDE, "params": {**_P, "min_prom": 15.0}},
    "A_prom15_conc30": {"search": WIDE, "params": {**_P, "min_prom": 15.0, "min_conc": 0.30}},
    "A_prom12_conc25": {"search": WIDE, "params": {**_P, "min_prom": 12.0, "min_conc": 0.25}},
    "A_prom12_conc20": {"search": WIDE, "params": {**_P, "min_prom": 12.0, "min_conc": 0.20}},
    # B: local concentration (lconc) in place of most of the wide-band conc
    "B_lconc60": {"search": WIDE, "params": {**_P, "min_prom": 15.0, "min_conc": 0.20, "min_lconc": 0.6}},
    # D: second pass at the referee's pitch (search f0 +- 250 Hz, blows within f0 +- 200 Hz)
    "D_conc30_prom15_snr15": {"search": LOCKED, "params": {**_P, "min_conc": 0.30, "min_prom": 15.0, "min_snr": 15.0},
                              "lock": (200.0, LOCKED)},
    "D_conc25_prom12": {"search": LOCKED, "params": {**_P, "min_conc": 0.25, "min_prom": 12.0}, "lock": (200.0, LOCKED)},
    "D_conc25_prom12_hyst": {"search": LOCKED, "params": {**_P, "min_conc": 0.25, "min_prom": 12.0, "core_conc": 0.35,
                                                          "core_prom": 18.0}, "lock": (200.0, LOCKED)},
    "D_conc20_prom12": {"search": LOCKED, "params": {**_P, "min_conc": 0.20, "min_prom": 12.0}, "lock": (200.0, LOCKED)},
    "D_conc20_prom12_snr15": {"search": LOCKED, "params": {**_P, "min_conc": 0.20, "min_prom": 12.0, "min_snr": 15.0},
                              "lock": (200.0, LOCKED)},
}


def played(frames):
    frames = np.asarray(frames)
    return np.any([(frames >= a) & (frames <= b) for a, b in PLAYED], axis=0)


def detect_variant(spec, feats):
    """
    Whistle table of one variant, with its tactical frame.
    """
    params = dict(spec["params"])
    if "lock" in spec:
        half, _ = spec["lock"]
        strict = whistle.whistles(feats[WIDE], PRESET["params"])
        params["pitch_lo_hz"], params["pitch_hi_hz"] = whistle.referee_pitch(strict, half)
    found = whistle.whistles(feats[tuple(spec["search"])], params)
    found["frame"] = align.to_tactical(found["start_s"].to_numpy(), "broadcast")
    return found


def measure(frames, freqs, intervals, f0, typed=None, window_s=3.0):
    """
    Dict of the comparison measures of whistle frames against a model's stoppages.
    """
    frames = np.asarray(frames, np.int64)
    keep = played(frames)
    frames, freqs = frames[keep], np.asarray(freqs)[keep]
    n_total = PLAYED[-1][1] + 1
    graded = np.zeros(n_total, bool)
    for a, b in PLAYED:
        graded[a:b + 1] = True
    w = int(window_s * FPS)
    starts, stops = intervals["start"].to_numpy(), intervals["stop"].to_numpy()

    def near_mask(points):
        m = np.zeros(n_total, bool)
        for s in points:
            m[max(s - w, 0):s + w + 1] = True
        return m

    near_all = near_mask(starts)
    if typed is None:
        typed = pd.DataFrame(columns=["start", "type"])
    near_foul = near_mask(typed.loc[typed["type"].eq("whistle"), "start"].to_numpy())
    near_restart = near_mask(typed.loc[typed["type"].isin(["throw_in", "goal_kick", "corner"]), "start"].to_numpy())
    dead = np.zeros(n_total, bool)
    for a, b in zip(starts, stops):
        dead[a:b] = True
    from scipy.ndimage import binary_dilation
    deep_alive = ~binary_dilation(dead, iterations=5 * FPS)
    edges = near_all | near_mask(stops)
    minutes = graded.sum() / FPS / 60
    chance = (near_all & graded).sum() / graded.sum()
    n = len(frames)
    near = int(near_all[frames].sum())

    def excess(mask):
        return int(mask[frames].sum()) - n * (mask & graded).sum() / graded.sum()

    return {"whistles": n, "per_min": n / minutes,
            "near_start": near / n if n else np.nan, "chance": chance,
            "excess": near - n * chance,
            "excess_foul": excess(near_foul), "excess_restart": excess(near_restart),
            "starts_hit": int(sum(bool((np.abs(frames - s) <= w).any()) for s in starts)),
            "deep_alive": int(deep_alive[frames].sum()),
            "inside_dead": int((dead[frames] & ~edges[frames]).sum()),
            "pitch_ok": float(np.mean(np.abs(freqs - f0) <= 100)) if n else np.nan}


def insert_whistles(x, sr, avoid_s, levels_db, pitches, every_s=20.0, seed=0):
    """
    Add synthetic pea whistles to the audio, about one every every_s seconds and away from avoid_s.
    Returns the new audio and a table of what was inserted.
    """
    rng = np.random.default_rng(seed)
    y = x.astype(np.float64).copy()
    rows = []
    avoid = np.sort(np.asarray(avoid_s, float))
    for c in np.arange(30.0, len(x) / sr - 30.0, every_s):
        start = c + rng.uniform(-5, 5)
        if len(avoid) and np.abs(avoid - start).min() < 3.0:
            continue
        level = float(rng.choice(levels_db))
        a = int(start * sr)
        ref = float(np.sqrt(np.mean(x[a - sr:a + sr].astype(np.float64) ** 2)) + 1e-9)
        blows = [rng.uniform(0.15, 1.0)] if rng.uniform() > 0.2 else [rng.uniform(0.12, 0.3)] * 2
        t0, f_used = start, []
        for d in blows:
            f = float(rng.choice(pitches)) + rng.normal(0, 30)
            m = int(d * sr)
            t = np.arange(m) / sr
            ff = f + 80 * np.sin(2 * np.pi * rng.uniform(35, 55) * t)
            ph = 2 * np.pi * np.cumsum(ff) / sr
            sig = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * synthetic._envelope(m, sr // 100, sr // 30)
            s = int(t0 * sr)
            y[s:s + m] += sig / sig.std() * ref * 10 ** (level / 20)
            f_used.append(f)
            t0 += d + 0.15
        rows.append((start, t0 - 0.15, level, float(np.mean(f_used))))
    return np.clip(y, -1, 1).astype(np.float32), pd.DataFrame(rows, columns=["start_s", "end_s", "level_db", "freq_hz"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="ens_seeds3", help="reference BAR folder in cache/dead_alive (no whistle input)")
    ap.add_argument("--types-model", default="tcn_bar_types7_w01",
                    help="BAR folder whose intervals.csv has stoppage types (no whistle input)")
    ap.add_argument("--variants", nargs="*", default=None, help="default: all of VARIANTS")
    ap.add_argument("--insert", action="store_true", help="also measure recall on inserted synthetic whistles")
    ap.add_argument("--levels", type=float, nargs="*", default=[-25, -20, -15, -10, -5])
    ap.add_argument("--out", default="dead_alive_detection/results/whistle_variants")
    args = ap.parse_args(argv)
    names = args.variants or list(VARIANTS)
    sr, x = whistle.read_wav(str(WAV))
    bands = sorted({tuple(VARIANTS[n]["search"]) for n in names} | {WIDE})
    feats = {b: whistle.frame_features(x, sr, whistle.BAND, b) for b in bands}
    intervals = pd.read_csv(Path("cache/dead_alive") / args.model / "intervals.csv")
    typed = pd.read_csv(Path("cache/dead_alive") / args.types_model / "intervals.csv")
    preset = detect_variant(VARIANTS["preset"], feats)
    f0 = float(np.median(preset["freq_hz"]))
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        found = detect_variant(VARIANTS[n], feats)
        found.to_csv(VARIANT_DIR / f"{n}.csv", index=False, float_format="%.3f")
        rows.append({"variant": n, **measure(found["frame"], found["freq_hz"], intervals, f0, typed)})
    table = pd.DataFrame(rows).set_index("variant")
    if args.insert:
        y, truth = insert_whistles(x, sr, preset["start_s"], args.levels, preset["freq_hz"].to_numpy())
        ifeats = {b: whistle.frame_features(y, sr, whistle.BAND, b) for b in bands}
        for n in names:
            found = detect_variant(VARIANTS[n], ifeats)
            starts = found["start_s"].to_numpy()
            hit = np.array([bool(len(starts)) and np.abs(starts - s).min() <= 0.3 for s in truth["start_s"]])
            on_pitch = np.abs(truth["freq_hz"].to_numpy() - f0) <= 200
            for lv in args.levels:
                sel = (truth["level_db"].to_numpy() == lv) & on_pitch
                table.loc[n, f"recall_{int(lv)}"] = hit[sel].mean()
            table.loc[n, "recall_off_pitch"] = hit[~on_pitch].mean()
        table.attrs["inserted"] = len(truth)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(out / "metrics.csv")
    (out / "variants.json").write_text(json.dumps({n: VARIANTS[n] for n in names}, indent=1, default=list))
    pd.set_option("display.width", 200)
    print(f"reference model {args.model}, types {args.types_model}, referee pitch {f0:.0f} Hz"
          + (f", {table.attrs['inserted']} inserted whistles" if args.insert else ""))
    print(table.round(3).to_string())
    print(f"-> {out / 'metrics.csv'}, {VARIANT_DIR}/<variant>.csv")


if __name__ == "__main__":
    main()
