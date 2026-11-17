"""
Made-up match audio with known referee whistles, to test audio/whistle.py without real audio.

A scene is crowd noise, commentary, crowd and fan whistling, plus referee whistles at known times.
The same arguments always give the same scene.
"""

import numpy as np
import pandas as pd

SR = 16000


def _envelope(n, attack, release):
    """
    Amplitude ramp that rises over attack samples and falls over release.
    """
    e = np.ones(n)
    a, r = min(attack, n // 2), min(release, n // 2)
    e[:a] = np.linspace(0, 1, a, endpoint=False)
    if r:
        e[n - r:] = np.linspace(1, 0, r)
    return e


def pink_noise(n, rng):
    """
    Unit-RMS noise with a 1/f spectrum.
    """
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def crowd(n, sr, rng):
    """
    Pink noise whose loudness drifts by a few dB every few seconds.
    """
    t = np.arange(n) / sr
    drift = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6.3))
    return pink_noise(n, rng) * drift


def _formants(f):
    """
    Relative amplitude of a voice harmonic at frequency f (Hz).
    """
    peaks = ((700, 1.0, 150), (1200, 0.6, 200), (2600, 0.2, 300), (3500, 0.1, 350))
    g = sum(a * np.exp(-0.5 * ((f - c) / w) ** 2) for c, a, w in peaks)
    return (g + 0.02) / (1 + f / 1000.0)


def commentary(n, sr, rng, talk_frac=0.75):
    """
    A voice talking in 1-6 s utterances with pauses, about talk_frac of the time.
    """
    out = np.zeros(n)
    i = int(rng.uniform(0, 1) * sr)
    while i < n:
        m = min(int(rng.uniform(1, 6) * sr), n - i)
        t = np.arange(m) / sr
        # Pitch glides between random targets every ~0.25 s (intonation), around a speaker mean.
        knots = np.arange(0, m + sr // 4, sr // 4)
        f0 = np.interp(np.arange(m), knots, rng.uniform(100, 220, len(knots)))
        phase = 2 * np.pi * np.cumsum(f0) / sr
        voice = np.zeros(m)
        for h in range(1, int(5500 / 100) + 1):
            amp = _formants(h * f0) * (h * f0 < sr / 2)
            voice += amp * np.sin(h * phase)
        syllables = 0.5 * (1 - np.cos(2 * np.pi * rng.uniform(3, 5) * t))   # ~4 syllables a s
        out[i:i + m] = voice * syllables * _envelope(m, sr // 50, sr // 20)
        pause = rng.exponential((1 - talk_frac) / talk_frac * 3.5)
        i += m + int(pause * sr)
    return out / max(out.std(), 1e-12)


def chorus(n, sr, rng, bursts_per_min=2.0, tones=25):
    """
    The crowd whistling: bursts of many weak tones at 1.5-4 kHz.
    """
    out = np.zeros(n)
    for _ in range(rng.poisson(bursts_per_min * n / sr / 60)):
        start, m = int(rng.uniform(0, n)), int(rng.uniform(2, 8) * sr)
        m = min(m, n - start)
        if m <= 0:
            continue
        t = np.arange(m) / sr
        burst = sum(np.sin(2 * np.pi * (rng.uniform(1500, 4000) * t
                                        + rng.uniform(50, 300) / 1.5 * np.sin(1.5 * t)))
                    for _ in range(tones))
        out[start:start + m] += burst / tones * _envelope(m, sr // 2, sr // 2)
    return out


def fan_whistle(dur_s, sr, rng):
    """
    One fan whistling: a loud tone that glides by several hundred Hz.
    """
    m = int(dur_s * sr)
    t = np.arange(m) / sr
    f = rng.uniform(1500, 3000) + rng.choice([-1, 1]) * rng.uniform(400, 900) * t / dur_s
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * _envelope(m, sr // 30, sr // 15)


def referee_whistle(dur_s, sr, rng, kind="pea"):
    """
    One blow of a referee's whistle and its fundamental (Hz).
    """
    m = int(dur_s * sr)
    t = np.arange(m) / sr
    f0 = rng.uniform(2800, 4200)
    if kind == "pea":
        f = f0 + 80 * np.sin(2 * np.pi * rng.uniform(35, 55) * t)
        phase = 2 * np.pi * np.cumsum(f) / sr
        x = np.sin(phase) + 0.25 * np.sin(2 * phase)
    elif kind == "fox":
        x = sum(a * np.sin(2 * np.pi * f0 * r * t) for r, a in ((1.0, 1.0), (1.045, 0.6), (1.09, 0.4)))
    else:
        raise ValueError(f"unknown whistle kind {kind!r}")
    return x * _envelope(m, sr // 100, sr // 30), f0


def scene(seconds=120.0, seed=0, snr_db=-5.0, n_whistles=12, kind="pea", talk=True,
          chorus_per_min=2.0, fans_per_min=1.0, sr=SR):
    """
    A made-up scene: the audio and a table of its true whistles.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    bg = crowd(n, sr, rng)
    if talk:
        bg = bg + 2.0 * commentary(n, sr, rng)
    bg = bg + 0.7 * chorus(n, sr, rng, chorus_per_min)
    for _ in range(rng.poisson(fans_per_min * seconds / 60)):
        w = 0.8 * fan_whistle(rng.uniform(0.4, 1.5), sr, rng)
        s = int(rng.uniform(0, n - len(w)))
        bg[s:s + len(w)] += w
    bg_rms = bg.std()

    # Whistles on an even grid with jitter, so they never overlap; some are blown twice.
    rows, x = [], bg.copy()
    slots = np.linspace(2, seconds - 4, n_whistles)
    for c in slots:
        start = c + rng.uniform(-0.5, 0.5)
        pattern = rng.choice(["short", "long", "double"], p=[0.6, 0.2, 0.2])
        blows = {"short": [rng.uniform(0.15, 0.5)], "long": [rng.uniform(0.8, 2.0)],
                 "double": [rng.uniform(0.12, 0.3), rng.uniform(0.12, 0.3)]}[pattern]
        t0 = start
        for k, d in enumerate(blows):
            w, f0 = referee_whistle(d, sr, rng, kind)
            w = w / w.std() * bg_rms * 10 ** (snr_db / 20)
            s = int(t0 * sr)
            x[s:s + len(w)] += w
            t0 += d + 0.15
        rows.append((start, t0 - 0.15, f0, pattern))
    truth = pd.DataFrame(rows, columns=["start_s", "end_s", "freq_hz", "pattern"])
    peak = np.abs(x).max()
    return (x / peak * 0.5).astype(np.float32), truth
