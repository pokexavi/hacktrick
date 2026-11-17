"""Make clean DFL tracking look like our automatic BAR-ATH tracking, at training time.

DFL is optical tracking with quality control: the ball is on every frame and every player keeps one
identity all match. Our ATD loses the ball on ~41% of graded frames and its players come as
tracker slots that drop, reappear and change person. A model trained on clean DFL inputs would
never have seen either, so it would fail on our data in exactly the frames that matter.

Two steps:

  profile(match)            measure the noise of a processed match -- aggregate numbers only
                            (fractions and run-length quantiles), so it is safe to print and keep
  degrade(match, prof, rng) a new Match: DFL positions, BAR-like gaps, slots and jitter

    python external/noise.py            # profile BAR-ATH (and one DFL match, for contrast) and
                                        # write data/processed/noise_profile.json

degrade() is random by design -- call it with a fresh rng every epoch, and the model sees a
different corruption of the same match each time.

How the corruption is built
---------------------------
Ball:     alternating seen / lost runs. Lost-run lengths are drawn from BAR's own distribution. The
          moment a gap starts is not uniform: its hazard is raised while the ball is in the air,
          in a crowd, or dead -- where BAR loses it -- and the base rate is solved so the overall
          lost fraction matches BAR's.
Players:  each real player is cut into tracklets with BAR's tracklet-length distribution and gaps
          sized to BAR's mean number of tracked players; each tracklet goes into a random free slot
          of its team, so slot k is no longer a person -- as in the ATD.
Jitter:   Gaussian noise sized from the difference in frame-to-frame roughness between sources.
Ball z:   removed (BAR has none).

Augmentations that are free and keep labels right: swap_teams (H <-> A, rotated so home still
attacks +x) and mirror_y.
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from external import schema  # noqa: E402

# Params
QUANTILES = np.linspace(0, 1, 41)       # run-length distributions are stored as 41 quantiles
AIR_Z_M = 1.8                            # ball higher than this counts as "in the air"
CROWD_RADIUS_M = 3.0
CROWD_PLAYERS = 3                        # this many players within the radius is a crowd
# Gap-start hazard multipliers. BAR offers no ball position inside its gaps, so these cannot be
# measured directly; DEAD_BOOST is solved from the public labels when profile() has them, the
# other two are informed guesses -- the knobs to tune first.
AIR_BOOST = 3.0
CROWD_BOOST = 2.0
DEFAULT_DEAD_BOOST = 2.0


# -- runs ------------------------------------------------------------------------------------------

def runs(mask, period):
    """Lengths of the True runs of mask, never crossing a period boundary, inside periods > 0."""
    out = []
    for p in (1, 2):
        m = mask[period == p].astype(np.int8)
        edges = np.diff(np.concatenate(([0], m, [0])))
        out.append(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    return np.concatenate(out)


def quantiles(lengths):
    return np.quantile(lengths, QUANTILES).tolist() if len(lengths) else None


def sample(qs, size, rng):
    """Draw from a distribution stored as QUANTILES (inverse-CDF interpolation), at least 1."""
    return np.maximum(1, np.round(np.interp(rng.random(size), QUANTILES, qs))).astype(np.int64)


# -- profile ---------------------------------------------------------------------------------------

def _roughness(xy, period):
    """Median |second difference| of positions, in metres per frame^2 -- a jitter fingerprint."""
    second = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    same = (period[2:] == period[:-2]) & (period[2:] > 0)
    values = np.abs(second[same]).ravel()
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def profile(match):
    """Aggregate noise statistics of one processed match. Nothing frame-level leaves this."""
    graded = match.period > 0
    period = match.period
    seen = np.isfinite(match.ball[:, 0])
    out = {"source": match.source, "name": match.name,
           "ball_lost_frac": float(1 - seen[graded].mean()),
           "ball_gap_q": quantiles(runs(~seen & graded, period)),
           "ball_seen_q": quantiles(runs(seen & graded, period))}

    labelled = graded & (match.label != "X")
    alive = labelled & np.isin(match.label, ("H", "A"))
    dead = labelled & (match.label == "D")
    out["labelled_frames"] = int(labelled.sum())
    out["ball_lost_frac_alive"] = float(1 - seen[alive].mean()) if alive.any() else None
    out["ball_lost_frac_dead"] = float(1 - seen[dead].mean()) if dead.any() else None

    for side, xy in (("home", match.home), ("away", match.away)):
        present = np.isfinite(xy[..., 0])
        counts = present[graded].sum(1)
        pieces = np.concatenate([runs(present[:, k] & graded, period) for k in range(xy.shape[1])])
        out[side] = {"n_slots": int(xy.shape[1]),
                     "tracked_mean": float(counts.mean()),
                     "tracked_q": quantiles(counts),
                     "over_11_frac": float((counts > 11).mean()),
                     "tracklet_q": quantiles(pieces),
                     "tracklets_per_minute": float(len(pieces) / (graded.sum() / schema.FPS / 60))}
    out["roughness_m"] = _roughness(np.concatenate([match.home, match.away], 1), period)
    out["ball_roughness_m"] = _roughness(match.ball[:, None, :2], period)
    return out


# -- degrade ---------------------------------------------------------------------------------------

def _crowded(match):
    ball = match.ball[:, None, :2]
    players = np.concatenate([match.home, match.away], 1)
    with np.errstate(invalid="ignore"):
        near = (np.sqrt(((players - ball) ** 2).sum(-1)) < CROWD_RADIUS_M).sum(1)
    return near >= CROWD_PLAYERS


def ball_hazard_weight(match, dead_boost):
    """Relative chance, per frame, that a ball gap starts there."""
    weight = np.ones(len(match.frame))
    z = match.ball[:, 2]
    weight += AIR_BOOST * (np.nan_to_num(z) > AIR_Z_M)
    weight += CROWD_BOOST * _crowded(match)
    weight *= np.where(match.label == "D", dead_boost, 1.0)
    return weight


def _ball_mask(weight, period, prof, rng, rate):
    """Seen mask from gap-start hazard rate*weight and gap lengths from the profile."""
    seen = np.ones(len(weight), dtype=bool)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        cum = np.concatenate(([0.0], np.cumsum(rate * weight[idx])))
        t = 0
        while t < len(idx):
            target = cum[t] + rng.exponential()
            start = int(np.searchsorted(cum, target)) - 1
            if start >= len(idx):
                break
            length = int(sample(prof["ball_gap_q"], 1, rng)[0])
            seen[idx[start:start + length]] = False
            t = start + length
    return seen


def solve_ball_rate(match, prof, dead_boost, rng, lost_frac=None, iterations=12):
    """Base hazard rate giving this lost fraction on this match (bisection).

    lost_frac defaults to the profile's match-wide figure. BAR's own varies from ~12% to ~80%
    between 5-minute blocks, so sampling it per call is a cheap and honest augmentation.
    """
    target = prof["ball_lost_frac"] if lost_frac is None else lost_frac
    weight = ball_hazard_weight(match, dead_boost)
    graded = match.period > 0
    low, high = 1e-5, 1.0
    for _ in range(iterations):
        mid = np.sqrt(low * high)
        lost = 1 - _ball_mask(weight, match.period, prof, rng, mid)[graded].mean()
        low, high = (mid, high) if lost < target else (low, mid)
    return float(np.sqrt(low * high))


def _fragment(xy, period, team_prof, n_slots, rng):
    """(T, n_slots, 2): every player cut into tracklets, each dropped into a random free slot."""
    n_frames, n_players = xy.shape[:2]
    out = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    real = np.isfinite(xy[..., 0])
    on_pitch = max(real[period > 0].sum(1).mean(), 1e-6)
    keep = min(team_prof["tracked_mean"] / on_pitch, 0.999)
    mean_piece = float(np.mean(sample(team_prof["tracklet_q"], 2000, rng)))
    mean_gap = mean_piece * (1 - keep) / keep

    pieces = []                                        # (start, stop, player)
    for k in range(n_players):
        t = int(rng.integers(0, max(int(mean_piece), 1)))
        while t < n_frames:
            length = int(sample(team_prof["tracklet_q"], 1, rng)[0])
            pieces.append((t, min(t + length, n_frames), k))
            t += length + int(rng.exponential(mean_gap)) + 1
    pieces.sort()
    free_at = np.zeros(n_slots, dtype=np.int64)
    for start, stop, k in pieces:
        free = np.flatnonzero(free_at <= start)
        if not len(free):
            continue
        slot = rng.choice(free)
        out[start:stop, slot] = xy[start:stop, k]
        free_at[slot] = stop
    out[~np.isfinite(out[..., 0])] = np.nan
    return out


def degrade(match, prof, rng, clean_roughness=None, ball_rate=None, dead_boost=None,
            lost_frac=None):
    """A BAR-like copy of a clean match. Labels, frames and periods are untouched.

    prof: the BAR profile (profile() of the BAR match, or noise_profile.json).
    clean_roughness: profile(match)["roughness_m"] -- pass it to avoid recomputing per call.
    ball_rate: solve_ball_rate(...) for this match -- likewise worth caching per match.
    lost_frac: target fraction of lost-ball frames (default: BAR's match-wide figure). Ignored
        when ball_rate is given.
    """
    dead_boost = dead_boost or prof.get("dead_boost", DEFAULT_DEAD_BOOST)
    if ball_rate is None:
        ball_rate = solve_ball_rate(match, prof, dead_boost, rng, lost_frac)
    seen = _ball_mask(ball_hazard_weight(match, dead_boost), match.period, prof, rng, ball_rate)

    if clean_roughness is None:
        clean_roughness = _roughness(np.concatenate([match.home, match.away], 1), match.period)
    # The second difference of white noise of sd s has sd s*sqrt(6); medians scale alike.
    extra = max(prof["roughness_m"] ** 2 - clean_roughness ** 2, 0.0) ** 0.5 / np.sqrt(6) / 0.6745

    home = _fragment(match.home, match.period, prof["home"], prof["home"]["n_slots"], rng)
    away = _fragment(match.away, match.period, prof["away"], prof["away"]["n_slots"], rng)
    home += rng.normal(0, extra, home.shape).astype(np.float32)
    away += rng.normal(0, extra, away.shape).astype(np.float32)
    ball = match.ball.copy()
    ball[~seen, :2] = np.nan
    ball[:, 2] = np.nan
    return replace(match, home=home, away=away, ball=ball,
                   meta=dict(match.meta, degraded=True))


# -- label-safe augmentations ----------------------------------------------------------------------

def swap_teams(match):
    """Away becomes home: arrays swapped, H <-> A, and rotated so the new home attacks +x."""
    label = match.label.copy()
    label[match.label == "H"], label[match.label == "A"] = "A", "H"
    ball = match.ball.copy()
    ball[:, :2] *= -1
    meta = dict(match.meta, n_home=match.meta["n_away"], n_away=match.meta["n_home"],
                swapped=not match.meta.get("swapped", False))
    return replace(match, home=-match.away, away=-match.home, ball=ball, label=label, meta=meta)


def mirror_y(match):
    """Reflect across the long axis. Possession does not care about left and right."""
    flip = np.array([1, -1], dtype=np.float32)
    ball = match.ball.copy()
    ball[:, 1] *= -1
    return replace(match, home=match.home * flip, away=match.away * flip, ball=ball)


# -- main ------------------------------------------------------------------------------------------

def _solve_dead_boost(prof):
    """Gap-start boost on dead frames that reproduces BAR's dead/alive lost-ball ratio, roughly.

    In a renewal process with equal gap lengths the lost fraction grows with the hazard, so the
    odds ratio of lost-vs-seen between dead and alive frames is a fair first estimate.
    """
    alive, dead = prof.get("ball_lost_frac_alive"), prof.get("ball_lost_frac_dead")
    if not alive or not dead or alive >= 1 or dead >= 1:
        return DEFAULT_DEAD_BOOST
    return float((dead / (1 - dead)) / (alive / (1 - alive)))


def main():
    bar = schema.processed_paths("bar")
    if not bar:
        sys.exit("no processed BAR-ATH file -- run external/convert_atd.py first")
    prof = profile(schema.load_match(bar[0]))
    prof["dead_boost"] = _solve_dead_boost(prof)
    path = os.path.join(schema.bar_processed_dir(), "noise_profile.json")
    with open(path, "w") as handle:
        json.dump(prof, handle, indent=2)
    print("wrote {}".format(path))

    def brief(p):
        return ("ball lost {:.1%} (alive {}, dead {}) | tracked home {:.1f} away {:.1f} | "
                "tracklet median {:.0f} s | roughness {:.4f} m").format(
            p["ball_lost_frac"],
            "{:.1%}".format(p["ball_lost_frac_alive"]) if p["ball_lost_frac_alive"] is not None
            else "-",
            "{:.1%}".format(p["ball_lost_frac_dead"]) if p["ball_lost_frac_dead"] is not None
            else "-",
            p["home"]["tracked_mean"], p["away"]["tracked_mean"],
            p["home"]["tracklet_q"][20] / schema.FPS, p["roughness_m"])

    print("BAR-ATH  " + brief(prof))
    print("         dead-ball gap boost {:.2f}".format(prof["dead_boost"]))
    dfl = schema.processed_paths("dfl")
    if dfl:
        print("DFL      " + brief(profile(schema.load_match(dfl[0]))))


if __name__ == "__main__":
    main()
