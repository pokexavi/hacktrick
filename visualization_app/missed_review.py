"""Where a Dead/Alive model may have missed a stoppage: a list of places to watch, all automatic.

The other review pages start from a stoppage the model predicted. This one starts from the frames
the model calls alive and collects every automatic sign that something stopped there:

    models     other models (seeds, variants) that do call it Dead
    p_dead     the reference model's own p_dead rises, but not enough to become a stoppage
    whistle    a detected whistle in play (`preset`, or a more permissive detector)
    ball gap   the tracked ball is missing for a while
    tagging    a Smart Tagging set piece whose window holds no stoppage at all

Signs close in time are joined into one candidate. A candidate's score is how many different kinds
of sign it has: nothing here is ground truth, it only orders what to watch first.
No I/O and no match data: arrays and tables in, tables out (tests use made-up ones).
"""
import numpy as np
import pandas as pd

SIGNS = ("models", "p_dead", "whistle", "ball gap", "tagging")
COLUMNS = ["start", "stop", "seconds", "score", "signs", "votes", "peak_p", "whistles", "tagged",
           "to_stoppage_s"]


def runs_of(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def model_signs(alive, voters, fps=25, min_s=2.0):
    """Runs of at least min_s where the reference is alive and some other model says Dead.
    voters: bool array (models, frames). strength = the largest share of models saying Dead."""
    if voters is None or not len(voters):
        return []
    share = np.mean(voters, axis=0)
    return [{"sign": "models", "start": a, "stop": b, "strength": float(share[a:b].max())}
            for a, b in runs_of(alive & (share > 0)) if b - a >= min_s * fps]


def p_dead_signs(alive, p_dead, fps=25, threshold=0.35, min_s=2.0):
    """Runs of at least min_s where the reference is alive with p_dead above the threshold."""
    p = np.nan_to_num(np.asarray(p_dead, dtype=float), nan=0.0)
    return [{"sign": "p_dead", "start": a, "stop": b, "strength": float(p[a:b].max())}
            for a, b in runs_of(alive & (p >= threshold)) if b - a >= min_s * fps]


def whistle_signs(alive, frames, label, fps=25, length_s=1.0):
    """One sign per whistle that falls on an alive frame. label: which detector."""
    n = len(alive)
    return [{"sign": "whistle", "start": int(f), "stop": min(int(f + length_s * fps), n), "strength": 1.0,
             "label": label} for f in np.asarray(frames, dtype=int) if 0 <= f < n and alive[f]]


def ball_gap_signs(alive, ball_observed, fps=25, min_s=5.0):
    """Runs of at least min_s where the reference is alive and the ball is not tracked."""
    seen = np.nan_to_num(np.asarray(ball_observed, dtype=float), nan=0.0) > 0
    return [{"sign": "ball gap", "start": a, "stop": b, "strength": (b - a) / fps}
            for a, b in runs_of(alive & ~seen) if b - a >= min_s * fps]


def _distance_to_dead(start, stop, dead_runs, fps):
    """Seconds from [start, stop) to the nearest stoppage (0 if they touch)."""
    gaps = [max(a - stop, start - b, 0) for a, b in dead_runs]
    return round(min(gaps) / fps, 1) if gaps else np.inf


def candidates(signs, dead, tagged=None, fps=25, join_s=5.0):
    """One row per group of signs less than join_s apart, most signs first (see COLUMNS).
    dead: bool per frame (the reference's Dead). tagged: table (start, stop, kind) of the Smart
    Tagging events with no stoppage in their window; they are ~35 s windows, too wide to join signs,
    so an event only names the candidates it overlaps, and is a candidate of its own if there are none."""
    dead_runs = runs_of(dead)
    groups = []
    for sign in sorted(signs, key=lambda s: s["start"]):
        if groups and sign["start"] - groups[-1]["stop"] < join_s * fps:
            groups[-1]["items"].append(sign)
            groups[-1]["stop"] = max(groups[-1]["stop"], sign["stop"])
        else:
            groups.append({"start": sign["start"], "stop": sign["stop"], "items": [sign]})
    for group in groups:
        group["tags"] = []
    if tagged is not None:
        for event in tagged.itertuples():
            hit = [g for g in groups if g["stop"] > event.start and g["start"] < event.stop]
            for group in hit:
                group["tags"].append(event.kind)
            if not hit:
                groups.append({"start": int(event.start), "stop": int(event.stop), "items": [],
                               "tags": [event.kind]})
    rows = []
    for group in groups:
        items = group["items"]
        kinds = [k for k in SIGNS[:4] if any(i["sign"] == k for i in items)] + (["tagging"] if group["tags"] else [])
        votes = [i["strength"] for i in items if i["sign"] == "models"]
        peaks = [i["strength"] for i in items if i["sign"] == "p_dead"]
        heard = sorted({i["label"] for i in items if i["sign"] == "whistle"})
        rows.append({"start": group["start"], "stop": group["stop"],
                     "seconds": round((group["stop"] - group["start"]) / fps, 1), "score": len(kinds),
                     "signs": ", ".join(kinds), "votes": round(max(votes), 2) if votes else 0.0,
                     "peak_p": round(max(peaks), 2) if peaks else 0.0, "whistles": ", ".join(heard),
                     "tagged": ", ".join(group["tags"]),
                     "to_stoppage_s": _distance_to_dead(group["start"], group["stop"], dead_runs, fps)})
    table = pd.DataFrame(rows, columns=COLUMNS)
    return table.sort_values(["score", "votes", "peak_p"], ascending=False, kind="stable").reset_index(drop=True)


def long_alive(alive, fps=25, min_s=90.0):
    """(start, stop, seconds) of the alive stretches of at least min_s, longest first."""
    rows = [{"start": a, "stop": b, "seconds": round((b - a) / fps, 1)}
            for a, b in runs_of(alive) if b - a >= min_s * fps]
    table = pd.DataFrame(rows, columns=["start", "stop", "seconds"])
    return table.sort_values("seconds", ascending=False, kind="stable").reset_index(drop=True)


def pieces(mask, seconds, fps=25):
    """The True runs of mask cut into [start, stop) pieces of at most `seconds`, in order."""
    size = int(seconds * fps)
    return [(s, min(s + size, b)) for a, b in runs_of(mask) for s in range(a, b, size)]
