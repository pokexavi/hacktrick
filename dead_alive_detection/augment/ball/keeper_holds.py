"""
Find when a keeper holds the ball in live play in a DFL match.

BAR-ATH often loses the ball when a keeper catches it, and the model then takes it for a
stoppage. Hiding the ball during these holds in DFL training copies teaches it that play is
still live. Holds come from the tracking; the event feed is only used to check them.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from dead_alive_detection.evaluation.stoppage_types import event_paths
from dead_alive_detection.loading.features import BOX_DEPTH_M, BOX_HALF_WIDTH_M, keepers
from dead_alive_detection.loading.tensors import STEPS_PER_S, match_tensors
from external import schema

# Params
MAX_S = 15.0          # a claim and a release farther apart than this are not one hold
HOLD_M = 2.0          # the keeper this close to the ball, in his box: he has it (as fit_hidden)
MIN_S = 1.5           # shorter runs are not a hold (a pass back through his feet)
BRIDGE_S = 0.4        # gaps of at most this long inside a run are closed
FPS = schema.FPS
STEP = 5


def keeper_ids(meta_path):
    """
    Ids of the players listed as goalkeepers in the match information.
    """
    root = ET.parse(meta_path).getroot()
    return {p.attrib["PersonId"] for p in root.iter("Player") if p.attrib.get("PlayingPosition") == "TW"}


def _time(event):
    return datetime.fromisoformat(event.attrib["EventTime"])


def keeper_events(meta_path, events_path):
    """
    Kickoff times per period and the keepers' claims and releases from the event feed, in time order.
    """
    keepers = keeper_ids(meta_path)
    kickoffs, found = {}, []
    root = ET.parse(events_path).getroot()
    for event in root.iter("Event"):
        for action in event.iter():
            a = action.attrib
            if action.tag == "KickOff" and a.get("GameSection") in ("firstHalf", "secondHalf"):
                kickoffs[1 if a["GameSection"] == "firstHalf" else 2] = _time(event)
            elif action.tag == "BallClaiming" and a.get("Type") in ("BallHeld", "BallClaimed") \
                    and a.get("Player") in keepers:
                found.append((_time(event), "claim"))
            elif action.tag == "SavedShot":
                found.append((_time(event), "claim"))
            elif action.tag == "Play" and a.get("GoalKeeperAction") and a.get("Player") in keepers:
                found.append((_time(event), "release"))
    return kickoffs, sorted(found, key=lambda x: x[0])


def _frame(when, kickoffs, first_frame):
    """
    Frame of a datetime, counted from the latest kickoff before it.
    """
    period = max((p for p, t in kickoffs.items() if t <= when), default=None)
    if period is None:
        return None
    return int(first_frame[period] + round((when - kickoffs[period]).total_seconds() * FPS))


def event_holds(match):
    """
    Claim-to-release windows from the event feed. Only for checking; training uses holds().
    """
    paths = event_paths(match.name)
    if paths is None or len(paths) != 2:     # Metrica has no such events
        return []
    kickoffs, found = keeper_events(*paths)
    first = {p: int(np.flatnonzero(match.period == p)[0]) for p in (1, 2) if p in kickoffs}
    n = len(match.frame)
    out, claim = [], None
    for when, kind in found:
        if kind == "claim":
            claim = when
        elif claim is not None:
            if 0 < (when - claim).total_seconds() <= MAX_S:
                a, b = _frame(claim, kickoffs, first), _frame(when, kickoffs, first)
                if a is not None and b is not None and 0 <= a < b <= n \
                        and (match.label[a:b] != "D").all() and (match.label[a:b] != "X").all():
                    out.append((a, b))
            claim = None
    return out


def claim_times(match):
    """
    Frames of the keepers' claims and saved shots in the event feed.
    """
    paths = event_paths(match.name)
    if paths is None or len(paths) != 2:
        return []
    kickoffs, found = keeper_events(*paths)
    first = {p: int(np.flatnonzero(match.period == p)[0]) for p in (1, 2) if p in kickoffs}
    return [f for f in (_frame(w, kickoffs, first) for w, k in found if k == "claim") if f is not None]


def holds(match, near_m=HOLD_M, min_s=MIN_S, bridge_s=BRIDGE_S):
    """
    (start, stop) frames where the keeper has the ball in live play, from the tracking.
    Needs a clean ball, so DFL only.
    """
    t = match_tensors(match)
    half = t["half_pitch"]
    ball = t["ball"][:, :2] * half
    gk = keepers(t)
    side = (ball[:, 0] > 0).astype(int)
    rows = np.arange(len(ball))
    with np.errstate(invalid="ignore"):
        near = np.linalg.norm(ball - gk[rows, side, :2], axis=1) < near_m
    in_box = ((half[0] - np.abs(ball[:, 0]) < BOX_DEPTH_M) & (np.abs(ball[:, 1]) < BOX_HALF_WIDTH_M))
    live = ~t["dead"] & t["scored"] & t["ball_seen"]
    hold = near & in_box & live
    change = np.flatnonzero(hold[1:] != hold[:-1]) + 1
    bounds = np.r_[0, change, len(hold)]
    bridge = int(round(bridge_s * STEPS_PER_S))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not hold[a] and b - a <= bridge and live[a:b].all() and t["period"][a] == t["period"][min(b, len(hold) - 1)]:
            hold[a:b] = True
    change = np.flatnonzero((hold[1:] != hold[:-1]) | (t["period"][1:] != t["period"][:-1])) + 1
    bounds = np.r_[0, change, len(hold)]
    frames = t["row"]
    minimum = round(min_s * STEPS_PER_S)
    return [(int(frames[a]), int(frames[b - 1] + STEP)) for a, b in zip(bounds[:-1], bounds[1:])
            if hold[a] and b - a >= minimum]


def mask(windows, rows, rng, section):
    """
    Mask of the steps whose ball is hidden, one random draw per hold.
    """
    hidden = np.zeros(len(rows), dtype=bool)
    for a, b in windows:
        u = rng.random(3)                      # a fixed number of draws per hold, whatever happens
        if u[0] >= section["prob"]:
            continue
        lead = section["lead_s"][0] + u[1] * (section["lead_s"][1] - section["lead_s"][0])
        tail = section["tail_s"][0] + u[2] * (section["tail_s"][1] - section["tail_s"][0])
        lo = np.searchsorted(rows, a + lead * FPS, side="left")
        hi = np.searchsorted(rows, b + tail * FPS, side="left")
        hidden[lo:hi] = True
    return hidden


def _summary():
    """
    Print, per DFL match, the holds found and how they compare with the event feed.
    """
    for path in schema.processed_paths("dfl"):
        match = schema.load_match(path)
        if event_paths(match.name) is None or len(event_paths(match.name)) != 2:
            continue
        windows = holds(match)
        lengths = np.array([(b - a) / FPS for a, b in windows])
        claims = claim_times(match)
        covered = sum(any(a - 75 <= c <= b + 75 for a, b in windows) for c in claims)
        print("{}: {} holds, median {:.1f} s, max {:.1f} s, total {:.0f} s; event claims/saves {}, "
              "within 3 s of a hold {}; event claim-to-release windows {}".format(
                  match.name, len(windows), np.median(lengths) if len(lengths) else 0,
                  lengths.max() if len(lengths) else 0, lengths.sum(), len(claims), covered,
                  len(event_holds(match))))


if __name__ == "__main__":
    _summary()
