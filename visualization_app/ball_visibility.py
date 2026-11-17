"""Ball-presence analysis from explicit ATD lines; no filesystem access."""
from dataclasses import dataclass
import math


def parse_visibility(lines):
    """Require consecutive one-based frame IDs; visibility needs finite x AND y.

    Missing rows are rejected rather than silently interpreted as missing-ball frames.
    The speed channel does not determine visibility.
    """
    visible = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split(":")
        if len(parts) != 3 or int(parts[0]) != len(visible) + 1:
            raise ValueError("ATD must have consecutive frame IDs starting at 1.")
        ball = parts[2].split(",")
        if len(ball) != 3:
            raise ValueError("ATD ball values must contain x,y,s.")
        x, y = float(ball[0]), float(ball[1])
        visible.append(math.isfinite(x) and math.isfinite(y))
    if not visible:
        raise ValueError("ATD contains no frames.")
    return visible


@dataclass(frozen=True)
class MissingRun:
    half: str
    start: int
    end: int  # exclusive; the last missing frame is end - 1

    @property
    def frames(self):
        return self.end - self.start


def missing_runs(visible, ranges):
    """Find consecutive missing-ball frames within disjoint half-open ranges.

    Each range is ``(name, start, end)`` in zero-based video frames. Runs never
    cross a range boundary, so halftime cannot become a single long loss.
    """
    runs = []
    previous_end = 0
    for name, start, end in sorted(ranges, key=lambda item: item[1]):
        if not 0 <= start <= end <= len(visible) or start < previous_end:
            raise ValueError("Analysis ranges must be valid, disjoint, and within ATD frames.")
        previous_end = end
        run_start = None
        for frame in range(start, end):
            if not visible[frame]:
                if run_start is None:
                    run_start = frame
            elif run_start is not None:
                runs.append(MissingRun(name, run_start, frame))
                run_start = None
        if run_start is not None:
            runs.append(MissingRun(name, run_start, end))
    return runs


def clip_window(run, *, fps, total_frames, end_focus, pre_roll_s, clip_length_s):
    """A short video range around either boundary of a missing-ball run."""
    if fps <= 0 or total_frames <= 0 or clip_length_s <= 0 or pre_roll_s < 0:
        raise ValueError("Invalid video length or clip settings.")
    focus = run.end if end_focus and run.end < total_frames else (
        run.end - 1 if end_focus else run.start
    )
    focus = min(max(focus, 0), total_frames - 1)
    length = max(1, round(clip_length_s * fps))
    start = max(0, focus - round(pre_roll_s * fps))
    end = min(total_frames, start + length)
    if end - start < length:
        start = max(0, end - length)
    return start, end

