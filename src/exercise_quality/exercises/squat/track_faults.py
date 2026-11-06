"""Edit an observed squat track to contain a known technique fault.

For testing the rubric: each function changes the tracked points the way
the fault would look from the side and leaves the rest alone. These are
geometric edits, not simulations: balance is not re-solved, so they are
not suitable as training data for anything that should obey physics.
Only ``shallow`` and ``forward_lean`` faults can be made physically, by
changing the squat style in the generator (see tests/test_rubric.py).
"""

import numpy as np

from exercise_quality.core.observed import ObservedTrack, segment_lengths
from exercise_quality.exercises.squat import reps as R

UPPER = ("shoulder", "elbow", "wrist", "ear")


def _replace(track, points):
    return ObservedTrack(points=points, fps=track.fps, units=track.units, visibility=track.visibility,
                         source=track.source + "+fault", yaw_scale=track.yaw_scale)


def _bump(n, start, peak, end):
    """0 outside [start, end], rising smoothly to 1 at ``peak``."""
    w = np.zeros(n)
    up = np.arange(start, peak + 1)
    down = np.arange(peak, end + 1)
    w[up] = 0.5 - 0.5 * np.cos(np.pi * (up - start) / max(peak - start, 1))
    w[down] = 0.5 + 0.5 * np.cos(np.pi * (down - peak) / max(end - peak, 1))
    return w


def _only_rep(track):
    found = R.find_reps(R.smoothed(track))
    if len(found) != 1:
        raise ValueError("track must contain exactly one rep")
    return found[0]


def heel_lift(track, rise=0.25):
    """Heels come up by ``rise`` x foot length around the bottom, toes stay down."""
    rep = _only_rep(track)
    foot = float(np.median(segment_lengths(track)["foot"]))
    w = _bump(len(track), rep.start, rep.lowest, rep.end)
    points = dict(track.points)
    points["heel"] = track["heel"] + np.stack([0 * w, rise * foot * w], axis=-1)
    return _replace(track, points)


def hips_rise_first(track, extra_lean=10.0):
    """Trunk tips ``extra_lean`` degrees further forward in the first half of the ascent."""
    rep = _only_rep(track)
    mid = (rep.ascent_start + rep.end) // 2
    w = _bump(len(track), rep.ascent_start, (rep.ascent_start + mid) // 2, mid)
    angle = -np.radians(extra_lean) * w          # clockwise = forward lean for a person facing +x
    c, s = np.cos(angle), np.sin(angle)
    points = dict(track.points)
    hip = track["hip"]
    for name in UPPER:
        d = track[name] - hip
        points[name] = hip + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=-1)
    return _replace(track, points)


def hitch(track, back_up=0.08):
    """A stall-and-rebound half way down: the body rises ``back_up`` x hip height, then continues."""
    rep = _only_rep(track)
    h = R.hip_height(R.smoothed(track))
    mid = rep.start + int(np.argmin(np.abs(h[rep.start:rep.lowest] - 0.5 * (h[rep.start] + h[rep.lowest]))))
    target = h[mid] + back_up
    j = mid - 1
    while j > rep.start and h[j] < target:
        j -= 1
    back = np.arange(mid, j - 1, -1)        # replay frames mid -> j backwards (rising)
    order = np.concatenate([np.arange(0, mid + 1), back, np.arange(j, len(track))])
    return _replace(track, {k: v[order] for k, v in track.points.items()})
