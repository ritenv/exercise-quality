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


def _rotate(vectors, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * vectors[:, 0] - s * vectors[:, 1], s * vectors[:, 0] + c * vectors[:, 1]], axis=-1)


def soft_lockout(track, knee=15.0):
    """The knees stop ``knee`` degrees short of straight at the top after the rep.

    From the ascent on, any frame bent less than ``knee`` is bent to it:
    the shank tips forward about the ankle and the thigh back about the
    knee by half the shortfall each, and everything above the hip moves
    with the hip.
    """
    from exercise_quality.core.observed import observed_angles

    rep = _only_rep(track)
    flexion = observed_angles(track)["knee_flexion"]
    short = np.where(np.arange(len(track)) >= rep.ascent_start, np.maximum(knee - flexion, 0.0), 0.0)
    half = np.radians(short) / 2
    ankle, knee_pt, hip = track["ankle"], track["knee"], track["hip"]
    new_knee = ankle + _rotate(knee_pt - ankle, -half)       # clockwise: shank leans forward
    new_hip = new_knee + _rotate(hip - knee_pt, half)          # counter-clockwise: thigh back
    points = dict(track.points)
    points["knee"] = new_knee
    for name in ("hip",) + UPPER:
        points[name] = track[name] + (new_hip - hip)
    return _replace(track, points)


def sit_back(track, degrees=25.0):
    """The trunk tips ``degrees`` further back at the bottom, putting the weight on the heels.

    The extra backward rotation about the hip grows with depth (fraction of
    the rep's hip drop), so it is the same on the way down and up at equal
    height and does not look like hips rising first.
    """
    rep = _only_rep(track)
    h = R.hip_height(R.smoothed(track))
    depth = np.clip((h[rep.start] - h) / (h[rep.start] - h[rep.lowest]), 0.0, 1.0)
    depth[: rep.start] = 0.0
    depth[rep.end + 1:] = 0.0
    angle = np.radians(degrees) * depth          # counter-clockwise = backward for a person facing +x
    points = dict(track.points)
    hip = track["hip"]
    for name in UPPER:
        points[name] = hip + _rotate(track[name] - hip, angle)
    return _replace(track, points)
