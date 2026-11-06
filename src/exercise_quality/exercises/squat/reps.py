"""Find squat reps in an observed track and measure each one.

Works the same on real (tracked) and synthetic tracks, so the two can be
compared like for like. The signal is the hip's height above the ankle as
a fraction of its standing value, which ignores camera panning and image
scale. A rep is a dip in that signal deep enough to be a squat.

Phase boundaries use a fraction of each rep's own excursion
(``PHASE_THRESHOLD``): the descent starts when the hip has dropped 5% of
the way to its lowest point and reaches the bottom at 95%; the ascent
mirrors that. A fixed-percentage threshold is the usual way to time
phases when the start is not perfectly still. The 5% figure is a
convention, not taken from a study.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from exercise_quality.core.observed import observed_angles, segment_lengths
from exercise_quality.core.signal import lowpass

PHASE_THRESHOLD = 0.05
# A dip counts as a squat rep if the hip drops at least this fraction of
# its standing height above the ankle and the knee bends at least
# MIN_KNEE_FLEXION. Bending over to pick something up drops the hip less
# and bends the knee less.
MIN_DROP = 0.15
MIN_KNEE_FLEXION = 45.0


@dataclass(frozen=True)
class Rep:
    start: int          # descent starts
    bottom_reached: int
    ascent_start: int
    end: int            # ascent ends
    lowest: int         # frame of the lowest hip


def smoothed(track, cutoff_hz=6.0):
    return track.map_points(lambda v: lowpass(v, track.fps, cutoff_hz))


def hip_height(track):
    """Hip height above the ankle, as a fraction of its standing (95th percentile) value."""
    h = track["hip"][:, 1] - track["ankle"][:, 1]
    return h / np.percentile(h, 95)


def find_reps(track, angles=None):
    """Reps in a (smoothed) track, in time order."""
    angles = angles or observed_angles(track)
    h = hip_height(track)
    drop = 1.0 - h
    bottoms, _ = find_peaks(drop, prominence=MIN_DROP, distance=int(0.5 * track.fps))
    bottoms = [b for b in bottoms if angles["knee_flexion"][b] >= MIN_KNEE_FLEXION]
    reps = []
    for j, b in enumerate(bottoms):
        lo = bottoms[j - 1] if j > 0 else 0
        hi = bottoms[j + 1] if j + 1 < len(bottoms) else len(h) - 1
        top_before = lo + int(np.argmax(h[lo:b + 1]))
        top_after = b + int(np.argmax(h[b:hi + 1]))
        rep = _phases(h, top_before, b, top_after)
        if rep is not None:
            reps.append(rep)
    return reps


def _phases(h, top_before, lowest, top_after):
    def crossing(segment, level, first):
        idx = np.flatnonzero(segment <= level)
        if idx.size == 0:
            return None
        return int(idx[0] if first else idx[-1])

    low = h[lowest]
    down = h[top_before:lowest + 1]
    up = h[lowest:top_after + 1]
    exc_down = h[top_before] - low
    exc_up = h[top_after] - low
    if exc_down <= 0 or exc_up <= 0:
        return None
    start = crossing(down, h[top_before] - PHASE_THRESHOLD * exc_down, True)
    reached = crossing(down, low + PHASE_THRESHOLD * exc_down, True)
    leave = crossing(up, low + PHASE_THRESHOLD * exc_up, False)
    end = crossing(up, h[top_after] - PHASE_THRESHOLD * exc_up, False)
    if None in (start, reached, leave, end):
        return None
    return Rep(start=top_before + start, bottom_reached=top_before + reached,
               ascent_start=lowest + leave, end=lowest + end, lowest=lowest)


def _at_knee(knee, values, target):
    """Value when the knee first reaches ``target`` (degrees) on the way down."""
    if target > knee.max():
        return np.nan
    i = int(np.argmax(knee >= target))
    if i == 0:
        return float(values[0])
    frac = (target - knee[i - 1]) / (knee[i] - knee[i - 1])
    return float(values[i - 1] + frac * (values[i] - values[i - 1]))


def rep_features(track, rep, angles=None, leg_length=None):
    """Measurements of one rep, all from tracker-visible points.

    Lengths are normalised so units and image scale drop out: horizontal
    hip and knee travel by leg length (hip to ankle, standing), heel rise
    by foot length.
    """
    angles = angles or observed_angles(track)
    fps = track.fps
    lengths = segment_lengths(track)
    if leg_length is None:
        leg_length = float(np.median(lengths["thigh"] + lengths["shank"]))
    foot_length = float(np.median(lengths["foot"]))
    down = slice(rep.start, rep.lowest + 1)
    whole = slice(rep.start, rep.end + 1)
    knee_down = angles["knee_flexion"][down]
    b = int(rep.start + np.argmax(knee_down))       # deepest knee bend on the way down
    hip_x = track["hip"][:, 0] - track["ankle"][:, 0]
    knee_x = track["knee"][:, 0] - track["toe"][:, 0]
    heel_rise = (track["heel"][:, 1] - track["toe"][:, 1])
    heel_rise = heel_rise - np.median(heel_rise[rep.start:rep.start + max(2, int(0.2 * fps))])

    def peak_ms(name):
        values = angles[name][down]
        return 1000.0 * (int(np.argmax(values)) - (rep.lowest - rep.start)) / fps

    trunk = angles["trunk_inclination"]
    # from the start of the rise, so a hold at the bottom does not count
    ascent = slice(rep.ascent_start, rep.end + 1)
    return {
        "knee_peak": float(angles["knee_flexion"][b]),
        "thigh_at_bottom": float(angles["thigh_inclination"][b]),
        "shank_at_bottom": float(angles["shank_inclination"][b]),
        "ankle_at_bottom": float(angles["ankle_dorsiflexion"][b]),
        "trunk_at_bottom": float(trunk[b]),
        "trunk_peak": float(trunk[whole].max()),
        "trunk_minus_shank_at_bottom": float(angles["trunk_minus_shank"][b]),
        "hip_flexion_at_bottom": float(angles["hip_flexion"][b]),
        "neck_at_bottom": float(angles["neck_inclination"][b]),
        "trunk_at_knee60": _at_knee(knee_down, trunk[down], 60.0),
        "trunk_at_knee90": _at_knee(knee_down, trunk[down], 90.0),
        "ankle_at_knee60": _at_knee(knee_down, angles["ankle_dorsiflexion"][down], 60.0),
        "ankle_at_knee90": _at_knee(knee_down, angles["ankle_dorsiflexion"][down], 90.0),
        # trunk lean gained after the lowest point: hips rising before the chest
        "trunk_rise_after_bottom": float(trunk[ascent].max() - trunk[rep.ascent_start]),
        "hip_back_at_bottom": float((hip_x[rep.start] - hip_x[b]) / leg_length),
        "knee_past_toe_at_bottom": float(knee_x[b] / foot_length),
        "heel_rise_max": float(heel_rise[whole].max() / foot_length),
        "descent_s": (rep.bottom_reached - rep.start) / fps,
        "bottom_s": (rep.ascent_start - rep.bottom_reached) / fps,
        "ascent_s": (rep.end - rep.ascent_start) / fps,
        "hip_drop": float(1.0 - hip_height(track)[rep.lowest]),
        "trunk_peak_ms": peak_ms("trunk_inclination"),
        "knee_peak_ms": peak_ms("knee_flexion"),
        "ankle_peak_ms": peak_ms("ankle_dorsiflexion"),
        "shoulder_flexion_at_bottom": float(angles["shoulder_flexion"][b]),
        "elbow_flexion_at_bottom": float(angles["elbow_flexion"][b]),
    }
