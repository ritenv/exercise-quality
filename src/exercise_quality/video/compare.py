"""Compare tracked real reps with synthetic reps of matching people.

For each real rep the generator is asked for the same task: same depth
(peak knee flexion), same tempo, same arm position, done by ANSUR II
people whose proportions match the person in the video. Everything else
(ankle use, trunk preference, hip end range, weight position) is drawn
from the validated population spreads exactly as in the literature
cohorts. Both sides are reduced to the tracker's view and measured with
the same code (``exercises/squat/reps.py``), so the real rep can be
placed within the distribution of what the model thinks people of that
build do.

Body matching uses scale-free ratios of tracker-visible lengths in
upright standing (thigh, trunk, foot and neck, each over leg length).
Tracker landmarks are not joint centres (MediaPipe's hip sits on the
outside of the hip, its shoulder near the top of the shoulder), so the
ratios carry a definition bias the model cannot remove; they are used
only to rank ANSUR people by similarity, and the report shows where the
video ratios sit in the ANSUR distribution.
"""

from dataclasses import replace

import numpy as np

from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.observed_balance import com_fraction, projected_body
from exercise_quality.core.observed import from_model_landmarks, observed_angles, segment_lengths
from exercise_quality.core.sagittal_model import JointAngles, forward_kinematics
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import reps as R
from exercise_quality.exercises.squat.balanced import ArmPosture, generate_feasible_squat
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_style

RATIOS = ("thigh", "trunk", "foot", "neck")


def standing_proportions(track, angles=None, upright_knee=10.0):
    """Ratios of segment lengths to leg length (thigh + shank), upright frames only."""
    angles = angles or observed_angles(track)
    lengths = segment_lengths(track)
    upright = (np.abs(angles["knee_flexion"]) < upright_knee) & (np.abs(angles["trunk_inclination"]) < 15)
    if upright.sum() < 1:
        upright = angles["knee_flexion"] <= np.percentile(angles["knee_flexion"], 10)
    leg = np.median((lengths["thigh"] + lengths["shank"])[upright])
    return {k: float(np.median(lengths[k][upright]) / leg) for k in RATIOS}


def model_proportions(body):
    standing = forward_kinematics(body, JointAngles.symmetric())
    track = from_model_landmarks(standing, fps=1.0)
    lengths = segment_lengths(track)
    leg = lengths["thigh"][0] + lengths["shank"][0]
    return {k: float(lengths[k][0] / leg) for k in RATIOS}


_TABLES = {}


def _proportion_table(population, sex):
    key = (id(population), sex)
    if key not in _TABLES:
        bodies = population.bodies(sex)
        table = np.array([[model_proportions(b)[r] for r in RATIOS] for b in bodies])
        _TABLES[key] = (population, bodies, table)    # keep population alive so its id stays unique
    return _TABLES[key][1], _TABLES[key][2]


def matched_bodies(population, ratios, sex, k=40, weights=None):
    """The k ANSUR people of ``sex`` whose proportions are closest.

    Distance is in units of each ratio's population SD, so a ratio that
    varies little between people does not dominate. Returns (bodies,
    population ratio table, standardised distance of the best match).
    """
    weights = weights or {r: 1.0 for r in RATIOS}
    bodies, table = _proportion_table(population, sex)
    sd = table.std(axis=0)
    target = np.array([ratios[r] for r in RATIOS])
    w = np.array([weights[r] for r in RATIOS])
    dist = np.sqrt((((table - target) / sd) ** 2 * w).sum(axis=1))
    order = np.argsort(dist)[:k]
    return [bodies[i] for i in order], table, float(dist[order[0]])


def measured_phases(timing, fps=200.0):
    """Descent, bottom and ascent durations of a timing's profile, measured as in reps.py."""
    t = timing.times(fps)
    h = 1.0 - timing.profile(t)
    lowest = int(np.argmin(h))
    rep = R._phases(h, 0, lowest, len(h) - 1)
    return np.array([(rep.bottom_reached - rep.start) / fps,
                     (rep.ascent_start - rep.bottom_reached) / fps,
                     (rep.end - rep.ascent_start) / fps])


def matched_timing(descent_s, bottom_s, ascent_s):
    """RepTiming whose measured phases match the real ones (least squares)."""
    from scipy.optimize import least_squares

    target = np.array([descent_s, bottom_s, ascent_s])

    def build(x):
        d, hold, a = np.abs(x) + np.array([0.2, 0.0, 0.2])
        return RepTiming(stand_before=0.6, descent=d, bottom_hold=hold, ascent=a, stand_after=0.6)

    def residual(x):
        return measured_phases(build(x)) - target

    x0 = np.array([descent_s / 0.64, 0.0, ascent_s / 0.64]) - np.array([0.2, 0.0, 0.2])
    best = least_squares(residual, x0, diff_step=0.05)
    return build(best.x)


def arm_posture(shoulder_flexion, elbow_flexion):
    """Arms fixed to the trunk at the measured angles (both in degrees)."""
    return ArmPosture("trunk", float(shoulder_flexion), float(max(elbow_flexion, 0.0)))


def synthetic_rep(body, knee_peak, timing, arms, seed, fps=30.0, tracker_noise=True,
                  return_track=False):
    """Generate one matched rep and measure it like a tracked one. None if it fails.

    With ``tracker_noise`` the exact landmarks first go through the
    MediaPipe-like noise model (core/tracker_noise.py).
    """
    rng = np.random.default_rng(seed)
    style = sample_style(rng, body, ISHIDA_PARALLEL)
    style = replace(style, peak_knee_flexion=float(knee_peak), timing=timing, arms=arms)
    seq = generate_feasible_squat(body, style, fps=fps)
    if not seq.is_feasible:
        return None
    track = from_model_landmarks(seq.landmarks, fps, source="synthetic")
    if tracker_noise:
        track = TN.apply(track, TN.TrackerNoise.sample(rng), rng)
    track = R.smoothed(track)
    angles = observed_angles(track)
    found = R.find_reps(track, angles)
    if len(found) != 1:
        return None
    features = R.rep_features(track, found[0], angles)
    features["com_at_bottom"] = com_at_bottom(body, track, found[0], angles)
    features["style_trunk_preference"] = style.trunk_peak_preference
    features["style_ankle_ratio"] = style.ankle_knee_ratio
    features["style_com_bottom"] = style.com_target_bottom
    if return_track:
        return features, track, found[0]
    return features


def com_at_bottom(body, track, rep, angles):
    """Estimated COM position on the foot (0 heel, 1 toe) at the deepest knee bend.

    Thigh and shank take their seen length (``projected_body``), so knees
    travelling out over the toes do not push the estimate back.
    """
    bottom = rep.start + int(np.argmax(angles["knee_flexion"][rep.start:rep.lowest + 1]))
    frame = {k: np.asarray(v)[bottom] for k, v in angles.items()}
    return float(com_fraction(projected_body(body, track, bottom, angles), frame))


def percentile(value, sample):
    sample = np.asarray([s for s in sample if np.isfinite(s)])
    if not np.isfinite(value) or sample.size == 0:
        return np.nan
    return float(100.0 * (np.mean(sample < value) + 0.5 * np.mean(sample == value)))
