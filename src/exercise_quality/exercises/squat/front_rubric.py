"""Front-view squat rubric: the two frontal-plane criteria of the Back Squat Assessment.

Grades a front-view MediaPipe track (``video.mediapipe_pose.load_raw``,
or ``video.camera.synthetic_raw`` for a 3D synthetic rep) with the same
code for both. The side-view rubric (``rubric.py``) covers the other
criteria; docs/front_rubric.md explains the choices and results.

Graded criteria (Myer et al. 2014, Strength Cond J 36(6):4-27, Table 1)
-----------------------------------------------------------------------
* hips_level (4): "Line of hips is parallel to ground in frontal plane
  throughout squat." Measured as the largest tilt of the line between
  MediaPipe's two hip points in the image, from the start of the descent
  to the end of the ascent (deg). Standard: 0. The tilt is a vertical
  measure, so a camera a few degrees off square barely changes it.
* knees_over_feet (5): "Lateral aspect of knee does not cross medial
  malleolus for either leg." For each leg, how far the knee centre moves
  towards the midline past the ankle centre, minus half the knee's width
  and half the ankle's width (``Body3D.knee_breadth``, ``ankle_breadth``):
  above 0, the outside of the knee is inside the inner ankle bone. Taken
  over the whole rep, worse leg, in cm (image pixels converted with the
  standing leg's image length against the body's leg length). Standard: 0.

Each criterion fails when its value exceeds the standard by more than
the measurement error (``TOLERANCE``: 95th percentile, from 3D reps
filmed by phones placed as people place them, ``sample_front_camera``,
against the true 3D values; ``measurement_tolerances`` recomputes them).
Deficits follow the BSA rule: a criterion counts once if it fails in 2
or more reps (or the only rep).

Reported, not graded: knee-to-ankle separation at the bottom (KASR),
hip shift, left-right depth difference, stance width, the camera angle.

Camera angle
------------
A knee ahead of the ankle shows up sideways in the image when the camera
is turned, one knee towards the midline and the other away. So the knee
criterion needs the camera near square. MediaPipe's 3D hips read the
angle too high: on the five front-view clips its hip axis was 11-16 deg
(mean 13) further round than the angle that makes the clip's left-right
knee angles match a symmetric person (docs/front_view_check.md). That
offset (``MEDIAPIPE_FRONT_YAW_BIAS``) is removed for real clips; synthetic
clips (which carry their ``camera``) have exact world landmarks. Beyond
``MAX_FRONT_YAW`` the result is marked unreliable. The hip check is less
sensitive to it (synthetic reps: about 0.14 deg of extra tilt per degree
off square) but at 20 deg that is already 3 deg.
"""

from dataclasses import dataclass, field

import numpy as np

from exercise_quality.core import model3d as M
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.video import front_view as F

CRITERIA = ("hips_level", "knees_over_feet")
# 95th percentile of the signed measurement error: mean of
# measurement_tolerances(n=300) with seeds 0 and 1 (each the pooled errors
# of two seeds of 150 reps).
TOLERANCE = {
    "hips_level": 4.9,          # deg; tracker drift, the largest tilt over a rep picks up its peaks
    "knees_over_feet": 3.1,     # cm
}
MEDIAPIPE_FRONT_YAW_BIAS = 13.0
MAX_FRONT_YAW = 10.0
MIN_REPS_FOR_RULE = 2
NOT_GRADED = {
    "knees in line with the toes": "MediaPipe's 3D foot directions are lopsided from the front (the right "
                                   "foot reads 25-35 deg more turned out than the left), so the toe line "
                                   "is not known; the knee's path is reported instead",
    "hip shift": "not a BSA criterion and no published norm; reported",
}


@dataclass
class FrontCriterion:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


@dataclass
class FrontRep:
    index: int
    criteria: dict
    features: dict


@dataclass
class FrontResult:
    reps: list
    deficits: dict
    pass_share: dict
    score: float
    reliable: bool
    warnings: list = field(default_factory=list)
    camera_angle: float = 0.0
    reported: dict = field(default_factory=dict)

    @property
    def deficit_total(self):
        return int(sum(self.deficits.values()))


_POPULATION = []


def _population():
    if not _POPULATION:
        _POPULATION.append(BodyPopulation.load())
    return _POPULATION[0]


def camera_angle(raw, segment=None):
    """Camera angle from square (deg), MediaPipe's hip axis with its front-view bias removed."""
    sl = slice(*segment) if segment else slice(None)
    yaw = float(np.nanmedian(F.view_yaw(raw)[sl]))
    return yaw if raw.get("camera") is not None else yaw - MEDIAPIPE_FRONT_YAW_BIAS


def frame_measures(xy, leg_px, body):
    """Per-frame hip tilt (deg, + = person's right hip lower) and knee crossing (cm, per leg)."""
    dy = xy[:, F.R_HIP, 1] - xy[:, F.L_HIP, 1]
    dx = xy[:, F.L_HIP, 0] - xy[:, F.R_HIP, 0]
    tilt = np.degrees(np.arctan2(dy, dx))
    metres_per_px = (body.thigh + body.shank) / leg_px
    half = 0.5 * (body.knee_breadth + body.ankle_breadth)
    # facing the camera, the person's right is image left: the right knee moving
    # towards the midline moves image right
    right = (xy[:, F.R_KNEE, 0] - xy[:, F.R_ANKLE, 0]) * metres_per_px - half
    left = (xy[:, F.L_ANKLE, 0] - xy[:, F.L_KNEE, 0]) * metres_per_px - half
    return tilt, 100.0 * right, 100.0 * left


def assess(raw, body=None, sex=None, segment=None):
    """Grade every rep in the front-view part of ``raw``. None if there is no front view.

    ``body``: a ``Body3D`` (knee and ankle widths, leg length); otherwise
    the median ANSUR II body of ``sex`` (default female).
    """
    if body is None:
        body = M.median_body3d(_population(), sex or "female")
    if segment is None:
        segs = F.front_segments(raw)
        if not segs:
            return None
        segment = max(segs, key=lambda s: s[1] - s[0])
    fps = float(raw["fps"])
    xy = F._smooth_image(raw, slice(*segment))
    if not F._facing_camera(xy):
        return None
    h = F.hip_height(xy)
    standing = h > 0.97
    leg_px = float(np.median((0.5 * (xy[:, F.L_ANKLE, 1] + xy[:, F.R_ANKLE, 1])
                              - 0.5 * (xy[:, F.L_HIP, 1] + xy[:, F.R_HIP, 1]))[standing]))
    tilt, right, left = frame_measures(xy, leg_px, body)
    ankle_sep = np.abs(xy[:, F.L_ANKLE, 0] - xy[:, F.R_ANKLE, 0])
    knee_sep = np.abs(xy[:, F.L_KNEE, 0] - xy[:, F.R_KNEE, 0])
    angle = camera_angle(raw, segment)
    warnings = []
    reliable = abs(angle) <= MAX_FRONT_YAW
    if not reliable:
        warnings.append(f"camera about {angle:+.0f} deg from square; a knee ahead of the ankle looks "
                        f"sideways from there and the hips sit at different distances, so neither "
                        f"check is trustworthy")
    reps = []
    for i, r in enumerate(F.find_reps(xy, fps)):
        sl = slice(r.start, r.end + 1)
        b = r.lowest
        values = {"hips_level": float(np.max(np.abs(tilt[sl]))),
                  "knees_over_feet": float(max(np.max(right[sl]), np.max(left[sl])))}
        criteria = {name: FrontCriterion(name, v, TOLERANCE[name], bool(v <= TOLERANCE[name]))
                    for name, v in values.items()}
        criteria["hips_level"].note = "largest hip-line tilt during the rep (deg); standard 0"
        criteria["knees_over_feet"].note = ("outside of the knee past the inner ankle bone, worse leg (cm); "
                                            "standard 0")
        mid_hip = 0.5 * (xy[b, F.L_HIP, 0] + xy[b, F.R_HIP, 0])
        mid_ankle = 0.5 * (xy[b, F.L_ANKLE, 0] + xy[b, F.R_ANKLE, 0])
        features = {
            "kasr_bottom": float(knee_sep[b] / ankle_sep[b]),
            "knee_crossing_right_cm": float(np.max(right[sl])),
            "knee_crossing_left_cm": float(np.max(left[sl])),
            "tilt_at_bottom": float(tilt[b]),
            "hip_shift": float(-(mid_hip - mid_ankle) / ankle_sep[b]),
            "depth": float(1.0 - h[b]),
        }
        reps.append(FrontRep(i, criteria, features))
    if not reps:
        return FrontResult([], {c: False for c in CRITERIA}, {c: 1.0 for c in CRITERIA}, 100.0, False,
                           warnings + ["no reps found"], angle)
    fails = {c: sum(not r.criteria[c].passed for r in reps) for c in CRITERIA}
    need = min(MIN_REPS_FOR_RULE, len(reps))
    deficits = {c: fails[c] >= need for c in CRITERIA}
    pass_share = {c: 1.0 - fails[c] / len(reps) for c in CRITERIA}
    score = 100.0 * float(np.mean(list(pass_share.values())))
    standing_tilt = float(np.median(tilt[standing]))
    reported = {
        "kasr_bottom": float(np.median([r.features["kasr_bottom"] for r in reps])),
        "hip_shift": float(np.median([r.features["hip_shift"] for r in reps])),
        "standing_tilt": standing_tilt,
        "stance_over_leg": float(np.median(ankle_sep[standing]) / leg_px),
    }
    return FrontResult(reps, deficits, pass_share, score, reliable, warnings, angle, reported)


# --- measurement error --------------------------------------------------------

# Where people put a phone for a front-view video (ASSUMPTION, uniform):
# within 8 deg of square, 3-5 m away, lens 0.4-1.3 m high.
FRONT_AZIMUTH_SPREAD = 8.0


def sample_front_camera(rng):
    from exercise_quality.video import camera as cam

    return cam.Camera(azimuth=float(rng.uniform(-FRONT_AZIMUTH_SPREAD, FRONT_AZIMUTH_SPREAD)),
                      distance=float(rng.uniform(*cam.SIDE_DISTANCE)),
                      lens_height=float(rng.uniform(*cam.SIDE_LENS_HEIGHT)))


def drop_hip(raw, degrees, side="right"):
    """A copy of ``raw`` with one MediaPipe hip point lowered, growing with depth.

    The hip line tilts by ``degrees`` at the bottom (``side`` hip lower),
    in proportion to how far the hips have dropped. Only the image points
    change. A test fault: the 3D model has no pelvic drop of its own.
    """
    out = dict(raw)
    image = raw["image"].copy()
    xy = image[:, :, :2]
    h = F.hip_height(xy)
    progress = np.clip((1.0 - h) / max(1.0 - h.min(), 1e-6), 0.0, 1.0)
    sep = np.abs(xy[:, F.L_HIP, 0] - xy[:, F.R_HIP, 0])
    i = F.R_HIP if side == "right" else F.L_HIP
    image[:, i, 1] += sep * np.tan(np.radians(degrees)) * progress
    out["image"] = image
    return out


def true_values(seq, start, end):
    """The criteria's true values from the 3D landmarks, frames ``start``..``end``."""
    L = seq.landmarks3d
    body = seq.body
    sl = slice(start, end + 1)
    hip = L["right_hip"][sl] - L["left_hip"][sl]          # +z = the person's right
    tilt = np.degrees(np.arctan2(-hip[:, 1], np.hypot(hip[:, 0], hip[:, 2])))
    half = 0.5 * (body.knee_breadth + body.ankle_breadth)
    right = -(L["right_knee"][sl, 2] - L["right_ankle"][sl, 2]) - half
    left = (L["left_knee"][sl, 2] - L["left_ankle"][sl, 2]) - half
    return {"hips_level": float(np.max(np.abs(tilt))),
            "knees_over_feet": 100.0 * float(max(np.max(right), np.max(left)))}


def synthetic_front_rep(rng, stance=None, protocol=None, camera=None, noise=True):
    """(sequence, raw) of one 3D rep filmed from the front, or None."""
    from dataclasses import replace

    from exercise_quality.core import tracker_noise as TN
    from exercise_quality.core.timing import RepTiming
    from exercise_quality.exercises.squat.balanced import generate_feasible_squat
    from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_stance, sample_style
    from exercise_quality.video import camera as cam

    body = M.sample_body3d(_population(), rng)
    style = sample_style(rng, body.sagittal, protocol or ISHIDA_PARALLEL)
    style = replace(style, timing=RepTiming(stand_before=0.8, descent=rng.uniform(1.0, 2.5),
                                            ascent=rng.uniform(0.8, 2.5), stand_after=0.8),
                    stance=stance or sample_stance(rng))
    seq = generate_feasible_squat(body, style, fps=30.0)
    if not seq.is_feasible:
        return None
    camera = camera or sample_front_camera(rng)
    raw = cam.synthetic_raw(seq, camera, noise=TN.TrackerNoise.sample(rng) if noise else None, rng=rng)
    return seq, raw


def measurement_tolerances(n=300, seed=0, quantile=95.0, raw=False):
    """Recompute ``TOLERANCE``: graded value minus the true 3D value over filmed synthetic reps."""
    from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH

    rng = np.random.default_rng(seed)
    errors = {c: [] for c in CRITERIA}
    while len(errors["hips_level"]) < n:
        protocol = ISHIDA_PARALLEL if rng.random() < 0.5 else ZAWADKA_MAX_DEPTH
        made = synthetic_front_rep(rng, protocol=protocol)
        if made is None:
            continue
        seq, film = made
        result = assess(film, body=seq.body, segment=(0, len(seq.times)))
        if result is None or len(result.reps) != 1:
            continue
        found = F.find_reps(F._smooth_image(film, slice(None)), float(film["fps"]))[0]
        truth = true_values(seq, found.start, found.end)
        for c in CRITERIA:
            errors[c].append(result.reps[0].criteria[c].value - truth[c])
    if raw:
        return errors
    # only errors that make a rep look worse than it is can fail a good rep
    return {c: float(np.percentile(v, quantile)) for c, v in errors.items()}
