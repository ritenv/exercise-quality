"""What a 2D pose tracker can see of a person, and angles measured from it.

Real video goes through MediaPipe Pose, which gives a fixed set of
surface landmarks. The synthetic model has more points (spine joints, the
ball of the foot) and exact joint centres. To compare the two fairly, or
to grade either with the same rubric, both are reduced to an
``ObservedTrack``: the near-side landmarks a tracker reports, in the
model's frame (person facing +x, y up), with any units and origin.

Only angles that can be computed from these points are measured here.
They are *tracker* definitions and differ from the model's anatomical
angles in ``sagittal_model.measure_joint_angles``:

* trunk is the hip-to-shoulder line, not the rib cage (no spine points);
* hip angle is the trunk line relative to the thigh, so it mixes true
  hip flexion with lumbar flexion and pelvic tilt;
* ankle is the shank relative to the foot's *standing* heel-to-toe line
  (``foot_reference``), not frame by frame: tracked heel and toe points
  jitter by several degrees and an oblique camera tilts the foot line by
  perspective, while a flat foot does not actually change pitch. Heel
  lift is measured separately, as the change in the foot line.

Mapping from the model (``from_model_landmarks``): every point is the
model landmark of the same name on the near (right) side, the tracker's
"foot index" is the toe tip, and the ear is placed on the C7-to-vertex
line at ``EAR_FRACTION`` of its length. That ear position is an
ASSUMPTION (tragion sits roughly mid-way up the head and neck in side
view); it is only used for head position.
"""

from dataclasses import dataclass, field, replace

import numpy as np

POINTS = ("ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe")
EAR_FRACTION = 0.55


@dataclass(frozen=True)
class ObservedTrack:
    points: dict            # name -> (T, 2), x forward, y up
    fps: float
    units: str = "m"
    visibility: dict = field(default_factory=dict)   # name -> (T,), 0..1
    source: str = ""
    # Horizontal foreshortening that has already been undone (1 = side-on).
    yaw_scale: float = 1.0

    def __post_init__(self):
        missing = [p for p in POINTS if p not in self.points]
        if missing:
            raise ValueError(f"missing points: {missing}")
        lengths = {len(v) for v in self.points.values()}
        if len(lengths) != 1:
            raise ValueError("all points need the same number of frames")

    def __len__(self):
        return len(self.points["hip"])

    @property
    def times(self):
        return np.arange(len(self)) / self.fps

    def __getitem__(self, name):
        return self.points[name]

    def map_points(self, fn):
        return replace(self, points={k: fn(v) for k, v in self.points.items()})

    def slice(self, start, stop):
        return replace(
            self,
            points={k: v[start:stop] for k, v in self.points.items()},
            visibility={k: v[start:stop] for k, v in self.visibility.items()},
        )


def from_model_landmarks(landmarks, fps, side="right", source="model"):
    """Reduce model landmarks (name -> (T, 2) or (2,)) to a tracker view."""
    lm = {k: np.atleast_2d(np.asarray(v, dtype=float)) for k, v in landmarks.items()}
    ear = lm["neck"] + EAR_FRACTION * (lm["vertex"] - lm["neck"])
    names = {"shoulder": "shoulder", "elbow": "elbow", "wrist": "wrist", "hip": "hip",
             "knee": "knee", "ankle": "ankle", "heel": "heel", "toe": "toe"}
    points = {k: lm[f"{side}_{v}"].copy() for k, v in names.items()}
    points["ear"] = ear
    return ObservedTrack(points=points, fps=float(fps), units="m", source=source)


def inclination(bottom, top):
    """Angle of bottom->top from vertical-up, degrees, positive when top is forward."""
    d = np.asarray(top) - np.asarray(bottom)
    return np.degrees(np.arctan2(d[..., 0], d[..., 1]))


def hanging_angle(top, bottom):
    """Angle of top->bottom from vertical-down, degrees, positive when swung forward."""
    d = np.asarray(bottom) - np.asarray(top)
    return np.degrees(np.arctan2(d[..., 0], -d[..., 1]))


def segment_lengths(track):
    p = track.points
    pairs = {"shank": ("ankle", "knee"), "thigh": ("knee", "hip"),
             "trunk": ("hip", "shoulder"), "neck": ("shoulder", "ear"),
             "foot": ("heel", "toe"), "upper_arm": ("shoulder", "elbow"),
             "forearm": ("elbow", "wrist")}
    return {k: np.linalg.norm(p[b] - p[a], axis=-1) for k, (a, b) in pairs.items()}


def wrap(angle):
    """Wrap degrees into (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(angle), 360.0)


def observed_angles(track, foot_pitch_reference=None):
    """Tracker-definition angles in degrees, each (T,).

    ``foot_pitch_reference``: standing foot-line pitch; measured from the
    track when None. ``heel_lift`` is the foot-line pitch above it.
    """
    p = track.points
    shank = inclination(p["ankle"], p["knee"])
    thigh = inclination(p["knee"], p["hip"])
    trunk = inclination(p["hip"], p["shoulder"])
    neck = inclination(p["shoulder"], p["ear"])
    foot = p["toe"] - p["heel"]
    foot_pitch = np.degrees(np.arctan2(-foot[..., 1], foot[..., 0]))
    if foot_pitch_reference is None:
        knee = shank - thigh
        upright = knee < 15.0
        reference = float(np.median(foot_pitch[upright] if upright.sum() >= 5 else foot_pitch))
    else:
        reference = float(foot_pitch_reference)
    upper_arm = hanging_angle(p["shoulder"], p["elbow"])
    forearm = hanging_angle(p["elbow"], p["wrist"])
    return {
        "shank_inclination": shank,
        "thigh_inclination": thigh,
        "trunk_inclination": trunk,
        "neck_inclination": neck,
        "foot_pitch": foot_pitch,
        "knee_flexion": shank - thigh,
        "ankle_dorsiflexion": shank - reference,
        "heel_lift": foot_pitch - reference,
        "hip_flexion": trunk - thigh,
        "trunk_minus_shank": trunk - shank,
        "shoulder_flexion": wrap(upper_arm + trunk),
        "elbow_flexion": wrap(forearm - upper_arm),
    }
