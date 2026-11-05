"""Side-view body model driven by anatomical joint angles.

The chain is rooted at a flat foot on the floor:

    foot (heel .. toe, on the floor)
      -> ankle (lateral malleolus height above the floor)
        -> knee -> hip -> lumbar (navel level) -> C7 ("neck") -> vertex
                                   shoulder (on the thorax line, below C7)
                                     -> elbow -> wrist -> hand (3rd metacarpal head)

The trunk is two rigid segments: the pelvis (hips to navel level) and the
thorax (navel level to C7), joined at a single lumbar joint. Real lumbar
flexion is spread over five vertebrae; one joint at the navel is the
simplest split that matches de Leva's segment data.

Left and right sides overlap in a side view, so there is one leg and one
arm, each carrying the mass of both sides (see core/balance.py).

All joint angles are zero in upright standing with the arms hanging, and
positive in the usual anatomical direction. See docs/conventions.md.

Every function accepts scalar or array angles. With arrays (for example
many frames, or many candidate trunk angles) every landmark comes back
with shape (..., 2).
"""

from dataclasses import dataclass, fields

import numpy as np

from exercise_quality.core.geometry import (
    down_vector,
    inclination_from_vertical,
    up_vector,
    wrap_degrees,
)

LANDMARKS = (
    "heel", "toe", "ankle", "knee", "hip", "lumbar", "neck", "vertex",
    "shoulder", "elbow", "wrist", "hand",
)

# de Leva segment name -> (origin landmark, other landmark). The order
# matters: de Leva measures each centre of mass from the origin.
SEGMENT_ENDPOINTS = {
    "head": ("vertex", "neck"),
    "thorax": ("neck", "lumbar"),
    "pelvis": ("lumbar", "hip"),
    "upper_arm": ("shoulder", "elbow"),
    "forearm": ("elbow", "wrist"),
    "hand": ("wrist", "hand"),
    "thigh": ("hip", "knee"),
    "shank": ("knee", "ankle"),
    "foot": ("heel", "toe"),
}

CONNECTIONS = (
    ("heel", "toe"), ("heel", "ankle"), ("ankle", "toe"),
    ("ankle", "knee"), ("knee", "hip"), ("hip", "lumbar"), ("lumbar", "neck"), ("neck", "vertex"),
    ("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "hand"),
)


@dataclass(frozen=True)
class JointAngles:
    """Anatomical joint angles in degrees. All zero = upright standing.

    ankle_dorsiflexion: shank tilting forward over a flat foot.
    knee_flexion: knee bending (0 straight).
    hip_flexion: pelvis tipping forward relative to the thigh (true hip
        flexion, pelvis vs femur).
    neck_flexion: head tilting forward relative to the thorax (negative
        is extension, for example looking ahead while the trunk leans).
    shoulder_flexion: upper arm raised forward relative to the thorax
        (90 = perpendicular to the thorax, pointing forward).
    elbow_flexion: forearm bending forward/up from the upper arm.
    wrist_flexion: hand relative to the forearm.
    lumbar_flexion: thorax bending forward relative to the pelvis
        (rounding the lower back; negative is extension).
    """

    ankle_dorsiflexion: float = 0.0
    knee_flexion: float = 0.0
    hip_flexion: float = 0.0
    neck_flexion: float = 0.0
    shoulder_flexion: float = 0.0
    elbow_flexion: float = 0.0
    wrist_flexion: float = 0.0
    lumbar_flexion: float = 0.0

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def segment_orientations(angles):
    """Absolute segment angles (degrees) from joint angles.

    Leg, pelvis, thorax and head: inclination from vertical-up, positive = leaning
    forward. Arm segments: angle from vertical-down, positive = swung
    forward.
    """
    a = angles
    shank = np.asarray(a.ankle_dorsiflexion, dtype=float)
    thigh = shank - a.knee_flexion
    pelvis = thigh + a.hip_flexion
    thorax = pelvis + a.lumbar_flexion
    head = thorax + a.neck_flexion
    # Hanging along a forward-leaning thorax means pointing down and
    # backwards, i.e. a negative angle from vertical-down.
    upper_arm = a.shoulder_flexion - thorax
    forearm = upper_arm + a.elbow_flexion
    hand = forearm + a.wrist_flexion
    return {
        "shank": shank, "thigh": thigh, "pelvis": pelvis, "thorax": thorax,
        "head": head,
        "upper_arm": upper_arm, "forearm": forearm, "hand": hand,
    }


def forward_kinematics(body, angles):
    """Landmark positions (metres) for a body in a given set of angles.

    Origin: on the floor directly below the ankle joint centre.
    """
    o = segment_orientations(angles)
    shape = np.broadcast(*o.values()).shape

    def fixed(x, y):
        return np.broadcast_to(np.array([x, y], dtype=float), shape + (2,))

    heel = fixed(-body.heel_offset, 0.0)
    toe = fixed(body.toe_offset, 0.0)
    ankle = fixed(0.0, body.ankle_height)

    knee = ankle + body.shank * up_vector(o["shank"])
    hip = knee + body.thigh * up_vector(o["thigh"])
    lumbar = hip + body.pelvis * up_vector(o["pelvis"])
    thorax_up = up_vector(o["thorax"])
    neck = lumbar + body.thorax * thorax_up
    vertex = neck + body.head * up_vector(o["head"])
    shoulder = neck - body.shoulder_drop * thorax_up
    elbow = shoulder + body.upper_arm * down_vector(o["upper_arm"])
    wrist = elbow + body.forearm * down_vector(o["forearm"])
    hand = wrist + body.hand * down_vector(o["hand"])

    return {
        "heel": heel, "toe": toe, "ankle": ankle, "knee": knee, "hip": hip,
        "lumbar": lumbar, "neck": neck, "vertex": vertex, "shoulder": shoulder,
        "elbow": elbow, "wrist": wrist, "hand": hand,
    }


def segment_lengths(body):
    """Length of every segment in SEGMENT_ENDPOINTS, in metres."""
    return {
        "head": body.head, "thorax": body.thorax, "pelvis": body.pelvis,
        "upper_arm": body.upper_arm, "forearm": body.forearm, "hand": body.hand,
        "thigh": body.thigh, "shank": body.shank, "foot": body.foot_length,
    }


def landmarks_to_array(landmarks):
    """Stack a landmark dict into an array of shape (..., len(LANDMARKS), 2)."""
    return np.stack([landmarks[name] for name in LANDMARKS], axis=-2)


def measure_joint_angles(landmarks):
    """Recover joint angles from landmark positions (inverse of the FK).

    Uses only geometry, so it can be applied to any landmarks with the
    same meaning, for example a pose estimator's output in future.
    """
    def incl(a, b):
        return inclination_from_vertical(landmarks[b] - landmarks[a])

    def arm(a, b):
        # angle from vertical-down, positive when swung forward
        v = landmarks[b] - landmarks[a]
        return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))

    # foot pitch, positive when the toe is higher than the heel; a raised
    # toe tilts the foot's normal backwards, which adds to dorsiflexion
    heel_to_toe = landmarks["toe"] - landmarks["heel"]
    foot = np.degrees(np.arctan2(heel_to_toe[..., 1], heel_to_toe[..., 0]))
    shank = incl("ankle", "knee")
    thigh = incl("knee", "hip")
    pelvis = incl("hip", "lumbar")
    thorax = incl("lumbar", "neck")
    head = incl("neck", "vertex")
    upper_arm = arm("shoulder", "elbow")
    forearm = arm("elbow", "wrist")
    hand = arm("wrist", "hand")
    w = wrap_degrees
    return {
        "ankle_dorsiflexion": w(shank + foot),
        "knee_flexion": w(shank - thigh),
        "hip_flexion": w(pelvis - thigh),
        "lumbar_flexion": w(thorax - pelvis),
        "neck_flexion": w(head - thorax),
        "shoulder_flexion": w(upper_arm + thorax),
        "elbow_flexion": w(forearm - upper_arm),
        "wrist_flexion": w(hand - forearm),
        # "trunk lean" in squat studies is the thorax against vertical
        "trunk_inclination": thorax,
        "pelvis_inclination": pelvis,
        "shank_inclination": shank,
        "thigh_inclination": thigh,
    }
