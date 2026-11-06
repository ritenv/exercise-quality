"""Side-view body model driven by anatomical joint angles.

The chain is rooted at the stance foot, with its toes flat on the floor:

    toes (ball .. toe tip, on the floor)
      -> rearfoot (heel .. ball; lifts at the heel with toe extension)
        -> ankle -> knee -> hip
             -> pelvis -> lumbar (navel) -> abdomen -> thoracolumbar (xiphoid)
                  -> thorax (rib cage) -> neck (C7) -> head (vertex)
                         shoulders (on the thorax line, below C7)
                           -> elbow -> wrist -> hand (3rd metacarpal head)
             -> other leg: hip -> knee -> ankle -> rearfoot -> toes

Trunk: three rigid segments matching de Leva's sub-segments. The pelvis
runs from the hip joint centres to navel level, the abdomen from the
navel to the xiphoid, the thorax (rib cage) from the xiphoid to C7.
Two spine joints join them: ``lower_lumbar`` at the navel and
``upper_lumbar`` at the xiphoid. Real flexion is spread over many
vertebrae; two joints let the rib cage turn further than the line from
navel to C7 when the back rounds, which is what marker-based studies
measure as the thorax.

Feet: the rearfoot (heel, ankle, ball of the foot) and the toes are
joined at the ball (metatarsophalangeal joint). Toe extension with the
toes flat lifts the heel.

Sides: the person faces +x, and the camera looks at their right side,
so "right" is the near side. Both hip joint centres, and both shoulder
joint centres, project onto the same point in the side view. Each limb
carries its own mass (see core/balance.py).

All joint angles are zero in upright standing with the arms hanging, and
positive in the usual anatomical direction. See docs/conventions.md.

Every function accepts scalar or array angles. With arrays (for example
many frames, or many candidate trunk angles) every landmark comes back
with shape (..., 2).
"""

from dataclasses import dataclass, field, fields

import numpy as np

from exercise_quality.core.geometry import (
    down_vector,
    inclination_from_vertical,
    up_vector,
    wrap_degrees,
)

SIDES = ("right", "left")
NEAR_SIDE = "right"

LEG_POINTS = ("heel", "ball", "toe", "ankle", "knee", "hip")
ARM_POINTS = ("shoulder", "elbow", "wrist", "hand")
MIDLINE_POINTS = ("lumbar", "thoracolumbar", "neck", "vertex")

LANDMARKS = (
    tuple(f"{s}_{p}" for s in SIDES for p in LEG_POINTS)
    + MIDLINE_POINTS
    + tuple(f"{s}_{p}" for s in SIDES for p in ARM_POINTS)
)

# Share of lumbar flexion taken at the upper joint (xiphoid level), the
# rest at the navel. The upper joint stands for the L1-L3 levels, the
# navel joint for L3-S1. Sagittal ranges of motion by level, as tabulated
# by Cheng et al. 2013 (Evid Based Spine Care J 4(1):6-12, Table 3) for
# their healthy group and for Pearcy 1984, Dvorak 1991 and White and
# Panjabi 1990, put 34-38% of lumbar motion at L1-L3. That table has no
# L5-S1 row; L5-S1 is taken as equal to L4-L5. Using motion ranges as the
# split during a squat is an ASSUMPTION.
UPPER_LUMBAR_SHARE = 0.35


def _segments():
    out = {
        "head": ("head", "vertex", "neck"),
        "thorax": ("thorax", "neck", "thoracolumbar"),
        "abdomen": ("abdomen", "thoracolumbar", "lumbar"),
        # both hip centres coincide in the side view
        "pelvis": ("pelvis", "lumbar", f"{NEAR_SIDE}_hip"),
    }
    for s in SIDES:
        out.update({
            f"{s}_upper_arm": ("upper_arm", f"{s}_shoulder", f"{s}_elbow"),
            f"{s}_forearm": ("forearm", f"{s}_elbow", f"{s}_wrist"),
            f"{s}_hand": ("hand", f"{s}_wrist", f"{s}_hand"),
            f"{s}_thigh": ("thigh", f"{s}_hip", f"{s}_knee"),
            f"{s}_shank": ("shank", f"{s}_knee", f"{s}_ankle"),
            # de Leva's whole foot; its mass rides on the rearfoot (see balance.py)
            f"{s}_foot": ("foot", f"{s}_heel", f"{s}_ball"),
        })
    return out


# model segment -> (de Leva segment, origin landmark, other landmark). The
# order matters: de Leva measures each centre of mass from the origin.
SEGMENTS = _segments()
SEGMENT_ENDPOINTS = {name: (a, b) for name, (_, a, b) in SEGMENTS.items()}


def _connections():
    out = []
    for s in SIDES:
        p = lambda n: f"{s}_{n}"  # noqa: E731
        out += [(p("heel"), p("ball")), (p("ball"), p("toe")), (p("heel"), p("ankle")),
                (p("ankle"), p("ball")), (p("ankle"), p("knee")), (p("knee"), p("hip")),
                (p("hip"), "lumbar"),
                (p("shoulder"), p("elbow")), (p("elbow"), p("wrist")), (p("wrist"), p("hand"))]
    out += [("lumbar", "thoracolumbar"), ("thoracolumbar", "neck"), ("neck", "vertex")]
    return tuple(out)


CONNECTIONS = _connections()


@dataclass(frozen=True)
class LegAngles:
    """hip_flexion: pelvis tipping forward relative to this thigh.
    knee_flexion: knee bending (0 straight).
    ankle_dorsiflexion: shank tilting forward over the rearfoot.
    toe_extension: toes bending up relative to the rearfoot; with the toes
        flat on the floor this lifts the heel.
    """

    hip_flexion: float = 0.0
    knee_flexion: float = 0.0
    ankle_dorsiflexion: float = 0.0
    toe_extension: float = 0.0


@dataclass(frozen=True)
class ArmAngles:
    """shoulder_flexion: upper arm raised forward relative to the thorax
        (90 = perpendicular to the thorax, pointing forward).
    elbow_flexion: forearm bending forward/up from the upper arm.
    wrist_flexion: hand relative to the forearm.
    """

    shoulder_flexion: float = 0.0
    elbow_flexion: float = 0.0
    wrist_flexion: float = 0.0


LEG_ANGLES = tuple(f.name for f in fields(LegAngles))
ARM_ANGLES = tuple(f.name for f in fields(ArmAngles))
SPINE_ANGLES = ("lower_lumbar_flexion", "upper_lumbar_flexion", "neck_flexion")


@dataclass(frozen=True)
class JointAngles:
    """Anatomical joint angles in degrees. All zero = upright standing.

    lower_lumbar_flexion: abdomen bending forward on the pelvis (navel).
    upper_lumbar_flexion: thorax bending forward on the abdomen (xiphoid).
    neck_flexion: head tilting forward relative to the thorax (negative
        is extension, for example looking ahead while the trunk leans).
    stance: the side whose foot is the root of the chain (toes flat on
        the floor). The other foot goes wherever its angles put it.

    ``JointAngles.symmetric(...)`` builds a pose with both sides equal.
    Reading a limb angle without a side (``angles.knee_flexion``) gives
    the mean of the two sides; ``angles.lumbar_flexion`` is the total of
    both spine joints (thorax relative to pelvis).
    """

    right_leg: LegAngles = field(default_factory=LegAngles)
    left_leg: LegAngles = field(default_factory=LegAngles)
    right_arm: ArmAngles = field(default_factory=ArmAngles)
    left_arm: ArmAngles = field(default_factory=ArmAngles)
    lower_lumbar_flexion: float = 0.0
    upper_lumbar_flexion: float = 0.0
    neck_flexion: float = 0.0
    stance: str = NEAR_SIDE

    def __post_init__(self):
        if self.stance not in SIDES:
            raise ValueError(f"stance must be one of {SIDES}")

    @classmethod
    def symmetric(cls, lumbar_flexion=None, lower_lumbar_flexion=0.0,
                  upper_lumbar_flexion=0.0, neck_flexion=0.0, **limb):
        """Both sides equal. ``lumbar_flexion``, if given, is the total spine
        flexion split between the two joints by UPPER_LUMBAR_SHARE."""
        unknown = set(limb) - set(LEG_ANGLES) - set(ARM_ANGLES)
        if unknown:
            raise TypeError(f"unknown angles: {sorted(unknown)}")
        if lumbar_flexion is not None:
            upper_lumbar_flexion = UPPER_LUMBAR_SHARE * np.asarray(lumbar_flexion, float)
            lower_lumbar_flexion = np.asarray(lumbar_flexion, float) - upper_lumbar_flexion
        leg = LegAngles(**{k: v for k, v in limb.items() if k in LEG_ANGLES})
        arm = ArmAngles(**{k: v for k, v in limb.items() if k in ARM_ANGLES})
        return cls(leg, leg, arm, arm, lower_lumbar_flexion, upper_lumbar_flexion, neck_flexion)

    def leg(self, side):
        return getattr(self, f"{side}_leg")

    def arm(self, side):
        return getattr(self, f"{side}_arm")

    def __getattr__(self, name):
        # bilateral mean for unprefixed limb angles
        if name in LEG_ANGLES:
            return 0.5 * (np.asarray(getattr(self.right_leg, name))
                          + np.asarray(getattr(self.left_leg, name)))
        if name in ARM_ANGLES:
            return 0.5 * (np.asarray(getattr(self.right_arm, name))
                          + np.asarray(getattr(self.left_arm, name)))
        raise AttributeError(name)

    @property
    def lumbar_flexion(self):
        return np.asarray(self.lower_lumbar_flexion) + np.asarray(self.upper_lumbar_flexion)

    def as_dict(self):
        """Flat dict of every joint angle, e.g. "right_knee_flexion"."""
        out = {}
        for s in SIDES:
            for n in LEG_ANGLES:
                out[f"{s}_{n}"] = getattr(self.leg(s), n)
            for n in ARM_ANGLES:
                out[f"{s}_{n}"] = getattr(self.arm(s), n)
        for n in SPINE_ANGLES:
            out[n] = getattr(self, n)
        return out

    @classmethod
    def from_dict(cls, values, stance=NEAR_SIDE):
        side = {}
        for s in SIDES:
            side[f"{s}_leg"] = LegAngles(**{n: values.get(f"{s}_{n}", 0.0) for n in LEG_ANGLES})
            side[f"{s}_arm"] = ArmAngles(**{n: values.get(f"{s}_{n}", 0.0) for n in ARM_ANGLES})
        return cls(**side, **{n: values.get(n, 0.0) for n in SPINE_ANGLES}, stance=stance)


def _other(side):
    return SIDES[1 - SIDES.index(side)]


def segment_orientations(angles):
    """Absolute segment angles (degrees) from joint angles.

    Leg, trunk and head segments: inclination from vertical-up, positive
    = leaning forward. Arm segments: angle from vertical-down, positive =
    swung forward. Feet ("{side}_rearfoot", "{side}_toes"): pitch from
    horizontal, positive when the back end is higher (heel up, or toe tip
    pointing down).
    """
    a = angles
    st, sw = a.stance, _other(a.stance)
    stance_leg, swing_leg = a.leg(st), a.leg(sw)
    o = {}
    # stance toes flat; toe extension rotates the rearfoot heel-up
    o[f"{st}_rearfoot"] = np.asarray(stance_leg.toe_extension, dtype=float)
    o[f"{st}_toes"] = o[f"{st}_rearfoot"] - stance_leg.toe_extension
    o[f"{st}_shank"] = o[f"{st}_rearfoot"] + stance_leg.ankle_dorsiflexion
    o[f"{st}_thigh"] = o[f"{st}_shank"] - stance_leg.knee_flexion
    o["pelvis"] = o[f"{st}_thigh"] + stance_leg.hip_flexion
    o["abdomen"] = o["pelvis"] + a.lower_lumbar_flexion
    o["thorax"] = o["abdomen"] + a.upper_lumbar_flexion
    o["head"] = o["thorax"] + a.neck_flexion
    o[f"{sw}_thigh"] = o["pelvis"] - swing_leg.hip_flexion
    o[f"{sw}_shank"] = o[f"{sw}_thigh"] + swing_leg.knee_flexion
    o[f"{sw}_rearfoot"] = o[f"{sw}_shank"] - swing_leg.ankle_dorsiflexion
    o[f"{sw}_toes"] = o[f"{sw}_rearfoot"] - swing_leg.toe_extension
    for s in SIDES:
        arm = a.arm(s)
        # Hanging along a forward-leaning thorax means pointing down and
        # backwards, i.e. a negative angle from vertical-down.
        o[f"{s}_upper_arm"] = arm.shoulder_flexion - o["thorax"]
        o[f"{s}_forearm"] = o[f"{s}_upper_arm"] + arm.elbow_flexion
        o[f"{s}_hand"] = o[f"{s}_forearm"] + arm.wrist_flexion
    return o


def _forward(pitch):
    """Unit vector along a foot segment, heel to toe, for a pitch (deg)."""
    r = np.radians(pitch)
    return np.stack([np.cos(r), -np.sin(r)], axis=-1)


def forward_kinematics(body, angles):
    """Landmark positions (metres) for a body in a given set of angles.

    Origin: on the floor directly below the stance ankle when that foot
    is flat. The stance ball of the foot stays fixed at (ball_offset, 0).
    """
    o = segment_orientations(angles)
    shape = np.broadcast(*o.values()).shape
    o = {k: np.broadcast_to(v, shape) for k, v in o.items()}
    st, sw = angles.stance, _other(angles.stance)
    lm = {}

    def foot_from_ball(side, ball):
        f, n = _forward(o[f"{side}_rearfoot"]), up_vector(o[f"{side}_rearfoot"])
        heel = ball - body.ball_length * f
        lm[f"{side}_ball"] = ball
        lm[f"{side}_heel"] = heel
        lm[f"{side}_ankle"] = heel + body.heel_offset * f + body.ankle_height * n
        lm[f"{side}_toe"] = ball + body.toe_length * _forward(o[f"{side}_toes"])

    foot_from_ball(st, np.broadcast_to(np.array([body.ball_offset, 0.0]), shape + (2,)))
    lm[f"{st}_knee"] = lm[f"{st}_ankle"] + body.shank * up_vector(o[f"{st}_shank"])
    hip = lm[f"{st}_knee"] + body.thigh * up_vector(o[f"{st}_thigh"])
    lm[f"{st}_hip"] = hip
    lm[f"{sw}_hip"] = hip

    lm[f"{sw}_knee"] = hip - body.thigh * up_vector(o[f"{sw}_thigh"])
    ankle = lm[f"{sw}_knee"] - body.shank * up_vector(o[f"{sw}_shank"])
    f, n = _forward(o[f"{sw}_rearfoot"]), up_vector(o[f"{sw}_rearfoot"])
    heel = ankle - body.heel_offset * f - body.ankle_height * n
    foot_from_ball(sw, heel + body.ball_length * f)

    lm["lumbar"] = hip + body.pelvis * up_vector(o["pelvis"])
    lm["thoracolumbar"] = lm["lumbar"] + body.abdomen * up_vector(o["abdomen"])
    thorax_up = up_vector(o["thorax"])
    neck = lm["thoracolumbar"] + body.thorax * thorax_up
    lm["neck"] = neck
    lm["vertex"] = neck + body.head * up_vector(o["head"])
    shoulder = neck - body.shoulder_drop * thorax_up
    for s in SIDES:
        lm[f"{s}_shoulder"] = shoulder
        lm[f"{s}_elbow"] = shoulder + body.upper_arm * down_vector(o[f"{s}_upper_arm"])
        lm[f"{s}_wrist"] = lm[f"{s}_elbow"] + body.forearm * down_vector(o[f"{s}_forearm"])
        lm[f"{s}_hand"] = lm[f"{s}_wrist"] + body.hand * down_vector(o[f"{s}_hand"])
    return {name: lm[name] for name in LANDMARKS}


def segment_lengths(body):
    """Length of every segment in SEGMENTS, in metres."""
    lengths = {"head": body.head, "thorax": body.thorax, "abdomen": body.abdomen,
               "pelvis": body.pelvis}
    for s in SIDES:
        lengths.update({
            f"{s}_upper_arm": body.upper_arm, f"{s}_forearm": body.forearm,
            f"{s}_hand": body.hand, f"{s}_thigh": body.thigh, f"{s}_shank": body.shank,
            f"{s}_foot": body.ball_length,
        })
    return lengths


def landmarks_to_array(landmarks):
    """Stack a landmark dict into an array of shape (..., len(LANDMARKS), 2)."""
    return np.stack([landmarks[name] for name in LANDMARKS], axis=-2)


def measure_joint_angles(landmarks):
    """Recover joint angles from landmark positions (inverse of the FK).

    Uses only geometry, so it can be applied to any landmarks with the
    same meaning, for example a pose estimator's output in future.

    Returns every joint angle per side ("right_knee_flexion"), the
    bilateral mean without a prefix ("knee_flexion"), the two spine
    joints, ``lumbar_flexion`` (thorax relative to pelvis, the "lumbar
    spine angle" of marker-based studies) and segment inclinations.
    """
    def incl(a, b):
        return inclination_from_vertical(landmarks[b] - landmarks[a])

    def arm(a, b):
        # angle from vertical-down, positive when swung forward
        v = landmarks[b] - landmarks[a]
        return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))

    def pitch(a, b):
        # positive when the back end is higher
        v = landmarks[b] - landmarks[a]
        return np.degrees(np.arctan2(-v[..., 1], v[..., 0]))

    w = wrap_degrees
    pelvis = incl(f"{NEAR_SIDE}_hip", "lumbar")
    abdomen = incl("lumbar", "thoracolumbar")
    thorax = incl("thoracolumbar", "neck")
    head = incl("neck", "vertex")
    out = {
        "lower_lumbar_flexion": w(abdomen - pelvis),
        "upper_lumbar_flexion": w(thorax - abdomen),
        "lumbar_flexion": w(thorax - pelvis),
        "neck_flexion": w(head - thorax),
        # "trunk lean" in squat studies is the thoracic segment against vertical
        "trunk_inclination": thorax,
        "abdomen_inclination": abdomen,
        "pelvis_inclination": pelvis,
    }
    for s in SIDES:
        p = lambda n: f"{s}_{n}"  # noqa: E731
        rearfoot = pitch(p("heel"), p("ball"))
        toes = pitch(p("ball"), p("toe"))
        shank = incl(p("ankle"), p("knee"))
        thigh = incl(p("knee"), p("hip"))
        upper_arm = arm(p("shoulder"), p("elbow"))
        forearm = arm(p("elbow"), p("wrist"))
        hand = arm(p("wrist"), p("hand"))
        out.update({
            p("toe_extension"): w(rearfoot - toes),
            p("ankle_dorsiflexion"): w(shank - rearfoot),
            p("knee_flexion"): w(shank - thigh),
            p("hip_flexion"): w(pelvis - thigh),
            p("shoulder_flexion"): w(upper_arm + thorax),
            p("elbow_flexion"): w(forearm - upper_arm),
            p("wrist_flexion"): w(hand - forearm),
            p("shank_inclination"): shank,
            p("thigh_inclination"): thigh,
            p("heel_height"): landmarks[p("heel")][..., 1],
        })
    for n in (*LEG_ANGLES, *ARM_ANGLES, "shank_inclination", "thigh_inclination"):
        out[n] = 0.5 * (out[f"right_{n}"] + out[f"left_{n}"])
    return out
