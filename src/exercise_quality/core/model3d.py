"""3D body model: the side-view model with feet placed apart and each leg in its own plane.

The validated side-view model (core/sagittal_model.py) stays the core.
Every joint angle keeps its meaning: knee flexion, ankle dorsiflexion and
the leg segment inclinations are measured within each leg's own plane,
the trunk within the pelvis's mid-plane. What 3D adds is where that
geometry sits in space:

* **Feet.** Both feet are flat on the floor, ``Stance.width_ratio`` x
  hip breadth apart (centre to centre), each turned out by
  ``Stance.toe_out``.
* **Leg plane.** Each leg (ankle, knee, hip) moves in a plane that
  contains the foot's long axis: the knee tracks over the toes, the
  coaching standard, and what 3D studies of healthy squats show (knee
  valgus/varus within a few degrees, see docs/model3d.md). The plane is
  tilted sideways about the foot axis so that the two hip joint centres
  land their measured distance apart (a closed chain through the
  pelvis). With feet wider than the hips the legs lean in, which is hip
  abduction.
* **Knee tracking fault.** ``Stance.knee_track_*`` turns a leg's plane
  medially about the vertical (knees caving in, dynamic valgus), growing
  with squat depth (``progress``).
* **Lateral shift.** ``Stance.lateral_shift`` moves the pelvis sideways
  (towards the person's right when positive), growing with depth.
* **Trunk.** Pelvis, abdomen, rib cage and head stay in the pelvis's
  mid-plane, at the side-view inclinations. The shoulders sit
  ``shoulder_width`` apart along the hip line.

Frame: x forward, y up, z to the person's right (right-handed). The
origin is on the floor, mid-way between the two ankles' floor points in
x, with both ankles above x = 0. The side view of the existing pipeline
is the (x, y) projection, seen from the person's right (+z).

With the ankles directly under the hip joint centres, no toe-out and no
faults, every (x, y) coordinate equals the side-view model's
(tests/test_model3d.py).

Body widths (``Body3D``) come from ANSUR II breadths, except the hip joint
centre separation, which ANSUR II does not measure (see ``make_body3d``).
"""

import zlib
from dataclasses import dataclass, replace

import numpy as np

from exercise_quality.core.balance import SupportRegion, model_segments
from exercise_quality.core.sagittal_body import SagittalBody
from exercise_quality.core.sagittal_model import LANDMARKS, SIDES, segment_orientations

# Hip joint centre separation, Mullaji et al. 2010 (Skeletal Radiol
# 39:363-368), CT of 200 adults: inter-femoral-head-centre distance
# 16.0 +- 0.8 cm, not influenced by height or BMI, correlated only weakly
# (r = 0.35) with the distance between the anterior superior iliac
# spines. The sex difference was significant, but the abstract gives no
# per-sex values, so one distribution is used for both (ASSUMPTION). The
# bicristal breadth (iliac crests) stands in for the inter-ASIS distance
# in the weak correlation (ASSUMPTION). Indian adults; ANSUR II is US Army.
HIP_JOINT_WIDTH = (0.160, 0.008)
HIP_JOINT_WIDTH_CORRELATION = 0.35

# Fraction along the foot (heel to toe tip) taken as the foot's centre
# when placing the feet: Sinclair et al. 2022 measured stance width
# between the feet's segment centres of mass. de Leva's foot centre of
# mass is used (core/deleva.py).
EXTRA_LANDMARKS = ("mid_hip", "right_ear", "left_ear", "nose")
LANDMARKS_3D = LANDMARKS + EXTRA_LANDMARKS
# not landmarks: each leg plane's normal (the knee flexion axis), for joint angles
LEG_NORMALS = ("right_leg_normal", "left_leg_normal")
# Ears at 55% of the way from C7 to the vertex (as core/observed.py), the
# nose 10 cm in front of the head axis at 50%: rough placements for the
# tracker's head points (ASSUMPTION).
EAR_FRACTION = 0.55
NOSE_FRACTION = 0.5
NOSE_FORWARD = 0.10
MAX_LEG_TILT_SIN = 0.95


@dataclass(frozen=True)
class Body3D:
    """A side-view body plus the breadths a 3D pose needs (metres).

    hip_joint_width: distance between the hip joint centres.
    shoulder_width: distance between the shoulder joint centres, taken as
        ANSUR II biacromial breadth (acromion to acromion; the joint
        centre is placed directly below the acromion, ASSUMPTION).
    hip_breadth: ANSUR II standing hip breadth, the skin width at the
        hips; stance widths are expressed relative to it.
    foot_breadth, head_breadth: ANSUR II.
    Every side-view attribute (``thigh``, ``mass``...) reads through to
    ``sagittal``.
    """

    sagittal: SagittalBody
    hip_joint_width: float
    shoulder_width: float
    hip_breadth: float
    foot_breadth: float
    head_breadth: float

    def __getattr__(self, name):
        if name == "sagittal":           # during unpickling, before fields exist
            raise AttributeError(name)
        return getattr(self.sagittal, name)


def make_body3d(population, sex, index):
    """Body3D for one ANSUR II person.

    The hip joint centre separation is drawn once per person (seeded by
    the subject id, so it is always the same for that person) from
    ``HIP_JOINT_WIDTH``, correlated 0.35 with the person's bicristal
    breadth (z-score within their sex).
    """
    row = population._by_sex[sex][index]
    sagittal = population.body(sex, index)
    crest = np.array([r["bicristal_breadth_m"] for r in population._by_sex[sex]])
    z = (row["bicristal_breadth_m"] - crest.mean()) / crest.std()
    rng = np.random.default_rng(zlib.crc32(f"hip-joint-width-{row['subject_id']}".encode()))
    r = HIP_JOINT_WIDTH_CORRELATION
    width = HIP_JOINT_WIDTH[0] + HIP_JOINT_WIDTH[1] * (r * z + np.sqrt(1 - r * r) * rng.normal())
    return Body3D(sagittal=sagittal, hip_joint_width=float(width),
                  shoulder_width=row["biacromial_breadth_m"], hip_breadth=row["hip_breadth_m"],
                  foot_breadth=row["foot_breadth_m"], head_breadth=row["head_breadth_m"])


def body3d_of(population, body):
    """The Body3D of an ANSUR II person given their side-view ``Body`` (matched by subject id)."""
    for i, row in enumerate(population._by_sex[body.sex]):
        if row["subject_id"] == body.subject_id:
            return make_body3d(population, body.sex, i)
    raise KeyError(f"no ANSUR II {body.sex} subject {body.subject_id!r}")


def sample_body3d(population, rng, sex=None, female_share=0.5):
    if sex is None:
        sex = "female" if rng.random() < female_share else "male"
    return make_body3d(population, sex, int(rng.integers(population.count(sex))))


def median_body3d(population, sex):
    m = population.median_measurements(sex)
    return Body3D(sagittal=population.median_body(sex), hip_joint_width=HIP_JOINT_WIDTH[0],
                  shoulder_width=m["biacromial_breadth_m"], hip_breadth=m["hip_breadth_m"],
                  foot_breadth=m["foot_breadth_m"], head_breadth=m["head_breadth_m"])


@dataclass(frozen=True)
class Stance:
    """How the feet are placed, plus frontal-plane faults.

    width_ratio: distance between the feet's centres / hip breadth.
    toe_out: each foot turned out from straight ahead (deg).
    knee_track_right / knee_track_left: medial rotation of that leg's
        plane about the vertical at the bottom of the squat (deg; positive
        = knee caving in towards the midline). Scaled by ``progress``.
    lateral_shift: sideways pelvis shift at the bottom (m, positive =
        towards the person's right). Scaled by ``progress``.
    """

    width_ratio: float = 1.23
    toe_out: float = 24.0
    knee_track_right: float = 0.0
    knee_track_left: float = 0.0
    lateral_shift: float = 0.0

    def knee_track(self, side):
        return getattr(self, f"knee_track_{side}")

    def under_hips(self, body):
        """The stance that puts each ankle directly under its hip joint centre, toes straight."""
        return replace(self, width_ratio=body.hip_joint_width / body.hip_breadth, toe_out=0.0)


def _sign(side):
    return 1.0 if side == "right" else -1.0


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _vec(x, y, z):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    return np.stack([x, y, z], axis=-1)


def _scale(a, v):
    return np.asarray(a, float)[..., None] * v


def _progress(progress, shape):
    p = np.asarray(progress, dtype=float)
    if p.ndim == 1 and len(shape) > 1:
        p = p.reshape(p.shape + (1,) * (len(shape) - 1))
    return np.broadcast_to(p, shape)


def foot_frame(body, stance, side):
    """Ankle floor point, foot axis (heel to toe) and medial horizontal unit vector."""
    s = _sign(side)
    theta = np.radians(stance.toe_out)
    forward = np.array([np.cos(theta), 0.0, s * np.sin(theta)])
    medial = np.array([np.sin(theta), 0.0, -s * np.cos(theta)])
    centre_from_heel = body.segments["foot"].com_fraction * body.foot_length
    half = 0.5 * stance.width_ratio * body.hip_breadth
    ankle_floor_z = s * half - (centre_from_heel - body.heel_offset) * forward[2]
    return np.array([0.0, 0.0, ankle_floor_z]), forward, medial


def forward_kinematics(body, angles, stance, progress=1.0, iterations=12):
    """Landmark positions (metres, shape (..., 3)) for a Body3D in a pose.

    ``angles`` is a side-view ``JointAngles``; ``progress`` (0 standing, 1
    at the bottom; scalar or one value per frame) scales the knee
    tracking and lateral shift faults. Returns every side-view landmark
    plus ``mid_hip``, ears and nose. ``"leg_tilt_sin"`` holds each leg's
    sideways tilt as a diagnostic (|value| near 1 means the hips cannot
    be reached).
    """
    o = segment_orientations(angles)
    shape = np.broadcast(*o.values()).shape
    o = {k: np.broadcast_to(np.asarray(v, float), shape) for k, v in o.items()}
    prog = _progress(progress, shape)
    up = np.array([0.0, 1.0, 0.0])
    lm, legs = {}, {}

    for side in SIDES:
        p = lambda n: f"{side}_{n}"  # noqa: E731
        a0, f, m = foot_frame(body, stance, side)
        ball = a0 + body.ball_offset * f
        rear = np.radians(o[p("rearfoot")])
        fwd = _scale(np.cos(rear), f) - _scale(np.sin(rear), up)
        nrm = _scale(np.sin(rear), f) + _scale(np.cos(rear), up)
        heel = ball - body.ball_length * fwd
        lm[p("ball")] = np.broadcast_to(ball, shape + (3,))
        lm[p("heel")] = heel
        lm[p("ankle")] = heel + body.heel_offset * fwd + body.ankle_height * nrm
        toes = np.radians(o[p("toes")])
        lm[p("toe")] = ball + body.toe_length * (_scale(np.cos(toes), f) - _scale(np.sin(toes), up))

        delta = np.radians(stance.knee_track(side)) * prog
        u = _scale(np.cos(delta), f) + _scale(np.sin(delta), m)
        mp = _scale(np.cos(delta), m) - _scale(np.sin(delta), f)
        b, g = np.radians(o[p("shank")]), np.radians(o[p("thigh")])
        a_u = body.shank * np.sin(b) + body.thigh * np.sin(g)
        a_v = body.shank * np.cos(b) + body.thigh * np.cos(g)
        legs[side] = (lm[p("ankle")], u, mp, a_u, a_v, b)

    shift = stance.lateral_shift * prog
    half = np.full(shape, 0.5 * body.hip_joint_width)
    for _ in range(iterations):
        hips, tilt = {}, {}
        for side in SIDES:
            ankle, u, mp, a_u, a_v, _ = legs[side]
            target = shift + _sign(side) * half
            sin_phi = (target - ankle[..., 2] - a_u * u[..., 2]) / (a_v * mp[..., 2])
            tilt[side] = sin_phi
            sin_phi = np.clip(sin_phi, -MAX_LEG_TILT_SIN, MAX_LEG_TILT_SIN)
            v = _scale(np.sqrt(1 - sin_phi**2), up) + _scale(sin_phi, mp)
            hips[side] = ankle + _scale(a_u, u) + _scale(a_v, v)
        gap = np.linalg.norm(hips["right"] - hips["left"], axis=-1)
        half = half * body.hip_joint_width / gap

    for side in SIDES:
        ankle, u, mp, a_u, a_v, b = legs[side]
        sin_phi = np.clip(tilt[side], -MAX_LEG_TILT_SIN, MAX_LEG_TILT_SIN)
        v = _scale(np.sqrt(1 - sin_phi**2), up) + _scale(sin_phi, mp)
        shank_dir = _scale(np.sin(b), u) + _scale(np.cos(b), v)
        lm[f"{side}_knee"] = ankle + _scale(np.full(shape, body.shank), shank_dir)
        lm[f"{side}_hip"] = hips[side]
        lm[f"{side}_leg_normal"] = _unit(np.cross(u, v))     # knee flexion axis, towards +z

    mid = 0.5 * (hips["right"] + hips["left"])
    lateral = _unit(hips["right"] - hips["left"])
    forward = _unit(np.cross(np.broadcast_to(up, lateral.shape), lateral))
    upward = np.cross(lateral, forward)

    def along(angle_deg):
        r = np.radians(angle_deg)
        return _scale(np.sin(r), forward) + _scale(np.cos(r), upward)

    def hanging(angle_deg):
        r = np.radians(angle_deg)
        return _scale(np.sin(r), forward) - _scale(np.cos(r), upward)

    lm["mid_hip"] = mid
    lm["lumbar"] = mid + body.pelvis * along(o["pelvis"])
    lm["thoracolumbar"] = lm["lumbar"] + body.abdomen * along(o["abdomen"])
    thorax_up = along(o["thorax"])
    lm["neck"] = lm["thoracolumbar"] + body.thorax * thorax_up
    head_up = along(o["head"])
    lm["vertex"] = lm["neck"] + body.head * head_up
    shoulder_mid = lm["neck"] - body.shoulder_drop * thorax_up
    for side in SIDES:
        sh = shoulder_mid + _sign(side) * 0.5 * body.shoulder_width * lateral
        lm[f"{side}_shoulder"] = sh
        lm[f"{side}_elbow"] = sh + body.upper_arm * hanging(o[f"{side}_upper_arm"])
        lm[f"{side}_wrist"] = lm[f"{side}_elbow"] + body.forearm * hanging(o[f"{side}_forearm"])
        lm[f"{side}_hand"] = lm[f"{side}_wrist"] + body.hand * hanging(o[f"{side}_hand"])
        lm[f"{side}_ear"] = (lm["neck"] + EAR_FRACTION * body.head * head_up
                             + _sign(side) * 0.5 * body.head_breadth * lateral)
    head_forward = _scale(np.cos(np.radians(o["head"])), forward) - _scale(np.sin(np.radians(o["head"])), upward)
    lm["nose"] = lm["neck"] + NOSE_FRACTION * body.head * head_up + NOSE_FORWARD * head_forward
    out = {name: lm[name] for name in LANDMARKS_3D + LEG_NORMALS}
    out["leg_tilt_sin"] = np.stack([tilt["right"], tilt["left"]], axis=-1)
    return out


def centre_of_mass(body, landmarks):
    """Whole-body centre of mass (..., 3); the pelvis runs from the navel to the mid-hip point."""
    segments = model_segments(body.sagittal)
    total = sum(s.mass for s in segments.values())
    acc = 0.0
    for name, s in segments.items():
        other = "mid_hip" if name == "pelvis" else s.other
        start = landmarks[s.origin]
        acc = acc + s.mass * (start + s.com_fraction * (landmarks[other] - start))
    return acc / total


def support_region(body, stance):
    """Front-to-back extent of the feet along x (both feet turned out equally).

    A fraction along this interval is the same fraction along each foot
    from heel to toe tip, so centre-of-pressure fractions keep their
    meaning (0 heel, 1 toe tip).
    """
    c = np.cos(np.radians(stance.toe_out))
    return SupportRegion(-body.heel_offset * c, body.toe_offset * c)


def side_view(landmarks):
    """(x, y) projection seen from the person's right, side-view landmark names only."""
    return {name: landmarks[name][..., :2] for name in LANDMARKS}


@dataclass(frozen=True)
class Kinematics3D:
    """What the squat generator needs from a body model, for a Body3D in a stance."""

    stance: Stance

    def forward_kinematics(self, body, angles, progress=1.0):
        return forward_kinematics(body, angles, self.stance, progress)

    def centre_of_mass(self, body, landmarks):
        return centre_of_mass(body, landmarks)

    def support_region(self, body):
        return support_region(body, self.stance)


def _frame(lateral, up):
    """Columns (lateral, anterior, up), orthonormal, from a lateral and a rough up axis."""
    lateral = _unit(lateral)
    up = _unit(up - np.sum(up * lateral, axis=-1, keepdims=True) * lateral)
    anterior = np.cross(up, lateral)
    return np.stack([lateral, anterior, up], axis=-1)


def joint_angles_3d(landmarks):
    """Hip angles as 3D motion capture reports them, and the knee's frontal-plane projection angle.

    Hip: thigh relative to pelvis, Cardan sequence flexion (about the
    lateral axis), then abduction (about the floating anterior axis), then
    rotation (about the thigh's long axis), the default in Visual3D, as
    used by Sinclair et al. 2022. Pelvis frame: lateral = left to right
    hip joint centre, up = mid-hip to navel. Thigh frame: up = knee to
    hip, lateral = the leg plane's normal (the knee flexion axis).
    Positive: flexion, abduction, internal rotation, on both sides.

    Knee frontal-plane projection angle (FPPA): seen from the front, how
    far the knee bends inwards off the hip-ankle line, degrees; positive
    = valgus (knee towards the midline), negative = knee outside the line.
    0 when the leg is straight from the front. Meaningful part-way down
    only: near parallel the thigh points at a front camera and its
    projection shrinks to almost nothing.
    """
    pelvis = _frame(landmarks["right_hip"] - landmarks["left_hip"], landmarks["lumbar"] - landmarks["mid_hip"])
    out = {}
    for side in SIDES:
        s = _sign(side)
        thigh = _frame(landmarks[f"{side}_leg_normal"], landmarks[f"{side}_hip"] - landmarks[f"{side}_knee"])
        r = np.swapaxes(pelvis, -1, -2) @ thigh
        b = np.degrees(np.arcsin(np.clip(r[..., 0, 2], -1.0, 1.0)))
        out[f"{side}_hip_flexion"] = np.degrees(np.arctan2(-r[..., 1, 2], r[..., 2, 2]))
        out[f"{side}_hip_abduction"] = -s * b
        out[f"{side}_hip_rotation"] = s * np.degrees(np.arctan2(-r[..., 0, 1], r[..., 0, 0]))
        # frontal projection: (z, y) seen from the front
        hip, knee, ankle = (landmarks[f"{side}_{n}"][..., [2, 1]] for n in ("hip", "knee", "ankle"))
        up_thigh = np.degrees(np.arctan2(hip[..., 0] - knee[..., 0], hip[..., 1] - knee[..., 1]))
        up_shank = np.degrees(np.arctan2(knee[..., 0] - ankle[..., 0], knee[..., 1] - ankle[..., 1]))
        # valgus (knee towards the midline): the hip is further out from the
        # knee than the knee from the ankle
        out[f"{side}_knee_fppa"] = s * (up_thigh - up_shank)
    for name in ("hip_flexion", "hip_abduction", "hip_rotation", "knee_fppa"):
        out[name] = 0.5 * (out[f"right_{name}"] + out[f"left_{name}"])
    return out
