"""Balance-constrained bodyweight squat generator.

How a frame is built
--------------------
1. The knee follows a minimum-jerk profile to ``peak_knee_flexion``.
2. The ankle dorsiflexes by ``ankle_knee_ratio`` x peak knee flexion in
   total, coupled to knee progress u = knee / peak knee as
   1 - (1 - u) ** ankle_coupling. With the coupling above 1 the ankle
   does more of its work early, as measured by Ishida et al. (2025).
   It never goes past the person's ankle limit.
3. Arms and head follow the chosen posture rules.
4. The trunk inclination is **solved** so that the whole-body centre of
   mass sits over a target point on the foot. Trunk lean is therefore an
   output of body proportions, mass distribution and balance, not an
   input. A long-femured person ends up leaning further forward without
   anyone programming that in.

Before the rep starts the standing posture is balanced too: the whole
body leans at the ankle (hips straight) until the centre of mass is over
the standing target. Quiet standing is controlled mostly at the ankle,
and this keeps hip flexion exactly zero at the first frame.

The balance is quasi-static: it treats every frame as if the body were
still. core/dynamics.py estimates how far the true centre of pressure
moves away from the centre of mass because of accelerations.

Where the defaults come from is documented in docs/model.md.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from exercise_quality.core.balance import centre_of_mass, support_region
from exercise_quality.core.sagittal_model import (
    JointAngles,
    forward_kinematics,
    measure_joint_angles,
)
from exercise_quality.core.timing import RepTiming

TRUNK_SEARCH_DEG = np.arange(-60.0, 130.0 + 1e-9, 1.0)


@dataclass(frozen=True)
class ArmPosture:
    """How the arms are held.

    reference="world": ``upper_arm_angle`` is fixed relative to the floor
        (angle from hanging straight down, 90 = parallel to the floor).
    reference="trunk": ``upper_arm_angle`` is shoulder flexion relative to
        the trunk, so the arms move with it.
    """

    reference: str
    upper_arm_angle: float
    elbow_flexion: float

    def __post_init__(self):
        if self.reference not in ("world", "trunk"):
            raise ValueError("reference must be 'world' or 'trunk'")


# Arms straight out, parallel to the floor (Zawadka et al. 2020 protocol).
ARMS_FORWARD = ArmPosture("world", 90.0, 0.0)
# Arms crossed over the chest (Kasahara et al. 2024 protocol). Crossed
# forearms point across the body, which a side view cannot show at full
# length, so this is a projection approximation chosen to put the arm
# mass where it really sits: upper arm close to the trunk, forearm and
# hand against the front of the chest (about 10-15 cm in front of the
# shoulder joint). The two angles are a geometric guess, not measured.
ARMS_CROSSED = ArmPosture("trunk", 20.0, 150.0)
ARMS_HANGING = ArmPosture("trunk", 0.0, 0.0)


@dataclass(frozen=True)
class JointLimits:
    """Largest angles the person can reach (degrees).

    hip_flexion is the hip angle (pelvis vs thigh, relative to standing)
    at which posterior pelvic tilt begins, Mata et al.'s measured
    quantity. Past it the hip only takes ``hip_end_range_gain`` of any
    further flexion and the pelvis tilts back. Both defaults are fitted
    population means (scripts/calibrate_lumbopelvic.py). The other values
    are caps that flag impossible poses. Person-specific values are
    sampled in protocols.py.
    """

    ankle_dorsiflexion: float = 45.0
    knee_flexion: float = 157.0
    hip_flexion: float = 85.0
    hip_end_range_gain: float = 0.1
    lumbar_flexion: float = 70.0


@dataclass(frozen=True)
class SquatStyle:
    """Everything about one rep that is not the body itself."""

    peak_knee_flexion: float = 120.0
    ankle_knee_ratio: float = 0.28
    ankle_coupling: float = 1.27
    # Preferred trunk lean at Ishida's peak knee angle (deg). None means no
    # preference, so the trunk alone absorbs balance (with share 0 that is
    # the original trunk-only model).
    trunk_peak_preference: float = 38.5
    trunk_coupling: float = 1.25
    # Share of the balance correction taken at the ankle; the trunk takes
    # the rest. 0 = trunk only.
    ankle_compensation_share: float = 0.5
    # Lumbopelvic rhythm: share of the total thigh-to-thorax flexion taken
    # at the hip before the hip limit; the rest is lumbar flexion.
    pelvis_share: float = 1.0
    head_follow: float = 0.5
    arms: ArmPosture = ARMS_CROSSED
    # Fraction of foot length from the heel that the centre of mass is
    # kept over at the bottom, and how much it moves between the start
    # and the bottom (bottom minus start).
    com_target_bottom: float = 0.363
    com_target_shift: float = -0.05
    timing: RepTiming = field(default_factory=RepTiming)
    limits: JointLimits = field(default_factory=JointLimits)

    @property
    def com_target_start(self):
        return self.com_target_bottom - self.com_target_shift

    def __post_init__(self):
        if self.peak_knee_flexion <= 0:
            raise ValueError("peak_knee_flexion must be positive")
        if not 0.0 <= self.head_follow <= 1.5:
            raise ValueError("head_follow should be between 0 and 1.5")
        if not 0.0 <= self.ankle_compensation_share <= 1.0:
            raise ValueError("ankle_compensation_share must be in [0, 1]")
        if not 0.0 < self.pelvis_share <= 1.0:
            raise ValueError("pelvis_share must be in (0, 1]")
        if self.ankle_coupling < 1.0:
            # below 1 the ankle would have infinite speed at the start
            raise ValueError("ankle_coupling must be at least 1")


@dataclass
class SquatSequence:
    body: object
    style: SquatStyle
    times: np.ndarray
    angles: JointAngles          # arrays of shape (T,)
    landmarks: dict              # name -> (T, 2)
    com: np.ndarray              # (T, 2)
    com_target_x: np.ndarray     # (T,)
    flags: tuple                 # problems that make the rep unrealistic
    balance_error: np.ndarray    # (T,) COM x minus target, metres
    notes: tuple = ()            # realistic but worth knowing, e.g. ankle_at_limit

    @property
    def support(self):
        return support_region(self.body)

    @property
    def fps(self):
        return 1.0 / (self.times[1] - self.times[0])

    @property
    def trunk_inclination(self):
        return self.measured()["trunk_inclination"]

    def measured(self):
        return measure_joint_angles(self.landmarks)

    def frame(self, i):
        return {name: xy[i] for name, xy in self.landmarks.items()}

    @property
    def is_feasible(self):
        return not self.flags


# Width (degrees) over which the pelvis hands over from following the
# thorax to being carried by the thigh. Smoothing avoids an instant change
# in pelvic speed, which would show up as a spike in the dynamics.
# ASSUMPTION: 8 deg; the hip still never exceeds its limit.
HIP_LIMIT_BLEND_DEG = 8.0


def smooth_min(a, b, width):
    """min(a, b) with the corner rounded over ``width`` (C2 continuous).

    Equal to the true minimum whenever |a - b| >= width and never above
    it, so a limit expressed with it is never exceeded.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    h = np.maximum(width - np.abs(a - b), 0.0) / width
    return np.minimum(a, b) - h**3 * width / 6.0


def pelvis_for_thorax(thorax, thigh, style):
    """Pelvis inclination given the thorax and thigh inclinations.

    Before the hip reaches end range, the hip takes ``pelvis_share`` of
    the total thigh-to-thorax flexion and the lumbar spine the rest
    (lumbopelvic rhythm; share 1 keeps the spine neutral). Past the end
    range onset ``limits.hip_flexion`` the hip keeps flexing, but only by
    ``limits.hip_end_range_gain`` of what is asked of it, as tissue
    compresses. The thigh therefore carries the pelvis into posterior
    tilt and the rest of the forward lean needed for balance comes from
    bending the lower back.

    This follows Mata et al. (2021): posterior pelvic tilt began at the
    same hip angle in squats as in passive supine hip flexion, i.e. when
    hip flexion approached end range. The end range is soft rather than a
    hard stop because measured hip flexion keeps rising until the bottom
    of the squat (Kasahara et al. 2024) and is larger in deeper squats
    (Zawadka et al. 2020 vs Kasahara). A constant rhythm before end range
    and a constant gain past it are ASSUMPTIONS.
    """
    thorax = np.asarray(thorax, dtype=float)
    thigh = np.asarray(thigh, dtype=float)
    limits = style.limits
    # hip flexion the trunk would ask for with this lumbopelvic rhythm
    # (exactly zero when standing)
    demand = style.pelvis_share * (thorax - thigh)
    past_end_range = limits.hip_flexion + limits.hip_end_range_gain * (demand - limits.hip_flexion)
    hip = smooth_min(demand, past_end_range, HIP_LIMIT_BLEND_DEG)
    return thigh + hip


def _angles_for_trunk(ankle, knee, trunk, style):
    """Joint angles for given ankle, knee and absolute thorax inclination."""
    thigh = ankle - knee
    pelvis = pelvis_for_thorax(trunk, thigh, style)
    shape = np.shape(pelvis)
    if style.arms.reference == "world":
        shoulder = style.arms.upper_arm_angle + trunk
    else:
        shoulder = np.broadcast_to(style.arms.upper_arm_angle, shape)
    return JointAngles(
        ankle_dorsiflexion=ankle,
        knee_flexion=knee,
        hip_flexion=pelvis - thigh,
        lumbar_flexion=trunk - pelvis,
        neck_flexion=(style.head_follow - 1.0) * trunk,
        shoulder_flexion=shoulder,
        elbow_flexion=np.broadcast_to(style.arms.elbow_flexion, shape),
        wrist_flexion=np.zeros(shape),
    )


def _com_x(body, ankle, knee, trunk, style):
    lm = forward_kinematics(body, _angles_for_trunk(ankle, knee, trunk, style))
    return centre_of_mass(body, lm)[..., 0]


def _bisect(fn, lo, hi, iterations=60):
    """Vectorised bisection; fn(lo) and fn(hi) must have opposite signs."""
    f_lo = fn(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(same, mid, lo)
        f_lo = np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def solve_standing_lean(body, style):
    """Ankle angle that balances upright standing (knees and hips straight)."""
    target = support_region(body).x_at(style.com_target_start)
    # trunk = ankle when knee and hip are straight
    def f(a):
        return _com_x(body, a, np.zeros_like(a), a, style) - target
    lo, hi = np.array(-20.0), np.array(20.0)
    if np.sign(f(lo)) == np.sign(f(hi)):
        raise ValueError("cannot balance standing over the requested target")
    return float(_bisect(f, lo, hi))


# Reference knee angle for the trunk preference curve: the peak knee
# flexion in Ishida et al. 2025, where trunk lean was measured.
TRUNK_REFERENCE_KNEE = 121.8

# How sharply the ankle stops at its limit when a correction pushes it
# there (degrees). ASSUMPTION, like HIP_LIMIT_BLEND_DEG.
ANKLE_LIMIT_BLEND_DEG = 4.0


def preferred_trunk(knee, style):
    """Trunk lean the person would choose at this knee angle, before any
    balance correction (degrees, excluding the standing lean).

    Same curve family as the ankle coupling, anchored at Ishida's peak
    knee angle; flat beyond it.
    """
    if style.trunk_peak_preference is None:
        return np.zeros_like(np.asarray(knee, dtype=float))
    u = np.clip(np.asarray(knee, dtype=float) / TRUNK_REFERENCE_KNEE, 0.0, 1.0)
    return style.trunk_peak_preference * (1.0 - (1.0 - u) ** style.trunk_coupling)


def split_correction(ankle_base, trunk_base, correction, style):
    """Ankle and thorax after a balance correction (degrees).

    The ankle takes ``ankle_compensation_share`` of the correction (never
    passing the ankle limit) and the thorax the rest. Share 0 is the
    original model where only the trunk adjusts.
    """
    share = style.ankle_compensation_share
    if share == 0.0:
        # the base pattern is already clipped to the limit
        ankle = np.asarray(ankle_base, dtype=float) + 0.0 * correction
    else:
        ankle = smooth_min(ankle_base + share * correction,
                           style.limits.ankle_dorsiflexion, ANKLE_LIMIT_BLEND_DEG)
    thorax = trunk_base + (1.0 - share) * correction
    return ankle, thorax


def solve_correction(body, ankle_base, knee, trunk_base, target_x, style):
    """Balance correction per frame that puts the COM x on ``target_x``.

    Searches a 1 degree grid for sign changes, follows the root nearest
    the previous frame (so the solution is continuous) and refines it by
    bisection. Frames without any root get NaN.
    """
    ankle_base, knee, trunk_base, target_x = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (ankle_base, knee, trunk_base, target_x)))

    def com_x(a, k, t, lam):
        ankle, thorax = split_correction(a, t, lam, style)
        return _com_x(body, ankle, k, thorax, style)

    grid = TRUNK_SEARCH_DEG
    values = com_x(ankle_base[:, None], knee[:, None], trunk_base[:, None], grid[None, :])
    values = values - target_x[:, None]

    lo = np.full(knee.shape, np.nan)
    previous = 0.0
    for i in range(len(knee)):
        crossings = np.nonzero(np.sign(values[i, :-1]) != np.sign(values[i, 1:]))[0]
        if crossings.size == 0:
            continue
        mids = grid[crossings] + 0.5
        k = crossings[np.argmin(np.abs(mids - previous))]
        lo[i] = grid[k]
        previous = grid[k] + 0.5

    found = ~np.isnan(lo)
    correction = np.full(knee.shape, np.nan)
    if found.any():
        a, kn, t, tx = ankle_base[found], knee[found], trunk_base[found], target_x[found]

        def f(lam):
            return com_x(a, kn, t, lam) - tx

        correction[found] = _bisect(f, lo[found], lo[found] + 1.0)
    return correction


def generate_squat(body, style=None, fps=100.0):
    """One balanced squat rep for ``body``. Returns a SquatSequence."""
    style = style or SquatStyle()
    timing = style.timing
    limits = style.limits
    support = support_region(body)
    flags, notes = [], []

    times = timing.times(fps)
    knee_profile = timing.profile(times)
    ankle_profile = 1.0 - (1.0 - np.clip(knee_profile, 0.0, 1.0)) ** style.ankle_coupling

    peak_knee = style.peak_knee_flexion
    if peak_knee > limits.knee_flexion:
        flags.append("knee_beyond_limit")

    for name in ("com_target_start", "com_target_bottom"):
        if not 0.0 < getattr(style, name) < 1.0:
            flags.append("com_target_outside_support")
            break

    lean = solve_standing_lean(body, style)
    wanted = style.ankle_knee_ratio * peak_knee
    available = limits.ankle_dorsiflexion - lean
    if wanted > available:
        # running out of ankle mobility is normal; the trunk compensates
        notes.append("ankle_at_limit")
    ankle_amplitude = max(0.0, min(wanted, available))

    knee = peak_knee * knee_profile
    ankle_base = lean + ankle_amplitude * ankle_profile
    trunk_base = lean + preferred_trunk(knee, style)
    target_fraction = (style.com_target_start
                       + (style.com_target_bottom - style.com_target_start)
                       * knee_profile)
    target_x = support.x_at(target_fraction)

    correction = solve_correction(body, ankle_base, knee, trunk_base, target_x, style)
    if np.isnan(correction).any():
        flags.append("balance_unreachable")
        correction = _fill_gaps(correction)
    ankle, trunk = split_correction(ankle_base, trunk_base, correction, style)
    if "ankle_at_limit" not in notes and np.max(ankle) > limits.ankle_dorsiflexion - 0.5:
        notes.append("ankle_at_limit")

    angles = _angles_for_trunk(ankle, knee, trunk, style)
    landmarks = forward_kinematics(body, angles)
    com = centre_of_mass(body, landmarks)

    if np.nanmax(angles.lumbar_flexion) > limits.lumbar_flexion:
        flags.append("lumbar_beyond_limit")

    return SquatSequence(
        body=body,
        style=style,
        times=times,
        angles=angles,
        landmarks=landmarks,
        com=com,
        com_target_x=target_x,
        flags=tuple(flags),
        balance_error=com[:, 0] - target_x,
        notes=tuple(notes),
    )


# Problems that a real person would solve by not going as deep.
MOBILITY_PROBLEMS = frozenset({"balance_unreachable", "lumbar_beyond_limit", "knee_beyond_limit"})


def generate_feasible_squat(body, style=None, fps=100.0, min_knee_flexion=60.0, step=2.0):
    """Like generate_squat, but a person who cannot reach the requested
    depth squats as deep as their body allows instead.

    Someone with stiff hips cannot reach parallel with heels down without
    rounding their back past its limit or falling backwards; in real life
    they stop higher. Depth is lowered in ``step`` degree increments until
    the rep is feasible, and the rep is noted "depth_limited_by_mobility".
    Problems that are not about mobility (an off-foot COM target) are
    returned unchanged.
    """
    style = style or SquatStyle()
    seq = generate_squat(body, style, fps)
    if seq.is_feasible or not set(seq.flags) <= MOBILITY_PROBLEMS:
        return seq
    depths = list(np.arange(style.peak_knee_flexion - step, min_knee_flexion, -step))
    for knee in depths + [min_knee_flexion]:
        attempt = generate_squat(body, replace(style, peak_knee_flexion=float(knee)), fps)
        if attempt.is_feasible:
            attempt.notes = attempt.notes + ("depth_limited_by_mobility",)
            return attempt
    return seq


def _fill_gaps(values):
    """Linearly fill NaNs so an infeasible rep can still be drawn."""
    values = values.copy()
    bad = np.isnan(values)
    if bad.all():
        return np.zeros_like(values)
    idx = np.arange(len(values))
    values[bad] = np.interp(idx[bad], idx[~bad], values[~bad])
    return values
