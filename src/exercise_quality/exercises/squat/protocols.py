"""Sample realistic squat styles for real bodies.

A protocol is the task people were asked to do in a study (depth target,
arm position, tempo). Sampling a style draws each person's own
strategy from the between-subject spread that study reported, so a
generated cohort can be compared directly with the published one.

Where a number is a modelling assumption instead of a measurement, the
comment says so. docs/model.md collects them all.
"""

from dataclasses import dataclass

import numpy as np

from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import literature as lit
from exercise_quality.exercises.squat.balanced import (
    ARMS_CROSSED,
    ARMS_FORWARD,
    ArmPosture,
    JointLimits,
    SquatStyle,
)

# Ratio of ankle dorsiflexion excursion to peak knee flexion. Ishida 2025:
# 33.8 / 121.8 = 0.278; the spread is their ankle SD over the knee peak.
# Kasahara 2024 (32.8 / 107.2 = 0.31) and Zawadka 2020 (about 30.5 / 122
# = 0.25) fall within one SD of this.
ANKLE_KNEE_RATIO = (lit.ISHIDA_2025["ankle_peak"].mean / lit.ISHIDA_2025["knee_peak"].mean,
                    lit.ISHIDA_2025["ankle_peak"].sd / lit.ISHIDA_2025["knee_peak"].mean)

# Kasahara 2024, correlation between ankle ROM and COM forward displacement.
ANKLE_SHIFT_CORRELATION = 0.398

# Ankle-knee coupling exponent gamma in ankle share = 1 - (1 - u)**gamma,
# u = knee / peak knee. Fitted to Ishida 2025: at 60 deg knee (u = 0.493)
# the ankle is at 19.3 / 33.8 = 0.571 of its peak, giving gamma = 1.25; at
# 90 deg (u = 0.739) it is at 0.822, giving gamma = 1.29. The mean of the
# two is used. The between-person SD (0.1) is an ASSUMPTION: the paper
# gives group means only.
ANKLE_COUPLING = (1.27, 0.1)

# Lumbopelvic rhythm and hip limit. Before the hip runs out of flexion the
# hip takes PELVIS_SHARE of the thigh-to-thorax flexion and the lumbar
# spine the rest; at the hip limit (relative to standing) the pelvis is
# carried back by the thigh. No study reports these two directly, so their means
# are FITTED to pelvis and hip measurements (Zawadka pelvis peak and hip
# ROM, Kasahara hip ROM) by scripts/calibrate_lumbopelvic.py; see
# docs/validation.md. Lumbar flexion, trunk lean and the timing of the
# pelvic tilt are then predictions. The hip limit spread is Mata 2021's
# between-subject SD; the pelvis share spread is an ASSUMPTION.
# First fit (hard stop, 80 people per cohort, 6 x 9 grid): share 1.0, limit 90 deg;
# all five target |z| < 0.9, and the cost rises steeply below a share of
# 0.9 (2.1 at 1.0, 4.7 at 0.8, 18.7 at 0.6). So the data favour a lower
# back that stays near neutral until the hip runs out, with lumbar
# flexion coming after, matching Mata's account of posterior pelvic tilt
# as a compensation. Mata's 110 deg is an absolute angle; 90 deg relative
# to standing implies about 20 deg of hip flexion in marker-based standing
# posture, which is plausible but not verified. The 0.05 spread in the
# share (clipped at 1) is an ASSUMPTION.
PELVIS_SHARE = (1.0, 0.05)
HIP_LIMIT = (85.0, lit.MATA_2021["hip_at_ppt_absolute"].sd)
# Soft end range. Final fit (gain x onset grid, after the ankle and trunk
# started sharing the balance correction, with Kasahara's hip timing and
# Mata's knee angle at the start of posterior tilt added as targets):
# gain 0.2 at 85 deg, cost 4.09 over 7 targets. Two targets pull against
# each other: Mata's knee angle wants a later onset (model z -1.19) and
# Kasahara's hip ROM an earlier one (z +1.26), so one onset plus a gain is
# a simplification of a more gradual or individual end range. Onset 85
# deg relative to standing vs Mata's absolute 110 deg implies about 25 deg
# of hip flexion in marker-based standing posture (not verified).
HIP_END_RANGE_GAIN = 0.2

# Preferred trunk lean (deg) at Ishida's peak knee angle: the published
# peak trunk inclination and its between-subject SD, used directly. The
# curve shape exponent is fitted to Ishida's 60 and 90 deg values in the
# same way as the ankle (1.32 and 1.18, mean 1.25).
TRUNK_PREFERENCE = (lit.ISHIDA_2025["trunk_peak"].mean, lit.ISHIDA_2025["trunk_peak"].sd)
TRUNK_COUPLING = 1.25

# Balance correction shared between ankle and trunk. ASSUMPTION: equal
# sharing (0.5). With the trunk taking all of it (0), people with low
# ankle use leant up to 80 deg and 17% could not reach parallel; any share
# from 0.25 to 0.75 gave realistic spreads (Ishida men: trunk peak SD 7.3
# vs 7.3 published, ankle SD 3.4 vs 4.5, knee SD 5.2 vs 5.7, nobody depth
# limited). Real people evidently spread the correction across joints.
ANKLE_COMPENSATION_SHARE = 0.5

# How much of the trunk lean the head follows (0 = head stays upright,
# 1 = head in line with the trunk). ASSUMPTION, no data used.
HEAD_FOLLOW_RANGE = (0.3, 0.7)


@dataclass(frozen=True)
class Protocol:
    name: str
    arms: ArmPosture
    knee_peak: tuple            # (mean, sd), or per-sex dict
    timing: RepTiming
    tempo_jitter: float = 0.0   # +- fraction applied to descent and ascent
    depth_is_maximal: bool = False
    note: str = ""


ISHIDA_PARALLEL = Protocol(
    name="ishida2025_parallel",
    # Arm position not stated in Ishida 2025. Kasahara 2024, from the same
    # lab and protocol citation, used arms crossed, so that is assumed.
    arms=ARMS_CROSSED,
    knee_peak=(lit.ISHIDA_2025["knee_peak"].mean, lit.ISHIDA_2025["knee_peak"].sd),
    timing=RepTiming(stand_before=1.0, descent=2.0, bottom_hold=0.0,
                     ascent=2.0, stand_after=1.0),
    note="thighs parallel, metronome 2 s down / 2 s up / 2 s standing",
)

KASAHARA_PARALLEL = Protocol(
    name="kasahara2024_parallel",
    arms=ARMS_CROSSED,
    knee_peak=(lit.KASAHARA_2024["knee_rom"].mean, lit.KASAHARA_2024["knee_rom"].sd),
    # ascent duration not reported; 2 s assumed to mirror the descent
    timing=RepTiming(stand_before=0.5, descent=2.0, ascent=2.0, stand_after=0.5),
    note="arms crossed, thighs parallel, 2 s descent",
)

ZAWADKA_MAX_DEPTH = Protocol(
    name="zawadka2020_max_depth",
    arms=ARMS_FORWARD,
    knee_peak={sex: (lit.ZAWADKA_2020[sex]["knee_rom"].mean, lit.ZAWADKA_2020[sex]["knee_rom"].sd)
               for sex in ("female", "male")},
    # preferred pace (not reported) with a 3 s hold at the bottom; 2 s
    # +- 25% per phase is an ASSUMPTION
    timing=RepTiming(stand_before=0.5, descent=2.0, bottom_hold=3.0,
                     ascent=2.0, stand_after=0.5),
    tempo_jitter=0.25,
    depth_is_maximal=True,
    note="maximal depth, arms forward, 3 s hold, preferred pace",
)

PROTOCOLS = {p.name: p for p in (ISHIDA_PARALLEL, KASAHARA_PARALLEL, ZAWADKA_MAX_DEPTH)}


def _normal(rng, mean_sd, lo=-np.inf, hi=np.inf):
    mean, sd = mean_sd
    return float(np.clip(rng.normal(mean, sd), lo, hi))


def sample_limits(rng, hip_limit=HIP_LIMIT, hip_gain=None):
    """Person-specific joint limits.

    Knee: Hemmerich 2006 maximum knee flexion in a heels-up squat.
    Hip: fitted mean (see HIP_LIMIT) with Mata 2021's spread.
    Ankle and lumbar: generous caps (ASSUMPTION). The sampled ankle ratio
    already reflects the mobility of the real people it came from, so no
    extra ankle restriction is added by default.
    """
    knee = _normal(rng, (lit.KNEE_FLEXION_MAX.mean, lit.KNEE_FLEXION_MAX.sd), 140, 170)
    hip = _normal(rng, hip_limit, 70, 130)
    gain = HIP_END_RANGE_GAIN if hip_gain is None else hip_gain
    return JointLimits(ankle_dorsiflexion=45.0, knee_flexion=knee, hip_flexion=hip,
                       hip_end_range_gain=gain, lumbar_flexion=70.0)


def sample_style(rng, body, protocol=ISHIDA_PARALLEL,
                 pelvis_share=PELVIS_SHARE, hip_limit=HIP_LIMIT, hip_gain=None):
    knee_dist = protocol.knee_peak
    if isinstance(knee_dist, dict):
        knee_dist = knee_dist[body.sex]
    limits = sample_limits(rng, hip_limit, hip_gain)
    knee_peak = _normal(rng, knee_dist, 60.0, 170.0)
    if protocol.depth_is_maximal:
        # a maximal squat cannot go past the person's own knee limit
        knee_peak = min(knee_peak, limits.knee_flexion)

    bottom = _normal(rng, (lit.ISHIDA_2025["cop_fraction"].mean,
                           lit.ISHIDA_2025["cop_fraction"].sd), 0.20, 0.60)

    # People with more ankle motion move their COM further forward during
    # the descent (Kasahara 2024: R = 0.398 between ankle ROM and COM
    # forward displacement). Sample the ankle ratio and the COM shift with
    # that correlation instead of independently.
    ratio_mean, ratio_sd = ANKLE_KNEE_RATIO
    z_ankle = rng.normal()
    z_other = rng.normal()
    ratio = float(np.clip(ratio_mean + ratio_sd * z_ankle, 0.05, 0.5))
    shift_pct = lit.KASAHARA_2024["com_ap_shift_pct_foot"]
    rho = ANKLE_SHIFT_CORRELATION
    shift = (shift_pct.mean + shift_pct.sd * (rho * z_ankle + np.sqrt(1 - rho**2) * z_other)) / 100.0
    shift = float(np.clip(shift, bottom - 0.65, bottom - 0.20))  # start stays on the foot

    timing = protocol.timing
    if protocol.tempo_jitter:
        j = protocol.tempo_jitter
        timing = RepTiming(
            stand_before=timing.stand_before,
            descent=timing.descent * rng.uniform(1 - j, 1 + j),
            bottom_hold=timing.bottom_hold,
            ascent=timing.ascent * rng.uniform(1 - j, 1 + j),
            stand_after=timing.stand_after,
        )

    return SquatStyle(
        peak_knee_flexion=knee_peak,
        ankle_knee_ratio=ratio,
        ankle_coupling=_normal(rng, ANKLE_COUPLING, 1.0, 1.6),
        head_follow=float(rng.uniform(*HEAD_FOLLOW_RANGE)),
        pelvis_share=_normal(rng, pelvis_share, 0.5, 1.0),
        trunk_peak_preference=_normal(rng, TRUNK_PREFERENCE, 5.0, 80.0),
        trunk_coupling=TRUNK_COUPLING,
        ankle_compensation_share=ANKLE_COMPENSATION_SHARE,
        arms=protocol.arms,
        com_target_bottom=bottom,
        com_target_shift=shift,
        timing=timing,
        limits=limits,
    )
