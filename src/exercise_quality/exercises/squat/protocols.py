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

# Standing COM position, fraction of foot length from the heel. The mean
# is Ishida's bottom position minus Kasahara's forward shift (0.363 + 0.05
# = 0.413), which keeps both measured means; it sits inside the 31-58%
# range Fujiwara et al. 2010 found in quiet standing. The SD is a quarter
# of that range (read as +-2 SD), an ASSUMPTION from a range.
STANDING_POSITION = (
    lit.ISHIDA_2025["cop_fraction"].mean - lit.KASAHARA_2024["com_ap_shift_pct_foot"].mean / 100.0,
    0.0675,
)

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
# Fitted together with the hip end range (see HIP_LIMIT below): 0.9. The
# very first fit, under a hard hip stop, had put it at 1.0 (neutral lower
# back until end range). The 0.05 spread (clipped at 1) is an ASSUMPTION.
PELVIS_SHARE = (0.9, 0.05)
HIP_LIMIT = (105.0, lit.MATA_2021["hip_at_ppt_absolute"].sd)
# Current fit (scripts/calibrate_lumbopelvic.py, share x gain x onset grid,
# then checked past the grid edge): share 0.9, gain 0.1, onset 105 deg,
# cost 0.25, all six targets |z| < 0.35 (107.5 deg: 0.30, 110 deg: 0.41).
# Letting the lower back take about 10% of the flexion before end range
# resolved the tension earlier fits had between pelvis tilt, hip ROM and
# Mata's onset. Earlier fits (onset 85-97.5 deg) used an 8 deg end-range
# blend, a fixed share of 1.0, or Kasahara's hip ROM as a target. Onset
# 105 deg relative to standing vs Mata's absolute 110 deg implies about
# 5 deg of hip flexion in marker-based standing posture (not verified).
HIP_END_RANGE_GAIN = 0.1

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

# Postural sway (core/sway.py). A 2 mm COP-like wobble with a 0.33 Hz
# median frequency reproduces Yamamoto et al. 2015's quiet-standing COP
# (mean velocity 6.3 mm/s, median frequency 0.33 Hz) at 60 fps. Spread
# between people uses their between-subject SDs: log10 mean velocity SD
# 0.097 (velocity scales linearly with the amplitude) and median
# frequency SD 0.071 Hz. Applying quiet-standing sway during the squat
# itself is an ASSUMPTION; movement probably adds more.
SWAY_AMPLITUDE_M = 0.002
SWAY_LOG10_SD = 0.097
SWAY_MEDIAN_HZ = (0.329, 0.071)

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
    # Depth: the instruction was "thighs parallel to the floor", the same
    # as Ishida 2025 from the same lab (Kasahara cites that protocol), so
    # Ishida's measured peak knee flexion is used. Kasahara's own knee
    # figure (107 deg) is a range of motion from the starting angle; used
    # as a peak it left the model's thighs 18 deg above horizontal and the
    # COM drop at 18% of height against their measured 23%. With Ishida's
    # depth the thighs are within about 5 deg of parallel and the COM drop
    # matches. A slightly bent standing knee in their data would explain
    # the difference.
    knee_peak=(lit.ISHIDA_2025["knee_peak"].mean, lit.ISHIDA_2025["knee_peak"].sd),
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


# Per-person style draws are truncated at this many SDs. The published
# spreads come from 16-60 people and say nothing about the far tails;
# untruncated normal draws occasionally produced caricatures (for example
# a 3.3 SD trunk-lean preference combined with stiff hips gave 69 deg of
# lumbar flexion at parallel). ASSUMPTION.
TRUNCATE_SD = 2.5


def _normal(rng, mean_sd, lo=-np.inf, hi=np.inf):
    mean, sd = mean_sd
    z = float(np.clip(rng.normal(), -TRUNCATE_SD, TRUNCATE_SD))
    return float(np.clip(mean + sd * z, lo, hi))


def weight_position_correlations():
    """Correlation matrix of (ankle z, COM shift, standing position).

    Chosen so that every published statistic holds at once:
      * ankle vs shift: Kasahara's R = 0.398
      * standing vs shift: fixed by var(bottom) = var(standing + shift),
        i.e. cov(standing, shift) = (sd_bottom^2 - sd_stand^2 - sd_shift^2) / 2
      * ankle vs standing: assumed to act only through the shift
        (conditional independence), corr = product of the other two
    """
    sd_bottom = lit.ISHIDA_2025["cop_fraction"].sd
    sd_shift = lit.KASAHARA_2024["com_ap_shift_pct_foot"].sd / 100.0
    sd_stand = STANDING_POSITION[1]
    r_as = ANKLE_SHIFT_CORRELATION
    r_ss = (sd_bottom**2 - sd_stand**2 - sd_shift**2) / (2 * sd_stand * sd_shift)
    r_at = r_as * r_ss
    return np.array([[1.0, r_as, r_at], [r_as, 1.0, r_ss], [r_at, r_ss, 1.0]])


def _sample_ankle_and_weight_position(rng):
    """Ankle ratio, standing COM position and bottom COM position, jointly.

    Sampling them independently made the standing position far too
    variable (SD 12% of foot length; only 70% of people inside the 31-58%
    range Fujiwara et al. measured in quiet standing).
    """
    z = np.linalg.cholesky(weight_position_correlations()) @ rng.normal(size=3)
    z = np.clip(z, -TRUNCATE_SD, TRUNCATE_SD)
    ratio_mean, ratio_sd = ANKLE_KNEE_RATIO
    shift_pct = lit.KASAHARA_2024["com_ap_shift_pct_foot"]
    ratio = float(np.clip(ratio_mean + ratio_sd * z[0], 0.05, 0.5))
    shift = shift_pct.mean / 100.0 + shift_pct.sd / 100.0 * z[1]
    start = STANDING_POSITION[0] + STANDING_POSITION[1] * z[2]
    start = float(np.clip(start, 0.20, 0.65))
    bottom = float(np.clip(start + shift, 0.20, 0.60))
    return ratio, start, bottom


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

    ratio, start, bottom = _sample_ankle_and_weight_position(rng)
    shift = bottom - start

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
        sway_amplitude=SWAY_AMPLITUDE_M * 10 ** _normal(rng, (0.0, SWAY_LOG10_SD)),
        sway_median_hz=_normal(rng, SWAY_MEDIAN_HZ, 0.15, 0.6),
        sway_seed=int(rng.integers(2**31)),
        trunk_coupling=TRUNK_COUPLING,
        ankle_compensation_share=ANKLE_COMPENSATION_SHARE,
        arms=protocol.arms,
        com_target_bottom=bottom,
        com_target_shift=shift,
        timing=timing,
        limits=limits,
    )
