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


def sample_limits(rng):
    """Person-specific joint limits.

    Knee: Hemmerich 2006 maximum knee flexion in a heels-up squat.
    Ankle and hip(+lumbar): generous caps (ASSUMPTION). The sampled ankle
    ratio already reflects the mobility of the real people it came from,
    so no extra ankle restriction is added by default.
    """
    knee = _normal(rng, (lit.KNEE_FLEXION_MAX.mean, lit.KNEE_FLEXION_MAX.sd), 140, 170)
    return JointLimits(ankle_dorsiflexion=45.0, knee_flexion=knee, hip_flexion=150.0)


def sample_style(rng, body, protocol=ISHIDA_PARALLEL):
    knee_dist = protocol.knee_peak
    if isinstance(knee_dist, dict):
        knee_dist = knee_dist[body.sex]
    limits = sample_limits(rng)
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
        arms=protocol.arms,
        com_target_bottom=bottom,
        com_target_shift=shift,
        timing=timing,
        limits=limits,
    )
