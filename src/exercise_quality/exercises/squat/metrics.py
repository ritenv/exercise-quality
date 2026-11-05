"""Interpretable squat descriptors measured from landmarks.

These describe *what* a rep looks like. They are not quality scores: a
deep squat or a hip-dominant squat is neither good nor bad without
context (goals, anatomy, injury history).

Definitions follow Straub RK, Powers CM. "A biomechanical review of the
squat exercise: implications for clinical practice." Int J Sports Phys
Ther. 2024;19(4):490-501. doi:10.26603/001c.94600. That review is a
narrative (level 5) synthesis; its thresholds are operational
definitions, not measured population norms.
"""

import numpy as np

from exercise_quality.core.sagittal_model import measure_joint_angles

# Straub & Powers 2024 operational depth bins, by peak knee flexion (deg).
DEPTH_BINS = (
    ("partial", 0.0, 90.0),
    ("parallel", 90.0, 110.0),   # "medium (90-110 deg or thigh parallel)"
    ("deep", 110.0, 135.0),
)

# Trunk-tibia angle threshold (deg). Straub & Powers use 10 deg; it rests
# on Barrack et al. 2021, a regression in loaded barbell back squats
# where -8 deg gave equal hip and knee extensor demand. For bodyweight
# squats it is a heuristic, not a validated cut-off.
EXTENSOR_BIAS_THRESHOLD = 10.0


def depth_category(peak_knee_flexion):
    """'partial', 'parallel', 'deep' or 'beyond deep' (> 135 deg)."""
    for name, lo, hi in DEPTH_BINS:
        if lo <= peak_knee_flexion < hi:
            return name
    return "beyond deep" if peak_knee_flexion >= 135.0 else "none"


def trunk_tibia_angle(landmarks):
    """Trunk inclination minus shank (tibia) inclination, per frame (deg).

    Positive: the trunk leans further forward than the shin.
    """
    m = measure_joint_angles(landmarks)
    return m["trunk_inclination"] - m["shank_inclination"]


def extensor_bias(trunk_tibia_at_bottom):
    """'hip', 'knee' or 'neutral' extensor bias from the trunk-tibia angle."""
    if trunk_tibia_at_bottom > EXTENSOR_BIAS_THRESHOLD:
        return "hip"
    if trunk_tibia_at_bottom < -EXTENSOR_BIAS_THRESHOLD:
        return "knee"
    return "neutral"


def describe_rep(sequence):
    """A small dictionary of descriptors for one generated rep."""
    m = sequence.measured()
    bottom = int(np.argmax(m["knee_flexion"]))
    tt = trunk_tibia_angle(sequence.landmarks)
    pelvis = m["pelvis_inclination"][: bottom + 1]
    return {
        "peak_knee_flexion": float(m["knee_flexion"][bottom]),
        "depth": depth_category(float(m["knee_flexion"][bottom])),
        "thigh_below_horizontal_deg": float(-90.0 - m["thigh_inclination"][bottom]),
        "trunk_tibia_at_bottom": float(tt[bottom]),
        "extensor_bias": extensor_bias(float(tt[bottom])),
        "posterior_pelvic_tilt_deg": float(pelvis.max() - pelvis[-1]),
        "peak_lumbar_flexion": float(m["lumbar_flexion"].max()),
        "com_over_foot_at_bottom": float(sequence.support.fraction(sequence.com[bottom, 0])),
    }
