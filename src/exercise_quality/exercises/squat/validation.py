import numpy as np

from exercise_quality.core.geometry import angle_between


def validate_squat_pose(joints):
    ankle = joints["ankle"]
    knee = joints["knee"]
    hip = joints["hip"]
    shoulder = joints["shoulder"]

    issues = []

    # Interior knee angle (180 = straight leg); a zero-length segment has
    # no angle, which is reported as an issue instead of a fake number
    try:
        knee_angle = angle_between(ankle - knee, hip - knee)
    except ValueError:
        knee_angle = 0.0
        issues.append("degenerate_segment")

    # Torso inclination from vertical
    torso = shoulder - hip
    torso_lean = float(np.degrees(
        np.arctan2(torso[0], torso[1])
    ))

    # Shin inclination from vertical
    shin = knee - ankle
    shin_angle = float(np.degrees(
        np.arctan2(shin[0], shin[1])
    ))

    # Broad preliminary limits, not clinical standards
    if not 45 <= knee_angle <= 180.01:
        issues.append("knee_angle_out_of_range")

    if not -10 <= torso_lean <= 75:
        issues.append("torso_lean_out_of_range")

    if not -10 <= shin_angle <= 65:
        issues.append("shin_angle_out_of_range")

    if hip[1] <= 0:
        issues.append("hip_below_ground")

    if knee[0] < ankle[0] - 0.05:
        issues.append("knee_behind_ankle")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "knee_angle": knee_angle,
        "torso_lean": torso_lean,
        "shin_angle": shin_angle,
    }
