import numpy as np


def angle_between(v1, v2):
    """Smallest angle between two vectors, in degrees."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator < 1e-10:
        raise ValueError("Cannot measure a zero-length vector")

    cosine = np.clip(
        np.dot(v1, v2) / denominator,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def measure_pose(joints, body):
    ankle = joints["ankle"]
    knee = joints["knee"]
    hip = joints["hip"]
    shoulder = joints["shoulder"]

    # Interior knee angle: 180 degrees = straight
    knee_angle = angle_between(
        ankle - knee,
        hip - knee,
    )

    # Knee flexion: 0 degrees = straight
    knee_flexion = 180.0 - knee_angle

    # Segment inclinations relative to vertical
    shin = knee - ankle
    torso = shoulder - hip

    shin_inclination = np.degrees(
        np.arctan2(shin[0], shin[1])
    )

    torso_inclination = np.degrees(
        np.arctan2(torso[0], torso[1])
    )

    # Hip height relative to standing leg length
    standing_hip_height = body.shin + body.thigh
    hip_height = float(hip[1])

    hip_descent_ratio = (
        standing_hip_height - hip_height
    ) / standing_hip_height

    # Signed hip position relative to ankle
    hip_horizontal_offset = float(
        hip[0] - ankle[0]
    )

    return {
        "knee_flexion_deg": float(knee_flexion),
        "shin_inclination_deg": float(shin_inclination),
        "torso_inclination_deg": float(torso_inclination),
        "hip_height_m": hip_height,
        "hip_descent_ratio": float(hip_descent_ratio),
        "hip_horizontal_offset_m": hip_horizontal_offset,
    }
