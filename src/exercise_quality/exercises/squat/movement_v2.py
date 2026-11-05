
import numpy as np

from exercise_quality.core.body import BodyProportions


def smoothstep(t):
    """Smooth transition from 0 to 1."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def calculate_squat_pose_v2(
    progress,
    body=None,
    max_knee_flexion=110.0,
    max_shin_inclination=30.0,
    max_torso_inclination=35.0,
):
    """
    Generate a 2D squat pose using coordinated joint angles.

    progress: 0 = standing, 1 = bottom position

    All angles are in degrees.
    """
    if body is None:
        body = BodyProportions()

    p = smoothstep(progress)

    knee_flexion = max_knee_flexion * p
    shin_angle = max_shin_inclination * p
    torso_angle = max_torso_inclination * p

    # Ankle remains fixed
    ankle = np.array([0.0, 0.0])

    # Shin points upward and forward
    shin_rad = np.radians(shin_angle)

    knee = ankle + body.shin * np.array([
        np.sin(shin_rad),
        np.cos(shin_rad),
    ])

    # Thigh orientation is determined by knee flexion
    # relative to the shin.
    thigh_angle = shin_angle - knee_flexion

    thigh_rad = np.radians(thigh_angle)

    hip = knee + body.thigh * np.array([
        np.sin(thigh_rad),
        np.cos(thigh_rad),
    ])

    # Torso inclination
    torso_rad = np.radians(torso_angle)

    torso_vector = np.array([
        np.sin(torso_rad),
        np.cos(torso_rad),
    ])

    shoulder = hip + body.torso * torso_vector
    head = shoulder + body.neck * torso_vector

    toe = ankle + np.array([body.foot, 0.0])

    return {
        "ankle": ankle,
        "toe": toe,
        "knee": knee,
        "hip": hip,
        "shoulder": shoulder,
        "head": head,
    }
