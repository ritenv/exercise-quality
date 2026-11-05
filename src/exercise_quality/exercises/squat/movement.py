import numpy as np
from exercise_quality.core.body import BodyProportions
from exercise_quality.core.kinematics import solve_two_bone_ik

def calculate_squat_pose(
    progress,
    depth=1.0,
    hip_shift=0.25,
    torso_lean=35.0,
    body=None,
):
    """
    progress: 0 = standing, 1 = bottom
    """
    if body is None:
        body = BodyProportions()

    progress = float(np.clip(progress, 0, 1))
    depth = float(np.clip(depth, 0.0, 1.0))
    effective_progress = progress * depth

    ankle = np.array([0.0, 0.0])

    # Target hip trajectory
    standing_height = body.shin + body.thigh

    hip = np.array([
        -hip_shift * effective_progress,
        standing_height
        - body.thigh * effective_progress,
    ])

    # Solve knee position
    if effective_progress < 1e-8:
        knee = np.array([0.0, body.shin])
    else:
        knee = solve_two_bone_ik(
            ankle,
            hip,
            body.shin,
            body.thigh,
            bend_direction=1,
        )

    # Torso inclination
    torso_angle = np.radians(
        torso_lean * effective_progress
    )

    torso_vector = np.array([
        np.sin(torso_angle),
        np.cos(torso_angle),
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
