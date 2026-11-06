"""Where the centre of mass sits over the foot, estimated from a tracked pose.

A tracker gives the shank, thigh, hip-to-shoulder line, head and arms but
not the spine shape or the person's mass distribution. Given a body with
the person's proportions (for example an ANSUR II match), the pose is
rebuilt in the sagittal model and its centre of mass computed with the de
Leva segment parameters, as everywhere else in the project.

Assumptions:
* the foot is flat (ankle dorsiflexion = shank lean);
* the trunk is straight along the hip-to-shoulder line: pelvis, abdomen
  and rib cage all take its inclination (no lumbar flexion visible);
* the head is tilted like the shoulder-to-ear line;
* the arms are at the tracked shoulder and elbow angles.

The result is quasi-static: during a fast squat the centre of pressure
differs from the centre of mass by the body's horizontal acceleration
term, so this is where the mass is, not where the floor pushes.
``tests/test_observed_balance.py`` checks it against the generator's
exact COM on synthetic reps.
"""

import numpy as np

from exercise_quality.core.balance import centre_of_mass, support_region
from exercise_quality.core.observed import observed_angles
from exercise_quality.core.sagittal_model import JointAngles, forward_kinematics


def pose_from_angles(angles):
    """JointAngles (arrays allowed) from ``observed_angles`` output."""
    trunk = np.asarray(angles["trunk_inclination"], float)
    return JointAngles.symmetric(
        ankle_dorsiflexion=np.asarray(angles["shank_inclination"], float),
        knee_flexion=np.asarray(angles["knee_flexion"], float),
        hip_flexion=trunk - np.asarray(angles["thigh_inclination"], float),
        lumbar_flexion=0.0 * trunk,
        neck_flexion=np.asarray(angles["neck_inclination"], float) - trunk,
        shoulder_flexion=np.asarray(angles["shoulder_flexion"], float),
        elbow_flexion=np.asarray(angles["elbow_flexion"], float),
    )


def com_fraction(body, angles):
    """COM position along the foot (0 heel, 1 toe tip) for each frame."""
    landmarks = forward_kinematics(body, pose_from_angles(angles))
    com = centre_of_mass(body, landmarks)
    x = com[..., 0]
    return support_region(body).fraction(x)


def track_com_fraction(body, track, angles=None):
    return com_fraction(body, angles or observed_angles(track))
