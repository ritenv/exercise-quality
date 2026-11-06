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

from dataclasses import replace

import numpy as np

from exercise_quality.core.balance import centre_of_mass, support_region
from exercise_quality.core.observed import observed_angles, segment_lengths
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


def projected_body(body, track, frame, angles=None):
    """``body`` with thigh and shank shortened to how long they look at ``frame``.

    When the knees travel out over the toes, a side view sees the thigh
    (and a little of the shank) foreshortened. Rebuilding that pose with
    full segment lengths at the seen angles puts the hip too far back, and
    the centre of mass with it (by about 0.1 of foot length at the bottom
    of a squat, through the tracker-noise model). The sagittal position of
    the centre of mass only depends on where the segments project, and a
    segment's mass centre projects to the same fraction along the
    projected segment, so shortening the segments to their seen length
    gives the right answer. Standing lengths are the medians over frames
    with the knee bent less than 10 deg (95th percentile if there are
    none). Never lengthens a segment.
    """
    angles = angles or observed_angles(track)
    lengths = segment_lengths(track)
    upright = np.asarray(angles["knee_flexion"]) < 10.0
    scales = {}
    for name in ("thigh", "shank"):
        values = np.asarray(lengths[name], float)
        standing = np.nanmedian(values[upright]) if upright.sum() >= 5 else np.nanpercentile(values, 95)
        scales[name] = float(min(1.0, values[frame] / standing))
    return replace(body, thigh=body.thigh * scales["thigh"], shank=body.shank * scales["shank"])
