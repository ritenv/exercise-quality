"""Every generated pose must preserve the body's segment lengths.

Forward/inverse kinematics should only rotate segments, never stretch
or shrink them. This is a pure geometry invariant: passing it says
nothing about realism or balance, but failing it means a bug.
"""

import numpy as np
import pytest

from exercise_quality.core.body import BodyProportions
from exercise_quality.core.body_generator import generate_random_body
from exercise_quality.exercises.squat.movement import calculate_squat_pose
from exercise_quality.exercises.squat.movement_v2 import (
    calculate_squat_pose_v2,
)

TOLERANCE_M = 1e-9

# (start joint, end joint, BodyProportions field)
SEGMENTS = [
    ("ankle", "knee", "shin"),
    ("knee", "hip", "thigh"),
    ("hip", "shoulder", "torso"),
    ("shoulder", "head", "neck"),
    ("ankle", "toe", "foot"),
]

BODIES = [BodyProportions()] + [generate_random_body(seed) for seed in range(50)]
PROGRESS_VALUES = np.concatenate([[0.0, 1e-9, 1e-6], np.linspace(0, 1, 31)])


def assert_segment_lengths(joints, body, context):
    for start, end, field in SEGMENTS:
        actual = np.linalg.norm(joints[end] - joints[start])
        expected = getattr(body, field)
        assert abs(actual - expected) < TOLERANCE_M, (
            f"{context}: {start}->{end} is {actual:.6f} m, "
            f"expected {field}={expected:.6f} m"
        )


def assert_finite(joints, context):
    for name, position in joints.items():
        assert np.all(np.isfinite(position)), f"{context}: {name} not finite"


@pytest.mark.parametrize("body_index", range(len(BODIES)))
def test_v1_preserves_segment_lengths(body_index):
    body = BODIES[body_index]
    for progress in PROGRESS_VALUES:
        joints = calculate_squat_pose(progress, body=body)
        context = f"V1 body={body_index} progress={progress}"
        assert_finite(joints, context)
        assert_segment_lengths(joints, body, context)


def test_v1_preserves_segment_lengths_across_parameters():
    rng = np.random.default_rng(0)
    for i in range(200):
        body = generate_random_body(seed=i)
        depth = rng.uniform(0.4, 1.0)
        hip_shift = rng.uniform(0.10, 0.35)
        torso_lean = rng.uniform(15, 60)
        for progress in PROGRESS_VALUES:
            joints = calculate_squat_pose(
                progress,
                depth=depth,
                hip_shift=hip_shift,
                torso_lean=torso_lean,
                body=body,
            )
            context = (
                f"V1 seed={i} depth={depth:.2f} hip_shift={hip_shift:.2f} "
                f"torso_lean={torso_lean:.1f} progress={progress}"
            )
            assert_finite(joints, context)
            assert_segment_lengths(joints, body, context)


@pytest.mark.parametrize("body_index", range(len(BODIES)))
def test_v2_preserves_segment_lengths(body_index):
    body = BODIES[body_index]
    for progress in PROGRESS_VALUES:
        joints = calculate_squat_pose_v2(progress, body=body)
        context = f"V2 body={body_index} progress={progress}"
        assert_finite(joints, context)
        assert_segment_lengths(joints, body, context)


@pytest.mark.parametrize("generator", [calculate_squat_pose, calculate_squat_pose_v2])
def test_standing_pose_is_vertical_and_continuous(generator):
    """At progress 0 the leg is straight up; a tiny step must barely move it.

    This catches the old bug where the standing knee used the default
    shin length instead of the body's own.
    """
    body = generate_random_body(seed=42)
    standing = generator(0.0, body=body)
    nearly_standing = generator(1e-6, body=body)

    np.testing.assert_allclose(standing["knee"], [0.0, body.shin], atol=TOLERANCE_M)
    np.testing.assert_allclose(
        standing["hip"], [0.0, body.shin + body.thigh], atol=TOLERANCE_M
    )
    for name in standing:
        jump = np.linalg.norm(nearly_standing[name] - standing[name])
        assert jump < 1e-3, f"{name} jumped {jump:.6f} m between progress 0 and 1e-6"
