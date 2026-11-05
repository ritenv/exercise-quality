"""Forward kinematics of the foot-rooted sagittal model."""

import numpy as np
import pytest

from exercise_quality.core.geometry import cross2d
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import (
    LANDMARKS,
    SEGMENT_ENDPOINTS,
    JointAngles,
    forward_kinematics,
    measure_joint_angles,
)

TOL = 1e-9


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module")
def body(population):
    return population.median_body("male")


def test_standing_pose_is_stacked_vertically(body):
    lm = forward_kinematics(body, JointAngles())
    for name in ("ankle", "knee", "hip", "neck", "vertex"):
        assert lm[name][0] == pytest.approx(0.0, abs=TOL)
    assert lm["ankle"][1] == pytest.approx(body.ankle_height)
    assert lm["hip"][1] == pytest.approx(body.standing_hip_height)
    assert lm["vertex"][1] == pytest.approx(body.stature)
    # arms hang straight down from the shoulder
    assert lm["hand"][0] == pytest.approx(0.0, abs=TOL)
    assert lm["shoulder"][1] - lm["hand"][1] == pytest.approx(
        body.upper_arm + body.forearm + body.hand)
    # foot on the floor, heel behind, toe in front
    assert lm["heel"][1] == lm["toe"][1] == 0.0
    assert lm["heel"][0] == pytest.approx(-body.heel_offset)
    assert lm["toe"][0] == pytest.approx(body.toe_offset)


def test_ankle_dorsiflexion_moves_knee_forward(body):
    lm = forward_kinematics(body, JointAngles(ankle_dorsiflexion=30))
    shin = lm["knee"] - lm["ankle"]
    assert shin[0] == pytest.approx(body.shank * np.sin(np.radians(30)))
    assert shin[1] == pytest.approx(body.shank * np.cos(np.radians(30)))


def test_knee_flexion_sends_hip_backwards(body):
    lm = forward_kinematics(body, JointAngles(knee_flexion=90))
    thigh = lm["hip"] - lm["knee"]
    np.testing.assert_allclose(thigh, [-body.thigh, 0.0], atol=TOL)


def test_hip_flexion_folds_trunk_forward(body):
    lm = forward_kinematics(body, JointAngles(hip_flexion=90))
    np.testing.assert_allclose(lm["neck"] - lm["hip"], [body.trunk, 0.0], atol=TOL)


def test_squat_with_trunk_kept_vertical(body):
    # shank +30, knee 100 -> thigh -70, hip 70 -> trunk vertical
    lm = forward_kinematics(body, JointAngles(30, 100, 70))
    trunk = lm["neck"] - lm["hip"]
    assert trunk[0] == pytest.approx(0.0, abs=TOL)


def test_shoulder_flexion_90_points_arm_forward(body):
    lm = forward_kinematics(body, JointAngles(shoulder_flexion=90))
    np.testing.assert_allclose(
        lm["elbow"] - lm["shoulder"], [body.upper_arm, 0.0], atol=TOL)


def test_shoulder_flexion_is_relative_to_trunk(body):
    # with the trunk leaning 40 deg, 90 deg of shoulder flexion is
    # perpendicular to the trunk: 40 deg below horizontal
    lm = forward_kinematics(body, JointAngles(hip_flexion=40, shoulder_flexion=90))
    arm = lm["elbow"] - lm["shoulder"]
    trunk = lm["neck"] - lm["hip"]
    assert np.dot(arm, trunk) == pytest.approx(0.0, abs=TOL)
    assert arm[0] > 0 and arm[1] < 0


def test_elbow_flexion_bends_forearm_forward(body):
    lm = forward_kinematics(body, JointAngles(elbow_flexion=90))
    np.testing.assert_allclose(
        lm["wrist"] - lm["elbow"], [body.forearm, 0.0], atol=TOL)


def test_shoulder_hangs_below_neck_on_trunk_line(body):
    lm = forward_kinematics(body, JointAngles(10, 60, 50))
    a = lm["neck"] - lm["shoulder"]
    b = lm["neck"] - lm["hip"]
    assert cross2d(a, b) == pytest.approx(0.0, abs=TOL)
    assert np.linalg.norm(a) == pytest.approx(body.shoulder_drop)


def random_angles(rng, n):
    return JointAngles(
        ankle_dorsiflexion=rng.uniform(-20, 45, n),
        knee_flexion=rng.uniform(0, 160, n),
        hip_flexion=rng.uniform(-20, 150, n),
        neck_flexion=rng.uniform(-60, 60, n),
        shoulder_flexion=rng.uniform(-40, 180, n),
        elbow_flexion=rng.uniform(0, 150, n),
        wrist_flexion=rng.uniform(-60, 60, n),
    )


def test_segment_lengths_preserved_for_random_poses(population):
    rng = np.random.default_rng(3)
    for _ in range(40):
        b = population.sample(rng)
        lm = forward_kinematics(b, random_angles(rng, 50))
        length_of = {
            "head": b.head, "trunk": b.trunk, "upper_arm": b.upper_arm,
            "forearm": b.forearm, "hand": b.hand, "thigh": b.thigh,
            "shank": b.shank, "foot": b.foot_length,
        }
        for seg, (a, c) in SEGMENT_ENDPOINTS.items():
            lengths = np.linalg.norm(lm[c] - lm[a], axis=-1)
            np.testing.assert_allclose(lengths, length_of[seg], atol=TOL)
        assert np.all(np.isfinite(np.stack([lm[k] for k in LANDMARKS])))


def test_measured_angles_round_trip(population):
    rng = np.random.default_rng(4)
    b = population.sample(rng)
    angles = random_angles(rng, 200)
    measured = measure_joint_angles(forward_kinematics(b, angles))
    for name, value in angles.as_dict().items():
        np.testing.assert_allclose(measured[name], value, atol=1e-8, err_msg=name)


def test_array_angles_match_scalar_calls(body):
    rng = np.random.default_rng(5)
    angles = random_angles(rng, 5)
    batch = forward_kinematics(body, angles)
    for i in range(5):
        single = forward_kinematics(body, JointAngles(
            **{k: float(v[i]) for k, v in angles.as_dict().items()}))
        for name in LANDMARKS:
            np.testing.assert_allclose(batch[name][i], single[name], atol=TOL)
