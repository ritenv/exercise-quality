"""Forward kinematics of the foot-rooted sagittal model."""

import numpy as np
import pytest

from exercise_quality.core.geometry import cross2d
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import (
    LANDMARKS,
    SEGMENT_ENDPOINTS,
    SIDES,
    UPPER_LUMBAR_SHARE,
    ArmAngles,
    JointAngles,
    LegAngles,
    forward_kinematics,
    measure_joint_angles,
    segment_lengths,
)

TOL = 1e-9
S = JointAngles.symmetric


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module")
def body(population):
    return population.median_body("male")


def test_standing_pose_is_stacked_vertically(body):
    lm = forward_kinematics(body, S())
    for name in ("right_ankle", "right_knee", "right_hip", "lumbar", "thoracolumbar",
                 "neck", "vertex"):
        assert lm[name][0] == pytest.approx(0.0, abs=TOL)
    assert lm["right_ankle"][1] == pytest.approx(body.ankle_height)
    assert lm["right_hip"][1] == pytest.approx(body.standing_hip_height)
    assert lm["lumbar"][1] == pytest.approx(body.standing_hip_height + body.pelvis)
    assert lm["thoracolumbar"][1] == pytest.approx(
        body.standing_hip_height + body.pelvis + body.abdomen)
    assert lm["vertex"][1] == pytest.approx(body.stature)
    # arms hang straight down from the shoulder
    assert lm["right_hand"][0] == pytest.approx(0.0, abs=TOL)
    assert lm["right_shoulder"][1] - lm["right_hand"][1] == pytest.approx(
        body.upper_arm + body.forearm + body.hand)
    # foot flat: heel behind, ball and toe in front, all on the floor
    for p in ("heel", "ball", "toe"):
        assert lm[f"right_{p}"][1] == pytest.approx(0.0, abs=TOL)
    assert lm["right_heel"][0] == pytest.approx(-body.heel_offset)
    assert lm["right_ball"][0] == pytest.approx(body.ball_offset)
    assert lm["right_toe"][0] == pytest.approx(body.toe_offset)


def test_symmetric_pose_puts_both_sides_on_top_of_each_other(body):
    lm = forward_kinematics(body, S(ankle_dorsiflexion=25, knee_flexion=100, hip_flexion=80,
                                    toe_extension=10, shoulder_flexion=70, elbow_flexion=30,
                                    lumbar_flexion=20))
    for name in LANDMARKS:
        if name.startswith("right_"):
            np.testing.assert_allclose(lm[name], lm["left_" + name[6:]], atol=TOL)


def test_ankle_dorsiflexion_moves_knee_forward(body):
    lm = forward_kinematics(body, S(ankle_dorsiflexion=30))
    shin = lm["right_knee"] - lm["right_ankle"]
    assert shin[0] == pytest.approx(body.shank * np.sin(np.radians(30)))
    assert shin[1] == pytest.approx(body.shank * np.cos(np.radians(30)))


def test_knee_flexion_sends_hip_backwards(body):
    lm = forward_kinematics(body, S(knee_flexion=90))
    thigh = lm["right_hip"] - lm["right_knee"]
    np.testing.assert_allclose(thigh, [-body.thigh, 0.0], atol=TOL)


def test_hip_flexion_folds_trunk_forward(body):
    lm = forward_kinematics(body, S(hip_flexion=90))
    np.testing.assert_allclose(lm["neck"] - lm["right_hip"], [body.trunk, 0.0], atol=TOL)


def test_squat_with_trunk_kept_vertical(body):
    # shank +30, knee 100 -> thigh -70, hip 70 -> trunk vertical
    lm = forward_kinematics(body, S(ankle_dorsiflexion=30, knee_flexion=100, hip_flexion=70))
    trunk = lm["neck"] - lm["right_hip"]
    assert trunk[0] == pytest.approx(0.0, abs=TOL)


def test_shoulder_flexion_90_points_arm_forward(body):
    lm = forward_kinematics(body, S(shoulder_flexion=90))
    np.testing.assert_allclose(
        lm["right_elbow"] - lm["right_shoulder"], [body.upper_arm, 0.0], atol=TOL)


def test_shoulder_flexion_is_relative_to_thorax(body):
    # with the thorax leaning 40 deg, 90 deg of shoulder flexion is
    # perpendicular to it: 40 deg below horizontal
    lm = forward_kinematics(body, S(hip_flexion=30, upper_lumbar_flexion=10, shoulder_flexion=90))
    arm = lm["right_elbow"] - lm["right_shoulder"]
    thorax = lm["neck"] - lm["thoracolumbar"]
    assert np.dot(arm, thorax) == pytest.approx(0.0, abs=TOL)
    assert arm[0] > 0 and arm[1] < 0


def test_elbow_flexion_bends_forearm_forward(body):
    lm = forward_kinematics(body, S(elbow_flexion=90))
    np.testing.assert_allclose(
        lm["right_wrist"] - lm["right_elbow"], [body.forearm, 0.0], atol=TOL)


def test_shoulder_hangs_below_neck_on_thorax_line(body):
    lm = forward_kinematics(body, S(ankle_dorsiflexion=10, knee_flexion=60, hip_flexion=50,
                                    lumbar_flexion=30))
    a = lm["neck"] - lm["right_shoulder"]
    b = lm["neck"] - lm["thoracolumbar"]
    assert cross2d(a, b) == pytest.approx(0.0, abs=TOL)
    assert np.linalg.norm(a) == pytest.approx(body.shoulder_drop)


def test_toe_extension_lifts_the_heel_about_the_ball(body):
    flat = forward_kinematics(body, S())
    up = forward_kinematics(body, S(toe_extension=30, ankle_dorsiflexion=-30))
    # ball and toes stay where they were
    np.testing.assert_allclose(up["right_ball"], flat["right_ball"], atol=TOL)
    np.testing.assert_allclose(up["right_toe"], flat["right_toe"], atol=TOL)
    # heel rises by ball length x sin(30)
    assert up["right_heel"][1] == pytest.approx(body.ball_length * 0.5)
    # plantarflexing by the same amount keeps the shank vertical
    shin = up["right_knee"] - up["right_ankle"]
    assert shin[0] == pytest.approx(0.0, abs=TOL)
    assert up["right_ankle"][1] > flat["right_ankle"][1]


def random_leg(rng, n):
    return LegAngles(hip_flexion=rng.uniform(-20, 150, n), knee_flexion=rng.uniform(0, 160, n),
                     ankle_dorsiflexion=rng.uniform(-40, 45, n),
                     toe_extension=rng.uniform(0, 60, n))


def random_arm(rng, n):
    return ArmAngles(shoulder_flexion=rng.uniform(-40, 180, n),
                     elbow_flexion=rng.uniform(0, 150, n), wrist_flexion=rng.uniform(-60, 60, n))


def random_angles(rng, n, stance="right"):
    """Independent random angles on each side (an asymmetric pose)."""
    return JointAngles(
        right_leg=random_leg(rng, n), left_leg=random_leg(rng, n),
        right_arm=random_arm(rng, n), left_arm=random_arm(rng, n),
        lower_lumbar_flexion=rng.uniform(-20, 45, n),
        upper_lumbar_flexion=rng.uniform(-15, 30, n),
        neck_flexion=rng.uniform(-60, 60, n),
        stance=stance,
    )


@pytest.mark.parametrize("stance", SIDES)
def test_segment_lengths_preserved_for_random_poses(population, stance):
    rng = np.random.default_rng(3)
    for _ in range(40):
        b = population.sample(rng)
        lm = forward_kinematics(b, random_angles(rng, 50, stance))
        length_of = segment_lengths(b)
        for seg, (a, c) in SEGMENT_ENDPOINTS.items():
            lengths = np.linalg.norm(lm[c] - lm[a], axis=-1)
            np.testing.assert_allclose(lengths, length_of[seg], atol=TOL, err_msg=seg)
        for s in SIDES:
            toes = np.linalg.norm(lm[f"{s}_toe"] - lm[f"{s}_ball"], axis=-1)
            np.testing.assert_allclose(toes, b.toe_length, atol=TOL)
        assert np.all(np.isfinite(np.stack([lm[k] for k in LANDMARKS])))


@pytest.mark.parametrize("stance", SIDES)
def test_stance_toes_stay_flat_and_still(body, stance):
    rng = np.random.default_rng(8)
    lm = forward_kinematics(body, random_angles(rng, 100, stance))
    np.testing.assert_allclose(lm[f"{stance}_ball"], np.tile([body.ball_offset, 0.0], (100, 1)),
                               atol=TOL)
    np.testing.assert_allclose(lm[f"{stance}_toe"][:, 1], 0.0, atol=TOL)


@pytest.mark.parametrize("stance", SIDES)
def test_measured_angles_round_trip(population, stance):
    rng = np.random.default_rng(4)
    b = population.sample(rng)
    angles = random_angles(rng, 200, stance)
    measured = measure_joint_angles(forward_kinematics(b, angles))
    for name, value in angles.as_dict().items():
        np.testing.assert_allclose(measured[name], value, atol=1e-8, err_msg=name)
    np.testing.assert_allclose(measured["lumbar_flexion"], angles.lumbar_flexion, atol=1e-8)
    np.testing.assert_allclose(measured["knee_flexion"], angles.knee_flexion, atol=1e-8)


def test_from_dict_inverts_as_dict():
    angles = random_angles(np.random.default_rng(6), 3, "left")
    again = JointAngles.from_dict(angles.as_dict(), stance="left")
    for name, value in again.as_dict().items():
        np.testing.assert_array_equal(value, angles.as_dict()[name])


def test_array_angles_match_scalar_calls(body):
    rng = np.random.default_rng(5)
    angles = random_angles(rng, 5)
    batch = forward_kinematics(body, angles)
    for i in range(5):
        single = forward_kinematics(body, JointAngles.from_dict(
            {k: float(v[i]) for k, v in angles.as_dict().items()}))
        for name in LANDMARKS:
            np.testing.assert_allclose(batch[name][i], single[name], atol=TOL)


def test_spine_joints_bend_only_what_is_above_them(body):
    straight = forward_kinematics(body, S(hip_flexion=20))
    bent = forward_kinematics(body, S(hip_flexion=20, lower_lumbar_flexion=20,
                                      upper_lumbar_flexion=10))
    for name in ("right_ankle", "right_knee", "right_hip", "lumbar"):
        np.testing.assert_allclose(bent[name], straight[name], atol=TOL)
    m = measure_joint_angles(bent)
    assert m["pelvis_inclination"] == pytest.approx(20.0)
    assert m["abdomen_inclination"] == pytest.approx(40.0)
    assert m["trunk_inclination"] == pytest.approx(50.0)
    assert m["lumbar_flexion"] == pytest.approx(30.0)


def test_symmetric_lumbar_flexion_is_shared_between_spine_joints():
    a = S(lumbar_flexion=40.0)
    assert a.upper_lumbar_flexion == pytest.approx(40.0 * UPPER_LUMBAR_SHARE)
    assert a.lumbar_flexion == pytest.approx(40.0)


def test_bending_the_lower_back_turns_the_rib_cage_more_than_the_c7_line(body):
    """Why the spine has two joints: when the lower back rounds, the rib
    cage (what studies measure as the thorax) turns further than a
    straight line from the navel to C7."""
    lm = forward_kinematics(body, S(hip_flexion=60, lumbar_flexion=30))
    m = measure_joint_angles(lm)
    chord = np.degrees(np.arctan2(*(lm["neck"] - lm["lumbar"])))
    assert m["trunk_inclination"] > chord + 4.0


def test_unprefixed_limb_angle_is_the_mean_of_both_sides():
    a = JointAngles(right_leg=LegAngles(knee_flexion=100), left_leg=LegAngles(knee_flexion=60))
    assert a.knee_flexion == pytest.approx(80.0)
    with pytest.raises(AttributeError):
        a.not_an_angle  # noqa: B018
    with pytest.raises(TypeError):
        S(knee=10)
    with pytest.raises(ValueError):
        JointAngles(stance="middle")
