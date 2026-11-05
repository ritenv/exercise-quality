import numpy as np
import pytest

from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import JointAngles, forward_kinematics
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat
from exercise_quality.exercises.squat.metrics import (
    depth_category,
    describe_rep,
    extensor_bias,
    trunk_tibia_angle,
)


@pytest.fixture(scope="module")
def body():
    return BodyPopulation.load().median_body("male")


@pytest.mark.parametrize("knee, expected", [
    (45, "partial"), (89.9, "partial"), (90, "parallel"), (109, "parallel"),
    (110, "deep"), (134, "deep"), (140, "beyond deep"),
])
def test_depth_bins(knee, expected):
    assert depth_category(knee) == expected


@pytest.mark.parametrize("value, expected", [(15, "hip"), (-15, "knee"), (0, "neutral"),
                                             (10, "neutral"), (-10, "neutral")])
def test_extensor_bias(value, expected):
    assert extensor_bias(value) == expected


def test_trunk_tibia_angle_on_a_known_pose(body):
    # shank 30 deg, knee 100 -> thigh -70, hip 100 -> pelvis 30, lumbar 10 -> thorax 40
    lm = forward_kinematics(body, JointAngles(30, 100, 100, lumbar_flexion=10))
    assert trunk_tibia_angle(lm) == pytest.approx(10.0)


def test_describe_rep(body):
    d = describe_rep(generate_squat(body, SquatStyle(peak_knee_flexion=120)))
    assert d["depth"] == "deep"
    assert d["peak_knee_flexion"] == pytest.approx(120.0)
    assert 0.0 < d["com_over_foot_at_bottom"] < 1.0
    assert d["extensor_bias"] in ("hip", "knee", "neutral")
    # thigh is roughly horizontal at 120 deg knee with ~35 deg of ankle
    assert abs(d["thigh_below_horizontal_deg"]) < 15
    assert np.isfinite(d["trunk_tibia_at_bottom"])
