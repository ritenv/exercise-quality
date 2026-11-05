"""Centre of mass and support region."""

import numpy as np
import pytest

from exercise_quality.core import deleva
from exercise_quality.core.balance import (
    centre_of_mass,
    segment_centres,
    segment_masses,
    support_region,
)
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import JointAngles, forward_kinematics


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module", params=deleva.SEXES)
def body(request, population):
    return population.median_body(request.param)


def test_segment_masses_add_up_to_body_mass(body):
    assert sum(segment_masses(body).values()) == pytest.approx(body.mass, rel=2e-4)


def test_standing_com_is_over_the_ankle_except_for_the_feet(body):
    """Standing straight, everything above the ankle is stacked at x=0. Only
    the feet sit forward: their centre is 40-44% of foot length from the
    heel while the ankle is at 21%."""
    com = centre_of_mass(body, forward_kinematics(body, JointAngles()))
    m = segment_masses(body)
    foot_x = (body.segments["foot"].com_fraction - 0.21) * body.foot_length
    assert com[0] == pytest.approx(m["foot"] * foot_x / sum(m.values()), abs=1e-12)
    assert 0 < com[0] < 0.002


def test_standing_com_height_is_plausible(population):
    """Textbook figures put the standing centre of mass a little above half
    of stature. This is a sanity range, not a validation."""
    for sex in deleva.SEXES:
        for body in population.bodies(sex)[::50]:
            com = centre_of_mass(body, forward_kinematics(body, JointAngles()))
            assert 0.53 < com[1] / body.stature < 0.60, (sex, com[1] / body.stature)


def test_segment_centres_follow_de_leva_direction(body):
    """de Leva measures from the origin landmark: trunk from C7, thigh from
    the hip. Getting the direction backwards would move the trunk centre
    by several centimetres, so check it explicitly."""
    lm = forward_kinematics(body, JointAngles())
    c = segment_centres(body, lm)
    trunk_from_neck = lm["neck"][1] - c["trunk"][1]
    assert trunk_from_neck == pytest.approx(
        body.segments["trunk"].com_fraction * body.trunk)
    thigh_from_hip = lm["hip"][1] - c["thigh"][1]
    assert thigh_from_hip == pytest.approx(
        body.segments["thigh"].com_fraction * body.thigh)
    head_from_vertex = lm["vertex"][1] - c["head"][1]
    assert head_from_vertex == pytest.approx(
        body.segments["head"].com_fraction * body.head)


def test_raising_arms_moves_com_by_the_arm_moment(body):
    """Lifting both arms forward moves the whole-body centre of mass by
    (arm mass x arm centre shift) / body mass, and nothing else moves."""
    down = forward_kinematics(body, JointAngles())
    forward = forward_kinematics(body, JointAngles(shoulder_flexion=90))
    m = segment_masses(body)
    c_down = segment_centres(body, down)
    c_fwd = segment_centres(body, forward)
    arm = ("upper_arm", "forearm", "hand")
    expected = sum(m[s] * (c_fwd[s] - c_down[s]) for s in arm) / sum(m.values())
    actual = centre_of_mass(body, forward) - centre_of_mass(body, down)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual[0] > 0.02  # arms are ~10% of mass, reaching ~0.3 m forward


def test_com_is_translation_consistent(body):
    """Mirroring the whole body front to back mirrors the COM x."""
    rng = np.random.default_rng(0)
    angles = JointAngles(*rng.uniform(-30, 60, size=7))
    lm = forward_kinematics(body, angles)
    mirrored = {k: v * np.array([-1.0, 1.0]) for k, v in lm.items()}
    a = centre_of_mass(body, lm)
    b = centre_of_mass(body, mirrored)
    assert b[0] == pytest.approx(-a[0])
    assert b[1] == pytest.approx(a[1])


def test_batched_com_matches_single(body):
    rng = np.random.default_rng(1)
    vals = rng.uniform(-20, 90, size=(7, 10))
    batch = centre_of_mass(body, forward_kinematics(body, JointAngles(*vals)))
    for i in range(10):
        single = centre_of_mass(body, forward_kinematics(body, JointAngles(*vals[:, i])))
        np.testing.assert_allclose(batch[i], single, atol=1e-12)


def test_support_region_fractions(body):
    s = support_region(body)
    assert s.length == pytest.approx(body.foot_length)
    assert s.fraction(s.heel_x) == pytest.approx(0.0)
    assert s.fraction(s.toe_x) == pytest.approx(1.0)
    assert s.fraction(0.0) == pytest.approx(body.heel_offset / body.foot_length)
    assert s.x_at(s.fraction(0.05)) == pytest.approx(0.05)
    assert s.contains(0.0)
    assert not s.contains(s.toe_x + 0.01)
    assert s.margin(s.heel_x - 0.02) == pytest.approx(-0.02)


def test_crouching_with_vertical_trunk_falls_backwards(body):
    """A deep knee bend with the shank and trunk vertical puts the hips and
    thighs far behind the heel. It should be detected as unbalanced."""
    lm = forward_kinematics(body, JointAngles(knee_flexion=110, hip_flexion=110))
    com = centre_of_mass(body, lm)
    assert not support_region(body).contains(com[0])
    assert com[0] < -body.heel_offset
