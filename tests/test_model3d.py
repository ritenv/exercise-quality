"""3D body model: reduces to the side-view model, keeps lengths, places faults where they belong."""

from dataclasses import replace

import numpy as np
import pytest

from exercise_quality.core import model3d as M
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import LANDMARKS
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module")
def body(population):
    return M.make_body3d(population, "female", 10)


def rep(body, stance, **style):
    return generate_squat(body, SquatStyle(stance=stance, **style), fps=30.0)


def test_feet_under_the_hips_reproduces_the_side_view_model(body):
    flat = rep(body, M.Stance().under_hips(body))
    side = generate_squat(body.sagittal, SquatStyle(), fps=30.0)
    assert flat.is_feasible and side.is_feasible
    for name in LANDMARKS:
        np.testing.assert_allclose(flat.landmarks[name], side.landmarks[name], atol=1e-9, err_msg=name)
    np.testing.assert_allclose(flat.com, side.com, atol=1e-9)


def test_closed_chain_keeps_hip_width_and_segment_lengths(body):
    for stance in (M.Stance(), M.Stance(width_ratio=1.6, toe_out=35.0),
                   M.Stance(knee_track_right=15.0, lateral_shift=0.05)):
        seq = rep(body, stance)
        lm = seq.landmarks3d
        width = np.linalg.norm(lm["right_hip"] - lm["left_hip"], axis=-1)
        np.testing.assert_allclose(width, body.hip_joint_width, rtol=1e-4)
        for side in ("right", "left"):
            thigh = np.linalg.norm(lm[f"{side}_hip"] - lm[f"{side}_knee"], axis=-1)
            shank = np.linalg.norm(lm[f"{side}_knee"] - lm[f"{side}_ankle"], axis=-1)
            np.testing.assert_allclose(thigh, body.thigh, rtol=1e-9)
            np.testing.assert_allclose(shank, body.shank, rtol=1e-9)
            np.testing.assert_allclose(lm[f"{side}_heel"][:, 1], 0.0, atol=1e-12)   # heels down


def test_feet_wider_than_the_hips_lean_the_legs_in(body):
    seq = rep(body, M.Stance(width_ratio=1.23, toe_out=0.0))
    lm = seq.landmarks3d
    ankle_gap = lm["right_ankle"][0, 2] - lm["left_ankle"][0, 2]
    assert ankle_gap > body.hip_joint_width
    # the right hip is medial to the right ankle: the leg leans in
    assert lm["right_hip"][0, 2] < lm["right_ankle"][0, 2]
    assert np.allclose(lm["mid_hip"][:, 2], 0.0, atol=1e-9)


def test_knees_track_over_the_toes_unless_they_cave_in(body):
    good = rep(body, M.Stance(toe_out=20.0))
    caved = rep(body, M.Stance(toe_out=20.0, knee_track_right=30.0, knee_track_left=30.0))
    b = int(np.argmax(good.angles.knee_flexion))

    def outside_hip_ankle_line(seq):
        # seen from the front: how far the right knee sits outside the hip-ankle line
        lm = {k: v[b] for k, v in seq.landmarks3d.items()}
        hip, knee, ankle = lm["right_hip"], lm["right_knee"], lm["right_ankle"]
        line_z = ankle[2] + (hip[2] - ankle[2]) * (knee[1] - ankle[1]) / (hip[1] - ankle[1])
        return knee[2] - line_z

    # healthy: the knee moves out over the turned-out foot; caved: inside the line
    assert outside_hip_ankle_line(good) > 0.02
    assert outside_hip_ankle_line(caved) < 0.0
    # at the start nothing differs
    np.testing.assert_allclose(caved.landmarks3d["right_knee"][0], good.landmarks3d["right_knee"][0], atol=1e-6)


def test_lateral_shift_moves_the_centre_of_mass_sideways(body):
    centred = rep(body, M.Stance())
    shifted = rep(body, M.Stance(lateral_shift=0.06))
    b = int(np.argmax(centred.angles.knee_flexion))
    assert abs(centred.com3d[b, 2]) < 1e-6
    assert shifted.com3d[b, 2] > 0.03


def test_the_forward_balance_is_solved_on_the_3d_centre_of_mass(body):
    seq = rep(body, M.Stance(width_ratio=1.5, toe_out=30.0))
    assert seq.is_feasible
    assert np.max(np.abs(seq.balance_error)) < 1e-6
    support = M.support_region(body, M.Stance(width_ratio=1.5, toe_out=30.0))
    frac = support.fraction(seq.com3d[:, 0])
    assert np.all((frac > 0.2) & (frac < 0.6))


def test_bodies_are_plausible_and_reproducible(population):
    a = M.make_body3d(population, "male", 5)
    assert a == M.make_body3d(population, "male", 5)
    widths = np.array([M.make_body3d(population, "male", i).hip_joint_width for i in range(400)])
    assert 0.15 < widths.mean() < 0.17 and 0.005 < widths.std() < 0.011
    assert 0.3 < a.hip_breadth < 0.45 and a.shoulder_width > a.hip_joint_width
    # side-view attributes read through
    assert a.thigh == a.sagittal.thigh and a.mass == a.sagittal.mass


def test_style_without_stance_is_unchanged(body):
    seq = generate_squat(body.sagittal, replace(SquatStyle(), stance=None), fps=30.0)
    assert not seq.is_3d and seq.landmarks3d is None


def test_hip_angles_and_frontal_knee_angle(body):
    from exercise_quality.core.model3d import joint_angles_3d

    under = joint_angles_3d(rep(body, M.Stance().under_hips(body)).landmarks3d)
    assert abs(under["hip_abduction"][0]) < 1e-6 and abs(under["knee_fppa"][0]) < 1e-6
    wide_seq = rep(body, M.Stance(width_ratio=1.4, toe_out=0.0))
    wide = joint_angles_3d(wide_seq.landmarks3d)
    lm = {k: v[0] for k, v in wide_seq.landmarks3d.items()}
    lean = np.degrees(np.arctan2(lm["right_ankle"][2] - lm["right_hip"][2], lm["right_hip"][1] - lm["right_ankle"][1]))
    assert wide["right_hip_abduction"][0] == pytest.approx(lean, abs=0.5)
    assert wide["left_hip_abduction"][0] == pytest.approx(wide["right_hip_abduction"][0], abs=1e-9)
    # 3D hip flexion at the bottom is close to the in-plane value with straight feet
    seq = rep(body, M.Stance(width_ratio=1.0, toe_out=0.0))
    j = joint_angles_3d(seq.landmarks3d)
    b = int(np.argmax(seq.angles.knee_flexion))
    assert j["hip_flexion"][b] == pytest.approx(seq.measured()["hip_flexion"][b], abs=3.0)
    # part-way down: knees out over turned-out toes read negative, caved-in knees positive
    out_seq = rep(body, M.Stance(toe_out=25.0))
    in_seq = rep(body, M.Stance(toe_out=25.0, knee_track_right=35.0, knee_track_left=35.0))
    i = int(np.argmax(out_seq.angles.knee_flexion >= 60.0))
    assert joint_angles_3d(out_seq.landmarks3d)["knee_fppa"][i] < 0.0
    assert (joint_angles_3d(in_seq.landmarks3d)["knee_fppa"][i]
            > joint_angles_3d(out_seq.landmarks3d)["knee_fppa"][i] + 5.0)
