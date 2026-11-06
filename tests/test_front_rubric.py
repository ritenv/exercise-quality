"""Front-view squat rubric: BSA hip level and frontal knee position."""

from dataclasses import replace

import numpy as np
import pytest

from exercise_quality.core import model3d as M
from exercise_quality.exercises.squat import front_rubric as FR
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_feasible_squat
from exercise_quality.core.timing import RepTiming
from exercise_quality.video import camera as cam
from exercise_quality.video import mediapipe_pose as mpp


@pytest.fixture(scope="module")
def body():
    return M.median_body3d(FR._population(), "male")


def film(body, noise=False, camera=cam.Camera(azimuth=0.0, distance=4.0), **stance):
    style = SquatStyle(peak_knee_flexion=115.0, timing=RepTiming(stand_before=0.8, descent=1.5, ascent=1.5,
                                                                  stand_after=0.8),
                       stance=replace(M.Stance(), **stance))
    seq = generate_feasible_squat(body, style, fps=30.0)
    assert seq.is_feasible
    rng = np.random.default_rng(0)
    raw = cam.synthetic_raw(seq, camera, noise=cam.TN.TrackerNoise() if noise else None, rng=rng)
    return seq, raw


def grade(seq, raw):
    result = FR.assess(raw, body=seq.body, segment=(0, len(seq.times)))
    assert result is not None and len(result.reps) == 1
    return result


def test_a_clean_rep_passes_and_matches_the_true_values(body):
    seq, raw = film(body)
    result = grade(seq, raw)
    assert result.reliable and result.camera_angle == pytest.approx(0.0, abs=0.5)
    rep = result.reps[0]
    assert all(c.passed for c in rep.criteria.values())
    found = FR.F.find_reps(FR.F._smooth_image(raw, slice(None)), 30.0)[0]
    truth = FR.true_values(seq, found.start, found.end)
    assert rep.criteria["hips_level"].value == pytest.approx(truth["hips_level"], abs=0.3)
    assert rep.criteria["knees_over_feet"].value == pytest.approx(truth["knees_over_feet"], abs=1.0)


def test_knees_caving_past_the_ankles_fail_only_the_knee_check(body):
    seq, raw = film(body, knee_track_right=45.0, knee_track_left=45.0)
    rep = grade(seq, raw).reps[0]
    assert not rep.criteria["knees_over_feet"].passed
    assert rep.criteria["hips_level"].passed
    assert FR.true_values(seq, 0, len(seq.times) - 1)["knees_over_feet"] > 0
    assert rep.features["kasr_bottom"] < 1.0


def test_a_dropped_hip_fails_only_the_hip_check(body):
    seq, raw = film(body)
    rep = grade(seq, FR.drop_hip(raw, 10.0)).reps[0]
    assert not rep.criteria["hips_level"].passed
    assert rep.criteria["knees_over_feet"].passed
    assert rep.criteria["hips_level"].value == pytest.approx(10.0, abs=1.0)
    assert rep.features["tilt_at_bottom"] > 8.0          # + = the person's right hip lower


def test_a_turned_camera_is_flagged(body):
    seq, raw = film(body, camera=cam.Camera(azimuth=20.0, distance=4.0))
    result = grade(seq, raw)
    assert not result.reliable and result.camera_angle == pytest.approx(20.0, abs=2.0)


def test_real_front_clips():
    for name, sex in (("front_squat_1", "female"), ("front_squat_2", "female"), ("front_squat_5", "male")):
        result = FR.assess(mpp.load_raw(name), body=M.median_body3d(FR._population(), sex))
        assert result.reliable and abs(result.camera_angle) < 5.0, name
        assert result.deficit_total == 0, name
    angled = FR.assess(mpp.load_raw("front_squat_3"))
    assert not angled.reliable and angled.camera_angle < -10.0


def test_left_right_pairs_share_tracker_noise():
    rng = np.random.default_rng(1)
    still = np.zeros((900, 33, 2))
    shared = cam.image_noise(still, 30.0, 500.0, cam.TN.TrackerNoise(), rng)
    apart = cam.image_noise(still, 30.0, 500.0, cam.TN.TrackerNoise(), rng, pair_share=0.0)
    gap = lambda x: np.std(x[:, 24, 1] - x[:, 23, 1])        # noqa: E731
    assert gap(shared) < 0.6 * gap(apart)


def test_stored_tolerances_match_the_noise_model():
    fresh = FR.measurement_tolerances(n=80, seed=11)
    for key, stored in FR.TOLERANCE.items():
        assert 0.6 * stored < fresh[key] < 1.6 * stored, (key, stored, fresh[key])
