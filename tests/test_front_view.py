"""Front-view measures: signs and values on synthetic reps with known faults; resampling."""

import numpy as np
import pytest

from exercise_quality.core import model3d as M
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat.balanced import ARMS_CROSSED, SquatStyle, generate_squat
from exercise_quality.video import camera as cam
from exercise_quality.video import front_view as F
from exercise_quality.video import mediapipe_pose as mpp

TIMING = RepTiming(stand_before=0.8, descent=1.2, ascent=1.2, stand_after=0.8)


@pytest.fixture(scope="module")
def body():
    return M.make_body3d(BodyPopulation.load(), "female", 10)


def front(body, azimuth=0.0, **stance):
    seq = generate_squat(body, SquatStyle(stance=M.Stance(**stance), peak_knee_flexion=125.0,
                                          timing=TIMING, arms=ARMS_CROSSED), fps=30.0)
    raw = cam.synthetic_raw(seq, cam.Camera(azimuth=azimuth))
    m = F.measure(raw, segment=(0, len(seq.times)))
    assert len(m["reps"]) == 1
    return m, m["reps"][0]


def test_toe_out_and_view_angle_are_read_back(body):
    m, _ = front(body, azimuth=10.0, toe_out=30.0)
    assert m["toe_out"]["right"] == pytest.approx(30.0, abs=0.5)
    assert m["toe_out"]["left"] == pytest.approx(30.0, abs=0.5)
    assert m["view_yaw"] == pytest.approx(10.0, abs=0.5)


def test_a_square_symmetric_squat_reads_symmetric(body):
    _, r = front(body)
    assert r["fppa_right_half"] == pytest.approx(r["fppa_left_half"], abs=1e-6)
    assert abs(r["hip_shift"]) < 1e-6 and abs(r["pelvic_tilt"]) < 1e-6 and abs(r["depth_asymmetry"]) < 1e-6


def test_knees_caving_in_lower_the_knee_separation_and_raise_the_fppa(body):
    _, good = front(body)
    _, caved = front(body, knee_track_right=20.0, knee_track_left=20.0)
    assert caved["kasr_bottom"] < good["kasr_bottom"] - 0.3
    assert caved["fppa_right_half"] > good["fppa_right_half"] + 10.0


def test_signs_of_shift_and_tilt(body):
    _, r = front(body, lateral_shift=0.05)
    assert r["hip_shift"] > 0.05            # towards the person's right
    assert r["pelvic_tilt"] < 0.0           # their right hip is higher (the right leg is more upright)


def test_a_camera_off_square_makes_the_two_knees_look_different(body):
    _, square = front(body)
    _, turned = front(body, azimuth=15.0)
    assert abs(square["fppa_right_half"] - square["fppa_left_half"]) < 1e-6
    assert abs(turned["fppa_right_half"] - turned["fppa_left_half"]) > 20.0


def test_irregular_frame_times_are_resampled():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.choice([1 / 120, 1 / 60, 1 / 40, 1 / 20], size=200))
    signal = np.sin(2 * np.pi * 0.5 * t)
    raw = {"t": t, "fps": np.float64(45.0),
           "image": np.repeat(signal[:, None, None], 33, axis=1).repeat(4, axis=2),
           "world": np.zeros((200, 33, 3)), "presence": np.ones((200, 33))}
    out = mpp.resample_uniform(raw)
    assert out["fps"] == 30.0 and np.allclose(np.diff(out["t"]), 1 / 30)
    assert np.allclose(out["image"][:, 0, 0], np.sin(2 * np.pi * 0.5 * out["t"]), atol=0.02)
    regular = dict(raw, t=np.arange(200) / 45.0)
    assert mpp.resample_uniform(regular) is regular
