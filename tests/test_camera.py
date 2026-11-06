"""Virtual camera: projection, MediaPipe format, and the real-video pipeline on synthetic reps."""

import numpy as np
import pytest

from exercise_quality.core import model3d as M
from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.observed import from_model_landmarks, observed_angles
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import reps as R
from exercise_quality.exercises.squat import rubric
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_squat
from exercise_quality.video import camera as cam
from exercise_quality.video import mediapipe_pose as mpp


@pytest.fixture(scope="module")
def body():
    return M.make_body3d(BodyPopulation.load(), "male", 3)


@pytest.fixture(scope="module")
def seq(body):
    s = generate_squat(body, SquatStyle(stance=M.Stance(), peak_knee_flexion=120.0), fps=30.0)
    assert s.is_feasible
    return s


def test_a_far_side_camera_sees_the_orthographic_side_view(seq):
    far = cam.Camera(azimuth=90.0, distance=200.0, long_side_fov=1.0, lens_height=0.9)
    track = mpp.observed_track(cam.synthetic_raw(seq, far))
    assert track.source.endswith(":right") and track.yaw_scale > 0.999
    exact = observed_angles(R.smoothed(from_model_landmarks(seq.landmarks, seq.fps)))
    seen = observed_angles(R.smoothed(track))
    for name in ("knee_flexion", "trunk_inclination", "shank_inclination", "thigh_inclination"):
        np.testing.assert_allclose(seen[name], exact[name], atol=0.2, err_msg=name)


def test_a_near_camera_distorts_angles_by_perspective(seq):
    near = mpp.observed_track(cam.synthetic_raw(seq, cam.Camera(distance=2.0)))
    exact = observed_angles(from_model_landmarks(seq.landmarks, seq.fps))
    seen = observed_angles(near)
    b = int(np.argmax(exact["knee_flexion"]))
    assert abs(seen["knee_flexion"][b] - exact["knee_flexion"][b]) > 0.5


def test_camera_angle_is_read_back_from_world_landmarks(seq):
    for azimuth, expected in ((90.0, 1.0), (45.0, np.cos(np.radians(45.0))), (0.0, 0.0)):
        raw = cam.synthetic_raw(seq, cam.Camera(azimuth=azimuth))
        assert mpp.world_yaw_scale(raw) == pytest.approx(expected, abs=0.02)


def test_the_left_side_camera_tracks_the_left_leg(seq):
    track = mpp.observed_track(cam.synthetic_raw(seq, cam.Camera(azimuth=-90.0)))
    assert track.source.endswith(":left")
    assert np.nanmedian(track["toe"][:, 0] - track["heel"][:, 0]) > 0    # faces +x after mirroring


def test_the_rubric_runs_on_a_synthetic_video(seq, body):
    rng = np.random.default_rng(0)
    raw = cam.synthetic_raw(seq, cam.Camera(), noise=TN.TrackerNoise(), rng=rng)
    result = rubric.assess(mpp.observed_track(raw), body=body.sagittal)
    assert len(result.reps) == 1 and result.reliable
    front = rubric.assess(mpp.observed_track(cam.synthetic_raw(seq, cam.Camera(azimuth=0.0))),
                          body=body.sagittal)
    assert not front.reliable          # a front view is not a side view


def test_mediapipe_points_land_where_the_model_puts_them(seq):
    raw = cam.synthetic_raw(seq, cam.Camera(azimuth=0.0))
    # front view: the person's right hip (24) appears on the image left
    assert np.all(raw["image"][:, 24, 0] < raw["image"][:, 23, 0])
    width = np.linalg.norm(raw["world"][:, 24] - raw["world"][:, 23], axis=-1)
    b = seq.body
    expected = b.hip_joint_width + cam.MEDIAPIPE_HIP_OUTWARD * (b.hip_breadth - b.hip_joint_width)
    np.testing.assert_allclose(width, expected, rtol=1e-3)


def test_side_view_offset_and_lens_height_are_recovered():
    from exercise_quality.core.model3d import Stance, median_body3d
    from exercise_quality.core.sagittal_body import BodyPopulation
    from exercise_quality.exercises.squat.balanced import SquatStyle, generate_feasible_squat
    from exercise_quality.video import camera as cam, mediapipe_pose as mpp

    body = median_body3d(BodyPopulation.load(), "female")
    seq = generate_feasible_squat(body, SquatStyle(stance=Stance()), fps=30.0)
    for offset in (-40.0, -15.0, 15.0, 40.0):
        raw = cam.synthetic_raw(seq, cam.Camera(azimuth=90.0 - offset))
        assert mpp.side_view_offset(raw) == pytest.approx(offset, abs=1.0)
    for lens in (0.4, 1.2):
        raw = cam.synthetic_raw(seq, cam.Camera(lens_height=lens, distance=4.0))
        assert mpp.lens_height_estimate(raw) == pytest.approx(lens, abs=0.25)


def test_sampled_side_cameras_follow_the_filming_ranges():
    import numpy as np
    from exercise_quality.video import camera as cam

    rng = np.random.default_rng(0)
    cams = [cam.sample_side_camera(rng) for _ in range(200)]
    assert all(abs(c.azimuth - 90.0) <= cam.SIDE_AZIMUTH_SPREAD for c in cams)
    assert all(cam.SIDE_LENS_HEIGHT[0] <= c.lens_height <= cam.SIDE_LENS_HEIGHT[1] for c in cams)
    assert cam.sample_side_camera(rng, offset=20.0).azimuth == 70.0
