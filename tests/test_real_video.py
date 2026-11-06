"""Tracker view, rep detection, tracker noise and the stored real tracks."""

from dataclasses import replace

import numpy as np
import pytest

from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.balance import support_region
from exercise_quality.core.observed import (
    ObservedTrack, from_model_landmarks, observed_angles, segment_lengths, wrap,
)
from exercise_quality.core.observed_balance import track_com_fraction
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import JointAngles, forward_kinematics, measure_joint_angles
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import reps as R
from exercise_quality.exercises.squat.balanced import SquatStyle, generate_feasible_squat
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_style
from exercise_quality.video import compare as C
from exercise_quality.video import mediapipe_pose as mpp


@pytest.fixture(scope="module")
def population():
    return BodyPopulation.load()


@pytest.fixture(scope="module")
def body(population):
    return population.median_body("female")


@pytest.fixture(scope="module")
def squat(body):
    style = SquatStyle(peak_knee_flexion=115.0,
                       timing=RepTiming(stand_before=0.6, descent=1.5, ascent=1.5, stand_after=0.6))
    return generate_feasible_squat(body, style, fps=30.0)


# --- tracker view --------------------------------------------------------

def test_tracker_angles_match_model_angles_where_definitions_agree(squat):
    track = from_model_landmarks(squat.landmarks, squat.fps)
    seen = observed_angles(track)
    model = measure_joint_angles(squat.landmarks)
    for name in ("knee_flexion", "shank_inclination", "thigh_inclination", "ankle_dorsiflexion"):
        np.testing.assert_allclose(seen[name], model[f"right_{name}"], atol=1e-9)
    assert np.allclose(seen["heel_lift"], 0.0, atol=1e-9)


def test_trunk_line_equals_thorax_only_with_a_straight_back(body):
    straight = forward_kinematics(body, JointAngles.symmetric(hip_flexion=60.0, knee_flexion=90.0,
                                                              ankle_dorsiflexion=30.0))
    seen = observed_angles(from_model_landmarks(straight, 1.0))
    assert seen["trunk_inclination"][0] == pytest.approx(
        measure_joint_angles(straight)["trunk_inclination"], abs=1e-9)
    rounded = forward_kinematics(body, JointAngles.symmetric(hip_flexion=60.0, knee_flexion=90.0,
                                                             ankle_dorsiflexion=30.0, lumbar_flexion=20.0))
    chord = observed_angles(from_model_landmarks(rounded, 1.0))["trunk_inclination"][0]
    thorax = measure_joint_angles(rounded)["trunk_inclination"]
    assert chord < thorax - 5.0    # the hip-shoulder line under-reads a rounded back


def test_arm_angles_round_trip_and_wrap(body):
    pose = JointAngles.symmetric(shoulder_flexion=150.0, elbow_flexion=110.0, hip_flexion=20.0)
    seen = observed_angles(from_model_landmarks(forward_kinematics(body, pose), 1.0))
    assert seen["shoulder_flexion"][0] == pytest.approx(150.0, abs=1e-6)
    assert seen["elbow_flexion"][0] == pytest.approx(110.0, abs=1e-6)
    assert wrap(-253.0) == pytest.approx(107.0)
    assert wrap(180.0) == pytest.approx(180.0)


def test_track_needs_every_point():
    with pytest.raises(ValueError):
        ObservedTrack(points={"hip": np.zeros((3, 2))}, fps=30.0)


# --- reps ------------------------------------------------------------------

def test_one_synthetic_rep_is_found_with_sensible_phases(squat):
    track = R.smoothed(from_model_landmarks(squat.landmarks, squat.fps))
    found = R.find_reps(track)
    assert len(found) == 1
    rep = found[0]
    assert rep.start < rep.bottom_reached <= rep.lowest <= rep.ascent_start < rep.end
    f = R.rep_features(track, rep)
    assert f["knee_peak"] == pytest.approx(115.0, abs=1.5)
    assert 0.6 < f["descent_s"] < 1.5 and 0.6 < f["ascent_s"] < 1.5
    assert f["heel_rise_max"] == pytest.approx(0.0, abs=1e-3)


def test_back_to_back_reps_and_shallow_bends(squat):
    track = from_model_landmarks(squat.landmarks, squat.fps)
    three = track.map_points(lambda v: np.concatenate([v, v, v]))
    assert len(R.find_reps(R.smoothed(three))) == 3
    shallow = generate_feasible_squat(squat.body, replace(squat.style, peak_knee_flexion=35.0), fps=30.0)
    assert R.find_reps(R.smoothed(from_model_landmarks(shallow.landmarks, 30.0))) == []


def test_measured_phase_durations_follow_the_timing():
    slow = C.measured_phases(RepTiming(descent=2.0, ascent=2.0))
    fast = C.measured_phases(RepTiming(descent=1.0, ascent=1.0))
    assert slow[0] == pytest.approx(2 * fast[0], rel=0.05)
    matched = C.measured_phases(C.matched_timing(0.8, 0.3, 0.7))
    np.testing.assert_allclose(matched, [0.8, 0.3, 0.7], atol=0.05)


# --- tracker noise ----------------------------------------------------------

def test_noise_is_reproducible_and_scaled_to_the_body(squat):
    track = from_model_landmarks(squat.landmarks, squat.fps)
    a = TN.apply(track, TN.TrackerNoise(), np.random.default_rng(3))
    b = TN.apply(track, TN.TrackerNoise(), np.random.default_rng(3))
    np.testing.assert_array_equal(a["knee"], b["knee"])
    big = track.map_points(lambda v: 100.0 * v)
    c = TN.apply(big, TN.TrackerNoise(), np.random.default_rng(3))
    np.testing.assert_allclose(c["knee"], 100.0 * a["knee"], rtol=1e-9, atol=1e-9)


def test_matched_synthetic_reps_are_3d_and_filmed():
    body = C._population().body("female", 10)
    timing = C.matched_timing(0.8, 0.3, 0.7)
    out = C.synthetic_rep(body, 110.0, timing, C.arm_posture(150, 110), 3, return_track=True)
    assert out is not None
    features, track, _ = out
    assert track.units == "px" and track.source.startswith("mediapipe:synthetic")
    # knees travelling out over turned-out toes shorten the thigh a side camera sees
    lengths = segment_lengths(track)
    knee = observed_angles(track)["knee_flexion"]
    assert np.median(lengths["thigh"][knee > 90]) < 0.97 * np.median(lengths["thigh"][knee < 10])


def test_noise_model_reproduces_the_measured_jitter(squat):
    track = from_model_landmarks(squat.landmarks, squat.fps)
    long = track.map_points(lambda v: np.concatenate([v] * 8))
    noisy = TN.apply(long, TN.TrackerNoise(drift_scale=0.0), np.random.default_rng(1))
    smooth = R.smoothed(noisy)
    leg = float(np.median(segment_lengths(smooth)["thigh"] + segment_lengths(smooth)["shank"]))
    for name in ("hip", "knee", "ankle"):
        measured = np.std((noisy[name] - smooth[name]) / leg, axis=0)
        np.testing.assert_allclose(measured, TN.MEASURED_JITTER[name], rtol=0.2)


# --- balance from a tracked pose ---------------------------------------------

def test_com_from_tracker_view_matches_the_generator(population):
    rng = np.random.default_rng(5)
    for _ in range(5):
        body = population.sample(rng)
        seq = generate_feasible_squat(body, sample_style(rng, body, ISHIDA_PARALLEL), fps=30.0)
        exact = support_region(body).fraction(seq.com[:, 0])
        estimate = track_com_fraction(body, from_model_landmarks(seq.landmarks, 30.0))
        assert np.max(np.abs(estimate - exact)) < 0.1
        assert np.mean(np.abs(estimate - exact)) < 0.03


# --- matching ------------------------------------------------------------------

def test_body_matching_finds_the_same_person(population):
    target = population.body("male", 123)
    ratios = C.model_proportions(target)
    bodies, _, dist = C.matched_bodies(population, ratios, "male", k=3)
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert C.model_proportions(bodies[0]) == pytest.approx(ratios)


# --- stored real tracks ---------------------------------------------------------

def test_stored_tracks_load_and_face_forward():
    names = mpp.available()
    assert {"squat_1", "squat_2"} <= set(names)
    manifest = mpp.manifest()
    for name in names:
        assert len(manifest[name]["video_sha256"]) == 64
        track = mpp.observed_track(name)
        assert np.median(track["toe"][:, 0] - track["heel"][:, 0]) > 0
        assert np.median(track["shoulder"][:, 1]) > np.median(track["ankle"][:, 1])
        assert 0.0 < track.yaw_scale <= 1.0


def test_side_on_clip_needs_no_yaw_correction_and_has_six_reps():
    track = mpp.observed_track("squat_1")
    assert track.yaw_scale > 0.95
    assert len(R.find_reps(R.smoothed(track))) == 6
    oblique = mpp.observed_track("squat_2")
    assert oblique.yaw_scale < 0.8
    assert len(R.find_reps(R.smoothed(oblique))) == 13
