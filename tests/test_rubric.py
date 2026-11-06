"""Side-view squat rubric: thresholds, faults, build allowance, real clips."""

from dataclasses import replace

import numpy as np
import pytest

from exercise_quality.core.observed import from_model_landmarks
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import rubric
from exercise_quality.exercises.squat import track_faults as F
from exercise_quality.exercises.squat.balanced import ARMS_CROSSED, SquatStyle, generate_feasible_squat
from exercise_quality.video import compare as C
from exercise_quality.video import mediapipe_pose as mpp


@pytest.fixture(scope="module")
def population():
    return rubric._population()


@pytest.fixture(scope="module")
def body(population):
    return population.median_body("male")


def make_track(body, **style):
    base = SquatStyle(peak_knee_flexion=130.0, ankle_knee_ratio=0.3, trunk_peak_preference=30.0,
                      timing=RepTiming(stand_before=0.6, descent=1.5, ascent=1.5, stand_after=0.6))
    seq = generate_feasible_squat(body, replace(base, **style), fps=30.0)
    assert seq.is_feasible
    return from_model_landmarks(seq.landmarks, 30.0)


def criteria(track, body):
    result = rubric.assess(track, body=body)
    assert len(result.reps) == 1
    return {k: c.passed for k, c in result.reps[0].criteria.items()}


def test_a_deep_upright_squat_passes_everything(body):
    assert all(criteria(make_track(body), body).values())


def test_each_fault_fails_only_its_own_criterion(body):
    good = make_track(body)
    cases = {
        "depth": make_track(body, peak_knee_flexion=85.0),
        "heels": F.heel_lift(good, rise=0.25),
        "ascent": F.hips_rise_first(good, extra_lean=10.0),
        "descent": F.hitch(good, back_up=0.08),
        "trunk": make_track(body, trunk_peak_preference=65.0, ankle_knee_ratio=0.18),
    }
    for name, track in cases.items():
        passed = criteria(track, body)
        assert not passed[name], name
        others = {k: v for k, v in passed.items() if k != name and not (name == "depth" and k == "trunk")}
        assert all(others.values()), (name, passed)


def test_one_bad_rep_in_three_is_not_a_deficit(body):
    good = make_track(body)
    bad = F.heel_lift(good, rise=0.25)
    three = good.map_points(lambda v: v)
    three = replace(three, points={k: np.concatenate([good[k], bad[k], good[k]]) for k in good.points})
    result = rubric.assess(three, body=body)
    assert len(result.reps) == 3
    assert result.pass_share["heels"] == pytest.approx(2 / 3)
    assert not result.deficits["heels"]
    two_bad = replace(three, points={k: np.concatenate([bad[k], bad[k], good[k]]) for k in good.points})
    assert rubric.assess(two_bad, body=body).deficits["heels"]


def test_score_is_the_mean_pass_share(body):
    result = rubric.assess(make_track(body, peak_knee_flexion=85.0), body=body)
    assert result.score == pytest.approx(100 * np.mean(list(result.pass_share.values())))
    assert result.deficit_total == sum(result.deficits.values())


def test_balance_needs_less_trunk_lean_with_more_shin_lean(body):
    needs = [rubric.balanced_trunk_minus_shin(body, -90.0, shin, 20.0, 150.0, 0.0) for shin in (25, 30, 35, 40)]
    assert np.all(np.diff(needs) < 0)


def test_long_thighs_get_a_larger_trunk_allowance(population):
    bodies = population.bodies("female")[:600]
    ratio = np.array([C.model_proportions(b)["thigh"] / C.model_proportions(b)["trunk"] for b in bodies])
    long_thigh, short_thigh = bodies[int(np.argmax(ratio))], bodies[int(np.argmin(ratio))]
    features = {"thigh_at_bottom": -90.0, "shoulder_flexion_at_bottom": ARMS_CROSSED.upper_arm_angle,
                "elbow_flexion_at_bottom": ARMS_CROSSED.elbow_flexion, "neck_at_bottom": 0.0,
                "trunk_at_bottom": 0.0}
    need_long = rubric.balanced_trunk_minus_shin(long_thigh, -90.0, rubric.CUED_SHIN_LEAN, 20.0, 150.0, 0.0)
    need_short = rubric.balanced_trunk_minus_shin(short_thigh, -90.0, rubric.CUED_SHIN_LEAN, 20.0, 150.0, 0.0)
    assert need_long > need_short + 15.0
    assert rubric.trunk_allowance(long_thigh, features) >= rubric.trunk_allowance(short_thigh, features) >= 0.0


def test_stored_tolerances_match_the_noise_model():
    fresh = rubric.measurement_tolerances(n=150, seed=7)
    for key, stored in rubric.TOLERANCE.items():
        assert 0.6 * stored < fresh[key] < 1.6 * stored, (key, stored, fresh[key])


def test_real_clips():
    side_on = rubric.assess(mpp.observed_track("squat_1"), sex="female")
    assert side_on.reliable and len(side_on.reps) == 6
    assert side_on.deficits["depth"] and side_on.deficits["trunk"]
    assert not side_on.deficits["heels"] and not side_on.deficits["ascent"]
    # her lean is about what balance needs at her own shin angle (reps 1-4)
    for rep in side_on.reps[:4]:
        assert abs(rep.criteria["trunk"].value - rep.trunk_needed_at_own_shin) < 5.0
    oblique = rubric.assess(mpp.observed_track("squat_2"), sex="male")
    assert not oblique.reliable and len(oblique.warnings) == 2
