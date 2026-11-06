"""Run the squat form rubric on the real clips and test it on synthetic reps.

Usage:
    PYTHONPATH=src python scripts/rubric_report.py [N]

1. Real clips: every stored MediaPipe track, graded.
2. Synthetic cohorts (Ishida parallel, Zawadka maximal depth): 3D reps
   with sampled stances, filmed by phones placed as people place them
   (video/camera.py), with tracker jitter and drift: how often each
   criterion passes, and how often the grade from the filmed view agrees
   with the grade from a perfect side view (the label noise a model
   trained on these grades would face).
3. Known faults: edited into the filmed track before tracker noise; how
   often each is caught.
4. Build allowance: how the trunk allowance varies with proportions across
   ANSUR II at parallel depth.

docs/rubric.md explains the results.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial

import numpy as np

from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.model3d import sample_body3d
from exercise_quality.core.observed import from_model_landmarks
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat import rubric
from exercise_quality.exercises.squat import track_faults as F
from exercise_quality.exercises.squat.balanced import ARMS_CROSSED, generate_feasible_squat
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH, sample_stance, sample_style
from exercise_quality.video import camera as cam
from exercise_quality.video import compare as C
from exercise_quality.video import mediapipe_pose as mpp

CRITERIA = rubric.BSA_CRITERIA + ("balance", "lockout")      # consistency needs several reps
FAULTS = {
    "shallow (knee 85 deg)": ("depth", None),
    "heels lift 25% of foot": ("heels", partial(F.heel_lift, rise=0.25)),
    "heels lift 10% of foot": ("heels", partial(F.heel_lift, rise=0.10)),
    "hips rise first (+10 deg)": ("ascent", partial(F.hips_rise_first, extra_lean=10.0)),
    "hips rise first (+5 deg)": ("ascent", partial(F.hips_rise_first, extra_lean=5.0)),
    "hitch on the way down (8%)": ("descent", partial(F.hitch, back_up=0.08)),
    "hitch on the way down (4%)": ("descent", partial(F.hitch, back_up=0.04)),
    "forward lean (trunk pref 65, low ankle)": ("trunk", None),
    "knees 15 deg short of straight at the top": ("lockout", partial(F.soft_lockout, knee=15.0)),
    "knees 8 deg short of straight at the top": ("lockout", partial(F.soft_lockout, knee=8.0)),
    "sit back, trunk 35 deg further back": ("balance", partial(F.sit_back, degrees=35.0)),
    "sit back, trunk 20 deg further back": ("balance", partial(F.sit_back, degrees=20.0)),
}


def _rep_track(seed, protocol, fault=None):
    """(side-view body, perfect side view, filmed track, rng) of one 3D rep."""
    pop = rubric._population()
    rng = np.random.default_rng(seed)
    body3d = sample_body3d(pop, rng)
    body = body3d.sagittal
    style = sample_style(rng, body, protocol)
    style = replace(style, timing=RepTiming(stand_before=0.6, descent=rng.uniform(1.0, 2.5),
                                            ascent=rng.uniform(0.8, 2.5), stand_after=0.6),
                    stance=sample_stance(rng))
    if fault == "shallow (knee 85 deg)":
        style = replace(style, peak_knee_flexion=85.0)
    if fault == "forward lean (trunk pref 65, low ankle)":
        style = replace(style, trunk_peak_preference=65.0, ankle_knee_ratio=0.18)
    seq = generate_feasible_squat(body3d, style, fps=30.0)
    if not seq.is_feasible:
        return None
    filmed = cam.filmed_track(seq, cam.sample_side_camera(rng))
    return body, from_model_landmarks(seq.landmarks, 30.0), filmed, rng


def _grade(track, body):
    result = rubric.assess(track, body=body)
    if len(result.reps) != 1:
        return None
    rep = result.reps[0]
    return {k: c.passed for k, c in rep.criteria.items()}, result.score, rep.features["turnaround_acceleration_g"]


def cohort_job(args):
    seed, protocol = args
    made = _rep_track(seed, protocol)
    if made is None:
        return None
    body, exact, filmed, rng = made
    noisy = TN.apply(filmed, TN.TrackerNoise.sample(rng), rng)
    a, b = _grade(exact, body), _grade(noisy, body)
    if a is None or b is None:
        return None
    return a, b


def fault_job(args):
    seed, fault = args
    made = _rep_track(seed, ISHIDA_PARALLEL, fault)
    if made is None:
        return None
    body, _, filmed, rng = made
    criterion, edit = FAULTS[fault]
    try:
        faulty = edit(filmed) if edit else filmed
    except ValueError:
        return None
    noisy = TN.apply(faulty, TN.TrackerNoise.sample(rng), rng)
    graded = _grade(noisy, body)
    if graded is None:
        return None
    return not graded[0][criterion]


def real_clips():
    print("## Real clips")
    manifest = mpp.manifest()
    for name in mpp.available():
        info = manifest.get(name, {})
        if info.get("view", "").startswith("front"):
            continue        # the side-view rubric; front views have their own measures
        result = rubric.assess(mpp.observed_track(name), sex=info.get("apparent_sex"))
        print(f"\n### {name} ({info.get('view', '')})")
        print(f"reliable: {result.reliable}; score {result.score:.0f} (BSA only {result.bsa_score:.0f}); "
              f"deficits {result.deficit_total} of {len(result.deficits)} (BSA {result.bsa_deficit_total} of 5); "
              f"matched ANSUR {result.body_id}")
        for w in result.warnings:
            print(f"  warning: {w}")
        print("criterion  " + " ".join(f"rep{r.index + 1:<3d}" for r in result.reps) + "  pass share  deficit")
        for c in [c for c in rubric.BSA_CRITERIA + rubric.EXTRA_CRITERIA if c in result.deficits]:
            cells = " ".join(f"{r.criteria[c].value:6.2f}{'' if r.criteria[c].passed else '*'}".ljust(7)
                             for r in result.reps)
            print(f"{c:10s} {cells}  {result.pass_share[c]:9.2f}  {result.deficits[c]}")
        print("thresholds: " + ", ".join(f"{c} {crit.threshold:.2f}" for c, crit in result.reps[0].criteria.items()))
        print("centre of mass at the bottom: " + " ".join(f"{r.features['com_at_bottom']:5.2f}" for r in result.reps))
        print("trunk needed at own shin: " + " ".join(f"{r.trunk_needed_at_own_shin:5.1f}" for r in result.reps))
        print("reported: " + ", ".join(f"{k} {v:.2f}" for k, v in result.reported.items()))


def synthetic_cohorts(pool, n):
    print("\n## Synthetic cohorts: 3D reps filmed by sampled phones, with tracker noise")
    for protocol in (ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH):
        out = [o for o in pool.map(cohort_job, [(s, protocol) for s in range(n)]) if o]
        clean = np.array([[o[0][0][c] for c in CRITERIA] for o in out])
        noisy = np.array([[o[1][0][c] for c in CRITERIA] for o in out])
        scores = np.array([o[1][1] for o in out])
        print(f"\n{protocol.name}: {len(out)} reps")
        print(f"{'criterion':10s} {'pass (exact)':>12s} {'pass (tracked)':>14s} {'agreement':>10s}")
        for i, c in enumerate(CRITERIA):
            print(f"{c:10s} {clean[:, i].mean():12.2f} {noisy[:, i].mean():14.2f} "
                  f"{np.mean(clean[:, i] == noisy[:, i]):10.2f}")
        print(f"tracked score: median {np.median(scores):.0f}, 10th-90th percentile "
              f"{np.percentile(scores, 10):.0f}-{np.percentile(scores, 90):.0f}")
        exact_g = np.array([o[0][2] for o in out])
        seen_g = np.array([o[1][2] for o in out])
        print(f"turnaround acceleration (g), exact vs tracked: median {np.median(exact_g):.2f} vs "
              f"{np.median(seen_g):.2f}; 5-95% {np.percentile(exact_g, 5):.2f}-{np.percentile(exact_g, 95):.2f} vs "
              f"{np.percentile(seen_g, 5):.2f}-{np.percentile(seen_g, 95):.2f}; "
              f"correlation {np.corrcoef(exact_g, seen_g)[0, 1]:.2f}")


def fault_detection(pool, n):
    print("\n## Injected faults (filmed and tracked, Ishida-style reps)")
    print(f"{'fault':42s} {'criterion':10s} {'caught':>7s} {'n':>5s}")
    for fault, (criterion, _) in FAULTS.items():
        out = [o for o in pool.map(fault_job, [(s, fault) for s in range(n)]) if o is not None]
        print(f"{fault:42s} {criterion:10s} {np.mean(out):7.2f} {len(out):5d}")


def build_allowance(n=400):
    print("\n## Build allowance at parallel depth, arms crossed (ANSUR II)")
    pop = rubric._population()
    rng = np.random.default_rng(0)
    for sex in ("female", "male"):
        bodies = [pop.body(sex, int(i)) for i in rng.choice(pop.count(sex), n, replace=False)]
        ratio = np.array([C.model_proportions(b)["thigh"] / C.model_proportions(b)["trunk"] for b in bodies])
        allow, needed = [], []
        for b in bodies:
            need = rubric.balanced_trunk_minus_shin(b, -90.0, rubric.CUED_SHIN_LEAN, ARMS_CROSSED.upper_arm_angle,
                                                    ARMS_CROSSED.elbow_flexion, 0.0)
            needed.append(need)
            allow.append(max(0.0, need))
        needed, allow = np.array(needed), np.array(allow)
        print(f"{sex}: trunk-minus-shin needed at {rubric.CUED_SHIN_LEAN} deg shin: median {np.median(needed):.1f}, "
              f"5-95% {np.percentile(needed, 5):.1f} to {np.percentile(needed, 95):.1f}; "
              f"share with an allowance > 0: {np.mean(allow > 0):.2f}; "
              f"correlation with thigh/trunk ratio {np.corrcoef(ratio, needed)[0, 1]:.2f}")
        hi = np.argsort(ratio)[-n // 20:]
        lo = np.argsort(ratio)[:n // 20]
        print(f"   longest-thigh 5%: needed {np.median(needed[hi]):.1f}; shortest-thigh 5%: {np.median(needed[lo]):.1f}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    n = int(argv[0]) if argv else 200
    real_clips()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        synthetic_cohorts(pool, n)
        fault_detection(pool, n // 2)
    build_allowance()


if __name__ == "__main__":
    sys.exit(main())
