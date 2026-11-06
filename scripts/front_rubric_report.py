"""Run the front-view squat rubric on the real clips and test it on synthetic reps.

    PYTHONPATH=src python scripts/front_rubric_report.py [N]

1. Real clips: every stored front-view MediaPipe track, graded.
2. Synthetic cohorts: 3D reps with sampled stances, filmed from the front by
   phones placed as people place them (front_rubric.sample_front_camera),
   with tracker noise: how often each criterion passes, graded from the
   true 3D values and from the filmed view, and how often the two agree.
3. Known faults: knees caving in (model3d.Stance knee_track, both legs and
   one leg) and one hip dropping (an image edit, front_rubric.drop_hip):
   how often each is caught.

docs/front_rubric.md explains the results.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from exercise_quality.core import model3d as M
from exercise_quality.exercises.squat import front_rubric as FR
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, ZAWADKA_MAX_DEPTH, sample_stance
from exercise_quality.video import front_view as F
from exercise_quality.video import mediapipe_pose as mpp

FAULTS = {
    "knees cave in 15 deg (both)": ("knees_over_feet", {"knee_track_right": 15.0, "knee_track_left": 15.0}),
    "knees cave in 25 deg (both)": ("knees_over_feet", {"knee_track_right": 25.0, "knee_track_left": 25.0}),
    "knees cave in 35 deg (both)": ("knees_over_feet", {"knee_track_right": 35.0, "knee_track_left": 35.0}),
    "right knee caves in 25 deg": ("knees_over_feet", {"knee_track_right": 25.0}),
    "right knee caves in 35 deg": ("knees_over_feet", {"knee_track_right": 35.0}),
    "right hip drops 6 deg": ("hips_level", 6.0),
    "right hip drops 10 deg": ("hips_level", 10.0),
}
PROTOCOLS = {"ishida": ISHIDA_PARALLEL, "zawadka": ZAWADKA_MAX_DEPTH}


def _grade(seq, film):
    result = FR.assess(film, body=seq.body, segment=(0, len(seq.times)))
    if result is None or len(result.reps) != 1:
        return None
    return result.reps[0]


def cohort_job(args):
    seed, protocol = args
    rng = np.random.default_rng(seed)
    made = FR.synthetic_front_rep(rng, protocol=PROTOCOLS[protocol])
    if made is None:
        return None
    seq, film = made
    rep = _grade(seq, film)
    if rep is None:
        return None
    found = F.find_reps(F._smooth_image(film, slice(None)), float(film["fps"]))[0]
    truth = FR.true_values(seq, found.start, found.end)
    exact = {c: truth[c] <= FR.TOLERANCE[c] for c in FR.CRITERIA}
    seen = {c: rep.criteria[c].passed for c in FR.CRITERIA}
    return exact, seen, truth["knees_over_feet"], rep.features["kasr_bottom"]


def fault_job(args):
    seed, fault = args
    rng = np.random.default_rng(seed)
    criterion, spec = FAULTS[fault]
    stance = sample_stance(rng)
    if isinstance(spec, dict):
        stance = replace(stance, **spec)
    made = FR.synthetic_front_rep(rng, stance=stance)
    if made is None:
        return None
    seq, film = made
    if not isinstance(spec, dict):
        film = FR.drop_hip(film, spec)
    rep = _grade(seq, film)
    if rep is None:
        return None
    return not rep.criteria[criterion].passed


def real_clips():
    print("## Real clips")
    manifest = mpp.manifest()
    pop = FR._population()
    for name in mpp.available():
        info = manifest.get(name, {})
        if not info.get("view", "").startswith("front"):
            continue
        sex = info.get("apparent_sex", "female")
        result = FR.assess(mpp.load_raw(name), body=M.median_body3d(pop, sex))
        print(f"\n### {name} ({info.get('view', '')}; median {sex} body)")
        if result is None:
            print("no front view")
            continue
        print(f"reliable: {result.reliable}; camera {result.camera_angle:+.0f} deg from square; "
              f"score {result.score:.0f}; deficits {result.deficit_total} of {len(result.deficits)}")
        for w in result.warnings:
            print(f"  warning: {w}")
        print("criterion        " + " ".join(f"rep{r.index + 1:<4d}" for r in result.reps) + "  deficit")
        for c in FR.CRITERIA:
            cells = " ".join(f"{r.criteria[c].value:6.2f}{'' if r.criteria[c].passed else '*'}".ljust(8)
                             for r in result.reps)
            print(f"{c:16s} {cells}  {result.deficits[c]}")
        right = [r.features["knee_crossing_right_cm"] for r in result.reps]
        left = [r.features["knee_crossing_left_cm"] for r in result.reps]
        print(f"knee crossing right / left, median (cm): {np.median(right):.1f} / {np.median(left):.1f}")
        print("reported: " + ", ".join(f"{k} {v:.2f}" for k, v in result.reported.items()))


def synthetic_cohorts(pool, n):
    print("\n## Synthetic cohorts filmed from the front")
    for protocol in PROTOCOLS:
        out = [o for o in pool.map(cohort_job, [(s, protocol) for s in range(n)]) if o]
        print(f"\n{protocol}: {len(out)} reps")
        print(f"{'criterion':16s} {'pass (true)':>11s} {'pass (filmed)':>13s} {'agreement':>10s}")
        for c in FR.CRITERIA:
            a = np.array([o[0][c] for o in out])
            b = np.array([o[1][c] for o in out])
            print(f"{c:16s} {a.mean():11.2f} {b.mean():13.2f} {np.mean(a == b):10.2f}")
        cross = np.array([o[2] for o in out])
        kasr = np.array([o[3] for o in out])
        print(f"true knee crossing (cm): median {np.median(cross):.1f}, 95% {np.percentile(cross, 95):.1f}; "
              f"KASR at the bottom median {np.median(kasr):.2f}")


def fault_detection(pool, n):
    print("\n## Injected faults (filmed from the front, Ishida-style reps)")
    print(f"{'fault':32s} {'criterion':16s} {'caught':>7s} {'n':>5s}")
    for fault, (criterion, _) in FAULTS.items():
        out = [o for o in pool.map(fault_job, [(10_000 + s, fault) for s in range(n)]) if o is not None]
        print(f"{fault:32s} {criterion:16s} {np.mean(out):7.2f} {len(out):5d}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    n = int(argv[0]) if argv else 200
    real_clips()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        synthetic_cohorts(pool, n)
        fault_detection(pool, n // 2)


if __name__ == "__main__":
    sys.exit(main())
