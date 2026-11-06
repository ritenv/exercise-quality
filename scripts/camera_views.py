"""How camera placement changes what the side-view pipeline measures.

    PYTHONPATH=src python scripts/camera_views.py [N]

N 3D reps (Ishida parallel protocol, sampled stances) are filmed by
virtual phone cameras at several angles around the person and distances,
with tracker noise, and run through the same MediaPipe adapter and rubric
as real videos. For each placement: the error in the side-view angles
against the true joint angles (within each leg's plane) at the deepest
knee bend (knee and thigh: within the leg's plane; trunk: the
hip-to-shoulder line the tracker measures), how often the rubric's depth
and trunk grades agree with the
grades from a perfect side view, and how often the result is marked
reliable. docs/model3d.md explains the results.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.model3d import sample_body3d
from exercise_quality.core.observed import from_model_landmarks, observed_angles
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.exercises.squat import reps as R
from exercise_quality.exercises.squat import rubric
from exercise_quality.exercises.squat.balanced import generate_feasible_squat
from exercise_quality.exercises.squat.protocols import ISHIDA_PARALLEL, sample_stance, sample_style
from exercise_quality.video import camera as cam
from exercise_quality.video import mediapipe_pose as mpp

AZIMUTHS = (90.0, 75.0, 60.0, 45.0)
DISTANCES = (2.0, 3.5, 6.0)
GRADED = ("depth", "trunk")


def _bottom_angles(track):
    smooth = R.smoothed(track)
    a = observed_angles(smooth)
    found = R.find_reps(smooth, a)
    if len(found) != 1:
        return None
    b = rubric._bottom(a, found[0])
    return {k: float(a[k][b]) for k in ("knee_flexion", "trunk_inclination", "thigh_inclination")}


def job(seed):
    pop = BodyPopulation.load()
    rng = np.random.default_rng(seed)
    body = sample_body3d(pop, rng)
    style = replace(sample_style(rng, body, ISHIDA_PARALLEL), stance=sample_stance(rng))
    seq = generate_feasible_squat(body, style, fps=30.0)
    if not seq.is_feasible:
        return None
    truth = seq.measured()
    b = int(np.argmax(seq.angles.knee_flexion))
    # the tracker's trunk is the hip-to-shoulder line (core/observed.py); the
    # trunk stays in the body's mid-plane, so the side projection is its true lean
    v = seq.landmarks["right_shoulder"][b] - seq.landmarks["right_hip"][b]
    true = {"knee_flexion": truth["knee_flexion"][b], "trunk_inclination": np.degrees(np.arctan2(v[0], v[1])),
            "thigh_inclination": truth["thigh_inclination"][b]}
    exact = rubric.assess(from_model_landmarks(seq.landmarks, seq.fps), body=body.sagittal)
    if len(exact.reps) != 1:
        return None
    exact_grades = {g: exact.reps[0].criteria[g].passed for g in GRADED}
    out = {}
    noise = replace(TN.TrackerNoise.sample(rng), knee_out_deg=0.0)
    for az in AZIMUTHS:
        for d in DISTANCES:
            raw = cam.synthetic_raw(seq, cam.Camera(azimuth=az, distance=d), noise=noise, rng=rng)
            track = mpp.observed_track(raw)
            seen = _bottom_angles(track)
            result = rubric.assess(track, body=body.sagittal)
            if seen is None or len(result.reps) != 1:
                out[(az, d)] = None
                continue
            out[(az, d)] = {
                "error": {k: seen[k] - true[k] for k in true},
                "agree": {g: result.reps[0].criteria[g].passed == exact_grades[g] for g in GRADED},
                "reliable": result.reliable,
            }
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    n = int(argv[0]) if argv else 60
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = [r for r in pool.map(job, range(n)) if r]
    print(f"{len(results)} reps; error = side-view angle minus true joint angle at the deepest knee bend")
    print("(median, 5-95%); agreement = rubric grade vs a perfect orthographic side view\n")
    print(f"{'azimuth':>7s} {'dist':>5s} {'found':>5s} {'knee error':>20s} {'trunk error':>20s} "
          f"{'thigh error':>20s} {'depth agree':>11s} {'trunk agree':>11s} {'reliable':>8s}")
    for az in AZIMUTHS:
        for d in DISTANCES:
            cells = [r[(az, d)] for r in results if r[(az, d)] is not None]
            if not cells:
                continue

            def err(k):
                e = np.array([c["error"][k] for c in cells])
                return f"{np.median(e):+5.1f} ({np.percentile(e, 5):+5.1f},{np.percentile(e, 95):+5.1f})"

            agree = {g: np.mean([c["agree"][g] for c in cells]) for g in GRADED}
            reliable = np.mean([c["reliable"] for c in cells])
            print(f"{az:7.0f} {d:5.1f} {len(cells) / len(results):5.2f} {err('knee_flexion'):>20s} "
                  f"{err('trunk_inclination'):>20s} {err('thigh_inclination'):>20s} "
                  f"{agree['depth']:11.2f} {agree['trunk']:11.2f} {reliable:8.2f}")


if __name__ == "__main__":
    sys.exit(main())
