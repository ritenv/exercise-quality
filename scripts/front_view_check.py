"""Real front-view squats against matched 3D synthetic reps seen by the virtual camera.

    PYTHONPATH=src python scripts/front_view_check.py [bodies_per_clip]

For each stored clip with a front view (video/front_view.py), synthetic
people of the same apparent sex get the clip's stance width (ankle
separation over leg height), mean toe-out of the two feet, depth (hip
drop), tempo and arm position, and are filmed by the virtual camera at
the clip's measured view angle with tracker noise. The same front-view
measures are taken from both.

Then, per synthetic person, the knee tracking angle (model3d.Stance
knee_track, both legs) that reproduces the clip's median knee-to-ankle
separation ratio at the bottom is found: 0 = knees exactly over the toes
as the model assumes, positive = knees inside that line (valgus),
negative = knees pushed further out than the toes point.

docs/front_view_check.md explains the results.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from exercise_quality.core import model3d as M
from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.timing import RepTiming
from exercise_quality.exercises.squat.balanced import ARMS_CROSSED, SquatStyle, generate_squat
from exercise_quality.video import camera as cam
from exercise_quality.video import front_view as F
from exercise_quality.video import mediapipe_pose as mpp

KEYS = ("depth", "kasr_bottom", "kasr_half", "fppa_right_half", "fppa_left_half",
        "hip_shift", "pelvic_tilt", "depth_asymmetry")


def _median_rep(measures):
    return {k: float(np.median([r[k] for r in measures["reps"]])) for k in measures["reps"][0]}


def _film(body, stance, knee, timing, camera, noise=None, rng=None):
    style = SquatStyle(stance=stance, peak_knee_flexion=knee, timing=timing, arms=ARMS_CROSSED)
    seq = generate_squat(body, style, fps=30.0)
    if not seq.is_feasible:
        return None
    raw = cam.synthetic_raw(seq, camera, noise=noise, rng=rng)
    m = F.measure(raw, segment=(0, len(seq.times)))
    if m is None or len(m["reps"]) != 1:
        return None
    return m


def _solve(f, target, lo, hi, steps=8):
    """x in [lo, hi] with f(x) = target for a monotone f (by bisection on the sign)."""
    flo, fhi = f(lo), f(hi)
    for _ in range(6):          # an infeasible end (e.g. feet too wide to reach): shrink towards the other
        if flo is not None and fhi is not None:
            break
        if fhi is None:
            hi = lo + 0.75 * (hi - lo)
            fhi = f(hi)
        else:
            lo = hi - 0.75 * (hi - lo)
            flo = f(lo)
    if flo is None or fhi is None:
        return None
    increasing = fhi > flo
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm is None:
            return None
        if (fm < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def job(args):
    clip, seed = args
    real = clip["measures"]
    target = clip["target"]
    pop = BodyPopulation.load()
    rng = np.random.default_rng(seed)
    body = M.sample_body3d(pop, rng, sex=clip["sex"])
    timing = RepTiming(stand_before=0.8, descent=max(target["descent_s"] / 0.9, 0.4),
                       ascent=max(target["ascent_s"] / 0.9, 0.4), stand_after=0.8)
    # MediaPipe's 3D foot directions are lopsided from the front (right foot
    # reads 25-35 deg more turned out than the left in every clip), so the
    # mean of the two is used; a clip can override it (feet out of frame)
    toe = clip.get("toe_out", float(np.mean(list(real["toe_out"].values()))))
    asym = target["fppa_right_half"] - target["fppa_left_half"]

    def camera(azimuth):
        return cam.Camera(azimuth=azimuth, distance=clip.get("distance", 3.5),
                          lens_height=clip.get("lens_height", 1.0), aim_height=clip.get("aim_height", 0.9))

    def stance(width, track=0.0):
        return M.Stance(width_ratio=width, toe_out=toe, knee_track_right=track, knee_track_left=track)

    def measure(width, knee, track=0.0, azimuth=0.0):
        return _film(body, stance(width, track), knee, timing, camera(azimuth))

    def rep(m):
        return _median_rep(m) if m else None

    # The view angle: MediaPipe's 3D hip axis reads front views 13-15 deg off
    # when the image looks square (docs/front_view_check.md), so it is solved
    # instead from the left-right difference in the knee's frontal angle,
    # which grows by about 3 deg per degree of camera angle. Two rounds,
    # since stance, depth and knee tracking also change that difference.
    azimuth, width, knee, track = real["view_yaw"], 1.3, 120.0, 0.0
    for _ in range(2):
        width = _solve(lambda w: (m := measure(w, knee, track, azimuth)) and m["stance_over_leg"],
                       real["stance_over_leg"], 0.8, 2.4, steps=7)
        if width is None:
            return None
        knee = _solve(lambda k: (r := rep(measure(width, k, track, azimuth))) and r["depth"],
                      target["depth"], 80.0, 155.0, steps=7)
        if knee is None:
            return None
        track = _solve(lambda d: (r := rep(measure(width, knee, d, azimuth))) and r["kasr_bottom"],
                       target["kasr_bottom"], -40.0, 40.0, steps=8)
        if track is None:
            return None
        azimuth = _solve(lambda a: (r := rep(measure(width, knee, track, a)))
                         and r["fppa_right_half"] - r["fppa_left_half"], asym, -35.0, 35.0, steps=8)
        if azimuth is None:
            return None
    noise = TN.TrackerNoise.sample(rng)
    over_toes = _film(body, stance(width), knee, timing, camera(azimuth), noise, rng)
    fitted = _film(body, stance(width, track), knee, timing, camera(azimuth), noise, rng)
    return {
        "width_ratio": width, "knee": knee, "knee_track": track, "azimuth": azimuth,
        "over_toes": _median_rep(over_toes) if over_toes else None,
        "fitted": _median_rep(fitted) if fitted else None,
    }


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    n = int(argv[0]) if argv else 16
    info = mpp.manifest()
    clips = []
    for name in mpp.available():
        meta = info.get(name, {})
        if "front" not in meta.get("view", ""):
            continue
        measures = F.measure(mpp.load_raw(name))
        if not measures or not measures["reps"]:
            print(f"{name}: no front-view reps")
            continue
        clips.append({"name": name, "sex": meta.get("apparent_sex", "female"), "measures": measures,
                      "target": _median_rep(measures), **meta.get("camera", {})})
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for clip in clips:
            out = [o for o in pool.map(job, [(clip, s) for s in range(n)]) if o]
            m, t = clip["measures"], clip["target"]
            print(f"\n## {clip['name']} ({clip['sex']}, {len(m['reps'])} reps, view {m['view_yaw']:+.0f} deg, "
                  f"frames {m['segment'][0]}-{m['segment'][1]})")
            print(f"stance over leg {m['stance_over_leg']:.2f}, over MediaPipe hips {m['stance_over_hips']:.2f}; "
                  f"shoulders over hips {m['shoulders_over_hips']:.2f}; toe-out from world landmarks "
                  f"R {m['toe_out']['right']:.0f} L {m['toe_out']['left']:.0f} deg; used "
                  f"{clip.get('toe_out', np.mean(list(m['toe_out'].values()))):.0f}"
                  f"{' (set in the manifest)' if 'toe_out' in clip else ' (mean)'}")
            if not out:
                print("no matched synthetic reps")
                continue
            widths = np.array([o["width_ratio"] for o in out])
            knees = np.array([o["knee"] for o in out])
            az = np.array([o["azimuth"] for o in out])
            print(f"matched: {len(out)} people; stance width {np.median(widths):.2f} x hip breadth; "
                  f"peak knee {np.median(knees):.0f} deg; camera angle from image symmetry "
                  f"{np.median(az):+.0f} deg (5-95% {np.percentile(az, 5):+.0f} to {np.percentile(az, 95):+.0f}), "
                  f"MediaPipe 3D hips said {m['view_yaw']:+.0f}")
            print(f"{'measure':16s} {'real':>8s} {'knees over toes (5-50-95%)':>30s} {'fitted tracking (median)':>25s}")
            for k in KEYS:
                ot = np.array([o["over_toes"][k] for o in out if o["over_toes"]])
                ft = np.array([o["fitted"][k] for o in out if o["fitted"]])
                spread = (f"{np.percentile(ot, 5):7.2f} {np.median(ot):7.2f} {np.percentile(ot, 95):7.2f}"
                          if ot.size else "")
                print(f"{k:16s} {t[k]:8.2f} {spread:>30s} {np.median(ft) if ft.size else np.nan:25.2f}")
            tracks = np.array([o["knee_track"] for o in out if o["knee_track"] is not None])
            if tracks.size:
                print(f"knee tracking that matches the real knee separation: median {np.median(tracks):+.1f} deg "
                      f"(5-95% {np.percentile(tracks, 5):+.1f} to {np.percentile(tracks, 95):+.1f}); "
                      f"positive = knees inside the toe line")


if __name__ == "__main__":
    sys.exit(main())
