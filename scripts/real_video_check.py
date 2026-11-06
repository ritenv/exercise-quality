"""Compare tracked real squats with matched synthetic squats.

Usage:
    PYTHONPATH=src python scripts/real_video_check.py [N_PER_REP] [--figures DIR]

For every stored side-view MediaPipe track (src/exercise_quality/data/real_video):
  1. find the reps and measure them (exercises/squat/reps.py);
  2. pick the 40 ANSUR II people whose proportions best match the person;
  3. for each real rep, generate N_PER_REP 3D synthetic reps with the same
     depth, tempo and arm position (stance sampled), film them with a
     virtual phone at the clip's angle from side-on, add tracker jitter
     and drift and measure them the same way;
  4. report where each real rep falls in its synthetic distribution;
  5. measure the tracker's own noise and compare with the noise model.

docs/real_video_check.md explains the results.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from exercise_quality.core import tracker_noise as TN
from exercise_quality.core.observed import observed_angles, segment_lengths, POINTS
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.signal import lowpass
from exercise_quality.exercises.squat import reps as R
from exercise_quality.video import compare as C
from exercise_quality.video import mediapipe_pose as mpp

MATCH_WEIGHTS = {"thigh": 1.0, "trunk": 1.0, "foot": 0.0, "neck": 0.0}
COMPARED = [
    ("thigh_at_bottom", "thigh angle at the bottom (deg, -90 = parallel)"),
    ("hip_drop", "hip drop, fraction of standing hip height"),
    ("shank_at_bottom", "shin lean at the bottom (deg)"),
    ("trunk_at_bottom", "trunk lean at the bottom (deg)"),
    ("trunk_minus_shank_at_bottom", "trunk lean minus shin lean (deg)"),
    ("trunk_at_knee60", "trunk lean at 60 deg knee (deg)"),
    ("trunk_at_knee90", "trunk lean at 90 deg knee (deg)"),
    ("hip_back_at_bottom", "hip travel back, fraction of leg length"),
    ("knee_past_toe_at_bottom", "knee ahead of toe tip, fraction of foot"),
    ("com_at_bottom", "COM over the foot at the bottom (0 heel, 1 toe)"),
    ("trunk_extra_on_ascent", "trunk lean up minus down, same hip height (deg)"),
    ("heel_rise_max", "heel rise, fraction of foot length"),
    ("trunk_peak_ms", "trunk peak vs lowest hip (ms)"),
    ("descent_s", "descent (s)"),
    ("ascent_s", "ascent (s)"),
]


def _job(args):
    body, knee, timing, arms, seed, fps, offset = args
    return C.synthetic_rep(body, knee, timing, arms, seed, fps=fps, camera_offset=offset)


def calibrated_targets(body, rep_feats, fps, offset, seeds=range(6)):
    """Timing and knee target whose *measured* synthetic values match the real rep."""
    knee = rep_feats["knee_peak"]
    target = np.array([rep_feats["descent_s"], rep_feats["bottom_s"], rep_feats["ascent_s"]])
    timing = C.matched_timing(*target)
    arms = C.arm_posture(rep_feats["shoulder_flexion_at_bottom"], rep_feats["elbow_flexion_at_bottom"])
    for _ in range(2):
        runs = [C.synthetic_rep(body, knee, timing, arms, s, fps=fps, camera_offset=offset) for s in seeds]
        runs = [r for r in runs if r]
        if not runs:
            break
        got = np.array([[r["descent_s"], r["bottom_s"], r["ascent_s"]] for r in runs]).mean(0)
        wanted = C.measured_phases(timing) * target / np.maximum(got, 1e-3)
        timing = C.matched_timing(*wanted)
        knee += rep_feats["knee_peak"] - np.mean([r["knee_peak"] for r in runs])
    return knee, timing, arms


def tracker_noise_table(track):
    smooth = track.map_points(lambda v: lowpass(v, track.fps))
    lengths = segment_lengths(smooth)
    leg = float(np.median(lengths["thigh"] + lengths["shank"]))
    jitter = {p: float(np.mean(np.std((track[p] - smooth[p]) / leg, axis=0))) for p in POINTS}
    raw, filt = observed_angles(track), observed_angles(smooth)
    angle_jitter = {k: float(np.std(raw[k] - filt[k]))
                    for k in ("knee_flexion", "thigh_inclination", "shank_inclination", "trunk_inclination")}
    knee = filt["knee_flexion"]
    upright = knee < 10
    trend = {}
    for seg in ("thigh", "shank", "trunk"):
        y = lengths[seg] / np.median(lengths[seg][upright] if upright.sum() > 5 else lengths[seg])
        c = np.polyfit(knee / 100, y, 2)
        trend[seg] = float(np.polyval(c, 1.0))
    return jitter, angle_jitter, trend


def descent_curves(track, rep, angles):
    sl = slice(rep.start, rep.lowest + 1)
    return angles["knee_flexion"][sl], angles["trunk_inclination"][sl], angles["shank_inclination"][sl]


def figure(name, real_curves, synth_curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.arange(20, 131, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, idx, label in ((axes[0], 1, "trunk lean (deg)"), (axes[1], 2, "shin lean (deg)")):
        stack = []
        for k, *vals in synth_curves:
            v = vals[idx - 1]
            order = np.argsort(k)
            y = np.interp(grid, k[order], v[order], left=np.nan, right=np.nan)
            stack.append(y)
        stack = np.array(stack)
        with np.errstate(all="ignore"):
            lo, mid, hi = np.nanpercentile(stack, [5, 50, 95], axis=0)
        ok = np.sum(np.isfinite(stack), axis=0) >= 10
        ax.fill_between(grid[ok], lo[ok], hi[ok], color="tab:blue", alpha=0.2, label="synthetic 5-95%")
        ax.plot(grid[ok], mid[ok], color="tab:blue", label="synthetic median")
        for j, (k, *vals) in enumerate(real_curves):
            ax.plot(k, vals[idx - 1], color="tab:red", alpha=0.7, lw=1, label="real reps" if j == 0 else None)
        ax.set_xlabel("knee flexion (deg), descent")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name}: tracked reps vs matched synthetic reps (tracker view)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def check_clip(name, info, population, n_per_rep, pool, figures):
    track = mpp.observed_track(name)
    smooth = R.smoothed(track)
    angles = observed_angles(smooth)
    reps = R.find_reps(smooth, angles)
    feats = [R.rep_features(smooth, r, angles) for r in reps]
    sex = info.get("apparent_sex", "male")
    ratios = C.standing_proportions(smooth, angles)
    bodies, table, dist = C.matched_bodies(population, ratios, sex, weights=MATCH_WEIGHTS)
    for r, f in zip(reps, feats):
        f["com_at_bottom"] = float(np.median([C.com_at_bottom(b, smooth, r, angles) for b in bodies[:10]]))
    upright = angles["knee_flexion"] < 10
    standing_com = float(np.median([np.median(C.com_fraction(b, {k: v[upright] for k, v in angles.items()}))
                                    for b in bodies[:10]]))
    ratio_pct = {r: float(100 * np.mean(table[:, i] < ratios[r])) for i, r in enumerate(C.RATIOS)}

    # sensitivity: the same reps measured from the 3D world landmarks
    world = R.smoothed(mpp.world_plane_track(name))
    world_angles = observed_angles(world)
    world_feats = {}
    for r in R.find_reps(world, world_angles):
        match = [i for i, rr in enumerate(reps) if abs(rr.lowest - r.lowest) <= 5]
        if match:
            world_feats[match[0]] = R.rep_features(world, r, world_angles)

    print(f"\n## {name}  ({info.get('view', '')}; {info.get('arms', '')})")
    raw_clip = mpp.load_raw(name)
    offset = mpp.side_view_offset(raw_clip)
    lens = (f", lens about {mpp.lens_height_estimate(raw_clip):.1f} m high from floor perspective"
            if track.yaw_scale > 0.95 else "")
    print(f"yaw correction c = {track.yaw_scale:.3f}, camera {offset:+.0f} deg from side-on "
          f"(+ = towards the front){lens}; {len(reps)} reps; matched as {sex}")
    print("proportions (tracker view, over leg length) and ANSUR percentile:")
    for r in C.RATIOS:
        used = "matched" if MATCH_WEIGHTS[r] else "not used"
        print(f"  {r:6s} {ratios[r]:.3f}  ANSUR {sex} mean {table[:, C.RATIOS.index(r)].mean():.3f}"
              f"  pct {ratio_pct[r]:5.1f}  ({used})")
    print(f"closest ANSUR match distance {dist:.2f} SD")
    print(f"standing COM over the foot: {standing_com:.2f} (quiet standing, Fujiwara 2010: 0.31-0.58)")
    if track.yaw_scale < 0.95:
        raw = R.smoothed(mpp.observed_track(name, correct_yaw=False))
        raw_angles = observed_angles(raw)
        low = [R.rep_features(raw, r, raw_angles)["trunk_at_bottom"] for r in R.find_reps(raw, raw_angles)]
        print(f"oblique view: trunk lean at the bottom without yaw correction (a lower bound) "
              f"median {np.median(low):.1f} deg")

    results, real_curves, synth_curves = [], [], []
    for i, (rep, f) in enumerate(zip(reps, feats)):
        knee, timing, arms = calibrated_targets(bodies[0], f, track.fps, offset)
        jobs = [(bodies[j % len(bodies)], knee, timing, arms, 1000 * i + j, track.fps, offset)
                for j in range(n_per_rep)]
        synth = [s for s in pool.map(_job, jobs) if s]
        results.append((f, synth))
        real_curves.append(descent_curves(smooth, rep, angles))
        if i < 3:
            for j in range(min(15, n_per_rep)):
                out = C.synthetic_rep(bodies[j % len(bodies)], knee, timing, arms, 1000 * i + j,
                                      fps=track.fps, return_track=True, camera_offset=offset)
                if out:
                    _, st, sr = out
                    synth_curves.append(descent_curves(st, sr, observed_angles(st)))

    print(f"\n{'feature':48s} {'real median':>11s} {'synth median':>12s} {'median pct':>10s} {'outside 5-95':>12s} {'3D-image':>9s}")
    summary = {}
    for key, label in COMPARED:
        real = np.array([f[key] for f, _ in results])
        synth_med = np.array([np.nanmedian([s[key] for s in sy]) for _, sy in results])
        pcts = np.array([C.percentile(f[key], [s[key] for s in sy]) for f, sy in results])
        outside = int(np.sum((pcts < 5) | (pcts > 95)))
        diffs = [world_feats[i][key] - results[i][0][key] for i in world_feats if key in world_feats[i]]
        sens = f"{np.nanmedian(diffs):+.2f}" if diffs else "n/a"
        summary[key] = (np.nanmedian(real), np.nanmedian(synth_med), np.nanmedian(pcts), outside, sens)
        print(f"{label:48s} {np.nanmedian(real):11.2f} {np.nanmedian(synth_med):12.2f} "
              f"{np.nanmedian(pcts):10.0f} {outside:>6d}/{len(results):<5d} {sens:>9s}")

    print("\nper rep percentiles (trunk at bottom, shin at bottom, trunk-shin, hip back):")
    for i, (f, sy) in enumerate(results):
        row = [C.percentile(f[k], [s[k] for s in sy]) for k in
               ("trunk_at_bottom", "shank_at_bottom", "trunk_minus_shank_at_bottom", "hip_back_at_bottom")]
        print(f"  rep {i + 1} (t={reps[i].lowest / track.fps:4.1f}s, knee {f['knee_peak']:5.1f}, n={len(sy)}): "
              + "  ".join(f"{v:5.0f}" for v in row))

    if figures:
        os.makedirs(figures, exist_ok=True)
        figure(name, real_curves, synth_curves, os.path.join(figures, f"real_check_{name}.png"))
    return track


def noise_report(track, population):
    print("\n## Tracker noise: real (side-on clip) vs noise model on matched synthetic reps")
    jitter, angle_jitter, trend = tracker_noise_table(track)
    rng = np.random.default_rng(0)
    body = population.sample(np.random.default_rng(0), sex="female")
    synth_tables = []
    for seed in range(20):
        out = C.synthetic_rep(body, 110.0, C.matched_timing(0.8, 0.3, 0.7), C.arm_posture(150, 110),
                              seed, fps=track.fps, tracker_noise=False, return_track=True)
        if not out:
            continue
        clean = out[1]
        # six back-to-back reps, like the clip, before noise is added
        clean = clean.map_points(lambda v: np.concatenate([v] * 6))
        noisy = TN.apply(clean, TN.TrackerNoise(), rng)
        synth_tables.append(tracker_noise_table(noisy))
    def mean_of(i, key):
        return np.mean([t[i][key] for t in synth_tables])
    print(f"{'quantity':40s} {'real':>8s} {'model':>8s}  kind")
    for p in ("shoulder", "hip", "knee", "ankle"):
        print(f"{'jitter ' + p + ' (frac of leg)':40s} {jitter[p]:8.4f} {mean_of(0, p):8.4f}  fitted")
    for k in angle_jitter:
        print(f"{'angle jitter ' + k + ' (deg)':40s} {angle_jitter[k]:8.2f} {mean_of(1, k):8.2f}  not fitted")
    for seg in trend:
        print(f"{seg + ' length at 100 deg knee / standing':40s} {trend[seg]:8.3f} {mean_of(2, seg):8.3f}  "
              "not fitted" + (" (3D stance and camera)" if seg == "thigh" else ""))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("n_per_rep", nargs="?", type=int, default=60)
    parser.add_argument("--figures", default=None)
    args = parser.parse_args(argv)
    population = BodyPopulation.load()
    manifest = mpp.manifest()
    side_on = None
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for name in mpp.available():
            info = manifest.get(name, {})
            if info.get("view", "").startswith("front"):
                continue        # front views: scripts/front_view_check.py
            track = check_clip(name, info, population, args.n_per_rep, pool, args.figures)
            if side_on is None and track.yaw_scale > 0.95:
                side_on = track
    if side_on is not None:
        noise_report(side_on, population)


if __name__ == "__main__":
    sys.exit(main())
