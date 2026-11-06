"""Generate many reps across every protocol and check them all.

    PYTHONPATH=src python scripts/stress_test.py [n_reps]

Unit tests check a few bodies; this checks the tails. Each rep gets a
random real person and a style sampled under one of the protocols, and
the script reports anything that would look wrong in an animation: reps
that are infeasible or leave their feet, ground-force spikes, stretched
segments, left and right sides drifting apart, lifted heels and extreme
trunk lean. Runs in parallel, one rep per core at a time.

It found the 4 deg ankle limit hand-over acting like a wall (force jerk
up to 19 body weights per second, 3 of 1500 people off their feet).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from exercise_quality.core.dynamics import analyse_dynamics
from exercise_quality.core.sagittal_body import BodyPopulation
from exercise_quality.core.sagittal_model import LANDMARKS, SEGMENT_ENDPOINTS, segment_lengths
from exercise_quality.exercises.squat import protocols as p
from exercise_quality.exercises.squat.balanced import generate_feasible_squat

PROTOCOLS = [p.ISHIDA_PARALLEL, p.KASAHARA_PARALLEL, p.ZAWADKA_MAX_DEPTH]
TRUNK_REPORT_DEG = 65.0
JERK_REPORT_BW_S = 3.0


def check(seed):
    rng = np.random.default_rng(seed)
    protocol = PROTOCOLS[seed % len(PROTOCOLS)]
    body = BodyPopulation.load().sample(rng)
    start = time.time()
    seq = generate_feasible_squat(body, p.sample_style(rng, body, protocol))
    elapsed = time.time() - start
    m, d, lengths = seq.measured(), analyse_dynamics(seq), segment_lengths(body)
    lm = seq.landmarks
    return {
        "seed": seed, "protocol": protocol.name, "feasible": seq.is_feasible,
        "depth_limited": "depth_limited_by_mobility" in seq.notes,
        "on_foot": d.stays_on_foot,
        "force_min": float(d.vertical_force.min()), "force_max": float(d.vertical_force.max()),
        "jerk": float(np.abs(np.gradient(d.vertical_force, 1 / seq.fps)).max()),
        "length_error": max(float(np.abs(np.linalg.norm(lm[b] - lm[a], axis=-1) - lengths[n]).max())
                            for n, (a, b) in SEGMENT_ENDPOINTS.items()),
        "side_gap": max(float(np.abs(lm[k] - lm["left_" + k[6:]]).max())
                        for k in LANDMARKS if k.startswith("right_")),
        "heel_height": max(float(np.abs(lm[f"{s}_heel"][:, 1]).max()) for s in ("right", "left")),
        "finite": all(np.isfinite(v).all() for v in lm.values()),
        "trunk": float(m["trunk_inclination"].max()),
        "lumbar": float(m["lumbar_flexion"].max()),
        "seconds": elapsed,
    }


def main(n):
    start = time.time()
    with ProcessPoolExecutor() as pool:
        reps = list(pool.map(check, range(n), chunksize=10))

    def col(k):
        return np.array([r[k] for r in reps])

    print(f"{n} reps in {time.time() - start:.0f} s ({col('seconds').mean() * 1000:.0f} ms per rep per core)")
    print(f"feasible {col('feasible').mean():.1%}, depth limited {col('depth_limited').mean():.1%}, "
          f"on their feet {col('on_foot').mean():.1%}, all finite {col('finite').all()}")
    print(f"vertical force {col('force_min').min():.2f}-{col('force_max').max():.2f} BW, "
          f"jerk 95th percentile {np.percentile(col('jerk'), 95):.2f}, max {col('jerk').max():.2f} BW/s")
    print(f"worst segment length error {col('length_error').max():.1e} m, left-right gap "
          f"{col('side_gap').max():.1e} m, heel height {col('heel_height').max():.1e} m")
    trunk = col("trunk")
    print(f"peak trunk lean median {np.median(trunk):.1f}, 99th percentile {np.percentile(trunk, 99):.1f}, "
          f"max {trunk.max():.1f} deg; peak lumbar flexion max {col('lumbar').max():.1f} deg")
    flagged = [r for r in reps if not r["feasible"] or not r["on_foot"]
               or r["jerk"] > JERK_REPORT_BW_S or r["trunk"] > TRUNK_REPORT_DEG]
    for r in flagged:
        print(f"  seed {r['seed']} ({r['protocol']}): feasible={r['feasible']} on_foot={r['on_foot']} "
              f"jerk={r['jerk']:.1f} trunk={r['trunk']:.0f} lumbar={r['lumbar']:.0f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
