"""Make synthetic tracks look like MediaPipe output of a side-on video.

The generator gives exact joint positions; a pose tracker on real video
does not. A model trained on exact positions would meet a different kind
of input in use. This adds the three effects measured on the side-on
clip (squat_1, MediaPipe Pose heavy, 30 fps; ``scripts/real_video_check.py``):

1. **Jitter**: fast, nearly white noise on every landmark. Measured as the
   part of each tracked coordinate above 6 Hz (raw minus the 6 Hz
   low-pass), as a fraction of leg length. 0.2-0.5% on the legs and trunk,
   about 1% on arms held behind the head (partly hidden). Only the part of
   white noise above 6 Hz shows up in that measurement, (15 - 6) / 15 =
   60% of its power at 30 fps, so the injected SD is the measured one
   divided by sqrt(0.6). About 40% of the jitter variance is shared by all
   landmarks (residual correlation shoulder-ankle 0.38, hip-ankle 0.46 in
   x): the camera in that clip pans, and on the fixed-camera clip the same
   correlations are about 0.1, so the shared part is camera shake. It moves
   the whole figure and leaves angles alone.
2. **Drift**: slower wander of each landmark relative to the body. Real
   segments are rigid, so their tracked length should only change with
   pose; what is left after removing the pose trend and the change common
   to all segments (camera zoom) is drift: segment length SD 1.8%, half
   correlation time about 0.15 s. Assuming independent ends, each landmark
   wanders by 1.8% x (segment / leg, about 0.5) / sqrt(2) = 0.65% of leg
   length, as Gaussian-smoothed white noise with a 0.09 s kernel (whose
   autocorrelation exp(-t^2 / 4 s^2) halves at 0.15 s).
3. **Knees out of plane**: in a squat the knees travel out over the toes,
   and a side view cannot see the sideways part, so the thigh looks
   shorter. On the side-on clip the tracked thigh shrank to 0.91 of its
   standing length by 60 deg of knee flexion and stayed there (shank: no
   consistent change). cos(24 deg) = 0.91, so the thigh's sideways angle
   is modelled as rising to ``knee_out_deg`` (mean 24) by 60 deg of knee
   flexion. The hip and ankle stay where they are and the knee point is
   re-placed for the shorter apparent thigh, which also changes the
   apparent knee angle, as on real video.

Everything here comes from ONE person in ONE clip. The spreads
(``KNEE_OUT_DEG`` SD, the per-person scaling of jitter and drift) are
ASSUMPTIONS until more side-on videos are added. Tracker *bias* (how far
MediaPipe's hip or shoulder sits from the true joint centre) cannot be
measured without a reference system and is not modelled.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from exercise_quality.core.observed import POINTS, ObservedTrack, observed_angles, segment_lengths

# High-frequency jitter SD measured above 6 Hz, fraction of leg length,
# (x, y). squat_1, 30 fps.
MEASURED_JITTER = {
    "ear": (0.0028, 0.0054), "shoulder": (0.0030, 0.0043), "elbow": (0.0076, 0.0109),
    "wrist": (0.0056, 0.0083), "hip": (0.0028, 0.0035), "knee": (0.0022, 0.0016),
    "ankle": (0.0017, 0.0016), "heel": (0.0018, 0.0017), "toe": (0.0023, 0.0019),
}
HIGHPASS_POWER_SHARE = (15.0 - 6.0) / 15.0
JITTER = {k: tuple(v / np.sqrt(HIGHPASS_POWER_SHARE) for v in xy) for k, xy in MEASURED_JITTER.items()}
CAMERA_SHAKE_SHARE = 0.4
DRIFT_SEGMENT_SD = 0.018
_SEGMENT_OVER_LEG = {"ear": 0.23, "heel": 0.3, "toe": 0.3}
DRIFT = {name: DRIFT_SEGMENT_SD * _SEGMENT_OVER_LEG.get(name, 0.5) / np.sqrt(2) for name in POINTS}
DRIFT_KERNEL_S = 0.09
KNEE_OUT_DEG = (24.0, 8.0)       # mean measured; SD an ASSUMPTION
KNEE_OUT_FULL_AT = 60.0          # knee flexion by which the full sideways angle is reached
LEVEL_SD = 0.5                   # per-track spread of jitter/drift scale, log-normal sigma (ASSUMPTION)


@dataclass(frozen=True)
class TrackerNoise:
    jitter_scale: float = 1.0
    drift_scale: float = 1.0
    knee_out_deg: float = KNEE_OUT_DEG[0]

    @classmethod
    def sample(cls, rng):
        return cls(
            jitter_scale=float(np.exp(rng.normal(0.0, LEVEL_SD))),
            drift_scale=float(np.exp(rng.normal(0.0, LEVEL_SD))),
            knee_out_deg=float(np.clip(rng.normal(*KNEE_OUT_DEG), 0.0, 40.0)),
        )


def _replace_knee(hip, ankle, knee, thigh_scale):
    """Knee position for a thigh shortened by ``thigh_scale``, hip and ankle fixed.

    Keeps the shank length and the side the knee bends to (forward).
    """
    shank = np.linalg.norm(knee - ankle, axis=-1)
    thigh = np.linalg.norm(hip - knee, axis=-1) * thigh_scale
    d_vec = hip - ankle
    d = np.linalg.norm(d_vec, axis=-1)
    eps = 1e-12 * (shank + thigh)
    d = np.clip(d, np.abs(shank - thigh) + eps, shank + thigh - eps)
    along = (shank**2 - thigh**2 + d**2) / (2 * d)
    across = np.sqrt(np.maximum(shank**2 - along**2, 0.0))
    unit = d_vec / np.linalg.norm(d_vec, axis=-1, keepdims=True)
    normal = np.stack([unit[..., 1], -unit[..., 0]], axis=-1)   # points forward for an upright leg
    side = np.sign(np.sum((knee - ankle) * normal, axis=-1))
    side = np.where(side == 0, 1.0, side)
    moved = ankle + along[..., None] * unit + (side * across)[..., None] * normal
    return np.where((np.asarray(thigh_scale) < 1.0)[..., None], moved, knee)


def apply(track, noise, rng):
    """A noisy copy of ``track`` (any units; noise scales with its leg length)."""
    lengths = segment_lengths(track)
    leg = float(np.median(lengths["thigh"] + lengths["shank"]))
    points = {k: v.copy() for k, v in track.points.items()}

    if noise.knee_out_deg > 0:
        knee = observed_angles(track, foot_pitch_reference=0.0)["knee_flexion"]
        angle = np.radians(noise.knee_out_deg) * np.clip(knee / KNEE_OUT_FULL_AT, 0.0, 1.0)
        points["knee"] = _replace_knee(points["hip"], points["ankle"], points["knee"], np.cos(angle))

    n = len(track)
    sigma_frames = DRIFT_KERNEL_S * track.fps
    shake = rng.normal(size=(n, 2))
    for name in POINTS:
        white = (np.sqrt(CAMERA_SHAKE_SHARE) * shake
                 + np.sqrt(1.0 - CAMERA_SHAKE_SHARE) * rng.normal(size=(n, 2)))
        drift = gaussian_filter1d(rng.normal(size=(n, 2)), sigma_frames, axis=0, mode="reflect")
        drift /= max(float(drift.std()), 1e-12)
        jitter = np.asarray(JITTER[name]) * noise.jitter_scale
        points[name] = points[name] + leg * (white * jitter + drift * DRIFT[name] * noise.drift_scale)
    return ObservedTrack(points=points, fps=track.fps, units=track.units,
                         visibility=track.visibility, source=track.source + "+tracker_noise",
                         yaw_scale=track.yaw_scale)
