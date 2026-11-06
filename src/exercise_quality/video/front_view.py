"""Front-view measurements from MediaPipe tracks, real or synthetic.

Works on the raw format of ``mediapipe_pose.load_raw`` (and
``video.camera.synthetic_raw``), so a real clip and a synthetic rep
filmed by the virtual camera are measured by the same code.

Only frames where the camera is within ``MAX_YAW`` of straight in front
(or behind) are used. The view angle comes from MediaPipe's 3D "world"
hip axis, as for the side view (``mediapipe_pose.world_yaw_scale``).

Measures (image pixels unless noted, all as scale-free ratios):

* depth: hip height above the ankles, as a fraction of standing (the
  side-view rep finder's signal, which the front view sees equally well).
* stance: ankle separation over standing leg height (hip to ankle,
  vertical), and over hip separation.
* toe-out: angle of each foot (heel to "foot index") from the body's
  forward direction, from the 3D world landmarks (MediaPipe has to guess
  depth, so this is rough).
* knee separation at the bottom over ankle separation (knee-to-ankle
  separation ratio, KASR). Above 1: knees outside the ankles.
* knee frontal-plane projection angle (FPPA) on each side: how far the
  knee bends off the hip-to-ankle line in the image, positive = valgus
  (knee towards the midline). Taken at the frame where the hip is half
  way down on the descent (near parallel the thigh points at the camera
  and the angle means little; see docs/model3d.md).
* hip shift at the bottom: mid-hip minus mid-ankle, sideways, over ankle
  separation (positive = towards the person's right).
* pelvic tilt at the bottom: hip line angle from horizontal (deg,
  positive = the person's right hip lower).
* depth asymmetry: right minus left hip height above its own ankle at
  the bottom, over standing leg height.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from exercise_quality.core.signal import lowpass

MAX_YAW = 25.0
MIN_DROP = 0.15
PHASE = 0.05
# MediaPipe indices
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32


def view_yaw(raw):
    """Signed angle (deg) of the hip axis from the image plane, per frame; 0 = square to the camera."""
    w = raw["world"]
    axis = w[:, L_HIP] - w[:, R_HIP]
    return np.degrees(np.arctan2(axis[:, 2], np.abs(axis[:, 0])))


def front_segments(raw, max_yaw=MAX_YAW, min_seconds=2.0):
    """Frame ranges where the camera is within ``max_yaw`` of square to the hips."""
    fps = float(raw["fps"])
    yaw = lowpass(view_yaw(raw), fps, 1.0)
    w = raw["world"]
    facing = np.abs(w[:, L_HIP, 0] - w[:, R_HIP, 0]) > 0      # hip axis across the image
    ok = (np.abs(yaw) < max_yaw) & facing
    segments, start = [], None
    for i, good in enumerate(np.append(ok, False)):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= min_seconds * fps:
                segments.append((start, i))
            start = None
    return segments


@dataclass(frozen=True)
class FrontRep:
    start: int
    lowest: int
    end: int
    half_down: int


def _smooth_image(raw, sl):
    fps = float(raw["fps"])
    xy = raw["image"][sl, :, :2].astype(float)
    return lowpass(xy.reshape(len(xy), -1), fps, 6.0).reshape(xy.shape)


def hip_height(xy):
    """Hip height above the ankles (image, y down), fraction of its 95th percentile."""
    hip = 0.5 * (xy[:, L_HIP, 1] + xy[:, R_HIP, 1])
    ankle = 0.5 * (xy[:, L_ANKLE, 1] + xy[:, R_ANKLE, 1])
    h = ankle - hip
    return h / np.percentile(h, 95)


def find_reps(xy, fps):
    h = hip_height(xy)
    bottoms, _ = find_peaks(1.0 - h, prominence=MIN_DROP, distance=int(0.5 * fps))
    reps = []
    for j, b in enumerate(bottoms):
        lo = bottoms[j - 1] if j else 0
        hi = bottoms[j + 1] if j + 1 < len(bottoms) else len(h) - 1
        top0 = lo + int(np.argmax(h[lo:b + 1]))
        top1 = b + int(np.argmax(h[b:hi + 1]))
        drop = h[top0] - h[b]
        if drop < MIN_DROP or h[top1] - h[b] < MIN_DROP:
            continue
        down = np.flatnonzero(h[top0:b + 1] <= h[top0] - PHASE * drop)
        up = np.flatnonzero(h[b:top1 + 1] <= h[top1] - PHASE * (h[top1] - h[b]))
        half = np.flatnonzero(h[top0:b + 1] <= h[top0] - 0.5 * drop)
        reps.append(FrontRep(top0 + int(down[0]), b, b + int(up[-1]), top0 + int(half[0])))
    return reps


def _fppa(hip, knee, ankle, side):
    """Valgus-positive frontal-plane angle at the knee, image coordinates (x right, y down)."""
    def up_angle(a, b):        # direction a -> b from image-up, positive towards image right
        return np.degrees(np.arctan2(b[..., 0] - a[..., 0], a[..., 1] - b[..., 1]))
    thigh = up_angle(knee, hip)
    shank = up_angle(ankle, knee)
    # facing the camera, the person's right is image left: a right knee caving
    # in moves image right (towards the midline)
    return (shank - thigh) if side == "right" else (thigh - shank)


def _facing_camera(xy):
    """True if the person faces the camera (their right hip is on the image left)."""
    return np.nanmedian(xy[:, R_HIP, 0] - xy[:, L_HIP, 0]) < 0


def toe_out(raw, frames):
    """Each foot's angle out from the body's forward direction (deg), from world landmarks."""
    w = raw["world"][frames].astype(float)
    axis = w[:, L_HIP][:, [0, 2]] - w[:, R_HIP][:, [0, 2]]          # person's right -> left, in (x, z)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    # forward: horizontal, perpendicular to the hip axis, away from the heels
    forward = np.stack([axis[:, 1], -axis[:, 0]], axis=-1)
    feet = {"right": (R_HEEL, R_TOE, -axis), "left": (L_HEEL, L_TOE, axis)}
    out = {}
    for side, (heel, toe, outward) in feet.items():
        f = w[:, toe][:, [0, 2]] - w[:, heel][:, [0, 2]]
        if np.nanmedian(np.sum(f * forward, axis=-1)) < 0:
            forward = -forward
        out[side] = float(np.nanmedian(np.degrees(np.arctan2(np.sum(f * outward, axis=-1),
                                                             np.sum(f * forward, axis=-1)))))
    return out


def measure(raw, segment=None):
    """Front-view measures of every rep in ``segment`` (frame range); None if no front view."""
    if segment is None:
        segs = front_segments(raw)
        if not segs:
            return None
        segment = max(segs, key=lambda s: s[1] - s[0])
    sl = slice(*segment)
    fps = float(raw["fps"])
    xy = _smooth_image(raw, sl)
    if not _facing_camera(xy):
        return None          # filmed from behind: left and right swap; not handled
    reps = find_reps(xy, fps)
    h = hip_height(xy)
    standing = h > 0.97
    leg = np.median((0.5 * (xy[:, L_ANKLE, 1] + xy[:, R_ANKLE, 1])
                     - 0.5 * (xy[:, L_HIP, 1] + xy[:, R_HIP, 1]))[standing])
    ankle_sep = np.abs(xy[:, L_ANKLE, 0] - xy[:, R_ANKLE, 0])
    hip_sep = np.abs(xy[:, L_HIP, 0] - xy[:, R_HIP, 0])
    knee_sep = np.abs(xy[:, L_KNEE, 0] - xy[:, R_KNEE, 0])
    shoulder_sep = np.abs(xy[:, L_SHOULDER, 0] - xy[:, R_SHOULDER, 0])
    yaw = view_yaw(raw)[sl]
    out = {
        "segment": segment,
        "fps": fps,
        "view_yaw": float(np.nanmedian(yaw)),
        "stance_over_leg": float(np.median(ankle_sep[standing]) / leg),
        "stance_over_hips": float(np.median(ankle_sep[standing] / hip_sep[standing])),
        "shoulders_over_hips": float(np.median(shoulder_sep[standing] / hip_sep[standing])),
        "toe_out": toe_out(raw, np.arange(*segment)[standing]),
        "reps": [],
    }
    for r in reps:
        b, m = r.lowest, r.half_down
        mid_hip = 0.5 * (xy[b, L_HIP] + xy[b, R_HIP])
        mid_ankle = 0.5 * (xy[b, L_ANKLE] + xy[b, R_ANKLE])
        dy = xy[b, R_HIP, 1] - xy[b, L_HIP, 1]
        dx = xy[b, L_HIP, 0] - xy[b, R_HIP, 0]
        right_h = xy[b, R_ANKLE, 1] - xy[b, R_HIP, 1]
        left_h = xy[b, L_ANKLE, 1] - xy[b, L_HIP, 1]
        rep = {
            "depth": float(1.0 - h[b]),
            "kasr_bottom": float(knee_sep[b] / ankle_sep[b]),
            "kasr_half": float(knee_sep[m] / ankle_sep[m]),
            "fppa_right_half": float(_fppa(xy[m, R_HIP], xy[m, R_KNEE], xy[m, R_ANKLE], "right")),
            "fppa_left_half": float(_fppa(xy[m, L_HIP], xy[m, L_KNEE], xy[m, L_ANKLE], "left")),
            # facing the camera, the person's right is image left
            "hip_shift": float(-(mid_hip[0] - mid_ankle[0]) / ankle_sep[b]),
            "pelvic_tilt": float(np.degrees(np.arctan2(dy, dx))),
            "depth_asymmetry": float((right_h - left_h) / leg),
            "descent_s": (b - r.start) / fps,
            "ascent_s": (r.end - b) / fps,
        }
        out["reps"].append(rep)
    return out
