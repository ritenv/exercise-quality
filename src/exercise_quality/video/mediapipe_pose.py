"""Turn MediaPipe Pose tracks into ``ObservedTrack`` objects.

Tracks come from ``scripts/extract_video_landmarks.py`` and live in
``data/real_video/``. The person may face either way and either side may
be towards the camera; the adapter picks the near side (higher landmark
visibility), mirrors the image so the person faces +x, and flips y so it
points up.

Camera not side-on
------------------
If the camera is turned by ``theta`` away from a true side view (about
the vertical axis), forward distances in the sagittal plane shrink by
``c = cos(theta)`` in the image while vertical ones do not. ``c`` is
measured from MediaPipe's 3D "world" landmarks: the hip axis should point
straight along the camera axis in a side view, and its angle away from it
is ``theta``. Image x is then divided by ``c``.

Why not from the image alone: a thigh or shank swinging in the sagittal
plane traces an ellipse u^2/c^2 + v^2 = L^2, so fitting it would give c
without 3D. But on the side-on clip that fit gave 0.93 (world landmarks:
0.997) because the knees travel out over the toes in a squat, and the
sideways part is invisible from the side, shortening the thigh by about
7% at the bottom. The fit cannot tell that apart from camera yaw.

Why not take the angles from the 3D world landmarks directly: on the
side-on clip they differ from the image angles by 5-7 deg on average at
the knee and shank (frame-to-frame SD up to 8 deg), because MediaPipe has
to guess depth. The image positions are the measured quantity, so they
are kept and only the global yaw comes from 3D. ``world_plane_track``
gives the 3D version for sensitivity checks. Out-of-plane movement seen
obliquely (knees pushed out towards the camera) still leaks into the
corrected angles, so oblique clips are lower confidence.
"""

import json
from pathlib import Path

import numpy as np

from exercise_quality.core.observed import ObservedTrack

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_video"

# MediaPipe Pose landmark indices (left, right).
INDEX = {
    "ear": (7, 8), "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28), "heel": (29, 30),
    "toe": (31, 32),   # MediaPipe "foot index"
}
LEG_POINTS = ("hip", "knee", "ankle", "heel", "toe")


def manifest():
    path = DATA_DIR / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def available():
    return sorted(p.stem for p in DATA_DIR.glob("*.npz"))


UNIFORM_FPS = 30.0


def load_raw(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        return resample_uniform({k: data[k] for k in data.files})


def resample_uniform(raw, fps=UNIFORM_FPS, tolerance=0.1):
    """Resample a track with irregular frame times (``raw["t"]``) to a constant rate.

    Everything downstream assumes evenly spaced frames. Screen recordings
    and some phone videos are not: frame gaps of 8-58 ms within one clip
    were seen. If the gaps vary by more than ``tolerance`` of their median,
    every landmark is linearly interpolated onto an even ``fps`` grid
    (NaN frames stay NaN where both neighbours are missing).
    """
    t = raw.get("t")
    if t is None or len(t) < 3:
        return raw
    gaps = np.diff(t)
    if np.max(np.abs(gaps - np.median(gaps))) <= tolerance * np.median(gaps):
        return raw
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    out = dict(raw)
    for key in ("image", "world", "presence"):
        values = raw[key].astype(float)
        flat = values.reshape(len(t), -1)
        res = np.empty((len(grid), flat.shape[1]))
        for j in range(flat.shape[1]):
            ok = np.isfinite(flat[:, j])
            res[:, j] = np.interp(grid, t[ok], flat[ok, j]) if ok.sum() >= 2 else np.nan
        out[key] = res.reshape((len(grid),) + values.shape[1:])
    out["t"] = grid
    out["fps"] = np.float64(fps)
    out["resampled_from_irregular"] = True
    return out


def near_side(raw):
    image = raw["image"]
    vis = {side: np.nanmean(image[:, [INDEX[p][i] for p in LEG_POINTS], 3])
           for i, side in enumerate(("left", "right"))}
    return max(vis, key=vis.get), vis


def world_yaw_scale(raw):
    """cos of the angle between the camera axis and the hip axis, from world landmarks."""
    w = raw["world"]
    axis = w[:, INDEX["hip"][0]] - w[:, INDEX["hip"][1]]
    lateral_in_image = np.abs(axis[:, 0])
    depth = np.abs(axis[:, 2])
    theta = np.arctan2(lateral_in_image, depth)
    return float(np.nanmedian(np.cos(theta)))


def side_view_offset(raw):
    """Signed camera angle from side-on (deg): positive = towards the person's front.

    The size comes from ``world_yaw_scale``; the sign from the image: seen
    from in front of side-on, the far hip appears ahead of the near one (in
    the direction the feet point), and behind it when seen from behind.
    Checked on virtual-camera reps (tests/test_camera.py).
    """
    side, _ = near_side(raw)
    near, far = (1, 0) if side == "right" else (0, 1)
    image = raw["image"]
    facing = np.sign(np.nanmedian(image[:, INDEX["toe"][near], 0] - image[:, INDEX["heel"][near], 0]))
    ahead = np.nanmedian((image[:, INDEX["hip"][far], 0] - image[:, INDEX["hip"][near], 0]) * facing)
    angle = float(np.degrees(np.arccos(np.clip(world_yaw_scale(raw), 0.0, 1.0))))
    return angle if ahead >= 0 else -angle


# Far heel above the near heel in standing (fraction of the near leg's
# image height) against lens height, from virtual-camera reps filmed side-on
# 3-5 m away with sampled stances: 0.042 at 0.4 m, 0.082 at 0.8, 0.118 at
# 1.2, 0.144 at 1.5 (10-90% about +-0.015, i.e. +-0.15 m).
FAR_HEEL_PER_METRE = 0.093
FAR_HEEL_AT_FLOOR = 0.005


def lens_height_estimate(raw):
    """Rough camera lens height above the floor (m) from floor perspective, side views only.

    A lens above the floor sees the far foot higher in the image than the
    near one, more so the higher it is. Calibrated on virtual-camera reps
    (``FAR_HEEL_PER_METRE``), not on real cameras; MediaPipe's far-foot
    points may carry their own bias.
    """
    side, _ = near_side(raw)
    near, far = (1, 0) if side == "right" else (0, 1)
    image = raw["image"]
    leg = np.abs(image[:, INDEX["ankle"][near], 1] - image[:, INDEX["hip"][near], 1])
    standing = leg > np.nanpercentile(leg, 80)
    rise = np.nanmedian((image[standing, INDEX["heel"][near], 1] - image[standing, INDEX["heel"][far], 1])
                        / leg[standing])
    return float((rise - FAR_HEEL_AT_FLOOR) / FAR_HEEL_PER_METRE)


def _points(raw, side, source):
    i = 0 if side == "left" else 1
    return {k: source[:, idx[i]].astype(float) for k, idx in INDEX.items()}


def _face_forward(points):
    facing = np.nanmedian(points["toe"][:, 0] - points["heel"][:, 0])
    if facing < 0:
        points = {k: v * np.array([-1.0, 1.0]) for k, v in points.items()}
    return points


def _raw(name):
    """A stored clip by name, or a raw dict as load_raw returns (e.g. video/camera.py)."""
    if isinstance(name, dict):
        return name, name.get("name", "synthetic")
    return load_raw(name), name


def observed_track(name, side="auto", correct_yaw=True):
    """Load a stored track as an ObservedTrack in pixels (person faces +x, y up).

    ``name`` is a stored clip or a raw dict in the same format
    (``video.camera.synthetic_raw`` makes one from a 3D rep)."""
    raw, name = _raw(name)
    if side == "auto":
        side, _ = near_side(raw)
    fps = float(raw["fps"])
    i = 0 if side == "left" else 1
    points = {k: v[:, :2] * np.array([1.0, -1.0]) for k, v in _points(raw, side, raw["image"]).items()}
    visibility = {k: raw["image"][:, idx[i], 3].astype(float) for k, idx in INDEX.items()}
    points = _face_forward(points)
    scale = world_yaw_scale(raw) if correct_yaw else 1.0
    points = {k: v * np.array([1.0 / scale, 1.0]) for k, v in points.items()}
    return ObservedTrack(points=points, fps=fps, units="px", visibility=visibility,
                         source=f"mediapipe:{name}:{side}", yaw_scale=scale)


def world_plane_track(name, side="auto"):
    """The 3D world landmarks projected onto the body's sagittal plane (metres)."""
    raw, name = _raw(name)
    if side == "auto":
        side, _ = near_side(raw)
    world = _points(raw, side, raw["world"])
    w = raw["world"].astype(float)
    up = np.array([0.0, -1.0, 0.0])
    axis = np.nanmedian(w[:, INDEX["hip"][0]] - w[:, INDEX["hip"][1]], axis=0)
    axis -= axis.dot(up) * up
    axis /= np.linalg.norm(axis)
    forward = np.cross(up, axis)
    points = {k: np.stack([v @ forward, v @ up], axis=-1) for k, v in world.items()}
    return ObservedTrack(points=_face_forward(points), fps=float(raw["fps"]), units="m",
                         source=f"mediapipe-world:{name}:{side}")
