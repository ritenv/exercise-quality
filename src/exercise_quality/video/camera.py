"""A virtual camera: see a 3D rep from any angle, as MediaPipe Pose would report it.

``synthetic_raw`` turns a 3D ``SquatSequence`` into the same raw format
that ``scripts/extract_video_landmarks.py`` stores for real videos
(33 MediaPipe landmarks per frame, image pixels and "world" metres), so
synthetic and real clips go through exactly the same code from there on
(``video/mediapipe_pose.observed_track``, the rubric, the rep finder).

Camera
------
A pinhole camera placed ``distance`` metres from the point on the floor
between the feet, at ``azimuth`` degrees around the person (0 = straight
in front, 90 = their right side, -90 = their left side, 180 = behind),
with the lens ``lens_height`` above the floor, aimed at ``aim_height``
above the floor between the feet. Perspective is real: a near camera
sees near limbs bigger, and angles in the image differ from the true
ones. Iizuka et al. 2026 (Front Sports Act Living, PMC13347614) found a
webcam 2 m from the side underestimating hip and knee angles by about
11 deg against marker-based capture, mostly from this projection. The default is a phone in portrait: 1080 x 1920 pixels,
69 deg field of view along the long side (a typical 26 mm-equivalent
main camera; ASSUMPTION).

Landmarks
---------
MediaPipe's 33 points are filled from the model's: ears, shoulders,
elbows, wrists, hip joint centres, knees, ankles, heels and toe tips (for
"foot index"). The face points (eyes, mouth) are put at the nose and the
finger points at the hand (knuckle). MediaPipe's hip and shoulder points
are learned from human labels, not joint centres. Shoulders are put at
the acromions (ANSUR II biacromial breadth; ASSUMPTION). Hips are moved
out from the hip joint centres by ``MEDIAPIPE_HIP_OUTWARD`` of the way to
the skin at hip-breadth level: fitted on four real front-view clips
(three people), whose MediaPipe shoulder-over-hip width was 1.66-1.75,
against 2.3-2.6 for acromion width over hip-joint-centre width
(docs/front_view_check.md). Women gave 0.30, the man 0.42.

Visibility: 0.9 for points on the side facing the camera, 0.6 on the
far side (ASSUMPTION; real MediaPipe visibilities on the side-on clip
are about 0.9-1.0 near and 0.5-0.8 far).

Tracker noise
-------------
``TrackerNoise`` jitter and drift (core/tracker_noise.py, measured on a
side-on clip) are added in image pixels, scaled by the near leg's
length in the image, for every view. Its knees-out term is not applied:
the 3D geometry already has the knees travelling out. Whether MediaPipe
is as noisy from the front as from the side is not known (ASSUMPTION).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from exercise_quality.core import tracker_noise as TN

N_LANDMARKS = 33
# MediaPipe index -> model3d landmark
MEDIAPIPE_SOURCE = {
    0: "nose", 1: "nose", 2: "nose", 3: "nose", 4: "nose", 5: "nose", 6: "nose",
    7: "left_ear", 8: "right_ear", 9: "nose", 10: "nose",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_hand", 18: "right_hand", 19: "left_hand", 20: "right_hand", 21: "left_hand", 22: "right_hand",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_toe", 32: "right_toe",
}
# which tracker-noise point each MediaPipe landmark behaves like
_NOISE_POINT = {
    "nose": "ear", "ear": "ear", "shoulder": "shoulder", "elbow": "elbow", "wrist": "wrist",
    "hand": "wrist", "hip": "hip", "knee": "knee", "ankle": "ankle", "heel": "heel", "toe": "toe",
}
NEAR_VISIBILITY, FAR_VISIBILITY = 0.9, 0.6
MEDIAPIPE_HIP_OUTWARD = 0.35


@dataclass(frozen=True)
class Camera:
    azimuth: float = 90.0
    distance: float = 3.5
    lens_height: float = 1.0
    aim_height: float = 0.9
    width: int = 1080
    height: int = 1920
    long_side_fov: float = 69.0

    @property
    def focal_px(self):
        return 0.5 * max(self.width, self.height) / np.tan(np.radians(0.5 * self.long_side_fov))

    def pose(self):
        """Camera centre and rotation (rows: image right, image down, viewing direction)."""
        a = np.radians(self.azimuth)
        centre = np.array([self.distance * np.cos(a), self.lens_height, self.distance * np.sin(a)])
        forward = np.array([0.0, self.aim_height, 0.0]) - centre
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return centre, np.stack([right, down, forward])

    def to_camera(self, points):
        centre, rot = self.pose()
        return (np.asarray(points, float) - centre) @ rot.T

    def project(self, points):
        """Pixel coordinates (u right, v down) and depth (m) of points (..., 3)."""
        c = self.to_camera(points)
        f = self.focal_px
        u = 0.5 * self.width + f * c[..., 0] / c[..., 2]
        v = 0.5 * self.height + f * c[..., 1] / c[..., 2]
        return np.stack([u, v], axis=-1), c[..., 2]


def _mediapipe_array(landmarks, body=None):
    first = next(iter(landmarks.values()))
    out = np.zeros(first.shape[:-1] + (N_LANDMARKS, 3))
    for i, name in MEDIAPIPE_SOURCE.items():
        out[..., i, :] = landmarks[name]
    if body is not None:
        lateral = landmarks["right_hip"] - landmarks["left_hip"]
        lateral = lateral / np.linalg.norm(lateral, axis=-1, keepdims=True)
        shift = 0.5 * MEDIAPIPE_HIP_OUTWARD * (body.hip_breadth - body.hip_joint_width)
        out[..., 24, :] += shift * lateral
        out[..., 23, :] -= shift * lateral
    return out


def _side_of(name):
    return "left" if name.startswith("left_") else "right" if name.startswith("right_") else None


def image_noise(image_xy, fps, leg_px, noise, rng):
    """Tracker jitter and drift added to MediaPipe image points (T, 33, 2), in pixels."""
    n = image_xy.shape[0]
    out = image_xy.copy()
    shake = rng.normal(size=(n, 2))
    sigma = TN.DRIFT_KERNEL_S * fps
    for i, name in MEDIAPIPE_SOURCE.items():
        point = _NOISE_POINT[name.split("_", 1)[-1]]
        white = (np.sqrt(TN.CAMERA_SHAKE_SHARE) * shake
                 + np.sqrt(1.0 - TN.CAMERA_SHAKE_SHARE) * rng.normal(size=(n, 2)))
        drift = gaussian_filter1d(rng.normal(size=(n, 2)), sigma, axis=0, mode="reflect")
        drift /= max(float(drift.std()), 1e-12)
        out[:, i] += leg_px * (white * TN.JITTER[point] * noise.jitter_scale
                               + drift * TN.DRIFT[point] * noise.drift_scale)
    return out


def synthetic_raw(seq, camera=Camera(), noise=None, rng=None, name="synthetic"):
    """MediaPipe-format raw track of a 3D rep seen by ``camera``.

    ``noise``: a ``TrackerNoise`` (its knees-out term is ignored), or None
    for exact projections. Returns the dict ``mediapipe_pose.load_raw``
    would, plus ``camera`` and ``name``.
    """
    if not seq.is_3d:
        raise ValueError("needs a 3D rep (SquatStyle.stance and a Body3D)")
    points = _mediapipe_array(seq.landmarks3d, seq.body)
    image, depth = camera.project(points)
    if np.any(depth <= 0.1):
        raise ValueError("part of the body is behind or at the camera")
    _, rot = camera.pose()
    lateral = seq.landmarks3d["right_hip"] - seq.landmarks3d["left_hip"]
    facing_right = np.median(lateral @ -rot[2]) > 0      # the person's right side faces the camera
    visibility = np.empty(points.shape[:-1])
    for i, name in MEDIAPIPE_SOURCE.items():
        side = _side_of(name)
        near = side is None or (side == "right") == facing_right
        visibility[..., i] = NEAR_VISIBILITY if near else FAR_VISIBILITY
    if noise is not None:
        near_side = "right" if facing_right else "left"
        hip, knee, ankle = (image[0, [k for k, v in MEDIAPIPE_SOURCE.items() if v == f"{near_side}_{p}"][0]]
                            for p in ("hip", "knee", "ankle"))
        leg_px = float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))
        image = image_noise(image, seq.fps, leg_px, noise, rng or np.random.default_rng())
    mid_hip = 0.5 * (points[:, 23] + points[:, 24])
    world = (points - mid_hip[:, None, :]) @ rot.T       # MediaPipe world: x right, y down, z away, hip origin
    image_z = world[..., 2]        # MediaPipe's image z is also depth relative to the hips
    return {
        "image": np.concatenate([image, image_z[..., None], visibility[..., None]], axis=-1),
        "world": world,
        "presence": np.ones(points.shape[:-1]),
        "fps": np.float64(seq.fps),
        "width": np.int64(camera.width),
        "height": np.int64(camera.height),
        "camera": camera,
        "name": name,
    }
