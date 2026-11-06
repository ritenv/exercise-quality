"""Run MediaPipe Pose on a video and save the landmark tracks.

Usage:
    PYTHONPATH=src python scripts/extract_video_landmarks.py VIDEO [VIDEO ...]

Writes ``src/exercise_quality/data/real_video/<name>.npz`` and updates
``manifest.json`` next to it. The video itself is never copied: the
manifest records its file name and SHA-256 so the tracks can be traced
back to it.

Each .npz holds
    image:  (frames, 33, 4) x and y in pixels, MediaPipe's relative depth z
            (same scale as x), and visibility (0..1)
    world:  (frames, 33, 3) MediaPipe's metric 3D estimate, metres, origin
            between the hips (NaN where no person was found)
    presence: (frames, 33) landmark presence (0..1)
    fps, width, height
Frames where no person was detected are NaN.

The model file is downloaded to ``models/`` on first use (not in git).
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
MODEL_PATH = ROOT / "models" / "pose_landmarker_heavy.task"
OUT_DIR = ROOT / "src" / "exercise_quality" / "data" / "real_video"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def extract(video_path):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    image_rows, world_rows, presence_rows = [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(round(index * 1000.0 / fps)))
            if result.pose_landmarks:
                points = result.pose_landmarks[0]
                image_rows.append([[p.x * width, p.y * height, p.z * width, p.visibility] for p in points])
                presence_rows.append([p.presence for p in points])
                world_rows.append([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
            else:
                image_rows.append(np.full((33, 4), np.nan))
                presence_rows.append(np.full(33, np.nan))
                world_rows.append(np.full((33, 3), np.nan))
            index += 1
    capture.release()
    return {
        "image": np.asarray(image_rows, dtype=np.float32),
        "world": np.asarray(world_rows, dtype=np.float32),
        "presence": np.asarray(presence_rows, dtype=np.float32),
        "fps": np.float64(fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for video in args.videos:
        tracks = extract(video)
        name = video.stem
        np.savez_compressed(OUT_DIR / f"{name}.npz", **tracks)
        missing = int(np.isnan(tracks["image"][:, 0, 0]).sum())
        entry = manifest.get(name, {})
        entry.update({
            "video_file": video.name,
            "video_sha256": sha256(video),
            "frames": int(len(tracks["image"])),
            "frames_without_person": missing,
            "fps": float(tracks["fps"]),
            "width": int(tracks["width"]),
            "height": int(tracks["height"]),
            "mediapipe_version": mp.__version__,
            "model": MODEL_PATH.name,
            "model_sha256": sha256(MODEL_PATH),
        })
        manifest[name] = entry
        print(f"{name}: {entry['frames']} frames, {missing} without a person")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
