# Real-video check

Does a generated squat look like a real one once both are seen through a pose tracker? This compares two real videos, tracked with MediaPipe Pose, against synthetic reps of matched people, and measures the tracker's own noise.

Reproduce with:

```bash
PYTHONPATH=src python scripts/extract_video_landmarks.py VIDEO...          # only when adding videos
PYTHONPATH=src python scripts/real_video_check.py 60 --figures docs/figures  # about 20 s
```

## Data

| Clip | View | Person (apparent) | Arms | Reps | Notes |
|---|---|---|---|---|---|
| squat_1 | side-on (yaw scale 0.997); camera pans and zooms a little | woman, trainers | hands behind the head | 6 | trainer demo; the last rep is a "sit back" squat with vertical shins |
| squat_2 | about 48 deg from side-on (yaw scale 0.67) | man, barefoot | hands clasped at the chest | 13 | trainer demo; starts with a setup bend, which is correctly not counted as a rep |

Both are third-party videos. Only the MediaPipe landmark tracks are in the repository (`src/exercise_quality/data/real_video/`), with each video's SHA-256 in `manifest.json`. Sex is an assumption from appearance and is used only to pick which ANSUR II sample to match. MediaPipe Pose Landmarker heavy, VIDEO mode, MediaPipe 1.2.2.

## Method

1. **Tracker view.** Real tracks and synthetic reps are both reduced to the points MediaPipe gives on the near side (ear, shoulder, elbow, wrist, hip, knee, ankle, heel, toe), facing +x. They're then measured with the same code (`core/observed.py`, `exercises/squat/reps.py`), using a 6 Hz zero-lag low-pass filter. Trunk lean here is the hip-to-shoulder line, not the rib cage.
2. **Camera angle.** The yaw comes from MediaPipe's 3D landmarks: how far the hip axis is turned from the camera axis. Image x is divided by its cosine. Two alternatives were tested and rejected:
   - Fitting the thigh's swing ellipse in the image gave 0.93 on the side-on clip. The knees travel out over the toes, and that looks like yaw.
   - Taking angles straight from the 3D landmarks put the side-on clip's knee and shank 5-7 deg off the image values.
3. **Reps.** A rep is a dip in hip-above-ankle height of at least 15% with at least 45 deg of knee bend. Phases are measured at 5% and 95% of each rep's excursion.
4. **Matched synthetic reps.** The people are the 40 ANSUR II people closest in thigh and trunk proportions. Foot and neck ratios aren't used: MediaPipe's "foot index" isn't the toe tip, and the ear is placed by assumption. Each real rep gets 60 synthetic reps with:
   - the same peak knee bend, tempo (calibrated on the measured phases) and arm angles;
   - everything else sampled from the validated population spreads, including a stance (feet width and toe-out, [model3d.md](model3d.md));
   - done in 3D and filmed by a virtual phone at the clip's angle from side-on (`mediapipe_pose.side_view_offset`), at a sampled distance and height (`camera.sample_side_camera`), then read by the same pipeline as the real clip;
   - tracker jitter and drift added (below).
5. **Balance check.** The centre of mass over the foot is estimated from the tracked pose with the matched body (`core/observed_balance.py`). It assumes a flat foot and a straight trunk along the hip-shoulder line. The thigh and shank take the length they appear to have in that frame. When the knees travel out over the toes, the thigh looks shorter from the side, and rebuilding the pose with its full length would put the hip, and the centre of mass, about 0.1 foot lengths too far back. On exact synthetic reps the estimate is off by 0.006 ± 0.010 foot lengths; filmed by sampled phones with tracker noise, 95% of errors against the true 3D centre of mass are within 0.094.

## Results

### squat_1 (side-on): tempo, shin and balance match; the thigh reads deeper and the trunk leans more

| Measure (median over 6 reps) | Real | Synthetic | Real rep's percentile among matched synthetic |
|---|---|---|---|
| Thigh angle at the bottom (-90 = parallel) | -82.9 | -74.7 | 4 |
| Shin lean at the bottom | 28.3 | 32.3 | 11 |
| Hip drop (fraction of standing height) | 0.52 | 0.47 | 100 |
| **Trunk lean at the bottom** | **43.6** | **28.8** | **100 (6 of 6 reps above 95)** |
| Trunk lean at 60 / 90 deg knee | 20.8 / 31.8 | 21.2 / 27.8 | 51 / 82 |
| Hip travel back (fraction of leg) | 0.16 | 0.19 | 25 |
| Centre of mass over the foot at the bottom | 0.40 | 0.35 | 64 |
| Trunk lean up minus down at the same hip height | 4.1 | 1.0 | 93 (2 of 6 above 95) |
| Descent / ascent (s) | 0.80 / 0.70 | 0.71 / 0.60 | 58 / 67 |

The camera is 4 deg towards her front (from the image, `side_view_offset`).

![squat_1](figures/real_check_squat_1.png)

- **Tempo, balance and trunk lean up to 90 deg of knee bend match.** Her bottom position sits over mid-foot: 0.40-0.46 of foot length in reps 1-4, against 0.363 ± 0.08 in Ishida's force-plate data. So the tracked angles are physically consistent rather than tracker artefacts.
- **The trunk curve has a different shape.** The real lean keeps rising to the bottom; the model's flattens past 90 deg, so it ends 15 deg lower.
- **Her thigh reads deeper and her shin more upright than any camera height explains.** Matched synthetic reps filmed with the lens at a fixed height (4 reps, 40 synthetic each):

  | Lens height | Thigh at the bottom | Shin lean |
  |---|---|---|
  | 0.4 m | -78.5 | 31.7 |
  | 0.9 m | -75.4 | 32.6 |
  | 1.4 m | -72.9 | 33.9 |
  | real | -82.2 | 28.9 |

  Before this check moved to 3D, the synthetic reps were a perfect side view of a single-plane squat, plus a fitted "knees out 24 deg" noise term, and gave -80.0 (percentile 24). The 3D camera moves the synthetic thigh higher because the knees travel towards the lens. Floor perspective (the far foot higher in the image) suggests her phone was about 1.5 m up, which fits a handheld camera that pans. That would make the gap larger, not smaller.

  Possible causes:
  - her knees travel out less than the sampled stance gives;
  - MediaPipe places a side-view knee differently from the projected 3D knee;
  - the floor-perspective estimate is biased.

  One clip can't separate them. The tracked thigh shrinks the same way: 0.90 of its standing length at 100 deg of knee bend in the real clip, 0.94 in the filmed synthetic reps. Nothing was refitted.
- The sixth rep (vertical shins, centre of mass near the heel) is far outside the synthetic spread on shin lean (percentile 0), as a deliberate "sit back" demonstration should be.

### squat_2 (oblique): unreliable angles, but the trunk still leans far more

- **The synthetic reps are now filmed from the same angle** (48 deg towards his front), so both sides carry the same projection errors.
- **Even standing, the corrected pose puts the centre of mass at 0.76 of foot length.** People stand at 0.31-0.58 (Fujiwara 2010). No yaw value from 0.67 to 1.0 brings both standing and bottom into a plausible range, so the oblique geometry is distorted and its angles shouldn't be trusted one by one.
- **Filmed synthetic reps show why.** At 45 deg the yaw-corrected trunk reads 10 deg less than the truth ([model3d.md](model3d.md), camera placement). The shoulders sit further out from the body's mid-plane than MediaPipe's hip points do, and that sideways gap projects into the image.
- **Like for like, he leans far more than the model:** 59.7 deg at the bottom against 12.6 in the filmed synthetic reps (percentile 100 in every rep); 48.8 even uncorrected. This man is a hip-dominant squatter (hip travel back 0.37 against 0.17).
- Tempo matches (descent 1.2 s, ascent 0.4 s).

### Tracker noise model (`core/tracker_noise.py`)

Measured on the side-on clip and reproduced on synthetic reps:

| Quantity | Real | Model | |
|---|---|---|---|
| Jitter above 6 Hz, hip / knee / ankle (fraction of leg length) | 0.0032 / 0.0019 / 0.0017 | 0.0029 / 0.0018 / 0.0015 | fitted |
| Angle jitter, knee / thigh / shin / trunk (deg) | 0.52 / 0.41 / 0.20 / 0.25 | 0.45 / 0.34 / 0.21 / 0.28 | not fitted |
| Thigh length at 100 deg knee / standing | 0.90 | 0.94 | not fitted: 3D stance and filming |
| Shin, trunk length at 100 deg knee / standing | 0.98, 1.01 | 0.99, 0.98 | not fitted |

The model has two parts:
- **Jitter.** Nearly white noise, 0.2-1% of leg length. 40% of it is shared by all points, which is camera shake: that share is about 0.1 on the fixed-camera clip.
- **Drift.** Each landmark wanders by 1.8% of its segment's length, with a half-correlation time of 0.15 s.

**The knees-out projection is no longer a noise term.** It used to be modelled as a fitted 24 deg sideways thigh angle. Now the 3D model gives it from the stance, and the virtual phone's perspective adds to it. From a perfect side view the thigh shortens to 0.909 (real 0.90). Filmed by phones at sampled heights it shortens to 0.94, because the knee nearer the lens also looks bigger.

All of it comes from one person in one clip; the spreads are assumptions.

## What this tells us about the model

1. **Trunk lean late in the descent is probably too low.** Three independent hints now point the same way:
   - squat_1 (side-on, balanced) leans 15 deg more at the bottom;
   - squat_2 leans at least 19 deg more;
   - Zawadka's lumbar and hip ranges imply about 12 deg more rib-cage rotation at maximal depth than the model gives (gap 1 in [validation.md](validation.md)).

   The cause is the preferred-trunk curve. It's fitted to Ishida's 60 deg, 90 deg and peak values at a 2 s tempo, and it bends over early. Two trainer demos at 0.8-1.2 s descents aren't a population, and MediaPipe's hip point could drift at depth (not measurable here). So **nothing was refitted**. What's needed is more side-on videos, ideally of untrained people at a controlled tempo, or a study reporting trunk lean through the whole descent.
2. **Hip placement is a style, and styles vary more than the model allows.** squat_1 keeps the hips forward and leans; squat_2 sends them far back. The model's hip travel spread is narrow because hip position falls out of balance plus the preferred lean.
3. **Real people rise slightly hips-first; the model doesn't.** At the same hip height, both real people lean more on the way up than on the way down (4.1 and 3.0 deg against 1.3 and 1.1 in their matched synthetic reps, 96th and 93rd percentiles). The generator's ascent is the time-reverse of its descent, an assumption with no ascent data behind it (model.md). Tracker lag that differs between landmarks would add the same signature, so this needs a reference measurement before changing the model.
4. Tempo and bottom balance match on the side-on clip. Her thigh reads 8 deg deeper and her shin 4 deg more upright than the filmed synthetic reps, whatever the camera height (above).

## Limits of this check

- Two people, both trainers, both demonstrating. Not a sample of anyone.
- No reference system: MediaPipe landmark bias against true joint centres can't be measured, only its noise.
- One clip is oblique; the yaw correction and the noise model's spreads are calibrated on a single side-on clip.
- The phone's distance and height in each clip are unknown and are sampled; the lens height from floor perspective is calibrated only on virtual-camera reps.

Add side-on videos (full body in frame, phone at knee height, 3.5-5 m away, within 15 deg of side-on) to `scripts/extract_video_landmarks.py`, and the check picks them up automatically.
