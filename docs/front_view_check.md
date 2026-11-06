# Front-view check

This compares five real front-view squat videos with matched 3D synthetic reps, filmed by the virtual camera ([model3d.md](model3d.md)) and measured by the same code (`video/front_view.py`).

```bash
PYTHONPATH=src python scripts/extract_video_landmarks.py VIDEO...   # only when adding videos
PYTHONPATH=src python scripts/front_view_check.py 16                # about 2 minutes
```

## Data

| Clip | Person (apparent) | View | Reps used | Notes |
|---|---|---|---|---|
| front_squat_1 | woman | square | 7 | trainer video (screen recording), wide toes-out stance, hands at chest |
| front_squat_2 | woman | square | 5 | trainer video with a rep counter, wide stance |
| front_squat_3 | woman | about 22° off (mild side angle) | 6 | reps 102-104 of a high-rep challenge; probably the same person as clip 1 |
| front_squat_4 | man | front for 5.5 s, then the camera orbits to his left | 3 | only the front-facing part is used |
| front_squat_5 | man (the repository owner) | square, phone low and tilted up, portrait | 9 | squat press-outs with a 2.5 kg plate, treated as bodyweight; toes below the frame edge |

- **Storage.** Only MediaPipe landmark tracks are stored. `manifest.json` has each video's SHA-256.
- **Frame times.** Clips 1-4 are screen recordings with irregular frame times (gaps of 8-58 ms). Extraction now stores each frame's timestamp, and loading resamples to an even 30 fps (`mediapipe_pose.resample_uniform`).

## What is measured

Everything is measured in the image, as scale-free ratios:
- **Stance:** ankle separation over leg height.
- **Depth:** hip drop.
- **Knee-to-ankle separation ratio (KASR)** at the bottom and half way down. Above 1 means the knees are outside the ankles.
- **Knee frontal-plane projection angle (FPPA)** on each side, half way down. Positive means valgus.
- **Hip shift, pelvic tilt and left-right depth difference** at the bottom.

**Matching.** For each clip, 16 synthetic people of the same apparent sex get the clip's stance width, toe-out, depth, tempo and hands-at-chest arms. They're filmed by the virtual camera with tracker noise.

**One fitted parameter, knee tracking** (`Stance.knee_track`). It's how far the knee's path turns inward from the toe line: 0 means the knees track exactly over the toes, as the model assumes; negative means further out than the toes point. It's fitted per synthetic person to the clip's knee separation at the bottom. The other measures are then predictions.

## Two things MediaPipe gets wrong from the front

1. **The camera angle from its 3D hips.**
   - In clips 1, 2 and 5 the hip axis reads 13-15° off square, yet the two sides look the same in the image.
   - The left-right difference in the knee angle is a better guide: on synthetic reps it grows by about 3° per degree of camera angle.
   - Solved that way, clips 1, 2 and 5 are 2-3° off square, clip 3 is 22° (MediaPipe said 6°), and clip 4 is 13° (MediaPipe said 0°, as the camera starts to orbit).
2. **Foot directions from its 3D landmarks.**
   - In every clip the right foot reads 25-35° more turned out than the left, including clips that look symmetric.
   - The mean of the two feet (20-28°) is plausible and is used.
   - In clip 5 the toes are out of frame, so the population mean (24°, Sinclair 2022) is used instead.

**MediaPipe's hip points are not hip joint centres.** Its shoulder-over-hip width in the standing frames was 1.66-1.75 (1.47 in clip 5), against 2.3-2.6 for acromion width over hip-joint-centre width. So its hip points sit further out. The virtual camera now places them 35% of the way from the joint centre to the skin at hip-breadth level (`camera.MEDIAPIPE_HIP_OUTWARD`; fitted here: women 0.30, men 0.42).

## Results

| Clip | Camera angle | Stance (× hip breadth) | Knee tracking that matches | KASR at the bottom, real / knees over toes / fitted | FPPA right, left (half way), real / fitted |
|---|---|---|---|---|---|
| 1 | +2° | 1.46 | **-13.4°** (-15.0 to -11.3) | 1.18 / 0.87 / 1.18 | -48, -53 / -46, -51 |
| 2 | +2° | 1.53 | **-13.9°** (-15.9 to -11.9) | 1.15 / 0.82 / 1.15 | -49, -54 / -52, -58 |
| 3 | -22° | 1.63 | **-3.1°** (-5.0 to -1.0) | 1.07 / 1.01 / 1.07 | -70, -13 / -67, -11 |
| 4 | -13° | 1.36 | **-8.9°** (-11.8 to -6.6) | 1.66 / 1.37 / 1.66 | -76, -42 / -79, -48 |
| 5 | +3° | 1.38 | **-5.6°** (-7.7 to -3.4) | 1.43 / 1.26 / 1.42 | -50, -58 / -48, -56 |

The camera angle is solved from image symmetry. Knee tracking is the median over 16 matched people, with 5-95% in brackets.

- **Every person's knees travel further out than their toes point:** by 3-14°.
  - The model's default (knees exactly over the toes) puts the knees too close together by 0.1-0.35 of ankle width at the bottom.
  - The three trainer clips (1, 2, 4) push out most. That fits the common coaching cue "knees out".
- **One fitted number per clip reproduces the other measures.**
  - The knee angle on both sides half way down matches within 1-6° in every clip.
  - Knee separation half way down matches within 0.05 (0.07 for clip 5).
- **Hip shift, pelvic tilt and depth difference are small in the square-on clips** (|shift| ≤ 0.04 of ankle width, tilt within ±1.5°) and inside the synthetic spread. In the angled clips (3, 4) the view makes them unreliable. Clip 3's 3.6° pelvic tilt and 0.13 depth difference lie outside the matched synthetic spread at the same angle. That could be real asymmetry, or the view being modelled imperfectly, and one clip can't tell which.
- **These people squat wide:** 1.36-1.63 × hip breadth, against Sinclair's lifters at 1.23.

## What this means for the model

- **Knee tracking needs a spread.** The default should be knees somewhat outside the toe line rather than exactly over it. The five clips give -3 to -14°, four of them trainers. **Nothing was refitted.** Five people, four of them demonstrating technique, aren't a population, and the result depends on toe-out, which MediaPipe can't measure well from the front. More videos of everyday squatters would settle the spread.
- **A front-view rubric can't judge "knees in line with the toes" from MediaPipe alone,** because the toe direction is unreliable. It can reliably measure:
  - the knee's outward path relative to straight ahead (toe-out minus tracking: 25-40° in these clips);
  - knees moving inside the ankles (KASR below 1);
  - left-right differences, but only when the camera is within a few degrees of square, which image symmetry can check.

## Limits

- **People:** five, two of them probably the same person, four of them trainers demonstrating. Two clips are not square to the camera.
- **No reference measurement.** MediaPipe's front-view errors are inferred from symmetry, not measured against motion capture.
- **Camera placement is estimated:** distance and height in clip 5 are judged from the framing.
- **Tracker noise** in the synthetic reps comes from the side-view clip.
