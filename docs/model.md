# Squat model: how it works and where every number comes from

This describes the balance-constrained squat generator (`exercises/squat/balanced.py`) and everything it depends on. Conventions (axes, angle signs) are in [conventions.md](conventions.md). Results against published data are in [validation.md](validation.md).

## Pipeline

1. **Body.** A real person is drawn from ANSUR II (`core/sagittal_body.py`). Their measured standing heights (lateral malleolus, tibiale, trochanterion, C7, acromion) and limb lengths are converted to joint-centre positions with de Leva's landmark-to-joint offsets. Proportions are never randomised independently, so every body shape is one that was actually measured.
2. **Mass.** Each segment gets de Leva's sex-specific mass fraction and centre-of-mass location (`core/deleva.py`, `core/balance.py`). Left and right limbs overlap in the side view and are modelled as one limb with double mass.
3. **Style.** Each person gets a squat strategy drawn from published between-subject spreads (`exercises/squat/protocols.py`): depth, ankle use, where they keep their weight on the foot, arm position and tempo.
4. **Motion.** The knee follows a minimum-jerk curve in real seconds. The ankle follows the knee through a measured coupling. Arms and head follow posture rules.
5. **Balance.** For every frame, the trunk lean is **solved** so the whole-body centre of mass sits over the person's target point on the foot. Trunk lean is an output, so long thighs or stiff ankles automatically produce more forward lean.
6. **Check.** Joint limits are compared, and the true centre of pressure, including accelerations and segment rotation, is computed (`core/dynamics.py`) to confirm the person would actually stay on their feet.

## Sources

| What | Source | How it was obtained |
|---|---|---|
| Segment mass fractions, centre-of-mass locations, radii of gyration, joint-centre offsets | de Leva P. 1996, J Biomech 29(9):1223-1230, Tables 2 and 4 | Two independent secondary transcriptions that agree: BMClab/BMC `data/BSP_ZdeLeva.txt`, and samchon/automovie notes read from a scan of the paper. Mass fractions sum to 100.00% (men) and 99.99% (women). |
| Body dimensions | ANSUR II (Gordon et al. 2015, NATICK/TR-15/007), 4082 men and 1986 women | Public-domain CSVs from the Penn State OPEN Design Lab. Subset in `src/exercise_quality/data/`. |
| Movement smoothness | Flash and Hogan 1985, J Neurosci 5(7):1688-1703 | Minimum-jerk profile, plus a via-point version for reps with no pause at the bottom. |
| Depth, ankle use, ankle-knee coupling, weight position on the foot, tempo | Ishida et al. 2025, Orthop J Sports Med 13(3) | Full text via Europe PMC. Natural condition, 16 men. |
| Forward shift of the COM during descent, and its correlation with ankle motion (R = 0.398) | Kasahara et al. 2024, J Hum Kinet 93:29-39 | Full text via Europe PMC. |
| Maximal-depth squat ranges, arms-forward protocol | Zawadka et al. 2020, J Sports Sci Med 19:500-507 | Journal PDF. |
| Maximum knee flexion in squatting | Hemmerich et al. 2006, J Orthop Res 24:770-781 | **Abstract only.** |
| Heel position behind the ankle (21% of foot length) | 2024 Scientific Reports paper on shoe heel height, citing radiographic data | **Secondary only**, not checked against the original radiographs. |

## Assumptions and their status

"Data" means taken directly from a source above. "Fitted" means derived from published numbers by a stated calculation. "Assumption" means no data was used.

| Item | Value | Status |
|---|---|---|
| Joint centre heights from landmark heights | de Leva Table 2 offsets | Data |
| Ankle joint centre at lateral malleolus height | | Assumption (common convention) |
| Heel behind ankle | 21% of foot length, same for everyone | Secondary source |
| Upright standing is perfectly stacked (C7 directly above the hips, vertex above C7) | | Assumption. Real posture has C7 slightly behind the hips. |
| Trunk is one rigid segment from the hips to C7 | | **Assumption, and the main known weakness.** See validation.md. |
| Foot stays flat (no heel lift) | | Matches the "heels down" instruction in all three studies |
| Ankle excursion ratio (ankle / peak knee) | 0.278 ± 0.037 | Fitted (Ishida peak ankle / peak knee, SD / knee) |
| Ankle-knee coupling exponent | 1.27 ± 0.1 | Fitted mean (Ishida 60 and 90 deg points give 1.25 and 1.29). The SD is an assumption. |
| Weight position on the foot at the bottom | 36.3% ± 8% of foot length from heel | Data (Ishida natural COP) |
| COM shift from start to bottom | -5.0% ± 9.6% of foot length, correlated 0.398 with ankle | Data (Kasahara) |
| Standing posture | lean at the ankle until balanced, hips straight | Assumption (quiet standing is mostly ankle-controlled) |
| Head follows trunk lean | 30-70% | Assumption |
| Arms forward | parallel to the floor, elbows straight | Protocol (Zawadka) |
| Arms crossed | 20 deg shoulder, 150 deg elbow | Assumption. A side-view projection of a 3D posture, chosen to put arm mass against the chest. |
| Knee limit | 157 ± 6 deg | Data (Hemmerich abstract) |
| Ankle and hip limits | 45 deg and 150 deg caps | Assumption (catches impossible poses only) |
| Tempo | 2 s down, 2 s up | Protocol (Ishida, Kasahara). The ±25% jitter on the Zawadka protocol is an assumption. |
| Ascent mirrors descent | | Assumption, no ascent data used |

## Known limitations

- **Rigid trunk.** Real squats involve 20-45 deg of pelvic tilt and lumbar flexion (Zawadka 2020). The model folds all of it into one hip angle. This is the most likely cause of the under-leaning at the bottom and the too-small centre-of-mass drop seen in validation.
- **Trunk lean varies too much between simulated people** (SD about 13 deg vs about 7 deg measured). Most of it comes from variation in ankle use. In real people, ankle use and other compensations evidently cancel more than in this model. The cause is not yet known; see validation.md.
- **Side view only.** No knee valgus, stance width, foot turn-out or left-right asymmetry.
- **Quasi-static balance.** Each frame is balanced as if still. Dynamics are checked afterwards, not used to plan the motion. This is fine at normal tempos (centre of pressure within about 5 mm of the plan at 2 s per phase) but not for fast or bouncy reps.
- **Populations.** ANSUR II is US Army personnel. de Leva is young Russian PE students. The squat studies are young, healthy, mostly Japanese and Polish adults. Older, heavier, injured or very mobile people are not represented.
- **No muscles or joint torques**, so the model says nothing about effort, strength limits or loading.
