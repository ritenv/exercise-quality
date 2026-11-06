# Data

## ansur_ii_sagittal.csv

A subset of the ANSUR II public working databases (4082 men, 1986 women), with only the measurements the side-view body model needs. Lengths are converted to metres and mass to kilograms.

- Source: Gordon CC, Blackwell CL, Bradtmiller B, et al. 2015. *2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary Statistics*. NATICK/TR-15/007.
- Distributed by the Penn State OPEN Design Lab: https://www.openlab.psu.edu/ansur2/
- Licence: US government work, public domain.
- Regenerate with `scripts/build_ansur_subset.py` (instructions are in that file).

Columns: `subject_id`, `sex`, `age_years`, `stature_m`, `lateral_malleolus_height_m`, `tibiale_height_m`, `trochanterion_height_m`, `omphalion_height_m`, `cervicale_height_m`, `acromion_height_m`, `acromion_radiale_length_m`, `radiale_stylion_length_m`, `hand_length_m`, `ball_of_foot_length_m`, `foot_length_m`, `mass_kg`.

Caveats:

- These are US Army personnel. They are younger, leaner and fitter than the general population, and the data say nothing about older adults or children.
- The OPEN Design Lab notes that the raw `weightkg` column is actually in hectograms. The build script divides by 10.
- The last two raw columns (`Heightin`, `Weightlbs`) are probably self-reported and are not used.
- Men and women should be analysed separately. The model keeps them separate and uses the matching de Leva mass table for each.
