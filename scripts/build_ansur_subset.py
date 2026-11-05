"""Build the small ANSUR II subset used by the sagittal body model.

The raw ANSUR II public files are not committed. Download them from the
Penn State OPEN Design Lab (https://www.openlab.psu.edu/ansur2/):

    https://tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv
    https://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv

then run:

    python scripts/build_ansur_subset.py MALE.csv FEMALE.csv

The output keeps only the measurements needed to place joint centres in
a side-view model, converted to metres and kilograms. Rows whose heights
are not ordered the way a standing human must be (ankle below knee below
hip and so on) are dropped and counted.

Units in the raw files: millimetres, except body mass, which is stored in
hectograms even though the column is called "weightkg".
"""

import csv
import sys
from pathlib import Path

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "src" / "exercise_quality" / "data" / "ansur_ii_sagittal.csv"
)

# raw column -> output column (all lengths converted mm -> m)
LENGTH_COLUMNS = {
    "stature": "stature_m",
    "lateralmalleolusheight": "lateral_malleolus_height_m",
    "tibialheight": "tibiale_height_m",
    "trochanterionheight": "trochanterion_height_m",
    "cervicaleheight": "cervicale_height_m",
    "acromialheight": "acromion_height_m",
    "acromionradialelength": "acromion_radiale_length_m",
    "radialestylionlength": "radiale_stylion_length_m",
    "handlength": "hand_length_m",
    "footlength": "foot_length_m",
}

FIELDNAMES = ["subject_id", "sex", "age_years", *LENGTH_COLUMNS.values(), "mass_kg"]


def heights_are_ordered(row):
    return (
        0
        < row["lateral_malleolus_height_m"]
        < row["tibiale_height_m"]
        < row["trochanterion_height_m"]
        < row["acromion_height_m"]
        < row["cervicale_height_m"]
        < row["stature_m"]
    )


def read_rows(path, sex):
    # utf-8-sig strips the byte order mark some of the files start with
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        # header capitalisation differs between files (subjectid vs SubjectId)
        lookup = {name.lower(): name for name in reader.fieldnames}
        for raw in reader:
            row = {
                "subject_id": raw[lookup["subjectid"]],
                "sex": sex,
                "age_years": int(raw[lookup["age"]]),
                "mass_kg": float(raw[lookup["weightkg"]]) / 10.0,
            }
            for source, target in LENGTH_COLUMNS.items():
                row[target] = float(raw[lookup[source]]) / 1000.0
            yield row


def main(male_path, female_path):
    kept, dropped = [], {"male": 0, "female": 0}
    for path, sex in [(male_path, "male"), (female_path, "female")]:
        for row in read_rows(path, sex):
            if heights_are_ordered(row) and all(
                row[c] > 0 for c in LENGTH_COLUMNS.values()
            ):
                kept.append(row)
            else:
                dropped[sex] += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in kept:
            writer.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })

    counts = {s: sum(r["sex"] == s for r in kept) for s in ("male", "female")}
    print(f"Wrote {len(kept)} rows to {OUTPUT}")
    print(f"Kept: {counts}  Dropped (inconsistent heights): {dropped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
