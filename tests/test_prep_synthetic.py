"""Smoke-test the prep logic on tiny fake tables (no downloads needed).
Run:  python tests/test_prep_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prep_brfss  # noqa: E402
import prep_nhanes  # noqa: E402
import prep_vertebral  # noqa: E402

nan = np.nan

# ---------------- NHANES ----------------
demo = pd.DataFrame({"SEQN": [1, 2, 3, 4], "RIDAGEYR": [25, 33, 60, 41],
                     "RIAGENDR": [1, 2, 2, 1], "RIDRETH1": [3, 4, 1, 3],
                     "DMDEDUC2": [4, 9, 3, 5], "INDFMPIR": [2.1, 0.8, 5, nan],
                     "WTINT2YR": [1e4] * 4, "WTMEC2YR": [1e4] * 4})
arq = pd.DataFrame({
    "SEQN": [1, 2, 3, 4], "ARQ010": [1, 2, 1, 1],
    # person 1: low back pain, onset 22, >=3 mo, gradual, stiff 1-4h, better w/ exercise -> IBP
    # person 4: low back pain, onset 38, >=3 mo, sudden, no stiffness, not better -> not IBP
    "ARQ020D": [4, nan, 4, 4], "ARQ022D": [22, nan, 50, 38], "ARQ024D": [1, nan, 1, 1],
    "ARQ025D": [5, nan, 5, 1], "ARQ040": [4, nan, 4, 6], "ARQ080": [1, nan, 1, 2],
    "ARQ070": [1, nan, 2, 2], "ARQ030A": [1, nan, 2, 2], "ARQ110": [2, 2, 2, 1],
    "ARD125A": [2, 2, 2, 2], "ARQ125C": [2, 2, 2, 2], "ARQ125D": [2, 2, 2, 2],
    "ARQ125E": [1, 2, 2, 2],
})
mcq = pd.DataFrame({"SEQN": [1, 2, 3, 4], "MCQ160A": [1, 2, 1, 9]})
bmx = pd.DataFrame({"SEQN": [1, 2, 3, 4], "BMXBMI": [24, 31, 28, 22], "BMXWAIST": [80, 100, 95, 78]})

n = prep_nhanes.build({"DEMO_F": demo, "ARQ_F": arq, "MCQ_F": mcq, "BMX_F": bmx})
print(n[["SEQN", "age", "arthritis_dx", "ibp", "pain_low_back", "morning_stiffness_cat"]])
assert list(n["SEQN"]) == [1, 2, 4], "age filter 20-44 failed"
assert n.set_index("SEQN").loc[1, "ibp"] == 1
assert n.set_index("SEQN").loc[4, "ibp"] == 0
assert n.set_index("SEQN").loc[2, "ibp"] == 0  # no chronic pain
assert np.isnan(n.set_index("SEQN").loc[4, "arthritis_dx"])  # 9 = don't know
assert n.set_index("SEQN").loc[4, "morning_stiffness_cat"] == 0  # code 6 = none
assert not set(prep_nhanes.FEATURES_IBP) & set(prep_nhanes.BACK_SYMPTOMS), "IBP leakage"
print("NHANES ok\n")

# ---------------- BRFSS ----------------
raw = pd.DataFrame({
    "HAVARTH4": [1, 2, 7, 2], "_AGEG5YR": [5, 1, 3, 2], "SEXVAR": [2, 1, 1, 2],
    "_BMI5": [3150, 2210, nan, 2600], "PHYSHLTH": [88, 5, 77, 30],
    "MENTHLTH": [2, 88, 99, 88], "DIABETE4": [3, 1, 4, 2], "_RFDRHV9": [1, 2, 9, 1],
    "_LLCPWT": [100, 200, 300, 400],
})
b = prep_brfss.build(raw)
print(b[["arthritis_dx", "bmi", "phys_unhealthy_days", "ment_unhealthy_days", "diabetes", "heavy_drinker"]])
assert len(b) == 3  # row with HAVARTH4=7 dropped
assert b.loc[0, "bmi"] == 31.5 and b.loc[0, "phys_unhealthy_days"] == 0
assert b.loc[1, "heavy_drinker"] == 1 and b.loc[1, "diabetes"] == 1
assert b.loc[2, "phys_unhealthy_days"] == 30
print("BRFSS ok\n")

# ---------------- Vertebral ----------------
v = prep_vertebral.build(pd.DataFrame(
    [[63.0, 22.5, 39.6, 40.5, 98.7, -0.25, "DH"], [50.0, 10.0, 40.0, 40.0, 120.0, 0.1, "NO"]],
    columns=prep_vertebral.FEATURES + ["label"]))
assert list(v["abnormal"]) == [1, 0]
print("Vertebral ok\n\nall synthetic tests passed")
