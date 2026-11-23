"""
Build the BRFSS modelling table for young adults (18-44).

Target: arthritis_dx = HAVARTH4 == 1
  ("Has a doctor ... ever told you that you have some form of arthritis,
   rheumatoid arthritis, gout, lupus, or fibromyalgia?")

Leakage note: do NOT use ARTHEXER or any other item from the arthritis
module - those are only asked of people who already said yes. DIFFWALK
(difficulty walking) is kept in OPTIONAL_FEATURES because it is often a
*consequence* of arthritis; try models with and without it.

The raw file is ~1 GB, so it is read in chunks and only the needed columns
are kept.

Usage:
    python src/prep_brfss.py              # 2024
    python src/prep_brfss.py --year 2025
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "brfss"
OUT = ROOT / "data" / "processed"

# _AGEG5YR: 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44
YOUNG_AGE_GROUPS = [1, 2, 3, 4, 5]

RAW_COLS = [
    "HAVARTH4", "_AGEG5YR", "SEXVAR", "_RACE", "_EDUCAG", "_INCOMG1", "EMPLOY1",
    "MARITAL", "_BMI5", "_SMOKER3", "_TOTINDA", "GENHLTH", "PHYSHLTH", "MENTHLTH",
    "DIABETE4", "ADDEPEV3", "ASTHMA3", "CHCCOPD3", "CVDINFR4", "CHCKDNY2",
    "DIFFWALK", "_LLCPWT", "_STSTR", "_PSU", "_STATE",
]
HEAVY_DRINK_PREFIX = "_RFDRHV"  # suffix changes by year (_RFDRHV9 in 2024)

NUMERIC_FEATURES = ["bmi", "phys_unhealthy_days", "ment_unhealthy_days"]
ORDINAL_FEATURES = ["age_group", "education", "income", "general_health"]
BINARY_FEATURES = ["female", "physically_active", "diabetes", "depression",
                   "asthma", "copd", "heart_attack", "kidney_disease", "heavy_drinker"]
CATEGORICAL_FEATURES = ["race", "employment", "marital", "smoker_status"]
OPTIONAL_FEATURES = ["difficulty_walking"]  # possible reverse causation
FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def yes_no(s: pd.Series) -> pd.Series:
    """BRFSS 1=Yes, 2=No, 7/9=don't know/refused -> 1/0/NaN."""
    return s.map({1: 1.0, 2: 0.0})


def keep(s: pd.Series, valid) -> pd.Series:
    return s.where(s.isin(valid))


def days(s: pd.Series) -> pd.Series:
    """PHYSHLTH/MENTHLTH: 1-30 days, 88 = none, 77/99 = unknown."""
    return s.where(s <= 30).fillna(s.map({88: 0.0}))


def build(raw: pd.DataFrame) -> pd.DataFrame:
    c = lambda name: raw[name] if name in raw else pd.Series(np.nan, index=raw.index)  # noqa: E731
    drink_col = next((x for x in raw.columns if x.startswith(HEAVY_DRINK_PREFIX)), None)

    out = pd.DataFrame(index=raw.index)
    out["arthritis_dx"] = yes_no(c("HAVARTH4"))
    out["age_group"] = c("_AGEG5YR")
    out["female"] = c("SEXVAR").map({1: 0.0, 2: 1.0})
    out["race"] = keep(c("_RACE"), range(1, 9))
    out["education"] = keep(c("_EDUCAG"), range(1, 5))
    out["income"] = keep(c("_INCOMG1"), range(1, 8))
    out["employment"] = keep(c("EMPLOY1"), range(1, 9))
    out["marital"] = keep(c("MARITAL"), range(1, 7))
    out["bmi"] = c("_BMI5") / 100.0  # stored with 2 implied decimals
    out["smoker_status"] = keep(c("_SMOKER3"), range(1, 5))
    out["physically_active"] = c("_TOTINDA").map({1: 1.0, 2: 0.0})
    out["general_health"] = keep(c("GENHLTH"), range(1, 6))
    out["phys_unhealthy_days"] = days(c("PHYSHLTH"))
    out["ment_unhealthy_days"] = days(c("MENTHLTH"))
    # DIABETE4: 1 yes, 2 yes-only-pregnancy, 3 no, 4 prediabetes
    out["diabetes"] = c("DIABETE4").map({1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0})
    out["depression"] = yes_no(c("ADDEPEV3"))
    out["asthma"] = yes_no(c("ASTHMA3"))
    out["copd"] = yes_no(c("CHCCOPD3"))
    out["heart_attack"] = yes_no(c("CVDINFR4"))
    out["kidney_disease"] = yes_no(c("CHCKDNY2"))
    out["heavy_drinker"] = (raw[drink_col].map({1: 0.0, 2: 1.0})
                            if drink_col else np.nan)
    out["difficulty_walking"] = yes_no(c("DIFFWALK"))
    # survey design (keep for weighted prevalence; optional sample_weight in sklearn)
    out["weight"] = c("_LLCPWT")
    out["strata"] = c("_STSTR")
    out["psu"] = c("_PSU")
    out["state"] = c("_STATE")
    return out.dropna(subset=["arthritis_dx"]).reset_index(drop=True)


def read_young(path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    parts = []
    reader = pd.read_sas(path, format="xport", chunksize=chunksize)
    for i, chunk in enumerate(reader):
        chunk.columns = [x.upper() for x in chunk.columns]
        cols = [x for x in RAW_COLS if x in chunk.columns]
        cols += [x for x in chunk.columns if x.startswith(HEAVY_DRINK_PREFIX)]
        chunk = chunk[cols]
        parts.append(chunk[chunk["_AGEG5YR"].isin(YOUNG_AGE_GROUPS)])
        print(f"\r  read {(i + 1) * chunksize:,} rows", end="")
    print()
    missing = set(RAW_COLS) - set(parts[0].columns)
    if missing:
        print(f"  warning: not in this year's file: {sorted(missing)}")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    a = p.parse_args()
    raw = read_young(RAW / f"LLCP{a.year}.xpt")
    df = build(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"brfss{a.year}_young.csv"
    df.to_csv(dest, index=False)

    rate = df["arthritis_dx"].mean()
    wrate = np.average(df["arthritis_dx"], weights=df["weight"].fillna(0))
    print(f"rows (18-44, answered HAVARTH4): {len(df):,}")
    print(f"arthritis positives: {int(df['arthritis_dx'].sum()):,}  "
          f"unweighted rate={rate:.1%}  survey-weighted rate={wrate:.1%}")
    print(df.groupby("age_group")["arthritis_dx"].mean().rename("rate by age group").to_string())
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
