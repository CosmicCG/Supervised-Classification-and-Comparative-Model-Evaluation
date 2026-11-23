"""
Build the NHANES 2009-2010 modelling table for young adults (20-44).

The arthritis questionnaire (ARQ_F) was only given to ages 20-69, so 20 is the
floor for "young adult" here (BRFSS goes down to 18).

Two targets are produced:

  arthritis_dx  - doctor ever said you had arthritis (MCQ160A == 1).
                  Features: demographics, body measures, spinal mobility (ARX),
                  and the back-pain SYMPTOM items from ARQ. This asks:
                  "do symptoms + body measures predict a diagnosis?"

  ibp           - inflammatory back pain, approximate Calin (1977) criteria
                  built from ARQ items (see build_ibp_target). Because the
                  target is DERIVED from ARQ010-ARQ080, those items must NOT be
                  used as features for this target (that would be leakage).
                  Use FEATURES_IBP instead.

Output: data/processed/nhanes_young.csv  (+ prints class balance)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhanes"
OUT = ROOT / "data" / "processed"

AGE_MIN, AGE_MAX = 20, 44

# ARQ per-location suffixes: A neck, B upper back, C mid back, D low back,
# E buttocks, F hips, G rib cage. "Back" for IBP = B, C, D, E.
BACK_SITES = ["B", "C", "D", "E"]

# ---------------------------------------------------------------------------
# Feature lists (columns are created in build())
# ---------------------------------------------------------------------------
DEMO_BODY = [
    "age", "female", "race_eth", "education", "income_poverty_ratio",
    "bmi", "waist_cm", "ever_smoked",
]
SPINAL_MOBILITY = ["occiput_wall_cm", "chest_expansion_cm", "lumbar_flexion_cm"]
OTHER_SPA_FEATURES = [  # extra-spinal features associated with spondyloarthritis
    "heel_pain_2wk", "iritis_uveitis", "ulcerative_colitis", "crohns",
]
BACK_SYMPTOMS = [
    "chronic_pain_6wk", "pain_neck", "pain_upper_back", "pain_mid_back",
    "pain_low_back", "pain_buttock", "pain_hip", "n_pain_sites",
    "back_pain_3mo", "morning_stiffness_cat", "pain_better_with_exercise",
    "pain_wakes_at_night", "nsaid_ever",
]

FEATURES_ARTHRITIS = DEMO_BODY + SPINAL_MOBILITY + OTHER_SPA_FEATURES + BACK_SYMPTOMS
FEATURES_IBP = DEMO_BODY + SPINAL_MOBILITY + OTHER_SPA_FEATURES  # no ARQ0xx items!


# ---------------------------------------------------------------------------
def read_xpt(name: str) -> pd.DataFrame:
    df = pd.read_sas(RAW / f"{name}.xpt", format="xport")
    df.columns = [c.upper() for c in df.columns]
    df["SEQN"] = df["SEQN"].astype(int)
    return df


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-NaN series if CDC named it differently."""
    if name in df.columns:
        return df[name]
    print(f"  (warning: column {name} not found - filled with NaN)")
    return pd.Series(np.nan, index=df.index)


def yes_no(s: pd.Series) -> pd.Series:
    """NHANES 1=Yes, 2=No, 7/9=refused/don't know -> 1/0/NaN."""
    return s.map({1: 1.0, 2: 0.0})


def clean_age(s: pd.Series) -> pd.Series:
    """Ages coded 777/999/7777/9999 etc. are refused/don't know."""
    return s.where(s < 150)


def build_ibp_target(d: pd.DataFrame) -> pd.Series:
    """
    Approximate Calin criteria for inflammatory back pain:
      Requires chronic back pain (almost daily for >= 3 months at a back site),
      then IBP if >= 4 of these 5 are met:
        1. onset before age 40               (ARQ022x < 40 at a back site)
        2. insidious onset                   (ARQ025x >= 3: developed over >= 1 week)
        3. duration >= 3 months              (ARQ024x == 1)
        4. morning stiffness > 30 min        (ARQ040 in 3, 4, 5)
        5. improves with exercise            (ARQ080 == 1)
    Missing answers count as "criterion not met".

    Caveat to mention in the report: ARQ025 was skipped for people whose pain is
    still ongoing, so criterion 2 is often missing for current sufferers. This
    is an operational approximation, not a clinical diagnosis. Check it against
    the prevalence reported by Weisman et al. 2013 (NHANES 2009-10 IBP).
    """
    def any_site(prefix, cond):
        hits = [cond(clean_age(col(d, f"{prefix}{s}"))) for s in BACK_SITES]
        return pd.concat(hits, axis=1).any(axis=1)

    dur_3mo = any_site("ARQ024", lambda x: x == 1)
    onset_lt40 = any_site("ARQ022", lambda x: x < 40)
    insidious = any_site("ARQ025", lambda x: x.isin([3, 4, 5]))
    stiffness = col(d, "ARQ040").isin([3, 4, 5])
    exercise = col(d, "ARQ080") == 1

    n_criteria = (onset_lt40.astype(int) + insidious.astype(int) + dur_3mo.astype(int)
                  + stiffness.astype(int) + exercise.astype(int))
    ibp = (dur_3mo & (n_criteria >= 4)).astype(float)
    # Unknown if they didn't answer the chronic-pain gate question at all
    ibp[col(d, "ARQ010").isna() | col(d, "ARQ010").isin([7, 9])] = np.nan
    return ibp


def build(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """tables: dict of raw NHANES DataFrames keyed by file name (DEMO_F, ...)."""
    d = tables["DEMO_F"]
    for name in ["ARQ_F", "MCQ_F", "BMX_F", "ARX_F", "SMQ_F"]:
        if name in tables:
            d = d.merge(tables[name], on="SEQN", how="left")

    d = d[(d["RIDAGEYR"] >= AGE_MIN) & (d["RIDAGEYR"] <= AGE_MAX)].copy()
    # keep only people who actually completed the ARQ interview
    d = d[col(d, "ARQ010").notna()].copy()

    out = pd.DataFrame({"SEQN": d["SEQN"]})
    # --- demographics / body
    out["age"] = d["RIDAGEYR"]
    out["female"] = (col(d, "RIAGENDR") == 2).astype(float)
    out["race_eth"] = col(d, "RIDRETH1")            # categorical 1-5
    out["education"] = col(d, "DMDEDUC2").where(lambda x: x <= 5)  # 1-5, ordinal
    out["income_poverty_ratio"] = col(d, "INDFMPIR")
    out["bmi"] = col(d, "BMXBMI")
    out["waist_cm"] = col(d, "BMXWAIST")
    out["ever_smoked"] = yes_no(col(d, "SMQ020"))
    # --- spinal mobility (exam)
    out["occiput_wall_cm"] = col(d, "ARXO2WD")
    out["chest_expansion_cm"] = col(d, "ARDDINEX")
    out["lumbar_flexion_cm"] = col(d, "ARDLFTL")
    # --- other SpA-related history
    out["heel_pain_2wk"] = yes_no(col(d, "ARQ110"))
    out["iritis_uveitis"] = yes_no(col(d, "ARD125A"))
    out["ulcerative_colitis"] = yes_no(col(d, "ARQ125C"))
    out["crohns"] = yes_no(col(d, "ARQ125D"))
    # --- back symptoms (ARQ). Skipped items mean "no pain" -> fill 0
    chronic = yes_no(col(d, "ARQ010"))
    out["chronic_pain_6wk"] = chronic
    sites = {"A": "pain_neck", "B": "pain_upper_back", "C": "pain_mid_back",
             "D": "pain_low_back", "E": "pain_buttock", "F": "pain_hip"}
    for suf, name in sites.items():
        out[name] = col(d, f"ARQ020{suf}").notna().astype(float)
    out["n_pain_sites"] = out[list(sites.values())].sum(axis=1)
    out["back_pain_3mo"] = pd.concat(
        [col(d, f"ARQ024{s}") == 1 for s in BACK_SITES], axis=1).any(axis=1).astype(float)
    # morning stiffness: 0 = none / not asked, 1..5 = <10min .. >4h
    ms = col(d, "ARQ040").where(lambda x: x <= 6)
    out["morning_stiffness_cat"] = ms.replace(6, 0).fillna(0)
    out["pain_better_with_exercise"] = (col(d, "ARQ080") == 1).astype(float)
    out["pain_wakes_at_night"] = (col(d, "ARQ070") == 1).astype(float)
    out["nsaid_ever"] = pd.concat(
        [col(d, f"ARQ030{x}") == 1 for x in "ABCDE"], axis=1).any(axis=1).astype(float)

    # --- targets
    out["arthritis_dx"] = yes_no(col(d, "MCQ160A"))
    out["ibp"] = build_ibp_target(d)
    out["ankylosing_spondylitis_dx"] = yes_no(col(d, "ARQ125E"))  # too rare to model alone; for reference

    # --- survey weight (interview weight; use MEC weight WTMEC2YR if you rely on ARX/BMX)
    out["weight_interview"] = col(d, "WTINT2YR")
    out["weight_mec"] = col(d, "WTMEC2YR")
    return out.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(f"\nrows (age {AGE_MIN}-{AGE_MAX}, completed ARQ): {len(df)}")
    for t in ["arthritis_dx", "ibp", "ankylosing_spondylitis_dx"]:
        vc = df[t].value_counts(dropna=False)
        pos = vc.get(1.0, 0)
        print(f"  {t:28s} positives={pos:5d}  rate={pos / df[t].notna().sum():.1%}  "
              f"missing={df[t].isna().sum()}")
    miss = df[FEATURES_ARTHRITIS].isna().mean().sort_values(ascending=False).head(6)
    print("  most-missing features:\n" + miss.to_string())


def main() -> None:
    names = ["DEMO_F", "ARQ_F", "MCQ_F", "BMX_F", "ARX_F", "SMQ_F"]
    tables = {n: read_xpt(n) for n in names if (RAW / f"{n}.xpt").exists()}
    df = build(tables)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "nhanes_young.csv", index=False)
    summarize(df)
    print(f"\nsaved -> {OUT / 'nhanes_young.csv'}")


if __name__ == "__main__":
    main()
