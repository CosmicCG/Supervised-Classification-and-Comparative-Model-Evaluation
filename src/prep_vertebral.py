"""
Clean the UCI Vertebral Column dataset (310 patients, 6 biomechanical features).

Labels:
  3-class: DH (disk hernia), SL (spondylolisthesis), NO (normal)
  2-class: AB (abnormal = DH + SL), NO (normal)

Note for the report: this dataset has no age and is not an arthritis or
inflammatory label. It is used as a small-sample comparison, as the proposal
says, not as a direct test of the arthritis question.

Output: data/processed/vertebral.csv with columns
  6 features + label_3c + abnormal (1 = AB, 0 = NO)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "uci_vertebral"
OUT = ROOT / "data" / "processed"

FEATURES = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
            "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]


def load_dat(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=FEATURES + ["label"])


def build(df3: pd.DataFrame) -> pd.DataFrame:
    out = df3.rename(columns={"label": "label_3c"}).copy()
    out["abnormal"] = (out["label_3c"] != "NO").astype(int)
    return out


def main() -> None:
    df = build(load_dat(RAW / "column_3C.dat"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "vertebral.csv", index=False)
    print(f"rows: {len(df)}")
    print(df["label_3c"].value_counts().to_string())
    print(f"abnormal rate: {df['abnormal'].mean():.1%}")
    print(f"saved -> {OUT / 'vertebral.csv'}")


if __name__ == "__main__":
    main()
