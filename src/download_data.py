"""
Download raw data for the arthritis / inflammatory back pain ML project.

Sources (all public, no login needed):
  1. NHANES 2009-2010  (CDC/NCHS) - SAS transport (.xpt) files
  2. BRFSS             (CDC)      - zipped SAS transport file, default year 2024
  3. UCI Vertebral Column (id 212) - zip with column_2C.dat / column_3C.dat

Usage:
    python src/download_data.py                 # everything
    python src/download_data.py --only nhanes   # just one source
    python src/download_data.py --brfss-year 2025

Files go to data/raw/<source>/. Existing files are skipped, so re-running is safe.
Only the Python standard library is used here.
"""
import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# --- NHANES 2009-2010 (cycle suffix _F) ----------------------------------
# DEMO = demographics, ARQ = inflammatory back pain / arthritis questionnaire,
# MCQ  = medical conditions (doctor-diagnosed arthritis), BMX = body measures,
# ARX  = arthritis body measures (spinal mobility), SMQ = smoking.
NHANES_FILES = ["DEMO_F", "ARQ_F", "MCQ_F", "BMX_F", "ARX_F", "SMQ_F"]
NHANES_URLS = [
    # current CDC layout
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/{name}.xpt",
    # older layout, kept as a fallback
    "https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/{name}.XPT",
]

# --- BRFSS ------------------------------------------------------------------
BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"

# --- UCI Vertebral Column ---------------------------------------------------
UCI_VERTEBRAL_URL = "https://archive.ics.uci.edu/static/public/212/vertebral+column.zip"

# www.cdc.gov rejects requests that don't look like a browser (HTTP 403),
# so send ordinary browser headers.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cdc.gov/brfss/annual_data/annual_2024.html",
}


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
                print()
            tmp.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    return False


def download_nhanes() -> None:
    out = RAW / "nhanes"
    for name in NHANES_FILES:
        dest = out / f"{name}.xpt"
        if dest.exists():
            print(f"[nhanes] {name}: already present")
            continue
        print(f"[nhanes] {name}")
        if not any(fetch(u.format(name=name), dest) for u in NHANES_URLS):
            print(f"[nhanes] !! could not download {name}. Get it manually from "
                  f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2009-2010")


def download_brfss(year: int) -> None:
    out = RAW / "brfss"
    final = out / f"LLCP{year}.xpt"
    if final.exists():
        print(f"[brfss] {final.name}: already present")
        return
    zpath = out / f"LLCP{year}XPT.zip"
    if not zpath.exists():
        print(f"[brfss] {year} (~100 MB zipped, ~1 GB unzipped)")
        if not fetch(BRFSS_URL.format(year=year), zpath):
            print(f"[brfss] !! download failed. Manual fix:\n"
                  f"    1. In your browser open {BRFSS_URL.format(year=year)}\n"
                  f"    2. Save the zip as {zpath}\n"
                  f"    3. Re-run: python src/download_data.py --only brfss\n"
                  f"       (it will skip the download and extract the zip for you)")
            return
    with zipfile.ZipFile(zpath) as z:
        # CDC's file name inside the zip often has a trailing space ("LLCP2024.XPT ")
        member = next(m for m in z.namelist() if m.strip().upper().endswith(".XPT"))
        with z.open(member) as src, open(final, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"[brfss] extracted -> {final}")


def download_uci() -> None:
    out = RAW / "uci_vertebral"
    if (out / "column_2C.dat").exists():
        print("[uci] already present")
        return
    zpath = out / "vertebral_column.zip"
    print("[uci] vertebral column")
    if not fetch(UCI_VERTEBRAL_URL, zpath):
        print("[uci] !! download failed. Get it from "
              "https://archive.ics.uci.edu/dataset/212/vertebral+column")
        return
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if m.endswith(".dat"):
                (out / Path(m).name).write_bytes(z.read(m))
    print(f"[uci] extracted -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["nhanes", "brfss", "uci"])
    p.add_argument("--brfss-year", type=int, default=2024)
    a = p.parse_args()
    if a.only in (None, "nhanes"):
        download_nhanes()
    if a.only in (None, "uci"):
        download_uci()
    if a.only in (None, "brfss"):
        download_brfss(a.brfss_year)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
