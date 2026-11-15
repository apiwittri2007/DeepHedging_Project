"""Build data/raw/hk/<ticker>.csv (Date, Close) from a free public source.

Source: github.com/zjplab/US-Trading-Data (GPL-3.0), which refreshes daily
split/dividend-adjusted prices for Hang Seng Tech constituents from Yahoo Finance
via yfinance. Usage:

    git clone --depth 1 https://github.com/zjplab/US-Trading-Data /tmp/src
    python -I scripts/fetch_hk_data.py --src /tmp/src --out data/raw/hk
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert(src_csv: Path, out_csv: Path) -> int:
    rows = 0
    with src_csv.open() as fin, out_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["Date", "Close"])
        for r in reader:
            close = r.get("Close", "")
            if not close:
                continue
            writer.writerow([r["Date"][:10], f"{float(close):.6f}"])
            rows += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="clone of zjplab/US-Trading-Data")
    p.add_argument("--out", default="data/raw/hk")
    args = p.parse_args()
    src = Path(args.src) / "data" / "HangSengTech"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted(src.glob("*.HK.csv")):
        n = convert(f, out / f.name)
        total += n
        print(f"{f.stem}: {n} rows")
    print(f"done: {total} rows")


if __name__ == "__main__":
    main()
