"""
Score a CSV of customers and output a ranked call list (production model).

Usage:
    python src/score_customers.py --input customers.csv --output ranked.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bank_ml import OUT_DIR, load_artifacts, score_customers  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Rank customers by subscription probability")
    parser.add_argument("--input", "-i", required=True, help="Input CSV (UCI schema, semicolon or comma)")
    parser.add_argument("--output", "-o", default=str(OUT_DIR / "ranked_customers.csv"), help="Output ranked CSV")
    parser.add_argument("--top", type=int, default=None, help="Keep only top N rows")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    try:
        df = pd.read_csv(in_path, sep=";")
    except Exception:
        df = pd.read_csv(in_path)

    artifacts = load_artifacts(retrain_if_missing=True)
    prod = artifacts["results"]["v2_no_duration"]
    ranked = score_customers(df, prod["clf"], prod["preprocessor"])

    if args.top:
        ranked = ranked.head(args.top)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    print(f"Scored {len(ranked):,} customers -> {out_path}")
    print(f"Top probability: {ranked['subscription_probability'].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
