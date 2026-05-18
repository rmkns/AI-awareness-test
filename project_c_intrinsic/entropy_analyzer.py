from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


CATEGORIES = {
    "contr": "Prieštaros (CONTR)",
    "unknown": "Nežinojimas (UNK)",
    "refl": "Refleksija (REFL)",
    "ctx": "Kontekstas (CLARIFY)",
    "sad": "Dviprasmiškumas (SAD)",
}


def load_intrinsic(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "model": r["model"],
                "test_id": r["test_id"],
                "run": r["run"],
                "n_tokens": r["n_tokens"],
                "mean_entropy": r["mean_entropy"],
                "max_entropy": r["max_entropy"],
                "min_p_chosen": r["min_p_chosen"],
                "low_conf_count": r["low_conf_count"],
                "low_conf_ratio": r["low_conf_ratio"],
            })
    return pd.DataFrame(rows)


def categorize(test_id: str) -> str:
    parts = test_id.split("_", 2)
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def per_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "test_id"])
              .agg(mean_H=("mean_entropy", "mean"),
                   std_H=("mean_entropy", "std"),
                   max_H=("max_entropy", "mean"),
                   low_ratio=("low_conf_ratio", "mean"),
                   n_runs=("run", "count"))
              .reset_index())


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["test_id"].apply(categorize)
    return (df.groupby(["model", "category"])
              .agg(mean_H=("mean_entropy", "mean"),
                   std_H=("mean_entropy", "std"),
                   max_H=("max_entropy", "mean"),
                   low_ratio=("low_conf_ratio", "mean"),
                   n=("test_id", "count"))
              .reset_index())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsic", required=True,
                   help="JSONL produced by intrinsic_evaluator.py")
    p.add_argument("--output-dir", default="intrinsic_analysis")
    args = p.parse_args()

    df = load_intrinsic(Path(args.intrinsic))
    print(f"Loaded {len(df)} responses")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "raw_intrinsic.csv", index=False)

    test_summary = per_test_summary(df)
    test_summary.to_csv(out / "per_test_summary.csv", index=False)
    print("\n=== Per test (top 10 by mean entropy) ===")
    print(test_summary.sort_values("mean_H", ascending=False)
          .head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cat_summary = per_category_summary(df)
    cat_summary.to_csv(out / "per_category_summary.csv", index=False)
    print("\n=== Per category ===")
    print(cat_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
