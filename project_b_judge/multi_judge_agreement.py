from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rubric import INDICATORS


def fleiss_kappa(table: np.ndarray) -> float:
    n_subjects, _ = table.shape
    n_raters = table[0].sum()
    if n_raters < 2:
        return float("nan")
    p_j = table.sum(axis=0) / (n_subjects * n_raters)
    P_i = ((table * (table - 1)).sum(axis=1) /
           (n_raters * (n_raters - 1)))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    if P_e == 1.0:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def load_judge_jsonl(path: Path, judge_label: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            row = {
                "judge": judge_label,
                "model": r.get("model"),
                "test_id": r.get("test_id"),
                "run": r.get("run"),
            }
            for ind in INDICATORS:
                row[ind] = r.get("judge_scores", {}).get(ind)
            rows.append(row)
    return pd.DataFrame(rows)


def to_counts_table(df: pd.DataFrame, indicator: str,
                    n_categories: int = 3) -> np.ndarray:
    pivot = df.pivot_table(
        index=["model", "test_id", "run"],
        columns="judge",
        values=indicator,
        aggfunc="first",
    ).dropna()
    arr = pivot.to_numpy().astype(int)
    table = np.zeros((arr.shape[0], n_categories), dtype=int)
    for i in range(arr.shape[0]):
        for v in arr[i]:
            if 0 <= v < n_categories:
                table[i, v] += 1
    return table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged-files", nargs="+", required=True,
                   help="Multiple JSONL files, one per judge")
    p.add_argument("--judge-labels", nargs="*", default=None,
                   help="Optional labels parallel to --judged-files; "
                        "defaults to file stems")
    p.add_argument("--output-dir", default="analysis_multijudge")
    args = p.parse_args()

    files = [Path(f) for f in args.judged_files]
    labels = args.judge_labels or [f.stem for f in files]
    if len(labels) != len(files):
        raise SystemExit("--judge-labels must match --judged-files length")

    df = pd.concat([load_judge_jsonl(f, l) for f, l in zip(files, labels)],
                   ignore_index=True)
    print(f"Loaded {len(df)} rows across {len(files)} judges")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_judges_long.csv", index=False)

    rows = []
    for ind in INDICATORS:
        table = to_counts_table(df, ind)
        kappa = fleiss_kappa(table)
        rows.append({"indicator": ind,
                     "n_subjects": int(table.shape[0]),
                     "fleiss_kappa": kappa})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "fleiss_per_indicator.csv", index=False)
    print("\n=== Fleiss kappa per indicator ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
