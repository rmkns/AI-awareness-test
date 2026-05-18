from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


INDICATORS = ("UNK", "CONTR", "REFL", "CLARIFY", "URG")


def load_intrinsic(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "test_id": r["test_id"],
                "run": r["run"],
                "intrinsic_model": r["model"],
                "mean_H": r["mean_entropy"],
                "max_H": r["max_entropy"],
                "min_p": r["min_p_chosen"],
                "low_ratio": r["low_conf_ratio"],
            })
    return pd.DataFrame(rows)


def load_judged(path: Path, target_intrinsic_model: Optional[str] = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (target_intrinsic_model is not None and
                    r.get("model") != target_intrinsic_model):
                continue
            row = {
                "test_id": r["test_id"],
                "run": r["run"],
                "behavioral_model": r["model"],
            }
            kw = r.get("keyword_scores") or {}
            jd = r.get("judge_scores") or {}
            for ind in INDICATORS:
                row[f"kw_{ind}"] = kw.get(ind)
                row[f"j_{ind}"] = jd.get(ind)
            rows.append(row)
    return pd.DataFrame(rows)


def _pearson_safe(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman_safe(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    pairs = [
        ("mean_H", "kw_UNK", "pearson"),
        ("max_H", "kw_CONTR", "pearson"),
        ("low_ratio", "kw_CLARIFY", "pearson"),
        ("mean_H", "j_UNK", "spearman"),
        ("max_H", "j_CONTR", "spearman"),
        ("low_ratio", "j_CLARIFY", "spearman"),
    ]

    for intrinsic_col, target_col, kind in pairs:
        sub = df.dropna(subset=[intrinsic_col, target_col])
        if len(sub) < 3:
            continue
        x = sub[intrinsic_col].astype(float).to_numpy()
        y = sub[target_col].astype(float).to_numpy()
        if kind == "pearson":
            r, p = _pearson_safe(x, y)
        else:
            r, p = _spearman_safe(x, y)
        rows.append({
            "intrinsic_metric": intrinsic_col,
            "target": target_col,
            "method": kind,
            "n": len(sub),
            "r": r,
            "p_value": p,
            "x_mean": float(x.mean()),
            "y_mean": float(y.mean()),
        })
    return pd.DataFrame(rows)


def make_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 out_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping scatter plot")
        return
    sub = df.dropna(subset=[x_col, y_col])
    if len(sub) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub[x_col], sub[y_col], alpha=0.5, s=18)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsic", required=True,
                   help="JSONL from intrinsic_evaluator.py")
    p.add_argument("--judged", required=True,
                   help="JSONL from project_b_judge/evaluate_llm_judge.py "
                        "(contains both keyword and judge scores)")
    p.add_argument("--output-dir", default="triangulation_out")
    p.add_argument("--target-behavioral-model", default=None,
                   help="Filter judged to one Project A model "
                        "(e.g. 'llama3.2'); if omitted, all models pooled")
    args = p.parse_args()

    intr = load_intrinsic(Path(args.intrinsic))
    jud = load_judged(Path(args.judged),
                      target_intrinsic_model=args.target_behavioral_model)
    print(f"Intrinsic rows: {len(intr)}, judged rows: {len(jud)}")

    intr_agg = (intr.groupby("test_id")
                    .agg(mean_H=("mean_H", "mean"),
                         max_H=("max_H", "mean"),
                         low_ratio=("low_ratio", "mean"))
                    .reset_index())
    jud_agg = jud.groupby("test_id").mean(numeric_only=True).reset_index()

    merged = intr_agg.merge(jud_agg, on="test_id", how="inner")
    print(f"Merged on test_id: {len(merged)} rows")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "merged_for_correlation.csv", index=False)

    corr = compute_correlations(merged)
    corr.to_csv(out / "correlation_matrix.csv", index=False)
    print("\n=== Correlation matrix ===")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    make_scatter(merged, "mean_H", "kw_UNK",
                 out / "scatter_meanH_vs_kwUNK.png",
                 "Mean entropy vs keyword UNK signal")
    make_scatter(merged, "max_H", "j_CONTR",
                 out / "scatter_maxH_vs_jCONTR.png",
                 "Max entropy vs judge CONTR score")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
