from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from rubric import INDICATORS, binarize


def load_judged(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {
                "test_id": r["test_id"],
                "model": r["model"],
                "run": r["run"],
                "judge_model": r.get("judge_model", "unknown"),
                "test_prompt": r.get("test_prompt", ""),
                "model_response": r.get("model_response", ""),
                "judge_rationale": r["judge_scores"].get("rationale", ""),
            }
            kw = r.get("keyword_scores") or {}
            jd = r.get("judge_scores") or {}
            for ind in INDICATORS:
                v_kw = kw.get(ind, kw.get(ind.lower()))
                row[f"kw_{ind}"] = int(v_kw) if v_kw is not None else None
                row[f"j_{ind}"] = jd.get(ind)
                row[f"jb_{ind}"] = binarize(jd.get(ind))
            rows.append(row)
    return pd.DataFrame(rows)


def kappa_safe(y1: np.ndarray, y2: np.ndarray) -> float:
    if len(y1) == 0 or len(set(y1)) < 2 or len(set(y2)) < 2:
        return float("nan")
    return float(cohen_kappa_score(y1, y2))


def compute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ind in INDICATORS:
        for model in sorted(df["model"].dropna().unique()):
            sub = df[(df["model"] == model)].dropna(
                subset=[f"kw_{ind}", f"jb_{ind}"])
            if len(sub) == 0:
                continue
            kw = sub[f"kw_{ind}"].astype(int).to_numpy()
            jb = sub[f"jb_{ind}"].astype(int).to_numpy()
            cm = confusion_matrix(kw, jb, labels=[0, 1])
            kappa = kappa_safe(kw, jb)
            rows.append({
                "indicator": ind,
                "model": model,
                "n": len(sub),
                "kw_pos_rate": float(kw.mean()),
                "judge_pos_rate": float(jb.mean()),
                "agree_pct": float((kw == jb).mean()),
                "cohen_kappa": kappa,
                "kw_only_n": int(cm[1, 0]),
                "judge_only_n": int(cm[0, 1]),
                "both_n": int(cm[1, 1]),
                "neither_n": int(cm[0, 0]),
            })
    return pd.DataFrame(rows)


def compute_overall_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ind in INDICATORS:
        sub = df.dropna(subset=[f"kw_{ind}", f"jb_{ind}"])
        if len(sub) == 0:
            continue
        kw = sub[f"kw_{ind}"].astype(int).to_numpy()
        jb = sub[f"jb_{ind}"].astype(int).to_numpy()
        kappa = kappa_safe(kw, jb)
        rows.append({
            "indicator": ind,
            "n": len(sub),
            "kw_pos_rate": float(kw.mean()),
            "judge_pos_rate": float(jb.mean()),
            "agree_pct": float((kw == jb).mean()),
            "cohen_kappa": kappa,
        })
    return pd.DataFrame(rows)


def find_disagreements(df: pd.DataFrame, indicator: str,
                       n: int = 10, seed: int = 42) -> pd.DataFrame:
    sub = df.dropna(subset=[f"kw_{indicator}", f"jb_{indicator}"])
    diss = sub[sub[f"kw_{indicator}"] != sub[f"jb_{indicator}"]].copy()
    if len(diss) == 0:
        return diss
    return diss.sample(min(n, len(diss)), random_state=seed)


def render_markdown_summary(overall: pd.DataFrame,
                            per_model: pd.DataFrame) -> str:
    lines = ["# LLM-as-Judge vs. Keyword agreement summary", ""]
    lines.append("## Overall (all models pooled)\n")
    lines.append(overall.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Per (indicator × model)\n")
    lines.append(per_model.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Cohen's κ interpretation (Landis & Koch 1977)")
    lines.append("- κ < 0.20  → poor")
    lines.append("- 0.20 - 0.40 → fair")
    lines.append("- 0.40 - 0.60 → moderate")
    lines.append("- 0.60 - 0.80 → substantial")
    lines.append("- κ ≥ 0.80   → almost perfect")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", required=True,
                   help="JSONL produced by evaluate_llm_judge.py")
    p.add_argument("--output-dir", default="analysis_output",
                   help="Directory for produced CSV / markdown files")
    p.add_argument("--n-disagreements", type=int, default=10,
                   help="How many disagreement examples to sample per indicator")
    args = p.parse_args()

    judged_path = Path(args.judged)
    if not judged_path.exists():
        print(f"ERROR: file not found: {judged_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_judged(judged_path)
    print(f"Loaded {len(df)} responses from {judged_path}")
    df.to_csv(out_dir / "merged_scores.csv", index=False)

    overall = compute_overall_agreement(df)
    per_model = compute_agreement(df)
    overall.to_csv(out_dir / "agreement_overall.csv", index=False)
    per_model.to_csv(out_dir / "agreement_per_model.csv", index=False)

    print("\n=== Overall agreement (pooled) ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Per-model agreement ===")
    print(per_model.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    diss_frames = []
    for ind in INDICATORS:
        d = find_disagreements(df, ind, n=args.n_disagreements)
        if len(d) > 0:
            d = d.copy()
            d["indicator"] = ind
            d["disagreement_type"] = d.apply(
                lambda r, i=ind:
                    f"kw={int(r[f'kw_{i}'])}, judge={int(r[f'jb_{i}'])}", axis=1)
            cols = ["indicator", "disagreement_type", "model", "test_id", "run",
                    "test_prompt", "model_response", "judge_rationale"]
            diss_frames.append(d[cols])
    if diss_frames:
        diss = pd.concat(diss_frames, ignore_index=True)
        diss.to_csv(out_dir / "disagreements_sample.csv", index=False)
        print(f"\nSaved {len(diss)} disagreement examples")

    md = render_markdown_summary(overall, per_model)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"\nReport written to {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
