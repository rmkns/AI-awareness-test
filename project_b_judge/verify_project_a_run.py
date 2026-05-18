from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from evaluate_llm_judge import load_responses


DEFAULT_MODELS = ("gemma3", "llama32", "gptoss", "deepseek", "qwen3")


def candidate_runs(runs_root: Path) -> Iterable[Path]:
    paths = [
        p for p in runs_root.iterdir()
        if p.is_dir() and (p.name.startswith("A_only_") or
                           p.name.startswith("AB_only_") or
                           p.name.startswith("A_combined_"))
    ]
    yield from sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_session(alias_dir: Path) -> Optional[Path]:
    sessions = [p for p in alias_dir.glob("session_*") if p.is_dir()]
    if not sessions:
        return None
    return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def inspect_run(run_dir: Path, aliases: List[str], runs: int,
                tests_per_model: int) -> Dict:
    model_rows = []
    ok = True
    expected_per_model = tests_per_model * runs

    for alias in aliases:
        alias_dir = run_dir / "A" / alias
        session = latest_session(alias_dir) if alias_dir.exists() else None
        row = {
            "alias": alias,
            "alias_dir": str(alias_dir),
            "session": str(session) if session else None,
            "count": 0,
            "n_tests": 0,
            "runs": [],
            "expected": expected_per_model,
            "ok": False,
        }
        if session is not None:
            items = load_responses(session)
            selected = [
                it for it in items
                if isinstance(it.get("run"), int) and 1 <= it["run"] <= runs
            ]
            keys = {(it["test_id"], it["run"]) for it in selected}
            row.update({
                "count": len(keys),
                "n_tests": len({test_id for test_id, _run in keys}),
                "runs": sorted({run for _test_id, run in keys}),
            })
            row["ok"] = (
                row["count"] == expected_per_model
                and row["n_tests"] == tests_per_model
                and row["runs"] == list(range(1, runs + 1))
            )
        ok = ok and bool(row["ok"])
        model_rows.append(row)

    return {
        "run_dir": str(run_dir),
        "ok": ok,
        "runs_required": runs,
        "tests_per_model": tests_per_model,
        "expected_per_model": expected_per_model,
        "expected_total": expected_per_model * len(aliases),
        "models": model_rows,
    }


def write_env(path: Path, run_info: Dict) -> None:
    lines = ["@echo off\n", f"set \"A_RUN_DIR={run_info['run_dir']}\"\n"]
    for row in run_info["models"]:
        lines.append(f"set \"SESSION_{row['alias']}={row['session']}\"\n")
    path.write_text("".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--a-run-dir", default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tests-per-model", type=int, default=40)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--write-env", default=None)
    args = parser.parse_args()

    aliases = list(args.models)
    inspected: List[Dict] = []

    if args.a_run_dir:
        info = inspect_run(Path(args.a_run_dir), aliases, args.runs,
                           args.tests_per_model)
        inspected.append(info)
    else:
        for run_dir in candidate_runs(Path(args.runs_root)):
            info = inspect_run(run_dir, aliases, args.runs,
                               args.tests_per_model)
            inspected.append(info)
            if info["ok"]:
                break

    chosen = next((info for info in inspected if info["ok"]), None)
    payload = {"chosen": chosen, "inspected": inspected}

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if chosen is None:
        print("No complete Project A run found for the requested shape.")
        print(f"Need {len(aliases)} models * {args.tests_per_model} tests "
              f"* {args.runs} runs.")
        for info in inspected[:5]:
            print(f"- {info['run_dir']}: ok={info['ok']}")
            for row in info["models"]:
                print(f"  {row['alias']}: {row['count']}/{row['expected']} "
                      f"runs={row['runs']} ok={row['ok']}")
        return 2

    if args.write_env:
        write_env(Path(args.write_env), chosen)

    print(f"Using Project A run: {chosen['run_dir']}")
    for row in chosen["models"]:
        print(f"  {row['alias']}: {row['count']}/{row['expected']} "
              f"from {row['session']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
