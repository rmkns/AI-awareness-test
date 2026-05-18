from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_MODELS = ("gemma3", "llama32", "gptoss", "deepseek", "qwen3")
INDICATORS = ("UNK", "CONTR", "REFL", "CLARIFY", "URG")


def latest_session(alias_dir: Path) -> Optional[Path]:
    sessions = [p for p in alias_dir.glob("session_*") if p.is_dir()]
    if not sessions:
        return None
    return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def json_files(session: Path) -> Iterable[Path]:
    for path in sorted(session.glob("*.json")):
        if path.name in {"aggregate.json"}:
            continue
        yield path


def load_records(session: Path) -> Dict[Tuple[str, int], Tuple[Path, Dict]]:
    records: Dict[Tuple[str, int], Tuple[Path, Dict]] = {}
    for path in json_files(session):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Bad JSON: {path}: {exc}") from exc
        test_id = data.get("test_id")
        run = data.get("run") or data.get("repeat") or data.get("run_index")
        if not test_id or run is None:
            raise SystemExit(f"Missing test_id/run in {path}")
        key = (str(test_id), int(run))
        if key in records:
            raise SystemExit(f"Duplicate Project A record {key} in {session}")
        records[key] = (path, data)
    return records


def output_name(source_name: str, new_run: int) -> str:
    if re.search(r"_r\d+\.json$", source_name):
        return re.sub(r"_r\d+\.json$", f"_r{new_run}.json", source_name)
    return f"{Path(source_name).stem}_r{new_run}.json"


def normalize_record(data: Dict, *, new_run: int, source_path: Path,
                     source_session: Path, source_run_dir: Path,
                     source_original_run: int, segment: str) -> Dict:
    out = dict(data)
    out["repeat"] = new_run
    out["run"] = new_run
    out.setdefault("answer", out.get("final_answer", ""))
    out.setdefault("response", out.get("final_answer", ""))
    out.setdefault("prompt", out.get("full_dialogue", ""))
    out["combined_source"] = {
        "segment": segment,
        "source_run_dir": str(source_run_dir.resolve()),
        "source_session": str(source_session.resolve()),
        "source_file": str(source_path.resolve()),
        "source_original_run": source_original_run,
    }
    return out


def write_summary(session_dir: Path, rows: List[Dict]) -> None:
    summary_path = session_dir / "summary.csv"
    fieldnames = [
        "test_id", "title", "repeat", "source_original_run",
        "source_segment", *INDICATORS, "tags", "expect",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: Dict[str, Dict[str, int]] = {}
    for row in rows:
        title = row["title"]
        aggregate.setdefault(
            title,
            {"count": 0, "UNK": 0, "CONTR": 0, "REFL": 0, "CLARIFY": 0,
             "URG": 0},
        )
        aggregate[title]["count"] += 1
        for indicator in INDICATORS:
            aggregate[title][indicator] += int(row.get(indicator) or 0)

    (session_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def merge_alias(alias: str, base_run_dir: Path, extra_run_dir: Path,
                output_run_dir: Path, base_runs: int,
                extra_runs: int, tests_per_model: int) -> Dict:
    base_session = latest_session(base_run_dir / "A" / alias)
    extra_session = latest_session(extra_run_dir / "A" / alias)
    if base_session is None:
        raise SystemExit(f"No base session for {alias}")
    if extra_session is None:
        raise SystemExit(f"No extra session for {alias}")

    base_records = load_records(base_session)
    extra_records = load_records(extra_session)

    base_tests = {test_id for test_id, run in base_records if 1 <= run <= base_runs}
    extra_tests = {test_id for test_id, run in extra_records if 1 <= run <= extra_runs}
    if len(base_tests) != tests_per_model:
        raise SystemExit(f"{alias}: base has {len(base_tests)} tests, "
                         f"expected {tests_per_model}")
    if len(extra_tests) != tests_per_model:
        raise SystemExit(f"{alias}: extra has {len(extra_tests)} tests, "
                         f"expected {tests_per_model}")
    if base_tests != extra_tests:
        missing_extra = sorted(base_tests - extra_tests)[:10]
        missing_base = sorted(extra_tests - base_tests)[:10]
        raise SystemExit(
            f"{alias}: base/extra test_id mismatch. "
            f"missing_extra={missing_extra}, missing_base={missing_base}"
        )

    session_dir = (
        output_run_dir / "A" / alias /
        f"session_{alias}_combined_10runs_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=False)

    summary_rows: List[Dict] = []
    copied = 0

    for run in range(1, base_runs + 1):
        for test_id in sorted(base_tests):
            path, data = base_records[(test_id, run)]
            new_data = normalize_record(
                data,
                new_run=run,
                source_path=path,
                source_session=base_session,
                source_run_dir=base_run_dir,
                source_original_run=run,
                segment=f"base_1_{base_runs}",
            )
            out_path = session_dir / output_name(path.name, run)
            out_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            copied += 1
            scores = new_data.get("scores") or {}
            summary_rows.append({
                "test_id": new_data.get("test_id", test_id),
                "title": new_data.get("title", test_id),
                "repeat": run,
                "source_original_run": run,
                "source_segment": f"base_1_{base_runs}",
                **{indicator: int(scores.get(indicator) or 0)
                   for indicator in INDICATORS},
                "tags": "|".join(new_data.get("tags") or []),
                "expect": "|".join(new_data.get("expect") or []),
            })

    for old_run in range(1, extra_runs + 1):
        new_run = base_runs + old_run
        for test_id in sorted(extra_tests):
            path, data = extra_records[(test_id, old_run)]
            new_data = normalize_record(
                data,
                new_run=new_run,
                source_path=path,
                source_session=extra_session,
                source_run_dir=extra_run_dir,
                source_original_run=old_run,
                segment=f"extra_{base_runs + 1}_{base_runs + extra_runs}",
            )
            out_path = session_dir / output_name(path.name, new_run)
            out_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            copied += 1
            scores = new_data.get("scores") or {}
            summary_rows.append({
                "test_id": new_data.get("test_id", test_id),
                "title": new_data.get("title", test_id),
                "repeat": new_run,
                "source_original_run": old_run,
                "source_segment": f"extra_{base_runs + 1}_{base_runs + extra_runs}",
                **{indicator: int(scores.get(indicator) or 0)
                   for indicator in INDICATORS},
                "tags": "|".join(new_data.get("tags") or []),
                "expect": "|".join(new_data.get("expect") or []),
            })

    write_summary(session_dir, summary_rows)

    expected = tests_per_model * (base_runs + extra_runs)
    if copied != expected:
        raise SystemExit(f"{alias}: copied {copied}, expected {expected}")

    return {
        "alias": alias,
        "base_session": str(base_session.resolve()),
        "extra_session": str(extra_session.resolve()),
        "combined_session": str(session_dir.resolve()),
        "copied": copied,
        "expected": expected,
        "runs": list(range(1, base_runs + extra_runs + 1)),
        "tests": tests_per_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--extra-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--base-runs", type=int, default=7)
    parser.add_argument("--extra-runs", type=int, default=3)
    parser.add_argument("--tests-per-model", type=int, default=40)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    base_run_dir = Path(args.base_run_dir)
    extra_run_dir = Path(args.extra_run_dir)
    output_run_dir = Path(args.output_run_dir)
    if output_run_dir.exists():
        raise SystemExit(f"Output already exists: {output_run_dir}")
    (output_run_dir / "A").mkdir(parents=True, exist_ok=False)

    results = [
        merge_alias(alias, base_run_dir, extra_run_dir, output_run_dir,
                    args.base_runs, args.extra_runs, args.tests_per_model)
        for alias in args.models
    ]

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_run_dir": str(base_run_dir.resolve()),
        "extra_run_dir": str(extra_run_dir.resolve()),
        "output_run_dir": str(output_run_dir.resolve()),
        "base_runs": args.base_runs,
        "extra_runs": args.extra_runs,
        "total_runs": args.base_runs + args.extra_runs,
        "tests_per_model": args.tests_per_model,
        "models": results,
        "total_records": sum(row["copied"] for row in results),
    }
    (output_run_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_run_dir / "pipeline_manifest_A.txt").write_text(
        "\n".join([
            "PROJECT A COMBINED",
            f"ROOT={Path.cwd().resolve()}",
            f"BASE_RUN_DIR={base_run_dir.resolve()}",
            f"EXTRA_RUN_DIR={extra_run_dir.resolve()}",
            f"A_REPEATS={args.base_runs + args.extra_runs}",
            f"TESTS_PER_MODEL={args.tests_per_model}",
            f"TOTAL_RECORDS={manifest['total_records']}",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"Combined run created: {output_run_dir}")
    for row in results:
        print(f"  {row['alias']}: {row['copied']}/{row['expected']} -> "
              f"{row['combined_session']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
