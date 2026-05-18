from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from evaluate_llm_judge import already_judged, call_judge, load_responses


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def item_key(item: Dict) -> tuple:
    return (item.get("model"), item.get("test_id"), item.get("run"))


def parse_retry_after(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    text = str(exc)
    match = re.search(r"retry_after_seconds['\"]?:\s*([0-9.]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"Retry-After['\"]?:\s*['\"]?([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


def error_record(item: Dict, judge_model: str, attempt: int,
                 exc: Exception, final: bool) -> Dict:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return {
        "ts": now(),
        "judge_model": judge_model,
        "model": item.get("model"),
        "test_id": item.get("test_id"),
        "run": item.get("run"),
        "attempt": attempt,
        "final": final,
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "retry_after_seconds": parse_retry_after(exc),
        "message": str(exc)[:4000],
        "headers": dict(headers) if headers is not None else {},
    }


def write_progress(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def filter_items(items: List[Dict], runs: int) -> List[Dict]:
    selected = [
        it for it in items
        if isinstance(it.get("run"), int) and 1 <= it["run"] <= runs
    ]
    return sorted(selected, key=lambda it: (str(it["test_id"]), int(it["run"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--require-items", type=int, default=400)
    parser.add_argument("--failure-output", required=True)
    parser.add_argument("--progress-output", required=True)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--retry-base", type=float, default=30.0)
    parser.add_argument("--retry-max", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stop-after-consecutive-failures", type=int, default=15)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    failure_path = Path(args.failure_output)
    progress_path = Path(args.progress_output)

    items = filter_items(load_responses(input_dir), args.runs)
    seen = already_judged(output_path)
    todo = [it for it in items if item_key(it) not in seen]

    progress = {
        "started_at": now(),
        "updated_at": now(),
        "input_dir": str(input_dir),
        "output": str(output_path),
        "failure_output": str(failure_path),
        "judge_model": args.judge_model,
        "runs_required": args.runs,
        "items_loaded": len(items),
        "items_required": args.require_items,
        "already_judged": len(seen),
        "todo_initial": len(todo),
        "success_this_run": 0,
        "permanent_failures_this_run": 0,
        "status": "starting",
    }
    write_progress(progress_path, progress)

    if args.require_items and len(items) != args.require_items:
        progress.update({
            "updated_at": now(),
            "status": "bad_input_count",
            "message": (
                f"Loaded {len(items)} items, expected {args.require_items}."
            ),
        })
        write_progress(progress_path, progress)
        print(progress["message"])
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.api_base, api_key=args.api_key,
                    timeout=args.timeout)

    print(f"Loaded {len(items)} responses from {input_dir}")
    print(f"Total {len(items)}, already judged {len(seen)}, todo {len(todo)}")

    consecutive_failures = 0
    permanent_failures = 0
    successes = 0

    with output_path.open("a", encoding="utf-8") as out_file, \
            failure_path.open("a", encoding="utf-8") as fail_file:
        for index, item in enumerate(todo, 1):
            success = False
            last_exc: Optional[Exception] = None

            for attempt in range(1, args.max_retries + 2):
                try:
                    result = call_judge(client, args.judge_model, item)
                    result["judge_attempts"] = attempt
                    result["judge_finished_at"] = now()
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    successes += 1
                    consecutive_failures = 0
                    success = True
                    print(f"[{index:>4}/{len(todo)}] OK "
                          f"{item.get('model')} {item.get('test_id')} "
                          f"run={item.get('run')} attempts={attempt}")
                    break
                except Exception as exc:
                    last_exc = exc
                    is_final = attempt > args.max_retries
                    rec = error_record(item, args.judge_model, attempt, exc,
                                       is_final)
                    fail_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fail_file.flush()
                    print(f"[{index:>4}/{len(todo)}] ERROR "
                          f"{item.get('test_id')} run={item.get('run')} "
                          f"attempt={attempt}: {exc}")
                    if is_final:
                        break
                    retry_after = rec.get("retry_after_seconds")
                    delay = retry_after if retry_after else args.retry_base * attempt
                    delay = min(float(delay), args.retry_max)
                    time.sleep(delay)

            if not success:
                permanent_failures += 1
                consecutive_failures += 1
                if last_exc is not None:
                    print(f"Permanent failure: {item_key(item)}: {last_exc}")

            progress.update({
                "updated_at": now(),
                "status": "running",
                "success_this_run": successes,
                "permanent_failures_this_run": permanent_failures,
                "completed_this_run": successes + permanent_failures,
                "remaining_this_run": len(todo) - successes - permanent_failures,
                "last_item": {
                    "model": item.get("model"),
                    "test_id": item.get("test_id"),
                    "run": item.get("run"),
                    "success": success,
                },
            })
            write_progress(progress_path, progress)

            if (args.stop_after_consecutive_failures > 0 and
                    consecutive_failures >= args.stop_after_consecutive_failures):
                progress.update({
                    "updated_at": now(),
                    "status": "stopped_consecutive_failures",
                    "consecutive_failures": consecutive_failures,
                })
                write_progress(progress_path, progress)
                return 4

            if args.sleep > 0:
                time.sleep(args.sleep)

    final_status = "complete" if permanent_failures == 0 else "complete_with_failures"
    progress.update({
        "updated_at": now(),
        "status": final_status,
        "success_this_run": successes,
        "permanent_failures_this_run": permanent_failures,
    })
    write_progress(progress_path, progress)
    return 0 if permanent_failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
