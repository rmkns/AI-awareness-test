from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


def load_candidates(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise SystemExit(f"Bad candidate line, expected label|model: {raw}")
        label, model = [part.strip() for part in line.split("|", 1)]
        rows.append((label, model))
    return rows


def error_payload(exc: Exception) -> Dict:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return {
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "message": str(exc)[:2000],
        "headers": dict(headers) if headers is not None else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--min-working", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "model_check_results.jsonl"
    csv_path = out / "model_check_summary.csv"
    working_path = out / "working_models.txt"
    top_path = out / f"working_models_top{args.top_n}.txt"
    manifest_path = out / "model_check_manifest.json"

    candidates = load_candidates(Path(args.models_file))
    client = OpenAI(base_url=args.api_base, api_key=args.api_key,
                    timeout=args.timeout)

    prompt = (
        "Atsakyk tik JSON formatu: "
        "{\"ok\": true, \"model_role\": \"judge\", \"note\": \"trumpai\"}"
    )
    results: List[Dict] = []
    with jsonl_path.open("a", encoding="utf-8") as jf:
        for label, model in candidates:
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            row: Dict = {
                "label": label,
                "model": model,
                "started_at": started,
                "ok": False,
            }
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=args.max_tokens,
                )
                content = completion.choices[0].message.content or ""
                row.update({
                    "ok": True,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_model": getattr(completion, "model", None),
                    "response_preview": content[:500],
                    "usage": (
                        completion.usage.model_dump()
                        if getattr(completion, "usage", None) is not None
                        else None
                    ),
                })
                print(f"OK   {label} | {model}")
            except Exception as exc:
                row.update({
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": error_payload(exc),
                })
                print(f"FAIL {label} | {model}: {exc}")
            results.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=["label", "model", "ok", "response_model", "status_code",
                        "message"],
        )
        writer.writeheader()
        for row in results:
            err = row.get("error") or {}
            writer.writerow({
                "label": row["label"],
                "model": row["model"],
                "ok": row["ok"],
                "response_model": row.get("response_model", ""),
                "status_code": err.get("status_code", ""),
                "message": err.get("message", ""),
            })

    working = [(r["label"], r["model"]) for r in results if r.get("ok")]
    working_path.write_text(
        "".join(f"{label}|{model}\n" for label, model in working),
        encoding="utf-8",
    )
    top = working[: args.top_n]
    top_path.write_text(
        "".join(f"{label}|{model}\n" for label, model in top),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_file": str(Path(args.models_file).resolve()),
            "n_candidates": len(candidates),
            "n_working": len(working),
            "min_working": args.min_working,
            "top_n": args.top_n,
            "working_models_file": str(working_path.resolve()),
            "top_models_file": str(top_path.resolve()),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(working) < args.min_working:
        print(f"Only {len(working)} working models; need {args.min_working}.")
        return 2
    print(f"Working models: {len(working)}. Top file: {top_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
