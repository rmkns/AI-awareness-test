from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def load_candidates(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) not in (5, 6):
            raise SystemExit(
                "Bad candidate line, expected "
                f"label|model|quant|dtype|device[|trust_remote_code]: {raw}"
            )
        rows.append({
            "label": parts[0],
            "model": parts[1],
            "quantization": parts[2],
            "dtype": parts[3],
            "device": parts[4],
            "trust_remote_code": parts[5] if len(parts) == 6 else "0",
        })
    return rows


def quant_args(quantization: str) -> List[str]:
    if quantization == "4bit":
        return ["--load-in-4bit"]
    if quantization == "8bit":
        return ["--load-in-8bit"]
    if quantization == "none":
        return []
    raise SystemExit(f"Unknown quantization: {quantization}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-working", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-memory-gpu", default="8GiB")
    parser.add_argument("--max-memory-cpu", default="48GiB")
    parser.add_argument("--offload-root", default="")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "model_check_results.jsonl"
    csv_path = out / "model_check_summary.csv"
    working_path = out / "working_models.txt"
    top_path = out / f"working_models_top{args.top_n}.txt"
    logs_dir = out / "logs"
    smoke_dir = out / "smoke_outputs"
    logs_dir.mkdir(exist_ok=True)
    smoke_dir.mkdir(exist_ok=True)

    candidates = load_candidates(Path(args.models_file))
    results: List[Dict] = []

    with jsonl_path.open("a", encoding="utf-8") as jf:
        for candidate in candidates:
            label = candidate["label"]
            smoke_out = smoke_dir / f"{label}.jsonl"
            log_path = logs_dir / f"{label}.log"
            offload_folder = (
                Path(args.offload_root) / label
                if args.offload_root
                else out / "offload" / label
            )
            offload_folder.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python,
                args.evaluator,
                "--model", candidate["model"],
                "--tests-dir", args.tests_dir,
                "--output", str(smoke_out),
                "--runs", "1",
                "--max-tests", "1",
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", "0.0",
                "--device", candidate["device"],
                "--dtype", candidate["dtype"],
                "--max-memory-gpu", args.max_memory_gpu,
                "--max-memory-cpu", args.max_memory_cpu,
                "--offload-folder", str(offload_folder),
                "--attn-implementation", args.attn_implementation,
                *quant_args(candidate["quantization"]),
            ]
            if args.use_cache:
                cmd.append("--use-cache")
            if candidate.get("trust_remote_code") in {"1", "true", "yes"}:
                cmd.append("--trust-remote-code")
            row: Dict = {
                **candidate,
                "ok": False,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "smoke_output": str(smoke_out),
                "log": str(log_path),
            }
            print(f"TEST {label} | {candidate['model']}")
            with log_path.open("w", encoding="utf-8") as log:
                log.write("COMMAND:\n")
                log.write(" ".join(cmd) + "\n\n")
                try:
                    child_env = os.environ.copy()
                    child_env.setdefault("PYTHONUTF8", "1")
                    child_env.setdefault("PYTHONIOENCODING", "utf-8")
                    child_env.setdefault("PYTHONUNBUFFERED", "1")
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=args.timeout,
                        env=child_env,
                    )
                    log.write("STDOUT:\n")
                    log.write(proc.stdout)
                    log.write("\n\nSTDERR:\n")
                    log.write(proc.stderr)
                    row.update({
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-2000:],
                        "stderr_tail": proc.stderr[-2000:],
                    })
                    if proc.returncode == 0 and smoke_out.exists():
                        lines = [ln for ln in smoke_out.read_text(
                            encoding="utf-8").splitlines() if ln.strip()]
                        row["ok"] = len(lines) >= 1
                        row["records"] = len(lines)
                    else:
                        row["records"] = 0
                    print(("OK  " if row["ok"] else "FAIL") +
                          f" {label} rc={proc.returncode}")
                except subprocess.TimeoutExpired as exc:
                    log.write(f"\nTIMEOUT after {args.timeout}s\n")
                    row.update({
                        "returncode": "timeout",
                        "records": 0,
                        "error": str(exc)[:2000],
                    })
                    print(f"FAIL {label} timeout")

            row["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            results.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "label", "model", "quantization", "dtype", "device", "ok",
            "trust_remote_code", "returncode", "records", "log",
            "smoke_output",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    working = [row for row in results if row.get("ok")]
    working_path.write_text(
        "".join(
            f"{row['label']}|{row['model']}|{row['quantization']}|"
            f"{row['dtype']}|{row['device']}|"
            f"{row.get('trust_remote_code', '0')}\n"
            for row in working
        ),
        encoding="utf-8",
    )
    top = working[: args.top_n]
    top_path.write_text(
        "".join(
            f"{row['label']}|{row['model']}|{row['quantization']}|"
            f"{row['dtype']}|{row['device']}|"
            f"{row.get('trust_remote_code', '0')}\n"
            for row in top
        ),
        encoding="utf-8",
    )
    (out / "model_check_manifest.json").write_text(
        json.dumps({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_file": str(Path(args.models_file).resolve()),
            "n_candidates": len(candidates),
            "n_working": len(working),
            "min_working": args.min_working,
            "top_n": args.top_n,
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
