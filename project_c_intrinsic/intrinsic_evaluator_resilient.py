from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from intrinsic_evaluator import (
    generate_with_logits,
    load_model,
    load_test_prompts,
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def already_done(output_path: Path) -> set:
    seen = set()
    if not output_path.exists():
        return seen
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((row.get("test_id"), row.get("run")))
    return seen


def write_progress(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def quantization_from_args(args) -> str:
    if args.load_in_4bit:
        return "4bit"
    if args.load_in_8bit:
        return "8bit"
    return "none"


def failure_record(prompt: Dict, run: int, attempt: int, exc: Exception,
                   final: bool, model_label: str, model_name: str) -> Dict:
    return {
        "ts": now(),
        "model_label": model_label,
        "model": model_name,
        "test_id": prompt.get("test_id"),
        "run": run,
        "attempt": attempt,
        "final": final,
        "error_type": type(exc).__name__,
        "message": str(exc)[:4000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--failure-output", required=True)
    parser.add_argument("--progress-output", required=True)
    parser.add_argument("--require-items", type=int, default=400)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--keep-full-tokens", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-memory-gpu", default="")
    parser.add_argument("--max-memory-cpu", default="48GiB")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--use-cache", action="store_true")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load-in-4bit", action="store_true")
    quant.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    failure_path = Path(args.failure_output)
    progress_path = Path(args.progress_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_test_prompts(Path(args.tests_dir))
    if args.max_tests is not None:
        prompts = prompts[: args.max_tests]

    combos = [(prompt, run) for prompt in prompts for run in range(args.runs)]
    seen = already_done(output_path)
    todo = [(prompt, run) for prompt, run in combos
            if (prompt["test_id"], run) not in seen]

    progress = {
        "started_at": now(),
        "updated_at": now(),
        "model_label": args.model_label,
        "model": args.model,
        "quantization": quantization_from_args(args),
        "dtype": args.dtype,
        "device": args.device,
        "tests_dir": str(Path(args.tests_dir).resolve()),
        "output": str(output_path.resolve()),
        "failure_output": str(failure_path.resolve()),
        "runs": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "keep_full_tokens": args.keep_full_tokens,
        "trust_remote_code": args.trust_remote_code,
        "max_memory_gpu": args.max_memory_gpu,
        "max_memory_cpu": args.max_memory_cpu,
        "offload_folder": args.offload_folder,
        "attn_implementation": args.attn_implementation,
        "use_cache": args.use_cache,
        "items_required": args.require_items,
        "items_loaded": len(combos),
        "already_done": len(seen),
        "todo_initial": len(todo),
        "success_this_run": 0,
        "permanent_failures_this_run": 0,
        "status": "loading_model",
    }
    write_progress(progress_path, progress)

    if args.require_items and len(combos) != args.require_items:
        progress.update({
            "updated_at": now(),
            "status": "bad_input_count",
            "message": f"Loaded {len(combos)} combos, expected {args.require_items}.",
        })
        write_progress(progress_path, progress)
        print(progress["message"])
        return 2

    tokenizer, model = load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=quantization_from_args(args),
        trust_remote_code=args.trust_remote_code,
        max_memory_gpu=args.max_memory_gpu,
        max_memory_cpu=args.max_memory_cpu,
        offload_folder=args.offload_folder,
        attn_implementation=args.attn_implementation,
        use_cache=args.use_cache,
    )
    model_name = getattr(model.config, "_name_or_path", args.model)

    progress.update({"updated_at": now(), "status": "running"})
    write_progress(progress_path, progress)
    print(f"Total {len(combos)} combos, already done {len(seen)}, todo {len(todo)}")

    successes = 0
    failures = 0
    with output_path.open("a", encoding="utf-8") as out, \
            failure_path.open("a", encoding="utf-8") as fail:
        for index, (prompt, run) in enumerate(todo, 1):
            ok = False
            last_exc: Optional[Exception] = None
            for attempt in range(1, args.max_retries + 2):
                try:
                    t0 = time.time()
                    result = generate_with_logits(
                        tokenizer,
                        model,
                        prompt["prompt"],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    elapsed = time.time() - t0
                    record = {
                        "model": model_name,
                        "model_label": args.model_label,
                        "test_id": prompt["test_id"],
                        "run": run,
                        "response": result["response"],
                        "n_tokens": result["n_tokens"],
                        "mean_entropy": result["mean_entropy"],
                        "max_entropy": result["max_entropy"],
                        "min_p_chosen": result["min_p_chosen"],
                        "low_conf_count": result["low_conf_count"],
                        "low_conf_ratio": result["low_conf_ratio"],
                        "elapsed_s": elapsed,
                        "quantization": quantization_from_args(args),
                        "dtype": args.dtype,
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "trust_remote_code": args.trust_remote_code,
                        "max_memory_gpu": args.max_memory_gpu,
                        "max_memory_cpu": args.max_memory_cpu,
                        "offload_folder": args.offload_folder,
                        "attn_implementation": args.attn_implementation,
                        "finished_at": now(),
                    }
                    if args.keep_full_tokens:
                        record["tokens"] = result["tokens"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    successes += 1
                    ok = True
                    print(f"[{index:>4}/{len(todo)}] OK "
                          f"{prompt['test_id']} run={run} "
                          f"H={result['mean_entropy']:.3f} "
                          f"low={result['low_conf_ratio']*100:.1f}% "
                          f"({elapsed:.1f}s)")
                    break
                except Exception as exc:
                    last_exc = exc
                    is_final = attempt > args.max_retries
                    fail.write(json.dumps(
                        failure_record(prompt, run, attempt, exc, is_final,
                                       args.model_label, args.model),
                        ensure_ascii=False,
                    ) + "\n")
                    fail.flush()
                    print(f"[{index:>4}/{len(todo)}] ERROR "
                          f"{prompt['test_id']} run={run} "
                          f"attempt={attempt}: {exc}")
                    if not is_final:
                        time.sleep(args.retry_sleep)

            if not ok:
                failures += 1
                if last_exc is not None:
                    print(f"Permanent failure {prompt['test_id']} run={run}: "
                          f"{last_exc}")

            progress.update({
                "updated_at": now(),
                "status": "running",
                "success_this_run": successes,
                "permanent_failures_this_run": failures,
                "completed_this_run": successes + failures,
                "remaining_this_run": len(todo) - successes - failures,
                "last_item": {
                    "test_id": prompt["test_id"],
                    "run": run,
                    "success": ok,
                },
            })
            write_progress(progress_path, progress)

    progress.update({
        "updated_at": now(),
        "status": "complete" if failures == 0 else "complete_with_failures",
        "success_this_run": successes,
        "permanent_failures_this_run": failures,
    })
    write_progress(progress_path, progress)
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    raise SystemExit(main())
