from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: install the OpenAI Python SDK first:\n"
          "  pip install openai", file=sys.stderr)
    raise

from rubric import build_judge_prompt, parse_judge_response


def _load_one_result_file(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  skipping {path.name}: {exc}", file=sys.stderr)
        return None

    test_id = (data.get("test_id") or data.get("test") or
               data.get("test_name") or path.stem.split("_run")[0])
    model = data.get("model") or data.get("model_name", "unknown")
    run = int(data.get("run") or data.get("run_index") or data.get("repeat") or 0)
    prompt = (data.get("user_message") or data.get("prompt") or
              data.get("test_prompt") or data.get("user") or
              data.get("full_dialogue") or "")
    response = (data.get("assistant_response") or data.get("response") or
                data.get("answer") or data.get("final_answer") or
                data.get("assistant", ""))
    keyword_scores = (data.get("scores") or data.get("keyword_scores") or
                      data.get("indicators") or {})

    if not prompt or not response:
        return None
    return {
        "test_id": test_id,
        "model": model,
        "run": run,
        "test_prompt": prompt,
        "model_response": response,
        "keyword_scores": keyword_scores,
    }


def load_responses(results_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name.startswith("_"):
            continue
        item = _load_one_result_file(json_path)
        if item is not None:
            items.append(item)
    return items


def already_judged(output_path: Path) -> set:
    seen = set()
    if not output_path.exists():
        return seen
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                key = (r.get("model"), r.get("test_id"), r.get("run"))
                seen.add(key)
            except json.JSONDecodeError:
                continue
    return seen


def call_judge(client: OpenAI, judge_model: str, item: Dict,
               temperature: float = 0.0,
               max_tokens: int = 600) -> Dict:
    prompt = build_judge_prompt(
        test_prompt=item["test_prompt"],
        model_response=item["model_response"],
    )
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = completion.choices[0].message.content or ""
    parsed = parse_judge_response(raw)
    return {
        **item,
        "judge_model": judge_model,
        "judge_raw": raw,
        "judge_scores": parsed.to_dict(),
    }


def evaluate_all(items: List[Dict], client: OpenAI, judge_model: str,
                 output_path: Path, max_workers: int = 4,
                 sleep_between: float = 0.3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = already_judged(output_path)
    todo = [it for it in items
            if (it["model"], it["test_id"], it["run"]) not in seen]
    print(f"Total {len(items)}, already judged {len(seen)}, todo {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    out_file = output_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(call_judge, client, judge_model, it): it
                       for it in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    result = fut.result()
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    print(f"[{i:>4}/{len(todo)}] "
                          f"{result['model']:<14} "
                          f"test={result['test_id']:<32} "
                          f"run={result['run']}: {result['judge_scores']}")
                except Exception as exc:
                    print(f"  ERROR on {item.get('test_id')} "
                          f"run={item.get('run')}: {exc}", file=sys.stderr)
                    time.sleep(2)
                if sleep_between > 0:
                    time.sleep(sleep_between)
    finally:
        out_file.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory with Project A *.json result files")
    p.add_argument("--output", required=True,
                   help="Output JSONL path (created/appended)")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini",
                   help="Judge model identifier (e.g. openai/gpt-4o-mini, "
                        "anthropic/claude-3.5-sonnet, "
                        "meta-llama/llama-3.3-70b-instruct)")
    p.add_argument("--api-base", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL "
                        "(default OpenRouter; use OpenAI's URL for direct)")
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
                   help="API key (default: OPENROUTER_API_KEY env var)")
    p.add_argument("--max-items", type=int, default=None,
                   help="Limit number of items (debug)")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel workers (default 4)")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Sleep between requests (rate-limit safety)")
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: API key not provided. Set OPENROUTER_API_KEY or use --api-key",
              file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    items = load_responses(Path(args.input_dir))
    print(f"Loaded {len(items)} responses from {args.input_dir}")
    if args.max_items is not None:
        items = items[:args.max_items]

    evaluate_all(items, client, args.judge_model, Path(args.output),
                 max_workers=args.max_workers,
                 sleep_between=args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
