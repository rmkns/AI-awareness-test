from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: install torch + transformers:\n"
          "  pip install torch transformers accelerate", file=sys.stderr)
    raise


def load_model(model_name: str, device: str = "auto",
               dtype: str = "auto",
               quantization: str = "none",
               trust_remote_code: bool = False,
               max_memory_gpu: str = "",
               max_memory_cpu: str = "48GiB",
               offload_folder: str = "",
               attn_implementation: str = "eager",
               use_cache: bool = False):
    print(f"Loading {model_name} (device={device}, dtype={dtype}, "
          f"quantization={quantization})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = (torch.float16 if dtype == "fp16" else
                   torch.bfloat16 if dtype == "bf16" else
                   "auto")

    quant_config = None
    if quantization in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "bitsandbytes / BitsAndBytesConfig nepasiekiamas. "
                "Instaliuok: pip install bitsandbytes>=0.43"
            ) from e
        compute_dtype = torch.float16 if dtype != "bf16" else torch.bfloat16
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None

    if device == "cpu":
        kwargs = {"device_map": {"": "cpu"}}
    elif device == "cuda":
        kwargs = {"device_map": {"": "cuda:0"}}
    else:
        kwargs = {"device_map": device}
        if max_memory_gpu:
            kwargs["max_memory"] = {0: max_memory_gpu, "cpu": max_memory_cpu}
        if offload_folder:
            kwargs["offload_folder"] = offload_folder
            kwargs["offload_state_dict"] = True
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    model.config.use_cache = use_cache
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_with_logits(tokenizer, model, prompt: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.0) -> Dict:
    model_device = next(model.parameters()).device
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
            inputs = inputs["input_ids"]
        inputs = inputs.to(model_device)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        inputs = enc["input_ids"].to(model_device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0 else 1.0,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs.sequences[0, inputs.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    per_token: List[Dict] = []
    for pos, logits in enumerate(outputs.scores):
        probs = F.softmax(logits[0], dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-12))
        entropy = float(-(probs * log_probs).sum().item())

        top2 = torch.topk(probs, k=2)
        p1 = float(top2.values[0].item())
        p2 = float(top2.values[1].item())

        chosen = int(generated_ids[pos].item())
        per_token.append({
            "pos": pos,
            "token_id": chosen,
            "token_str": tokenizer.decode([chosen]),
            "entropy": entropy,
            "p_chosen": float(probs[chosen].item()),
            "top1_p": p1,
            "top2_p": p2,
            "margin": p1 - p2,
            "top1_token": tokenizer.decode([int(top2.indices[0].item())]),
        })

    if not per_token:
        return {
            "response": response_text,
            "tokens": [],
            "n_tokens": 0,
            "mean_entropy": 0.0,
            "max_entropy": 0.0,
            "min_p_chosen": 1.0,
            "low_conf_count": 0,
            "low_conf_ratio": 0.0,
        }

    n = len(per_token)
    mean_h = sum(t["entropy"] for t in per_token) / n
    max_h = max(t["entropy"] for t in per_token)
    min_p = min(t["p_chosen"] for t in per_token)
    low = sum(1 for t in per_token if t["p_chosen"] < 0.5)

    return {
        "response": response_text,
        "tokens": per_token,
        "n_tokens": n,
        "mean_entropy": mean_h,
        "max_entropy": max_h,
        "min_p_chosen": min_p,
        "low_conf_count": low,
        "low_conf_ratio": low / n,
    }


def load_test_prompts(tests_dir: Path) -> List[Dict]:
    prompts = []
    for path in sorted(tests_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        prompts.append({
            "test_id": path.stem,
            "prompt": text,
        })
    return prompts


def already_done(output_path: Path) -> set:
    seen = set()
    if not output_path.exists():
        return seen
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                seen.add((r.get("test_id"), r.get("run")))
            except json.JSONDecodeError:
                continue
    return seen


def evaluate_all(tokenizer, model, prompts: List[Dict],
                 output_path: Path, runs: int = 5,
                 max_new_tokens: int = 512,
                 temperature: float = 0.0,
                 keep_full_tokens: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = already_done(output_path)
    todo = [(p, r) for p in prompts for r in range(runs)
            if (p["test_id"], r) not in seen]
    print(f"Total {len(prompts)*runs} combos, "
          f"already done {len(seen)}, todo {len(todo)}")

    out = output_path.open("a", encoding="utf-8")
    try:
        for i, (p, run) in enumerate(todo, 1):
            t0 = time.time()
            result = generate_with_logits(
                tokenizer, model, p["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - t0

            record = {
                "model": model.config._name_or_path,
                "test_id": p["test_id"],
                "run": run,
                "response": result["response"],
                "n_tokens": result["n_tokens"],
                "mean_entropy": result["mean_entropy"],
                "max_entropy": result["max_entropy"],
                "min_p_chosen": result["min_p_chosen"],
                "low_conf_count": result["low_conf_count"],
                "low_conf_ratio": result["low_conf_ratio"],
                "elapsed_s": elapsed,
            }
            if keep_full_tokens:
                record["tokens"] = result["tokens"]

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(f"[{i:>4}/{len(todo)}] "
                  f"{p['test_id']:<32} run={run} "
                  f"Hmean={result['mean_entropy']:.3f} "
                  f"Hmax={result['max_entropy']:.3f} "
                  f"low%={result['low_conf_ratio']*100:.1f} "
                  f"({elapsed:.1f}s)")
    finally:
        out.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HuggingFace model id")
    p.add_argument("--tests-dir", required=True,
                   help="Directory with Project A *.txt test prompts")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--runs", type=int, default=5,
                   help="Repetitions per test (default 5)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy decoding")
    p.add_argument("--device", default="auto",
                   help="cuda / cpu / auto (default auto)")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "fp16", "bf16"])
    quant = p.add_mutually_exclusive_group()
    quant.add_argument("--load-in-4bit", action="store_true",
                       help="bitsandbytes nf4 (7B models fit in ~5 GB VRAM)")
    quant.add_argument("--load-in-8bit", action="store_true",
                       help="bitsandbytes 8-bit (7B models fit in ~8 GB VRAM)")
    p.add_argument("--keep-full-tokens", action="store_true",
                   help="Save per-token detail (large file!)")
    p.add_argument("--max-tests", type=int, default=None,
                   help="Limit number of tests (debug)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow custom model code from the HF repo")
    p.add_argument("--max-memory-gpu", default="",
                   help="Accelerate max_memory for GPU 0, e.g. 8GiB")
    p.add_argument("--max-memory-cpu", default="48GiB",
                   help="Accelerate CPU max_memory when offloading")
    p.add_argument("--offload-folder", default="",
                   help="Folder for accelerate CPU/disk offload")
    p.add_argument("--attn-implementation", default="eager",
                   help="Attention implementation, default eager for stability")
    p.add_argument("--use-cache", action="store_true",
                   help="Enable KV cache during generation")
    args = p.parse_args()

    quantization = ("4bit" if args.load_in_4bit
                    else "8bit" if args.load_in_8bit
                    else "none")
    tokenizer, model = load_model(args.model, device=args.device,
                                  dtype=args.dtype,
                                  quantization=quantization,
                                  trust_remote_code=args.trust_remote_code,
                                  max_memory_gpu=args.max_memory_gpu,
                                  max_memory_cpu=args.max_memory_cpu,
                                  offload_folder=args.offload_folder,
                                  attn_implementation=args.attn_implementation,
                                  use_cache=args.use_cache)
    prompts = load_test_prompts(Path(args.tests_dir))
    if args.max_tests is not None:
        prompts = prompts[:args.max_tests]
    print(f"Loaded {len(prompts)} test prompts")

    evaluate_all(tokenizer, model, prompts, Path(args.output),
                 runs=args.runs,
                 max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature,
                 keep_full_tokens=args.keep_full_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
