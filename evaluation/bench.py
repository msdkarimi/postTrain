"""Forgetting / regression benchmarks (lm-evaluation-harness) with FIXED settings, so every model is scored the same way.

    python -m evaluation.bench --model outputs/sft_lora/merged --name sft                 # full suite
    python -m evaluation.bench --model Qwen/Qwen3.5-0.8B-Base --name base --preset quick  # ~10x fewer questions, +-3-4 points of noise

Suite (what each one tells you):
  BFCL    function calling (single-turn AST + irrelevance)     our generation, BFCL's checker       (evaluation/bfcl.py; does the model call tools correctly?)
  IFEval  instruction following with verifiable constraints   generation, 0-shot, chat template   (does SFT hurt following instructions?)
  GSM8K   grade-school math word problems                      generation, 5-shot, chat template   (does SFT hurt reasoning?)
  MMLU    broad knowledge, multiple choice                      log-likelihood, 5-shot, no template (does SFT hurt knowledge?)
Run it for EVERY model you want to compare (base, instruct, each SFT variant) and compare the numbers; a small drop is normal, a
large one means catastrophic forgetting. Output: outputs/bench/<name>/bench.json (+ raw.json), read by evaluation/plot.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PRESETS = {"full": {"bfcl": None, "ifeval": None, "gsm8k": None, "mmlu": None},
           "quick": {"bfcl": 40, "ifeval": 150, "gsm8k": 200, "mmlu": 10}}      # limits are per category (BFCL) / per subject (MMLU)

# (display name, task, metric key prefix, filter or None)
HEADLINE = [("ifeval_prompt_strict", "ifeval", "prompt_level_strict_acc", None), ("ifeval_inst_strict", "ifeval", "inst_level_strict_acc", None),
            ("gsm8k_flexible", "gsm8k", "exact_match", "flexible-extract"), ("gsm8k_strict", "gsm8k", "exact_match", "strict-match"),
            ("mmlu", "mmlu", "acc", None)]


def extract_scores(results: dict) -> dict:
    """Pull the headline metrics (value, stderr, n) out of an lm-eval results dict."""
    res, nsamp = results["results"], results.get("n-samples", {})
    out = {}
    for name, task, metric, flt in HEADLINE:
        if task not in res:
            continue
        key = f"{metric},{flt or 'none'}"
        if key not in res[task]:
            continue
        stderr = res[task].get(f"{metric}_stderr,{flt or 'none'}")
        n = nsamp.get(task, {}).get("effective") if task in nsamp else sum(v.get("effective", 0) for k, v in nsamp.items() if k.startswith(task + "_"))
        out[name] = {"value": float(res[task][key]), "stderr": float(stderr) if isinstance(stderr, (int, float)) else None, "n": n or None}
    return out


def run_task(model_args: str, task: str, shots: int, chat: bool, limit, batch_size: int, device: str) -> dict:
    import lm_eval
    return lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=[task], num_fewshot=shots, batch_size=batch_size, device=device, limit=limit,
                                   apply_chat_template=chat, fewshot_as_multiturn=chat and shots > 0, log_samples=False,
                                   random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="outputs/bench")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--tasks", default="bfcl,ifeval,mmlu", help="any of bfcl,ifeval,mmlu,gsm8k (gsm8k is optional, off by default)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="override: at most N questions per task (N per subject for MMLU); for smoke tests")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype if device != "cpu" else "float32"
    margs = f"pretrained={args.model},dtype={dtype}" + (f",peft={args.adapter}" if args.adapter else "")
    plan = {"ifeval": (0, True), "gsm8k": (5, True), "mmlu": (5, False)}          # shots, chat template
    limits = {k: (args.limit or v) for k, v in PRESETS[args.preset].items()}
    scores, raw = {}, {}
    for task in [t for t in args.tasks.split(",") if t]:
        if task == "bfcl":
            from evaluation import bfcl
            from evaluation.generate import HFGenerator
            print(f"== bfcl: limit {limits['bfcl']} per category", flush=True)
            gen = HFGenerator(args.model, args.adapter, batch_size=args.batch_size, max_new_tokens=256)
            res = bfcl.evaluate(gen, limit=limits["bfcl"] or 0)
            scores.update(res["scores"])
            raw["bfcl"] = res["failures"]
            del gen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        shots, chat = plan["mmlu" if task.startswith("mmlu") else task]               # mmlu_<subject> allowed for quick checks
        lim = limits["mmlu" if task.startswith("mmlu") else task]
        print(f"== {task}: {shots}-shot, chat template {chat}, limit {lim}", flush=True)
        r = run_task(margs, task, shots, chat, lim, args.batch_size, device)
        scores.update(extract_scores(r))
        raw[task] = {"results": r["results"], "n-samples": r.get("n-samples", {})}
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    import lm_eval
    meta = {"model": args.model, "adapter": args.adapter, "preset": args.preset, "limits": limits, "settings": {k: {"shots": v[0], "chat_template": v[1]} for k, v in plan.items()},
            "lm_eval_version": getattr(lm_eval, "__version__", "?"), "scores": scores}
    (out / "bench.json").write_text(json.dumps(meta, indent=2))
    (out / "raw.json").write_text(json.dumps(raw, indent=2, default=str))
    print("\n" + "\n".join(f"{k:26s} {100 * v['value']:5.1f}%  (+-{100 * (v['stderr'] or 0):.1f}, n={v['n']})" for k, v in scores.items()))
    print(f"wrote {out}/bench.json")


if __name__ == "__main__":
    main()
