"""BFCL (Berkeley Function-Calling Leaderboard) single-turn function calling, scored with BFCL's own AST checker.

We use the `bfcl-eval` package only for its questions, ground truth and `ast_checker`; generation is ours (HF generate + the Qwen3.5
chat template with `tools=`), so nothing needs vLLM or a server. Categories (BFCL v4 "non-live" AST + relevance):
  simple_python           one function offered, one call expected
  multiple                several functions offered, one call expected (pick the right one)
  parallel                one function, several calls in one reply
  parallel_multiple       several functions, several calls
  irrelevance             no function fits: the right answer is NOT to call anything
Score = unweighted mean of the category accuracies (BFCL's own summary does the same). Multi-turn / agentic / live categories are not run.
Numbers are comparable between OUR runs of different models, not to the public leaderboard (different prompt and decoding).

    python -m evaluation.bfcl --model Qwen/Qwen3.5-0.8B --name qwen-instruct [--limit 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import types
from pathlib import Path

from evaluation.parse import parse_assistant_output

CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]
MODEL_KEY = "qwen3-0.6b"            # only used by BFCL's name-mangling hook; we stub the registry (see _load_checker)


def _load_checker():
    """Import BFCL's AST checker without its 40-package dependency tree (every vendor SDK) by stubbing the model registry."""
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.MODEL_CONFIG_MAPPING = collections.defaultdict(lambda: types.SimpleNamespace(underscore_to_dot=False))
    sys.modules.setdefault("bfcl_eval.constants.model_config", stub)
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    return ast_checker, Language.PYTHON


def _data_dir() -> Path:
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data"


def load_category(cat: str, limit: int = 0) -> list:
    d = _data_dir()
    rows = [json.loads(line) for line in (d / f"BFCL_v4_{cat}.json").read_text().splitlines() if line.strip()]
    truth = {}
    if cat != "irrelevance":
        lines = (d / "possible_answer" / f"BFCL_v4_{cat}.json").read_text().splitlines()
        truth = {r["id"]: r["ground_truth"] for r in map(json.loads, (x for x in lines if x.strip()))}
    rows = rows[:limit] if limit else rows
    return [{"id": r["id"], "messages": r["question"][0], "functions": r["function"], "truth": truth.get(r["id"])} for r in rows]


def to_tools(functions: list) -> list:
    out = []
    for f in functions:
        f = json.loads(json.dumps(f))
        if f.get("parameters", {}).get("type") == "dict":
            f["parameters"]["type"] = "object"
        out.append({"type": "function", "function": f})
    return out


class _Prompt:
    def __init__(self, messages, tools):
        self.messages, self.tools = messages, tools

    def prompt_text(self, tok):
        return tok.apply_chat_template(self.messages, tools=self.tools, tokenize=False, add_generation_prompt=True)


def score_item(item: dict, parsed_calls: list, cat: str, ast_checker, language) -> tuple:
    """-> (ok, reason)"""
    if cat == "irrelevance":
        return (not parsed_calls, "ok" if not parsed_calls else "called a function when none fits")
    if not parsed_calls:
        return False, "no function call"
    model_output = [{c["name"]: c["arguments"]} for c in parsed_calls]
    res = ast_checker(item["functions"], model_output, item["truth"], language, cat, MODEL_KEY)
    return bool(res["valid"]), "ok" if res["valid"] else "; ".join(map(str, res.get("error", [])))[:200]


def evaluate(generate, categories=CATEGORIES, limit: int = 0) -> dict:
    """generate(list[_Prompt]) -> list[str]. Returns {"scores": {...}, "failures": [...]}."""
    ast_checker, language = _load_checker()
    scores, failures, accs = {}, [], []
    for cat in categories:
        items = load_category(cat, limit)
        tools = [to_tools(it["functions"]) for it in items]
        outs = generate([_Prompt(it["messages"], t) for it, t in zip(items, tools)])
        n_ok = 0
        for it, t, text in zip(items, tools, outs):
            parsed = parse_assistant_output(text or "", t)
            ok, why = score_item(it, parsed.calls, cat, ast_checker, language)
            n_ok += ok
            if not ok and len(failures) < 200:
                failures.append({"id": it["id"], "reason": why, "output": (text or "")[:300]})
        acc = n_ok / max(len(items), 1)
        accs.append(acc)
        se = (acc * (1 - acc) / max(len(items), 1)) ** 0.5
        scores[f"bfcl_{cat}"] = {"value": acc, "stderr": se, "n": len(items)}
    mean = sum(accs) / max(len(accs), 1)
    n = sum(s["n"] for s in scores.values())
    se = (sum(s["stderr"] ** 2 for s in scores.values()) ** 0.5) / max(len(scores), 1)
    scores["bfcl_ast"] = {"value": mean, "stderr": se, "n": n}
    return {"scores": scores, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="outputs/bench")
    ap.add_argument("--limit", type=int, default=0, help="at most N questions per category (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    from evaluation.generate import HFGenerator
    gen = HFGenerator(args.model, args.adapter, batch_size=args.batch_size, max_new_tokens=256)
    res = evaluate(gen, limit=args.limit)
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    merge_scores(out, res["scores"], {"bfcl": {"categories": CATEGORIES, "limit": args.limit}})
    (out / "bfcl_failures.json").write_text(json.dumps(res["failures"], indent=2))
    print("\n".join(f"{k:26s} {100 * v['value']:5.1f}%  (n={v['n']})" for k, v in res["scores"].items()))


def merge_scores(out: Path, scores: dict, extra_settings: dict) -> None:
    """Add scores to outputs/bench/<name>/bench.json (shared with evaluation.bench) without clobbering what is already there."""
    p = out / "bench.json"
    meta = json.loads(p.read_text()) if p.exists() else {"preset": "n/a", "settings": {}, "scores": {}}
    meta["scores"].update(scores)
    meta["settings"].update(extra_settings)
    p.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
