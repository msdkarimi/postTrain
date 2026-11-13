"""Evaluate a model on the held-out agent tasks (data/raw/agent_test_tasks.jsonl).

    python -m evaluation.run_eval --model outputs/sft_lora/merged --out outputs/eval/sft
    python -m evaluation.run_eval --model Qwen/Qwen3.5-0.8B-Base --out outputs/eval/base --per-family 4
    python -m evaluation.run_eval --model Qwen/Qwen3.5-0.8B-Base --oracle --out outputs/eval/oracle   # harness self-test (must be 100%)
    python -m evaluation.compare outputs/eval/base outputs/eval/sft

Each task is regenerated from (family, seed), so the environment and the checker are exactly those used to build the data. The model
drives the episode with real tool calls; the checker decides pass/fail from the final state (files, calendar, drafts, tasks) and reply.

SAFETY: tasks of kind "sandbox" make the model run its own Python / bash on this machine (not isolated from files or network).
They are skipped unless you pass --allow-code-exec; do that only on a throwaway VM or in a container. The function-calling
("api") tasks run against an in-process simulation and are safe anywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from distill.build_trajectories import task_signature
from evaluation.agent import AgentRun


def summarize(results: list) -> dict:
    def block(rs):
        n = len(rs)
        return {"n": n, "pass": sum(r["ok"] for r in rs), "pass_rate": round(sum(r["ok"] for r in rs) / max(n, 1), 4)}

    def group(key):
        g = defaultdict(list)
        for r in results:
            g[r[key]].append(r)
        return {k: block(v) for k, v in sorted(g.items())}

    tags = defaultdict(list)
    for r in results:
        for t in r["tags"]:
            tags[t].append(r)
    n_msgs = max(sum(r["n_assistant_msgs"] for r in results), 1)
    reasons = defaultdict(int)
    for r in results:
        if not r["ok"]:
            reasons[r["reason"].split(":")[0]] += 1
    return {"overall": {**block(results), "format_error_rate": round(sum(r["format_errors"] for r in results) / n_msgs, 4),
                        "tool_error_rate": round(sum(r["tool_errors"] for r in results) / max(sum(r["n_calls"] for r in results), 1), 4),
                        "avg_tool_calls": round(sum(r["n_calls"] for r in results) / max(len(results), 1), 2),
                        "step_cap_hits": sum(r["reason"] == "step_cap" for r in results)},
            "by_kind": group("kind"), "by_category": group("category"), "by_difficulty": group("difficulty"), "by_family": group("family"),
            "by_skill": {t: block(v) for t, v in sorted(tags.items()) if len(v) >= 5}, "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}


def markdown(name: str, s: dict) -> str:
    o = s["overall"]
    lines = [f"# {name}", "", f"**Pass rate {100 * o['pass_rate']:.1f}%** ({o['pass']}/{o['n']}) | tool-call format errors per assistant message: {o['format_error_rate']:.3f} | "
             f"tool/API error rate per call: {o['tool_error_rate']:.3f} | avg tool calls: {o['avg_tool_calls']} | step-cap hits: {o['step_cap_hits']}", ""]
    for title, key in (("By kind", "by_kind"), ("By category", "by_category"), ("By difficulty", "by_difficulty"), ("By skill (tag)", "by_skill"), ("By family", "by_family")):
        lines += [f"## {title}", "", "| | n | pass rate |", "|---|---|---|"] + [f"| {k} | {v['n']} | {100 * v['pass_rate']:.1f}% |" for k, v in s[key].items()] + [""]
    lines += ["## Why tasks failed", ""] + [f"- {k}: {v}" for k, v in s["failure_reasons"].items()]
    return "\n".join(lines) + "\n"


def transcript(r: dict) -> str:
    out = [f"### {r['family']} seed {r['seed']} - FAILED ({r['reason']})"]
    for m in r["messages"][1:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(f"**assistant**: {m['content']}\n```\n" + "\n".join(json.dumps(c["function"]) for c in m["tool_calls"]) + "\n```")
        else:
            out.append(f"**{m['role']}**: {m['content'][:600]}")
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir or Hub id (with --oracle only its tokenizer is used)")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    ap.add_argument("--tasks", default="data/raw/agent_test_tasks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=0, help="max tasks per family (0 = all)")
    ap.add_argument("--families", default="", help="comma-separated family names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=8, help="tool-calling steps allowed per user turn")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-code-exec", action="store_true", help="also run sandbox tasks (model-written code executes on THIS machine)")
    ap.add_argument("--oracle", action="store_true", help="replay the scripted answers instead of a model (harness self-test)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.tasks)]
    want = set(filter(None, args.families.split(",")))
    by_fam, skipped = defaultdict(list), 0
    for r in rows:
        if want and r["family"] not in want:
            continue
        if r["kind"] == "sandbox" and not args.allow_code_exec:
            skipped += 1
            continue
        by_fam[r["family"]].append(r)
    chosen = [r for f in sorted(by_fam) for r in sorted(by_fam[f], key=lambda x: x["seed"])[: args.per_family or None]]
    if args.limit:
        chosen = chosen[: args.limit]
    if skipped:
        print(f"skipping {skipped} sandbox tasks (model-written code would run on this machine); pass --allow-code-exec on a throwaway VM to include them")
    print(f"{len(chosen)} tasks from {len(by_fam)} families")

    runs = []
    for r in chosen:
        run = AgentRun(r["family"], r["seed"], args.max_steps)
        if task_signature(run.task) != r["sig"]:
            raise SystemExit(f"{r['family']} seed {r['seed']}: the task definition changed since {args.tasks} was built; rebuild it with distill.build_trajectories")
        runs.append(run)

    if args.oracle:
        from transformers import AutoTokenizer
        from evaluation.generate import OracleGenerator
        gen = OracleGenerator(AutoTokenizer.from_pretrained(args.model))
    else:
        from evaluation.generate import HFGenerator
        gen = HFGenerator(args.model, args.adapter, args.dtype, args.batch_size, args.max_new_tokens)

    t0, active, wave = time.time(), list(runs), 0
    while active:
        wave += 1
        for run, text in zip(active, gen(active)):
            try:
                run.feed(text)
            except Exception as e:                                     # one broken task must not lose the whole evaluation; it is scored as failed
                print(f"  harness error on {run.family} seed {run.seed}: {type(e).__name__}: {e}", flush=True)
                run._finish(f"harness_error:{type(e).__name__}")
        active = [r for r in active if not r.done]
        print(f"  wave {wave}: {len(runs) - len(active)}/{len(runs)} finished ({time.time() - t0:.0f}s)", flush=True)

    results = [r.result for r in runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(results)
    s["model"], s["adapter"], s["oracle"], s["seconds"] = args.model, args.adapter, args.oracle, round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    (out / "summary.md").write_text(markdown(f"{args.model}{' + ' + args.adapter if args.adapter else ''}{' (ORACLE)' if args.oracle else ''}", s))
    fails = [r for r in results if not r["ok"]][:12]
    (out / "failures.md").write_text("\n\n---\n\n".join(transcript(r) for r in fails) or "no failures\n")
    o = s["overall"]
    print(f"\nPASS RATE {100 * o['pass_rate']:.1f}% ({o['pass']}/{o['n']}) | format errors/msg {o['format_error_rate']:.3f} | tool errors/call {o['tool_error_rate']:.3f}")
    print("by category:", {k: f"{100 * v['pass_rate']:.0f}%" for k, v in s["by_category"].items()})
    print(f"wrote {out}/ (summary.md, summary.json, results.jsonl, failures.md)")


if __name__ == "__main__":
    main()
