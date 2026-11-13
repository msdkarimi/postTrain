"""Build the verified agent-trajectory dataset (the "teacher" side of distillation).

    python -m distill.build_trajectories --n-train 40000 --n-test-per-family 8 --out data/raw

Two task kinds share one pipeline:
  * sandbox tasks  - run_python / run_bash in a throwaway workspace (envs/tasks/fam_*.py)
  * function-calling tasks - typed tools of a simulated workspace API: calendar, contacts, email, tasks, files, utilities
    (envs/tasks/fam_api_*.py); several tools offered per task (needed ones + distractors), parallel calls, API errors,
    multi-turn follow-ups, clarifications, no-tool and unsupported requests.

Per task: generate (seeded) -> run the scripted teacher solution FOR REAL -> independent checker -> hygiene filters (grounded
claims, no answer leakage, no trailing error, bounded size) -> dedup -> decontaminate against the held-out test prompts ->
write canonical JSONL (etl/schema.py), stats and a dataset card.

Outputs (under --out):
    agent_trajectories.jsonl   training data  (feeds etl/sources/local_jsonl.py)
    agent_test_tasks.jsonl     held-out evaluation tasks (prompt + files/world + tool names; the solution is NOT included)
    agent_stats.json           counts per family / category / difficulty / tag / tool, drop reasons
    DATASET_CARD.md            what is in the data and how it was made
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import envs.tasks  # noqa: F401  (registers families)
from distill.filters import hygiene, normalize_prompt
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, run_any
from envs.tasks.api_base import ApiTask
from envs.tasks.base import FAMILIES, make_task

TRAIN_SEED_OFFSET = 10_000          # train seeds never overlap the test seeds 0..N-1


def task_signature(task) -> str:
    """Identity of a task INSTANCE = its prompt text + the data it runs on (files or world). Two tasks with the same generic
    prompt but different data are different tasks; only identical prompt AND identical data count as duplicates / leakage."""
    data = task.world if isinstance(task, ApiTask) else task.files
    payload = normalize_prompt(task.prompt) + "|" + json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_one(args: tuple) -> dict:
    name, seed = args
    try:
        task = make_task(name, seed)
        r = random.Random(f"sys:{name}:{seed}")
        res = run_any(task, SYSTEM_PROMPTS[r.randrange(len(SYSTEM_PROMPTS))], API_SYSTEM_PROMPTS[r.randrange(len(API_SYSTEM_PROMPTS))])
        if not res["ok"]:
            return {"family": name, "seed": seed, "kept": False, "reason": f"checker:{str(res['reason'])[:50]}"}
        bad = hygiene(task, res)
        if bad:
            return {"family": name, "seed": seed, "kept": False, "reason": f"hygiene:{bad}"}
        msgs = res["messages"]
        n_calls = sum(len(m.get("tool_calls", [])) for m in msgs)
        meta = {"family": name, "category": task.category, "difficulty": task.difficulty, "tags": task.tags, "seed": seed,
                "kind": "api" if isinstance(task, ApiTask) else "sandbox", "n_tool_calls": n_calls,
                "n_turns": sum(1 for m in msgs if m["role"] == "user"), "n_tools_offered": len(res["tools"])}
        return {"family": name, "seed": seed, "kept": True, "prompt": task.prompt, "sig": task_signature(task),
                "record": {"id": f"agent/{name}/{seed}", "messages": msgs, "tools": res["tools"], "meta": meta}}
    except Exception as e:  # a generator bug must never silently poison the data
        return {"family": name, "seed": seed, "kept": False, "reason": f"exception:{type(e).__name__}:{str(e)[:60]}"}


def allocate(n_total: int) -> dict:
    wsum = sum(w for _, _, w in FAMILIES.values())
    return {name: max(8, round(n_total * w / wsum)) for name, (_, _, w) in FAMILIES.items()}


def test_row(name: str, seed: int) -> dict:
    t = make_task(name, seed)
    base = {"family": name, "seed": seed, "difficulty": t.difficulty, "category": t.category, "prompt": t.prompt, "sig": task_signature(t)}
    if isinstance(t, ApiTask):
        return {**base, "kind": "api", "tools": t.tools, "world": t.world, "turns": [x.prompt for x in t.turns]}
    return {**base, "kind": "sandbox", "files": t.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40000, help="task attempts (kept count is lower after filtering/dedup)")
    ap.add_argument("--n-test-per-family", type=int, default=8)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    test_rows, test_sigs = [], set()
    for name in sorted(FAMILIES):
        for seed in range(args.n_test_per_family):
            row = test_row(name, seed)
            test_sigs.add(row["sig"])
            test_rows.append(row)
    with (out / "agent_test_tasks.jsonl").open("w") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    alloc = allocate(args.n_train)
    jobs = [(name, TRAIN_SEED_OFFSET + k) for name, n in alloc.items() for k in range(n)]
    print(f"{len(FAMILIES)} families, {len(jobs)} task attempts, {args.workers} workers ...")
    attempted, drops, kept_by_family, rows, seen_sigs, uniq_prompts = Counter(), Counter(), Counter(), [], set(), set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(build_one, jobs, chunksize=32)):
            attempted[r["family"]] += 1
            if not r["kept"]:
                drops[r["reason"]] += 1
                continue
            if r["sig"] in test_sigs:
                drops["decontaminated_vs_test"] += 1
                continue
            if r["sig"] in seen_sigs:
                drops["duplicate_task"] += 1
                continue
            seen_sigs.add(r["sig"])
            uniq_prompts.add(normalize_prompt(r["prompt"]))
            kept_by_family[r["family"]] += 1
            rows.append(r["record"])
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} processed, {len(rows)} kept", flush=True)
    random.Random(0).shuffle(rows)
    with (out / "agent_trajectories.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- stats ---------------------------------------------------------------------------------------------------
    tag_counts, diff, cat, kind = Counter(), Counter(), Counter(), Counter()
    tool_use, n_calls, n_turns, parallel, errors, no_tool = Counter(), [], [], 0, 0, 0
    for r in rows:
        m = r["meta"]
        tag_counts.update(m["tags"])
        diff[m["difficulty"]] += 1
        cat[m["category"]] += 1
        kind[m["kind"]] += 1
        n_calls.append(m["n_tool_calls"])
        n_turns.append(m["n_turns"])
        no_tool += int(m["n_tool_calls"] == 0)
        for msg in r["messages"]:
            if msg["role"] == "assistant" and len(msg.get("tool_calls", [])) >= 2:
                parallel += 1
            for c in msg.get("tool_calls", []):
                tool_use[c["function"]["name"]] += 1
            if msg["role"] == "tool" and msg["content"].startswith('{"error"'):
                errors += 1
    stats = {"attempted": sum(attempted.values()), "kept": len(rows), "keep_rate": round(len(rows) / max(sum(attempted.values()), 1), 4),
             "per_family": {n: {"attempted": attempted[n], "kept": kept_by_family[n]} for n in sorted(FAMILIES)},
             "drop_reasons": dict(drops), "difficulty": dict(diff), "category": dict(cat), "kind": dict(kind), "tags": dict(tag_counts.most_common()),
             "tool_calls": {"total": sum(n_calls), "mean": round(statistics.mean(n_calls), 2), "max": max(n_calls), "by_tool": dict(tool_use.most_common())},
             "turns": {"multi_turn": sum(1 for t in n_turns if t > 1), "max": max(n_turns)},
             "parallel_call_messages": parallel, "api_error_responses": errors, "no_tool_trajectories": no_tool,
             "unique_prompts": len(uniq_prompts), "n_families": len(FAMILIES), "test_tasks": len(test_rows), "seconds": round(time.time() - t0, 1)}
    (out / "agent_stats.json").write_text(json.dumps(stats, indent=2))
    from distill.card import write_card
    write_card(out / "DATASET_CARD.md", stats)
    print(json.dumps({k: stats[k] for k in ("attempted", "kept", "keep_rate", "kind", "difficulty", "category", "drop_reasons")}, indent=2))
    print(f"\nwrote {len(rows)} trajectories + {len(test_rows)} test tasks to {out}  ({stats['seconds']}s)")


if __name__ == "__main__":
    main()
