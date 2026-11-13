"""Teacher-authored tasks: new, realistic sandbox tasks written by an LLM teacher (not by a template), verified by execution.

    python -m distill.teacher_tasks --tasks 'distill/teacher_data/tasks_*.yaml' --out data/raw/agent_teacher_tasks.jsonl

Each YAML entry is a complete task: a natural prompt, the files it runs on, the teacher's solution steps (brief reasoning + a real
run_bash / run_python call each) and the closing reply. Nothing is trusted: `build` executes every step in a throwaway workspace,
checks the declared outcome (files and/or reply) independently of the solution, and applies the same hygiene filters as the
procedural data (replies grounded in observed output, no leaked answers, no trailing error, bounded size).

    - id: t001
      category: log_analysis            # becomes the family  teacher_<category>
      difficulty: easy|medium|hard
      tags: [bash, grep]
      prompt: "..."
      files: {logs/app.log: "line1\\nline2\\n"}
      steps: [{plan: "...", tool: run_bash, arg: "..."}]       # add `fails: true` for an intentionally failing first attempt
      final: "..."
      grounded: ["42"]                  # strings the reply asserts; each must appear in a tool output
      check:
        files: {out.txt: {equals: "x\\n"}, report.md: {contains: ["a", "b"]}}
        absent: [tmp.txt]
        run: [{cmd: "python3 -c 'import m; assert m.f(2) == 4'", stdout_contains: [ok]}]   # executed in the final workspace
        reply_contains: ["42"]
        reply_not_contains: ["error"]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from distill.filters import hygiene, normalize_prompt
from envs.rollout import SYSTEM_PROMPTS, run_scripted
from envs.tasks.base import Step, Task


def make_check(spec: dict):
    def check(ws: Path, reply: str):
        for rel, rule in (spec.get("files") or {}).items():
            p = ws / rel
            if not p.exists():
                return False, f"missing:{rel}"
            text = p.read_text(errors="replace")
            if "equals" in rule and text.strip() != str(rule["equals"]).strip():
                return False, f"content_mismatch:{rel}: got {text.strip()[:80]!r}"
            for s in rule.get("contains", []):
                if str(s) not in text:
                    return False, f"missing_text:{rel}:{s}"
            for s in rule.get("not_contains", []):
                if str(s) in text:
                    return False, f"forbidden_text:{rel}:{s}"
        for run in spec.get("run", []):                      # executable acceptance tests, independent of the teacher's own checks
            try:
                r = subprocess.run(["bash", "-c", run["cmd"]], cwd=ws, capture_output=True, text=True, timeout=run.get("timeout", 15))
            except subprocess.TimeoutExpired:
                return False, f"run_timeout:{run['cmd'][:40]}"
            if r.returncode != run.get("exit", 0):
                return False, f"run_exit_{r.returncode}:{run['cmd'][:40]}:{(r.stderr or r.stdout)[-120:]}"
            if "stdout_equals" in run and r.stdout.strip() != str(run["stdout_equals"]).strip():
                return False, f"run_stdout:{r.stdout.strip()[:80]!r}"
            for sub in run.get("stdout_contains", []):
                if str(sub) not in r.stdout:
                    return False, f"run_stdout_missing:{sub}"
        for rel in spec.get("absent", []):
            if (ws / rel).exists():
                return False, f"should_not_exist:{rel}"
        for s in spec.get("reply_contains", []):
            if str(s).lower() not in reply.lower():
                return False, f"reply_missing:{s}"
        for s in spec.get("reply_not_contains", []):
            if str(s).lower() in reply.lower():
                return False, f"reply_forbidden:{s}"
        return True, "ok"
    return check


def expand(v) -> str:
    """File contents: 'x__REPEAT_5000' means 5000 times 'x' (for size-based tasks without 5 KB of YAML)."""
    m = re.fullmatch(r"(.)__REPEAT_(\d+)", str(v))
    return m.group(1) * int(m.group(2)) if m else str(v)


def to_task(e: dict) -> Task:
    steps = [Step(s["plan"].strip(), s["tool"], s["arg"].rstrip("\n") if s["tool"] == "run_bash" else s["arg"], expect_error=bool(s.get("fails")))
             for s in e["steps"]]
    return Task(family=f"teacher_{e['category']}", category=e["category"], difficulty=e.get("difficulty", "medium"), prompt=e["prompt"].strip(),
                files={k: expand(v) for k, v in (e.get("files") or {}).items()}, steps=steps, final=e["final"].strip(), check=make_check(e["check"]),
                grounded=[str(g) for g in e.get("grounded", [])], tags=list(e.get("tags", [])) + ["teacher_written"])


def build(entries: list, verbose=True) -> tuple:
    rows, drops, seen = [], Counter(), set()
    for e in entries:
        try:
            task = to_task(e)
        except Exception as ex:
            drops[f"bad_spec:{type(ex).__name__}"] += 1
            print("BAD SPEC", e.get("id"), ex)
            continue
        r = random.Random(e["id"])
        res = run_scripted(task, SYSTEM_PROMPTS[r.randrange(len(SYSTEM_PROMPTS))])
        why = None if res["ok"] else f"checker:{res['reason']}"
        why = why or (f"hygiene:{hygiene(task, res)}" if hygiene(task, res) else None)
        if why is None and normalize_prompt(task.prompt) in seen:
            why = "duplicate_prompt"
        if why:
            parts = why.split(":")
            drops[parts[0] + ":" + (parts[1][:30] if len(parts) > 1 else "")] += 1
            if verbose:
                print(f"REJECT {e['id']}: {why}")
            continue
        seen.add(normalize_prompt(task.prompt))
        msgs = res["messages"]
        from envs.rollout import TOOLS
        meta = {"family": task.family, "category": task.category, "difficulty": task.difficulty, "tags": task.tags, "seed": e["id"], "kind": "sandbox",
                "n_tool_calls": sum(len(m.get("tool_calls", [])) for m in msgs), "n_turns": 1, "n_tools_offered": len(TOOLS), "teacher_written": True}
        rows.append({"id": f"agent/teacher/{e['id']}", "messages": msgs, "tools": TOOLS, "meta": meta})
    return rows, drops


_PLAIN = re.compile(r"^(\s*(?:- )?(?:prompt|final|plan|id|category|arg)): (?![|>\"'\[{&*!%@`])(.+?)\s*$", re.M)


def load_yaml(path: str) -> list:
    """Authoring convenience: one-line prompt/final/plan values may contain ': ' or quotes without YAML escaping."""
    text = _PLAIN.sub(lambda m: f"{m.group(1)}: '" + m.group(2).replace("'", "''") + "'", Path(path).read_text())
    return yaml.safe_load(text) or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="distill/teacher_data/tasks_*.yaml")
    ap.add_argument("--out", default="data/raw/agent_teacher_tasks.jsonl")
    a = ap.parse_args()
    entries = [e for p in sorted(glob.glob(a.tasks)) for e in load_yaml(p)]
    rows, drops = build(entries)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)}/{len(entries)} tasks verified and written to {a.out}; rejected: {dict(drops)}")


if __name__ == "__main__":
    main()
