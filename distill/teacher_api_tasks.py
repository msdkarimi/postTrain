"""Teacher-authored FUNCTION-CALLING tasks (typed tools of the simulated workspace API), verified by execution.

    python -m distill.teacher_api_tasks --tasks 'distill/teacher_data/api_*.yaml' --out data/raw/agent_teacher_api.jsonl

Companion of distill/teacher_tasks.py for the API side. Each YAML entry gives a small hand-written world (contacts, events, inbox,
tasks, files), the tools to OFFER (needed ones + distractors), one or more turns, and the teacher's solution as steps of tool calls
(several calls in one step = parallel calls). `build` replays the calls against a fresh WorkspaceAPI, so every observation is what the
API really returned (incl. errors); the declared outcome is checked against the API's final STATE, not against the teacher's words.

    - id: a001
      category: scheduling
      tools: [get_current_date, contacts_search, calendar_find_free_slot, calendar_create_event, email_send]
      world: {today: "2025-03-03", contacts: [{name: Ana Ruiz, email: ana@acme.com, team: Sales}], events: [], inbox: [], tasks: [], files: {}}
      turns:
        - prompt: "..."
          steps:
            - plan: "reasoning before acting (brief chain of thought)"
              calls: [{tool: contacts_search, args: {query: Ana}}]      # add `fails: true` for an intended API error
          final: "..."
          grounded: ["09:00"]
      check:
        events: [{title: Pricing review, date: "2025-03-04", start: "11:00", attendees: [Ana Ruiz]}]
        no_events: [Old meeting]
        tasks: [{title: Prepare agenda, owner: Ana Ruiz, due: "2025-03-04"}]
        drafts: [{to: [ana@acme.com], subject_contains: Hello}]
        sent: 0                      # exact number of e-mails actually sent
        files: {notes.txt: {contains: [x]}}
        called: [calendar_create_event]
        not_called: [email_send]
        reply_contains: ["11:00"]
        reply_not_contains: ["sent"]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import random
from collections import Counter
from pathlib import Path

from distill.filters import hygiene, normalize_prompt
from distill.teacher_tasks import load_yaml
from envs.rollout import API_SYSTEM_PROMPTS, run_api
from envs.tasks.api_base import ApiStep, ApiTask, Call, Turn


def make_check(spec: dict):
    def check(api, replies):
        text = " ".join(replies).lower()
        for want in spec.get("events", []):
            ok = any(e["title"] == want["title"] and all(e.get(k) == want[k] for k in ("date", "start") if k in want)
                     and ("attendees" not in want or set(e["attendees"]) == set(want["attendees"])) and ("duration" not in want or True)
                     for e in api.events.values())
            if not ok:
                return False, f"missing_event:{want['title']}"
        for title in spec.get("no_events", []):
            if any(e["title"] == title for e in api.events.values()):
                return False, f"event_should_not_exist:{title}"
        for want in spec.get("tasks", []):
            if not any(all(t.get(k) == v for k, v in want.items()) for t in api.tasks):
                return False, f"missing_task:{want}"
        for want in spec.get("drafts", []):
            pool = list(api.drafts.values()) + list(api.sent)
            if not any(set(d["to"]) == set(want["to"]) and want.get("subject_contains", "").lower() in d["subject"].lower()
                       and all(s.lower() in d["body"].lower() for s in want.get("body_contains", [])) for d in pool):
                return False, f"missing_draft:{want}"
        if "sent" in spec and len(api.sent) != spec["sent"]:
            return False, f"sent_count:{len(api.sent)}"
        for path, rule in (spec.get("files") or {}).items():
            body = api.files.get(path)
            if body is None:
                return False, f"missing_file:{path}"
            if "equals" in rule and body.strip() != str(rule["equals"]).strip():
                return False, f"file_mismatch:{path}"
            for s in rule.get("contains", []):
                if str(s) not in body:
                    return False, f"file_missing_text:{path}:{s}"
        for path in spec.get("files_absent", []):
            if path in api.files:
                return False, f"file_should_not_exist:{path}"
        done = {name for name, _, ok in api.log if ok}
        called = {name for name, _, _ in api.log}
        for t in spec.get("called", []):
            if t not in done:
                return False, f"tool_not_called:{t}"
        for t in spec.get("not_called", []):
            if t in called:
                return False, f"tool_should_not_be_called:{t}"
        for s in spec.get("reply_contains", []):
            if str(s).lower() not in text:
                return False, f"reply_missing:{s}"
        for s in spec.get("reply_not_contains", []):
            if str(s).lower() in text:
                return False, f"reply_forbidden:{s}"
        return True, "ok"
    return check


def to_task(e: dict) -> ApiTask:
    turns_raw = e.get("turns") or [{k: e[k] for k in ("prompt", "steps", "final", "grounded") if k in e}]
    turns = []
    for t in turns_raw:
        steps = [ApiStep(s["plan"].strip(), [Call(c["tool"], c.get("args") or {}, expect_error=bool(c.get("fails"))) for c in s["calls"]]) for s in t.get("steps", [])]
        turns.append(Turn(t["prompt"].strip(), steps, t["final"].strip(), [str(g) for g in t.get("grounded", [])]))
    world = copy.deepcopy(e.get("world") or {})
    for k in ("contacts", "events", "inbox", "tasks"):
        world.setdefault(k, [])
    world.setdefault("files", {})
    return ApiTask(family=f"teacher_api_{e['category']}", category=e["category"], difficulty=e.get("difficulty", "medium"), world=world, tools=list(e["tools"]),
                   turns=turns, check=make_check(e.get("check") or {}), tags=list(e.get("tags", [])) + ["teacher_written", "function_calling"],
                   no_tools_ok=bool(e.get("no_tools")))


def build(entries: list, verbose=True) -> tuple:
    rows, drops, seen = [], Counter(), set()
    for e in entries:
        try:
            task = to_task(e)
        except Exception as ex:
            drops[f"bad_spec:{type(ex).__name__}"] += 1
            print("BAD SPEC", e.get("id"), repr(ex))
            continue
        r = random.Random(e["id"])
        res = run_api(task, API_SYSTEM_PROMPTS[r.randrange(len(API_SYSTEM_PROMPTS))])
        why = None if res["ok"] else f"checker:{res['reason']}"
        if why is None:
            bad = hygiene(task, res)
            why = f"hygiene:{bad}" if bad else None
        if why is None:
            called = {c.tool for t in task.turns for st in t.steps for c in st.calls}
            if called - set(task.tools):                       # a trajectory must never call a tool the model was not offered
                why = f"checker:tool_not_offered:{sorted(called - set(task.tools))}"
        if why is None and normalize_prompt(task.prompt) in seen:
            why = "duplicate:prompt"
        if why:
            drops[why[:48]] += 1
            if verbose:
                print(f"REJECT {e['id']}: {why}")
            continue
        seen.add(normalize_prompt(task.prompt))
        msgs = res["messages"]
        meta = {"family": task.family, "category": task.category, "difficulty": task.difficulty, "tags": task.tags, "seed": e["id"], "kind": "api",
                "n_tool_calls": sum(len(m.get("tool_calls", [])) for m in msgs), "n_turns": sum(1 for m in msgs if m["role"] == "user"),
                "n_tools_offered": len(res["tools"]), "teacher_written": True}
        rows.append({"id": f"agent/teacher_api/{e['id']}", "messages": msgs, "tools": res["tools"], "meta": meta})
    return rows, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="distill/teacher_data/api_*.yaml")
    ap.add_argument("--out", default="data/raw/agent_teacher_api.jsonl")
    a = ap.parse_args()
    entries = [e for p in sorted(glob.glob(a.tasks)) for e in load_yaml(p)]
    rows, drops = build(entries)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)}/{len(entries)} tasks verified and written to {a.out}; rejected: {dict(drops)}")


if __name__ == "__main__":
    main()
