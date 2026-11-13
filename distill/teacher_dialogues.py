"""Multi-turn conversations written by the teacher (me), verified turn by turn, -> data/raw/teacher_dialogues.jsonl

    python -m distill.teacher_dialogues            # verify every dialogue, write the kept ones, exit 1 if any failed
    python -m distill.teacher_dialogues --stats    # also print turns / skills

Why: the generated sources are templates. These are real conversations, each several user turns long, that mix the skills the benchmarks need inside ONE
session (survey: instruction following, chain-of-thought, multi-turn, context following, tool use, feedback / self-correction, honesty):
  - a format rule given once has to hold in later turns, or is changed / revised mid-conversation (IFEval)
  - tools stay offered for the whole session; some turns need a call, some reuse an earlier result, some need none (BFCL multi-turn)
  - step-by-step reasoning with what-if follow-ups, and users who push back on a correct answer (no caving)
  - questions on a given document, including ones the document does not answer
  - code that is written, then changed, with the final version executed

Dialogues live in distill/teacher_data/dialogues_*.txt:

    ### mt/a001 | skills=if_persist,cot | tools=math.percentage,weather.get_current
    @system
    optional system prompt
    @user
    text (any number of lines)
    @assistant
    text
    @check max_words n=40 ; lowercase ; has=42 ; hasnt=sorry
    @call math.percentage {"part": 3, "whole": 12}
    @result 25.0
    @assistant
    final text for that turn
    @pyassert f(3) == 9

`@check` applies to the assistant message above it: any distill.ifgen constraint (params as k=v; lists with |), plus has= / hasnt= / json / lines n=.
`@call` lines after each other are parallel calls of one assistant message (an `@assistant` text right before them becomes that message's content).
`@result` is the tool output. For pure functions (math, dates, units, ...) it is compared with the real implementation in distill.fc_gen, so a wrong
number is rejected. `@pyassert` runs the last ```python block above it and asserts the expression.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

from distill import fc_gen
from distill.ifgen import check_fn

FILES = "distill/teacher_data/dialogues_*.txt"
OUT = "data/raw/teacher_dialogues.jsonl"
PURE = {"math", "geometry", "units", "text", "dates", "lists", "stats", "time", "physics", "health", "home", "education", "science", "transport.fuel_cost",
        "transport.parking_fee", "finance.compound_interest", "finance.loan_payment", "finance.vat_price", "finance.tip_split", "finance.roi", "finance.break_even",
        "finance.salary_after_tax", "shop.apply_coupon", "plants"}
CATEGORY = {"tools": "function_calling", "if": "instruction_following", "cot": "reasoning", "context": "context_following", "honest": "refusal", "code": "coding",
            "mcq": "knowledge", "write": "writing", "extract": "extraction", "advice": "general"}
INT_KEYS = {"n", "lo", "hi", "i"}


def parse(path: str) -> list:
    out, cur = [], None
    mode = None
    for ln, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.rstrip("\n")
        m = re.match(r"^### (mt/\S+)\s*(.*)$", line)
        if m:
            head = dict(kv.strip().split("=", 1) for kv in m.group(2).split("|") if "=" in kv)
            cur = {"id": m.group(1), "skills": [s.strip() for s in head.get("skills", "").split(",") if s.strip()],
                   "tools": [s.strip() for s in head.get("tools", "").split(",") if s.strip()], "items": [], "file": f"{path}:{ln}"}
            out.append(cur)
            mode = None
            continue
        if cur is None:
            continue
        m = re.match(r"^@(system|user|assistant)\s*$", line)
        if m:
            cur["items"].append({"kind": m.group(1), "text": []})
            mode = "text"
            continue
        m = re.match(r"^@(call|result|check|pyassert)\s+(.*)$", line)
        if m:
            cur["items"].append({"kind": m.group(1), "arg": m.group(2).strip(), "ln": ln})
            mode = None
            continue
        if mode == "text":
            cur["items"][-1]["text"].append(line)
    return out


def _params(spec: str) -> tuple:
    """'max_words n=40' -> ('max_words', {'n': 40}) ; 'keywords ks=a|b' -> ('keywords', {'ks': ['a','b']})"""
    spec = spec.strip()
    name, _, rest = spec.partition(" ")
    if "=" in name:                                             # has=..., hasnt=...
        k, _, v = spec.partition("=")
        return k, {"v": v}
    p = {}
    for m in re.finditer(r"(\w+)=((?:(?! \w+=).)*)", rest):
        k, v = m.group(1), m.group(2).strip()
        p[k] = int(v) if k in INT_KEYS else (v.split("|") if k == "ks" else v)
    return name, p


def run_check(spec: str, text: str) -> bool:
    name, p = _params(spec)
    if name == "has":
        return p["v"].lower() in text.lower()
    if name == "hasnt":
        return p["v"].lower() not in text.lower()
    if name == "lines":
        return len([x for x in text.split("\n") if x.strip()]) == p["n"]
    if name == "jsonvalid":
        try:
            body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            json.loads(body)
            return True
        except Exception:
            return False
    return bool(check_fn(name)(text, p))


def _pure(name: str) -> bool:
    return name in PURE or name.split(".")[0] in PURE


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 0.011 * max(1.0, abs(b)) * 0.01 + 0.011
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def build_one(d: dict) -> tuple:
    """-> (record | None, [errors])"""
    errs, msgs, last_asst = [], [], None
    for name in d["tools"]:
        if name not in fc_gen.FUNCS:
            errs.append(f"unknown tool {name}")
    if errs:
        return None, errs
    tools = [fc_gen.schema(fc_gen.FUNCS[n]) for n in d["tools"]]
    pending = []                                                  # calls of the open assistant message that still wait for a result
    for it in d["items"]:
        k = it["kind"]
        if k in ("system", "user", "assistant"):
            text = "\n".join(it["text"]).strip("\n")
            if not text.strip():
                errs.append(f"empty {k} message")
            if pending:
                errs.append("call without result before next message")
            msgs.append({"role": k, "content": text})
            last_asst = msgs[-1] if k == "assistant" else last_asst
        elif k == "call":
            m = re.match(r"^(\S+)\s+(\{.*\})$", it["arg"])
            if not m:
                errs.append(f"line {it['ln']}: bad @call")
                continue
            try:
                args = json.loads(m.group(2))
            except Exception as e:
                errs.append(f"line {it['ln']}: call args not JSON ({e})")
                continue
            call = {"type": "function", "function": {"name": m.group(1), "arguments": args}}
            if msgs and msgs[-1]["role"] == "assistant" and (msgs[-1].get("tool_calls") is not None or not pending) and not msgs[-1].get("_closed"):
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            pending.append(call)
        elif k == "result":
            if not pending:
                errs.append(f"line {it['ln']}: @result without @call")
                continue
            call = pending.pop(0)
            for m_ in msgs:
                if m_["role"] == "assistant" and m_.get("tool_calls"):
                    m_["_closed"] = True
            try:
                val = json.loads(it["arg"])
            except Exception:
                errs.append(f"line {it['ln']}: result not JSON")
                continue
            name, args = call["function"]["name"], call["function"]["arguments"]
            f = fc_gen.FUNCS.get(name)
            if f and _pure(name) and not (isinstance(val, dict) and "error" in val):
                try:
                    real = f["run"](dict(args))
                    real = json.loads(json.dumps(real if not isinstance(real, float) else round(real, 4)))
                    if not _same(val, real):
                        errs.append(f"line {it['ln']}: {name} result {val!r} != real {real!r}")
                except Exception as e:
                    errs.append(f"line {it['ln']}: {name} could not run ({e})")
            msgs.append({"role": "tool", "content": json.dumps(val, ensure_ascii=False)})
        elif k == "check":
            if last_asst is None:
                errs.append(f"line {it['ln']}: @check before any assistant message")
                continue
            for spec in it["arg"].split(";"):
                if not spec.strip():
                    continue
                try:
                    ok = run_check(spec, last_asst["content"])
                except Exception as e:
                    ok = False
                    spec += f" (error {e})"
                if not ok:
                    errs.append(f"line {it['ln']}: check failed: {spec.strip()}")
        elif k == "pyassert":
            blocks = re.findall(r"```python\n(.*?)```", last_asst["content"] if last_asst else "", re.S)
            if not blocks:
                errs.append(f"line {it['ln']}: no python block for @pyassert")
                continue
            env: dict = {}
            try:
                exec(blocks[-1], env)                              # noqa: S102 - my own hand-written snippets
                if not eval(it["arg"], env):                       # noqa: S307
                    errs.append(f"line {it['ln']}: pyassert false: {it['arg']}")
            except Exception as e:
                errs.append(f"line {it['ln']}: pyassert error {type(e).__name__}: {e}")
    if pending:
        errs.append("call without result at end")
    for m_ in msgs:
        m_.pop("_closed", None)
    # structure
    body = [m_ for m_ in msgs if m_["role"] != "system"]
    if not body or body[0]["role"] != "user":
        errs.append("must start with a user turn")
    if msgs and msgs[-1]["role"] != "assistant" or (msgs and msgs[-1].get("tool_calls")):
        errs.append("must end with a plain assistant message")
    for a, b in zip(body, body[1:]):
        if a["role"] == "user" and b["role"] != "assistant":
            errs.append("user turn not followed by assistant")
        if a["role"] == "assistant" and not a.get("tool_calls") and b["role"] != "user":
            errs.append("assistant text followed by non-user")
        if a["role"] == "assistant" and a.get("tool_calls") and b["role"] != "tool":
            errs.append("call not followed by tool result")
    n_user = sum(1 for m_ in msgs if m_["role"] == "user")
    if n_user < 2:
        errs.append("not multi-turn (needs 2+ user turns)")
    rec = {"id": d["id"], "messages": msgs, "tools": tools}
    e = fc_gen.check(rec) if tools else None
    if e:
        errs.append(f"schema: {e}")
    if any(m_.get("tool_calls") for m_ in msgs) and not tools:
        errs.append("calls without offered tools")
    skills = d["skills"] or ["advice"]
    n_calls = sum(len(m_.get("tool_calls", [])) for m_ in msgs)
    rec["meta"] = {"family": "mt_" + skills[0], "category": CATEGORY.get(skills[0].split("_")[0], "general"), "difficulty": "hard" if n_user >= 4 or n_calls >= 3 else "medium",
                   "kind": "api" if tools else "chat", "tags": ["teacher_written", "multi_turn"] + skills, "n_tool_calls": n_calls, "n_turns": n_user,
                   "n_tools_offered": len(tools), "seed": 0}
    return (None if errs else rec), errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=FILES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    dialogues = [d for p in sorted(glob.glob(a.files)) for d in parse(p)]
    seen, kept, failed = set(), [], 0
    for d in dialogues:
        if d["id"] in seen:
            print(f"FAIL {d['id']} ({d['file']}): duplicate id")
            failed += 1
            continue
        seen.add(d["id"])
        rec, errs = build_one(d)
        if errs:
            failed += 1
            print(f"FAIL {d['id']} ({d['file']})")
            for e in errs:
                print("     ", e)
        else:
            kept.append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    turns = sum(r["meta"]["n_turns"] for r in kept)
    print(f"[teacher_dialogues] {len(kept)} dialogues kept, {failed} failed, {turns} user turns (= training examples) -> {a.out}")
    if a.stats:
        c = Counter(s for r in kept for s in r["meta"]["tags"][2:])
        print("  skills:", dict(c.most_common()))
        print("  with tools:", sum(1 for r in kept if r["tools"]), " tool calls:", sum(r["meta"]["n_tool_calls"] for r in kept))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
