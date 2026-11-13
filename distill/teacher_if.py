"""Instruction-following answers written by the teacher (me) instead of a teacher model: ingest + verify.

    python -m distill.teacher_if select --n 60 --skip-done          # print the next prompts to answer (stratified over constraint types)
    python -m distill.teacher_if build                               # verify every written answer, write data/raw/if_verified.jsonl

Answers live in distill/teacher_data/if_answers_*.txt as blocks:
    ### if/123
    <the answer, any number of lines>
    ### if/124
    ...
`build` keeps an answer only if EVERY programmatic constraint of its prompt passes (distill.ifgen.verify), so a slip in word count or format is
rejected, not trained on. The record format is identical to distill.teacher_sample's, so both sources can be mixed.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

from distill.ifgen import verify

PROMPTS = "data/raw/if_prompts.jsonl"
ANSWERS = "distill/teacher_data/if_answers_*.txt"


def load_answers(pattern: str = ANSWERS) -> dict:
    out = {}
    for p in sorted(glob.glob(pattern)):
        cur, buf = None, []
        for line in open(p):
            m = re.match(r"^### (if/\d+)\s*$", line)
            if m:
                if cur:
                    out[cur] = "\n".join(buf).strip("\n")
                cur, buf = m.group(1), []
            elif cur:
                buf.append(line.rstrip("\n"))
        if cur:
            out[cur] = "\n".join(buf).strip("\n")
    return out


def cmd_select(a) -> None:
    rows = [json.loads(l) for l in open(a.prompts)]
    done = set(load_answers())
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    use = Counter()
    picked = []
    for r in rows:
        if r["id"] in done:
            continue
        names = [c["name"] for c in r["constraints"]]
        mw = [c["params"]["n"] for c in r["constraints"] if c["name"] == "min_words"]
        if mw and mw[0] > a.max_min_words:
            continue
        if any(use[n] >= a.n * 0.14 + 1 for n in names):          # keep the constraint mix balanced inside a batch
            continue
        picked.append(r)
        use.update(names)
        if len(picked) >= a.n:
            break
    for r in picked:
        print(f"[{r['id']}] " + r["prompt"].replace("\n", " \\n "))


def cmd_build(a) -> None:
    prompts = {r["id"]: r for r in map(json.loads, open(a.prompts))}
    kept, bad = [], Counter()
    for pid, text in load_answers().items():
        r = prompts.get(pid)
        if r is None:
            bad["unknown_id"] += 1
            continue
        ok, failed = verify(text, r["constraints"])
        if not ok or len(text) < 5:
            bad["failed:" + ",".join(failed)] += 1
            print(f"FAIL {pid}: {failed}")
            continue
        kept.append({"id": f"ifverified/{pid}", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": text}], "tools": [],
                     "meta": {"family": f"if_{r['form']}", "category": "instruction_following", "difficulty": "medium", "kind": "instruct", "seed": pid,
                              "tags": ["verified_constraints", "teacher_written"] + [c["name"] for c in r["constraints"]], "n_tool_calls": 0, "n_turns": 1, "n_tools_offered": 0}})
    Path(a.out).write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    c = Counter(t for k in kept for t in k["meta"]["tags"][2:])
    print(f"kept {len(kept)} verified answers -> {a.out}; rejected {dict(bad)}; constraints covered: {dict(c.most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--max-min-words", type=int, default=150)
    s.add_argument("--prompts", default=PROMPTS)
    b = sub.add_parser("build")
    b.add_argument("--prompts", default=PROMPTS)
    b.add_argument("--out", default="data/raw/if_verified.jsonl")
    a = ap.parse_args()
    {"select": cmd_select, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
