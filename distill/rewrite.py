"""Teacher-written prompt rewrites ("expansion" distillation, done by an LLM instead of templates).

    python -m distill.rewrite export --n-per-family 19 --out distill/teacher_data/rewrite_todo.tsv     # id <TAB> original first user prompt
    # ... an LLM teacher writes distill/teacher_data/rewrites_*.tsv   (id <TAB> rewritten prompt, one per line) ...
    python -m distill.rewrite apply  --rewrites 'distill/teacher_data/rewrites_*.tsv' --out data/raw/agent_teacher_rewrites.jsonl

A rewrite only changes how the user ASKS (tone, wording, vagueness, extra context, typos), never WHAT is asked. The trajectory below it
(real tool calls, real observations, checker-verified) is reused unchanged, so `apply` rejects a rewrite unless every fact from the
original prompt survives: numbers, dates/times, quoted strings, file names / e-mails, and capitalised names.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

SRC = "data/raw/agent_trajectories.jsonl"
_NUM = re.compile(r"\d[\d,.:/-]*\d|\d")
_QUOTED = re.compile(r"\"([^\"]{2,})\"|(?<![\w])'([^'\n]{2,}?)'(?![\w])|`([^`]{2,})`")
_FILEISH = re.compile(r"[\w-]+(?:[./@_][\w-]+)+")
_WORD = re.compile(r"\S+")
SOFT_CAPS = {"Hey", "Hello", "Thanks", "Thank", "Please", "Could", "Can", "Would", "Quick", "Small", "Need", "Working", "Keep", "Let", "Hi", "And", "But", "Also"}


def key_tokens(prompt: str) -> list:
    keys = set(_NUM.findall(prompt)) | {g for m in _QUOTED.findall(prompt) for g in m if g} | set(_FILEISH.findall(prompt))
    prev = "."                                              # capitalised words count as names only in mid-sentence
    for tok in _WORD.findall(prompt):
        w = re.sub(r"'s$", "", tok.strip(".!?:,;()\"'"))
        if prev[-1] not in ".!?:" and prev.rstrip(",;").lower() not in {x.lower() for x in SOFT_CAPS} and re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in SOFT_CAPS:
            keys.add(w)
        prev = tok
    return sorted(k for k in keys if k.strip(".,:;"))


def missing(original: str, rewrite: str) -> list:
    low, flat = rewrite.lower(), rewrite.lower().replace(",", "")
    return [k for k in key_tokens(original) if k.lower() not in low and k.lower().replace(",", "") not in flat]


def first_user(rec: dict) -> str:
    return next(m["content"] for m in rec["messages"] if m["role"] == "user")


def load_src(path=SRC) -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def cmd_export(a) -> None:
    rng = random.Random(a.seed)
    by_fam = {}
    for rid, rec in load_src(a.src).items():
        by_fam.setdefault(rec["meta"]["family"], []).append(rid)
    ids = []
    for fam in sorted(by_fam):
        pool = sorted(by_fam[fam])
        rng.shuffle(pool)
        ids += pool[:a.n_per_family]
    recs = load_src(a.src)
    done = set()
    for p in glob.glob(a.skip_done or "__none__"):
        done |= {l.split("\t", 1)[0] for l in open(p) if "\t" in l}
    rows = [(i, " ".join(first_user(recs[i]).split())) for i in ids if i not in done]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(f"{i}\t{p}\n" for i, p in rows))
    print(f"exported {len(rows)} prompts from {len(by_fam)} families -> {a.out}")


def cmd_apply(a) -> None:
    recs = load_src(a.src)
    todo = [l.split("\t", 1)[0] for l in open(a.todo)] if Path(a.todo).exists() else []      # lets a teacher refer to a prompt by line number
    out, bad, seen = [], Counter(), set()
    for path in sorted(glob.glob(a.rewrites)):
        for ln, line in enumerate(open(path), 1):
            line = line.replace(" ||| ", "\t", 1)                 # teachers write "N ||| text" (no tab characters needed)
            if "\t" not in line:
                continue
            rid, new = line.rstrip("\n").split("\t", 1)
            new = new.strip()
            if rid.isdigit() and int(rid) <= len(todo):
                rid = todo[int(rid) - 1]
            if rid not in recs:
                bad["unknown_id"] += 1
                continue
            orig = first_user(recs[rid])
            if not 8 <= len(new) <= 1200:
                bad["bad_length"] += 1
            elif " ".join(new.split()).lower() == " ".join(orig.split()).lower():
                bad["unchanged"] += 1
            elif (miss := missing(orig, new)):
                bad["lost_fact"] += 1
                if a.report:
                    print(f"LOST {rid}: {miss} | {new[:100]}")
            elif (rid, new) in seen:
                bad["duplicate"] += 1
            else:
                seen.add((rid, new))
                rec = json.loads(json.dumps(recs[rid]))
                k = next(i for i, m in enumerate(rec["messages"]) if m["role"] == "user")
                rec["messages"][k]["content"] = new
                rec["id"] = f"{rid}/{a.suffix}"
                rec["meta"]["teacher_rewrite"] = True
                rec["meta"]["tags"] = list(rec["meta"]["tags"]) + ["teacher_rewrite"]
                out.append(rec)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"kept {len(out)} rewrites -> {a.out}; rejected: {dict(bad)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n-per-family", type=int, default=19)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--src", default=SRC)
    e.add_argument("--skip-done", default="")
    e.add_argument("--out", default="distill/teacher_data/rewrite_todo.tsv")
    p = sub.add_parser("apply")
    p.add_argument("--rewrites", default="distill/teacher_data/rewrites_*.tsv")
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default="data/raw/agent_teacher_rewrites.jsonl")
    p.add_argument("--report", action="store_true")
    p.add_argument("--suffix", default="teacher", help="id suffix of the new records (use another one for model-written rewrites)")
    p.add_argument("--todo", default="distill/teacher_data/rewrite_todo.tsv")
    a = ap.parse_args()
    {"export": cmd_export, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
