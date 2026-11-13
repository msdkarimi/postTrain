"""Use an open instruct model as an extra TEACHER on the GPU box (survey: "labeling / expansion" + "curation by verifiable feedback").

Two jobs, both ending in verified data only:

  python -m distill.teacher_sample if --teacher Qwen/Qwen3.5-9B --prompts data/raw/if_prompts.jsonl --out data/raw/if_verified.jsonl [--k 4]
      The teacher answers instruction-following prompts (distill/ifgen.py); an answer is kept only if EVERY programmatic constraint passes
      (word limits, bullet counts, required / banned words, casing, JSON, ...). Up to k samples per prompt, the first passing one is kept.

  python -m distill.teacher_sample paraphrase --teacher <model> --out distill/../data/teacher/model_rewrites.tsv [--n-per-family 150]
      The teacher rewrites the first user prompt of existing verified agent trajectories in natural, varied wording. My hand-written rewrites
      (distill/teacher_data) are the few-shot examples, and a rewrite is accepted only if every fact of the original survives
      (distill.rewrite.missing). Apply the result with:
        python -m distill.rewrite apply --rewrites data/teacher/model_rewrites.tsv --suffix tm --out data/raw/agent_teacher_model_rewrites.jsonl

The teacher is chosen with --teacher (default Qwen/Qwen3.8-27B: dense 27B, Apache-2.0, hybrid linear attention like the student; in bf16 it needs ~56 GB,
so it fits an 80 GB A100 and nothing else should run on that GPU meanwhile). A bigger model than the 0.8B student makes better data; the verifiers,
not the teacher's confidence, decide what is kept. Generation uses plain Hugging Face generate (no vLLM needed), batched, left-padded.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path
from typing import Callable

from distill.ifgen import verify
from distill.rewrite import first_user, key_tokens, load_src, missing

THINK = re.compile(r"<think>.*?</think>\s*", re.S)
IF_SYSTEM = ("You are a precise assistant. Follow EVERY instruction in the user's request exactly, including length limits, formatting rules, required "
             "and forbidden words. Reply with the answer only: no preamble, no explanation of what you did.")
PARA_SYSTEM = ("You rewrite a user's request to an AI assistant so it sounds like a different real person typed it (terse, chatty, polite, hurried, vague, "
               "with extra context...). Keep EVERY fact exactly: all numbers, dates, times, file names, e-mail addresses, quoted text and names, and keep what is "
               "asked for and what format is required. Do not answer the request. Output only the rewritten request.")


class ChatPrompt:
    """Adapter so evaluation.generate.HFGenerator can batch plain chat messages."""

    thinking = False                       # class-level switch set by --enable-thinking

    def __init__(self, messages):
        self.messages = messages

    def prompt_text(self, tok):
        try:
            return tok.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
        except TypeError:
            return tok.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)


def hf_generate(model: str, batch_size: int, max_new: int, temperature: float) -> Callable:
    from evaluation.generate import HFGenerator
    gen = HFGenerator(model, batch_size=batch_size, max_new_tokens=max_new, temperature=temperature, top_p=0.95)
    return lambda chats: [THINK.sub("", t or "").strip() for t in gen([ChatPrompt(c) for c in chats])]


# ------------------------------------------------------------------------------------------------------------------------------ IF
def run_if(rows: list, generate: Callable, k: int = 4) -> tuple:
    """-> (records, stats). generate(list[messages]) -> list[str]"""
    pending, kept, stats = {i: r for i, r in enumerate(rows)}, [], {"asked": 0, "passed": 0, "failed_by": {}}
    for _ in range(k):
        if not pending:
            break
        ids = list(pending)
        outs = generate([[{"role": "system", "content": IF_SYSTEM}, {"role": "user", "content": pending[i]["prompt"]}] for i in ids])
        for i, text in zip(ids, outs):
            stats["asked"] += 1
            ok, failed = verify(text, pending[i]["constraints"])
            if ok and len(text) > 5:
                r = pending.pop(i)
                stats["passed"] += 1
                kept.append({"id": f"ifverified/{r['id']}", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": text}], "tools": [],
                             "meta": {"family": f"if_{r['form']}", "category": "instruction_following", "difficulty": "medium", "kind": "instruct", "seed": r["id"],
                                      "tags": ["verified_constraints", "teacher_model"] + [c["name"] for c in r["constraints"]], "n_tool_calls": 0, "n_turns": 1, "n_tools_offered": 0}})
            else:
                for f in failed:
                    stats["failed_by"][f] = stats["failed_by"].get(f, 0) + 1
    return kept, stats


# ------------------------------------------------------------------------------------------------------------------------------ paraphrase
def load_shots(teacher_dir: str = "distill/teacher_data") -> list:
    todo = [l.split("\t") for l in open(f"{teacher_dir}/rewrite_todo.tsv").read().splitlines()]
    shots = []
    for p in sorted(glob.glob(f"{teacher_dir}/rewrites_*.tsv")):
        for line in open(p):
            line = line.replace(" ||| ", "\t", 1)
            if "\t" in line:
                i, new = line.rstrip("\n").split("\t", 1)
                if i.isdigit() and int(i) <= len(todo):
                    shots.append((todo[int(i) - 1][1], new.strip()))
    return [s for s in shots if not missing(*s)]


def paraphrase_messages(prompt: str, shots: list, rng: random.Random, n_shots: int = 6) -> list:
    msgs = [{"role": "system", "content": PARA_SYSTEM}]
    for orig, new in rng.sample(shots, min(n_shots, len(shots))):
        msgs += [{"role": "user", "content": orig}, {"role": "assistant", "content": new}]
    return msgs + [{"role": "user", "content": prompt}]


BAD_START = re.compile(r"^\s*(sure|here is|here's|rewritten|certainly|okay)\b", re.I)


def accept_rewrite(orig: str, new: str) -> bool:
    new = " ".join(new.split())
    if not new or BAD_START.match(new) or "\n\n" in new.strip():
        return False
    lo, hi = 0.45 * len(orig), 2.6 * len(orig) + 40
    if not lo <= len(new) <= hi or new.lower() == orig.lower():
        return False
    return not missing(orig, new)


def run_paraphrase(recs: dict, shots: list, generate: Callable, n_per_family: int, k: int, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    by_fam = {}
    for rid, rec in recs.items():
        by_fam.setdefault(rec["meta"]["family"], []).append(rid)
    ids = []
    for fam in sorted(by_fam):
        pool = sorted(by_fam[fam])
        rng.shuffle(pool)
        ids += pool[:n_per_family]
    lines, seen, asked = [], set(), 0
    todo = list(ids)
    for _ in range(k):
        if not todo:
            break
        chats = [paraphrase_messages(" ".join(first_user(recs[i]).split()), shots, rng) for i in todo]
        outs = generate(chats)
        nxt = []
        for rid, text in zip(todo, outs):
            asked += 1
            orig = first_user(recs[rid])
            new = " ".join((text or "").split())
            if accept_rewrite(orig, new) and new.lower() not in seen:
                seen.add(new.lower())
                lines.append(f"{rid} ||| {new}")
            else:
                nxt.append(rid)
        todo = nxt
    return lines, {"asked": asked, "kept": len(lines), "families": len(by_fam)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["if", "paraphrase"])
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B", help="Hub id or local path of the teacher chat model (default: Qwen/Qwen3.8-27B, dense 27B, Apache-2.0)")
    ap.add_argument("--enable-thinking", action="store_true", help="let the teacher think before answering (slower; needs a larger --max-new, e.g. 2500); the <think> block is stripped")
    ap.add_argument("--prompts", default="data/raw/if_prompts.jsonl")
    ap.add_argument("--src", default="data/raw/agent_trajectories.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="max attempts per prompt")
    ap.add_argument("--n-per-family", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--batch-size", type=int, default=32, help="27B bf16 = 56 GB of weights on an 80 GB A100; the hybrid model's cache is small, 32 fits")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    ChatPrompt.thinking = a.enable_thinking
    gen = hf_generate(a.teacher, a.batch_size, a.max_new, a.temperature)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.job == "if":
        rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
        rows = rows[:a.limit] if a.limit else rows
        kept, stats = run_if(rows, gen, a.k)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        print(f"kept {len(kept)}/{len(rows)} prompts ({100 * len(kept) / max(len(rows), 1):.0f}%); {stats}")
    else:
        recs = load_src(a.src)
        if a.limit:
            recs = dict(list(recs.items())[:a.limit])
        shots = load_shots()
        lines, stats = run_paraphrase(recs, shots, gen, a.n_per_family, a.k)
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"kept {len(lines)} rewrites; {stats} (shots: {len(shots)})")


if __name__ == "__main__":
    main()
