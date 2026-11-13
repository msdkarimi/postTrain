"""Local JSONL files that are already in canonical format (see etl/schema.py).

Use this for your OWN data, e.g. teacher-distilled sandbox trajectories that passed their checker.
Each line: {"id": optional, "messages": [...], "tools": optional}

params:
  paths: list of .jsonl or .jsonl.gz paths (relative to the repo root or absolute)
  source_name: label stored in the record (default: file stem)
  max_per_family: cap per meta.family (stratified: a small deterministic random sample of every family), default none
  max_examples: overall cap after stratification (deterministic shuffle), default none
  optional: true = a missing path is skipped with a warning instead of raising (for data produced later)
  max_per_shape: keep at most N examples that share the same "shape" (first user message and first assistant message with numbers, quoted text and
                 capitalised words masked) - removes near-identical template outputs before the per-family cap picks its sample
  max_per_prompt: keep at most N examples with the same first user message (a prompt answered many times in slightly different ways is still repetition)
  max_sentence_reuse: drop an example when one of its assistant sentences (25+ characters) has already appeared in this many accepted examples
"""
from __future__ import annotations

import gzip
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from etl.schema import SkipExample, make_record
from etl.sources import register


_MASK = [(re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`"), "Q"), (re.compile(r"\d+(?:[.,:/-]\d+)*"), "0"), (re.compile(r"\b[A-Z][a-z]{2,}\b"), "W")]


def shape_of(row: dict) -> str:
    """Template fingerprint of a conversation: first user and first assistant message, with the variable parts masked."""
    msgs = row.get("messages", [])
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    out = []
    for text in (user, asst):
        t = text[:160]
        for rx, rep in _MASK:
            t = rx.sub(rep, t)
        out.append(re.sub(r"\s+", " ", t.lower())[:110])
    return "|".join(out)


def sentences_of(row: dict) -> set:
    out = set()
    for m in row.get("messages", []):
        if m["role"] == "assistant":
            for sent in re.split(r"(?<=[.!?])\s+|\n+", m["content"]):
                sent = " ".join(sent.lower().split())
                if len(sent) >= 25:
                    out.add(sent)
    return out


def dedupe_templates(rows: list, rng, max_per_shape, max_sentence_reuse, max_per_prompt=None) -> tuple:
    """rows = [(label, i, row)]. Deterministic shuffle, then keep an example only while its shape and each of its sentences are under the caps."""
    rng.shuffle(rows)
    shapes, sents, prompts, kept, dropped = {}, {}, {}, [], 0
    for item in rows:
        row = item[2]
        key = shape_of(row) if max_per_shape else None
        ss = sentences_of(row) if max_sentence_reuse else set()
        pkey = " ".join(next((m["content"] for m in row.get("messages", []) if m["role"] == "user"), "").lower().split()) if max_per_prompt else None
        if (max_per_prompt and prompts.get(pkey, 0) >= max_per_prompt) or (max_per_shape and shapes.get(key, 0) >= max_per_shape) or (max_sentence_reuse and any(sents.get(x, 0) >= max_sentence_reuse for x in ss)):
            dropped += 1
            continue
        if max_per_prompt:
            prompts[pkey] = prompts.get(pkey, 0) + 1
        if max_per_shape:
            shapes[key] = shapes.get(key, 0) + 1
        for x in ss:
            sents[x] = sents.get(x, 0) + 1
        kept.append(item)
    return kept, dropped


@register("local_jsonl")
def local_jsonl(params: dict, seed: int) -> Iterator:
    rows, skipped = [], 0
    for p in params["paths"]:
        path = Path(p)
        if not path.exists():
            if params.get("optional"):                       # data that is produced later (e.g. on the GPU box): skip with a warning
                print(f"[local_jsonl] optional path missing, skipped: {path}")
                continue
            raise FileNotFoundError(f"local_jsonl path not found: {path}")
        label = params.get("source_name", path.stem)
        opener = (lambda q: gzip.open(q, "rt")) if path.suffix == ".gz" else (lambda q: q.open())
        with opener(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                rows.append((label, i, row))
    for _ in range(skipped):
        yield SkipExample("jsonl_bad_json")
    rng = random.Random(seed)
    if params.get("max_per_shape") or params.get("max_sentence_reuse") or params.get("max_per_prompt"):
        rows, dropped = dedupe_templates(rows, rng, params.get("max_per_shape"), params.get("max_sentence_reuse"), params.get("max_per_prompt"))
        print(f"[local_jsonl] {params.get('source_name', '?')}: dropped {dropped} near-duplicate template outputs, kept {len(rows)}")
    cap_f, cap_all = params.get("max_per_family"), params.get("max_examples")
    if cap_f:
        by_fam = defaultdict(list)
        for item in rows:
            by_fam[item[2].get("meta", {}).get("family", "?")].append(item)
        rows = []
        for fam in sorted(by_fam):
            items = by_fam[fam]
            rng.shuffle(items)
            rows += items[:cap_f]
    if cap_all and len(rows) > cap_all:
        rng.shuffle(rows)
        rows = rows[:cap_all]
    for label, i, row in rows:
        yield make_record(str(row.get("id", f"{label}/{i}")), label, row.get("messages", []), row.get("tools"))
