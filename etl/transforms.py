"""Cleaning, dedup and deterministic train/val split on canonical records."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable

from etl.schema import content_hash, validate


def clean(rec: dict) -> dict:
    """Strip stray whitespace; drop unknown keys."""
    msgs = []
    for m in rec["messages"]:
        out = {"role": m["role"], "content": m["content"].strip() if isinstance(m.get("content"), str) else m.get("content")}
        if m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        msgs.append(out)
    return {"id": rec["id"], "source": rec["source"], "messages": msgs, "tools": rec.get("tools")}


def clean_and_validate(records: Iterable[dict], drops: Counter) -> list[dict]:
    out = []
    for rec in records:
        rec = clean(rec)
        reason = validate(rec)
        if reason:
            drops[f"invalid:{reason}"] += 1
            continue
        out.append(rec)
    return out


def dedup(records: list[dict], drops: Counter) -> list[dict]:
    """Exact-duplicate removal on conversation content (keeps first occurrence)."""
    seen: set[str] = set()
    out = []
    for rec in records:
        h = content_hash(rec)
        if h in seen:
            drops["duplicate"] += 1
            continue
        seen.add(h)
        out.append(rec)
    return out


def _group_key(rec: dict) -> str:
    """Conversations that start from the same user prompt (e.g. the same task in several dataset configs, or one
    conversation expanded into several turns) must land in the SAME split, otherwise validation measures memorisation."""
    first_user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), None)
    return hashlib.sha1(first_user.encode("utf-8")).hexdigest() if first_user else content_hash(rec)


def split_train_val(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic split by FIRST USER PROMPT: the same prompt always lands in the same split, independent of
    ordering and of which dataset config it came from, so re-running the pipeline never leaks val prompts into train.
    A prompt REWRITE (id ending /teacher or /tm) reuses the group of the trajectory it was derived from, so a rewrite and its original never
    end up on opposite sides of the split."""
    group_of = {r["id"]: _group_key(r) for r in records if not r["id"].endswith(("/teacher", "/tm"))}
    train, val = [], []
    threshold = int(val_fraction * 10_000)
    for rec in records:
        rid = rec["id"]
        key = group_of.get(rid.rsplit("/", 1)[0], _group_key(rec)) if rid.endswith(("/teacher", "/tm")) else _group_key(rec)
        bucket = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16) % 10_000
        (val if bucket < threshold else train).append(rec)
    return train, val
