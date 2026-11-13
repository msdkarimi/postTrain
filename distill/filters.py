"""Quality filters applied to every verified trajectory (on top of the task checker)."""
from __future__ import annotations

import json
import re
from typing import Optional

FORBIDDEN = ["expected answer", "ground truth", "the checker", "hidden test", "as an ai", "the teacher", "verifier"]
MAX_CHARS = 14_000          # total characters of a trajectory (code + observations + text)
MAX_PLAN_CHARS = 450        # reasoning stays short: small students learn better from brief plans


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def hygiene(task, res: dict) -> Optional[str]:
    """Return None if the trajectory is clean, else a short reason."""
    msgs, obs = res["messages"], res["observations"]
    if not obs and not getattr(task, "no_tools_ok", False):
        return "no_tool_use"
    # 1) every value the final reply asserts must have been OBSERVED in a tool output (no unsupported claims)
    for tok in task.grounded:
        if not any(tok in o for o in obs):
            return "ungrounded_claim"
    # 2) the plan text must not state a result before a tool has shown it (no answer leakage / guessing)
    seen, user_text = 0, ""
    for m in msgs[1:]:
        if m["role"] == "user":
            user_text += " " + m["content"]                  # values the USER stated may be repeated in a plan
        elif m["role"] == "tool":
            seen += 1
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for tok in task.grounded:
                if len(tok) >= 4 and tok in m["content"] and tok not in user_text and not any(tok in o for o in obs[:seen]):
                    return "answer_leak_in_plan"
    # 3) must not end on an error
    if obs and ("[exit code" in obs[-1] or obs[-1].startswith('{"error"')):
        return "ends_with_error"
    # 4) no meta chatter about the data pipeline
    assistant_text = " ".join(m["content"] for m in msgs if m["role"] == "assistant").lower()
    if any(p in assistant_text for p in FORBIDDEN):
        return "meta_phrase"
    # 5) bounded sizes
    total = sum(len(m["content"]) + len(json.dumps(m.get("tool_calls", []))) for m in msgs)
    if total > MAX_CHARS:
        return "too_long"
    if any(m["role"] == "assistant" and m.get("tool_calls") and len(m["content"]) > MAX_PLAN_CHARS for m in msgs):
        return "plan_too_long"
    if any(m["role"] == "assistant" and not m.get("tool_calls") and len(m["content"]) > 1500 for m in msgs):
        return "reply_too_long"
    # 6) a final reply must exist and be non-trivial
    if len(msgs[-1]["content"].strip()) < 12 or msgs[-1].get("tool_calls"):
        return "bad_final"
    return None
