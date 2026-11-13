"""Function-calling task framework (ApiTask): typed tool calls against the simulated WorkspaceAPI.

An ApiTask has one or more TURNS (multi-turn conversations with follow-ups / clarifications). Each turn has steps; a step is
one assistant message with a short plan and one OR MORE tool calls (several calls in a step = parallel calls). Call arguments and
plans may be callables of the observations so far, so the scripted teacher reacts to what the API really returned.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional, Union

from envs.tasks.base import pick
from envs.tasks.common import DOMAINS, email_for, fmt_hhmm, people, rand_date
from envs.workspace_api import HI, LO, TOOL_DEFS

Text = Union[str, Callable[[list], str]]
TEAMS = ["Sales", "Engineering", "Support", "Finance", "Design", "Marketing"]
MEETING_TITLES = ["Project kickoff", "Weekly sync", "Planning session", "Design review", "Quarterly check-in", "Budget review",
                  "Customer workshop", "Roadmap discussion", "Retro", "Vendor demo", "Onboarding session", "Strategy meeting"]
BUSY_TITLES = ["Standup", "1:1 sync", "Sprint planning", "Customer call", "Lunch & learn", "Interview", "Security briefing", "Hiring panel",
               "Incident review", "Product sync", "Training session", "Team offsite prep"]


@dataclass
class Call:
    tool: str
    args: Union[dict, Callable[[list], dict]]
    expect_error: bool = False           # an intentional first attempt that must really fail with an API error


@dataclass
class ApiStep:
    plan: Text
    calls: list                          # >1 call = parallel tool calls in one assistant message


@dataclass
class Turn:
    prompt: str
    steps: list
    final: Text
    grounded: list = field(default_factory=list)


@dataclass
class ApiTask:
    family: str
    category: str
    difficulty: str
    world: dict
    tools: list                          # tool NAMES offered to the model (required ones + distractors)
    turns: list
    check: Callable                      # (WorkspaceAPI, [replies]) -> (ok, reason)
    tags: list = field(default_factory=list)
    no_tools_ok: bool = False            # direct answers / refusals / clarifications legitimately use no tool

    @property
    def prompt(self) -> str:
        return " || ".join(t.prompt for t in self.turns)

    @property
    def grounded(self) -> list:
        return [g for t in self.turns for g in t.grounded]


# ---- tool sets --------------------------------------------------------------------------------------------------------
def pick_tools(rng: random.Random, required: list, extra=(3, 7)) -> list:
    others = [t for t in TOOL_DEFS if t not in required]
    chosen = list(required) + rng.sample(others, min(len(others), rng.randint(*extra)))
    rng.shuffle(chosen)
    return chosen


# ---- worlds -----------------------------------------------------------------------------------------------------------
def busy_blocks(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        s = rng.choice(range(8 * 60, 17 * 60, 15))
        out.append((s, min(s + rng.choice([30, 45, 60, 90, 120]), 18 * 60)))
    return out


def make_world(rng: random.Random, n_people=(4, 7), busy=(2, 4), day: Optional[date] = None) -> dict:
    day = day or rand_date(rng, rng.choice([2025, 2026]))
    names = people(rng, rng.randint(*n_people))
    domain = rng.choice(DOMAINS)
    contacts = [{"name": nm, "email": email_for(nm, domain if rng.random() < 0.7 else rng.choice(DOMAINS)), "team": rng.choice(TEAMS)} for nm in names]
    events, k = [], 0
    for nm in names:
        for s, e in busy_blocks(rng, rng.randint(*busy)):
            k += 1
            events.append({"id": f"evt_{k:02d}", "title": rng.choice(BUSY_TITLES), "date": day.isoformat(), "start": fmt_hhmm(s), "end": fmt_hhmm(e), "attendees": [nm]})
        for s, e in busy_blocks(rng, rng.randint(1, 2)):                        # another day: must be ignored by day-specific queries
            k += 1
            events.append({"id": f"evt_{k:02d}", "title": rng.choice(BUSY_TITLES), "date": (day + timedelta(days=1)).isoformat(),
                           "start": fmt_hhmm(s), "end": fmt_hhmm(e), "attendees": [nm]})
    return {"today": (day - timedelta(days=rng.randint(1, 6))).isoformat(), "day": day.isoformat(), "contacts": contacts,
            "events": events, "inbox": [], "tasks": [], "files": {}}


def hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


def free_start(world_events: list, attendees: list, day: str, dur: int, lo=LO, hi=HI) -> Optional[int]:
    """Independent brute-force earliest common free start (minutes), over a list of event dicts."""
    busy = [(hm(e["start"]), hm(e["end"])) for e in world_events if e["date"] == day and set(e["attendees"]) & set(attendees)]
    for start in range(lo, hi - dur + 1):
        if all(not (start < e and s < start + dur) for s, e in busy):
            return start
    return None


def world_with_slot(rng, n_att=(2, 4), dur=None, want_late=True, **kw):
    """A world plus attendees and a duration such that a common slot exists (preferably not at 09:00)."""
    dur = dur or rng.choice([30, 45, 60])
    for attempt in range(80):
        w = make_world(rng, **kw)
        att = rng.sample([c["name"] for c in w["contacts"]], rng.randint(*n_att))
        slot = free_start(w["events"], att, w["day"], dur)
        if slot is not None and (slot > LO or not want_late or attempt > 60):
            return w, att, dur, slot
    raise RuntimeError("no feasible world")


def names_phrase(names: list) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def first(name: str) -> str:
    return name.split(" ", 1)[0]


# ---- observation helpers (used inside callable plans / args) --------------------------------------------------------------
def jfind(obs: list, key: str, nth: int = 0):
    """The nth parsed JSON observation that contains `key` at top level."""
    hits = [json.loads(o) for o in obs if o.startswith("{") and key in json.loads(o)]
    return hits[nth]


def email_of(world: dict, name: str) -> str:
    return next(c["email"] for c in world["contacts"] if c["name"] == name)
