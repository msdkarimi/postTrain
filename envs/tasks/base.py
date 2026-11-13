"""Task framework: a Task = prompt + input files + a scripted teacher solution + an independent checker.

The "teacher" is a set of solution programs written by hand (several strategies per task family, with short
ReAct-style plans). Every step is EXECUTED for real in the sandbox, so the observations are real; the checker
decides if the trajectory is kept (rejection sampling). Tasks are procedurally generated from a seed.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

Text = Union[str, Callable[[list], str]]


@dataclass
class Step:
    plan: Text                      # short reasoning shown before the action (may depend on earlier observations)
    tool: str                       # run_python | run_bash
    arg: Text                       # code / command (may depend on earlier observations)
    expect_error: bool = False      # an intentional first attempt that must really fail


@dataclass
class Followup:
    """A later user turn in a multi-turn classic task (same workspace)."""
    prompt: str
    steps: list
    final: str
    grounded: list = field(default_factory=list)


@dataclass
class Task:
    family: str
    category: str
    difficulty: str                 # easy | medium | hard
    prompt: str
    files: dict
    steps: list
    final: str                      # the closing reply
    check: Callable[[Path, str], tuple]   # (workspace dir, final reply) -> (ok, reason); independent of the solution
    grounded: list = field(default_factory=list)   # strings the final reply asserts; must occur in a tool observation
    tags: list = field(default_factory=list)
    followups: list = field(default_factory=list)  # later user turns (multi-turn tasks)


FAMILIES: dict = {}


def family(name: str, category: str, weight: float = 1.0):
    def deco(fn):
        FAMILIES[name] = (fn, category, weight)
        return fn
    return deco


def make_task(name: str, seed: int) -> Task:
    fn, _, _ = FAMILIES[name]
    return fn(random.Random(f"{name}:{seed}"))


# ---- phrasing helpers ----------------------------------------------------------------------------------------
def pick(rng: random.Random, options):
    return rng.choice(options)


def decorate(rng: random.Random, text: str) -> str:
    """Light, realistic variation in how users type."""
    if rng.random() < 0.10 and text[:1].isupper() and not text[:2].isupper():
        text = text[0].lower() + text[1:]
    if rng.random() < 0.10:
        text = text.rstrip(".")
    return text


INTROS = ["", "", "", "Hi! ", "Hey, ", "Quick one: ", "Could you help me with something? ", "Hello. ", "Need a hand here. ",
          "I'm in a rush, so please be efficient. ", "Working on my weekly report. ", "Small task: "]
OUTROS = ["", "", "", "", " Thanks!", " Thanks in advance.", " Let me know once it's done.", " Keep your reply short.",
          " Please double-check the result.", " Thank you!"]


def frame(rng: random.Random, body: str) -> str:
    return decorate(rng, f"{pick(rng, INTROS)}{body}{pick(rng, OUTROS)}".strip())


# ---- plan phrasing -------------------------------------------------------------------------------------------
PLAN_LS = ["Let me start by looking at what's in the workspace.", "First I'll list the files to see what I'm working with.",
           "I'll check the workspace contents first.", "Let me see which files are available.",
           "I should look at the files before doing anything."]
PLAN_PEEK = ["Now I'll look at the first few lines to understand the format.",
             "Let me peek at the structure of the data before writing any code.",
             "I need to see the column names and the data format first.",
             "Let me check how the file is laid out."]
PLAN_BOTH = ["I'll list the files and look at the start of the relevant ones to understand the layout.",
             "Let me inspect the workspace and the data format in one go.",
             "First, let me see the files and what the data looks like."]
PLAN_VERIFY = ["Let me verify the result before I answer.", "I'll double-check the output file.",
               "Quick check that the output looks right.", "Let me confirm what was written."]


def inspect_steps(rng: random.Random, peek: list, lines: int = 5) -> list:
    """Exploration first: the agent looks at the files before acting (so it can read the real schema)."""
    mode = rng.choice(["combined", "separate", "python"])
    if mode == "combined":
        cmd = "ls; " + "; ".join(f"echo '== {p}'; head -n {lines} '{p}'" for p in peek)
        return [Step(pick(rng, PLAN_BOTH), "run_bash", cmd)]
    if mode == "separate":
        return [Step(pick(rng, PLAN_LS), "run_bash", "ls"),
                Step(pick(rng, PLAN_PEEK), "run_bash", "; ".join(f"echo '== {p}'; head -n {lines} '{p}'" for p in peek))]
    code = ("import os\nfor name in sorted(os.listdir('.')):\n    print(name)\n"
            + "".join(f"print('== {p}')\nwith open('{p}') as f:\n    for _ in range({lines}):\n        line = f.readline()\n"
                     f"        if not line: break\n        print(line.rstrip())\n" for p in peek))
    return [Step(pick(rng, PLAN_BOTH), "run_python", code)]


PLAN_DO = ["Now I'll write a short script that does the calculation.", "I have what I need; let me compute it.",
           "Time to do the actual work with a small script.", "Now that I know the layout, I'll process the data.",
           "Let me write the code for this."]
PLAN_DO_BASH = ["I can do this with a quick shell command.", "A short command should handle this.",
                "Now I'll run the command that does the job."]


def verify_step(rng: random.Random, cmd: str, p: float = 0.6) -> list:
    """Self-verification: look at the result before answering (done in most trajectories)."""
    return [Step(pick(rng, PLAN_VERIFY), "run_bash", cmd)] if rng.random() < p else []


def last_error_line(obs: str) -> str:
    """The final line of a traceback / error message in an observation (grounds the model's reaction)."""
    body = obs.split("\n[exit code")[0]
    lines = [l for l in body.strip().split("\n") if l.strip()]
    return lines[-1].strip() if lines else ""


def with_recovery(rng: random.Random, good: Step, bugs: list, p: float = 0.25):
    """Maybe prepend an intentionally failing first attempt (a REAL error) followed by the fix.
    bugs = [(buggy_arg, plan_after_error(obs) -> str), ...]. Returns (steps, recovered)."""
    if bugs and rng.random() < p:
        bad_arg, fix_plan = rng.choice(bugs)
        return [Step(good.plan, good.tool, bad_arg, expect_error=True), Step(fix_plan, good.tool, good.arg)], True
    return [good], False


# ---- checker helpers -----------------------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def has_number(text: str, value: float, tol: float = 0.011) -> bool:
    return any(abs(n - value) <= tol for n in numbers_in(text))


def read_text(ws: Path, rel: str) -> Optional[str]:
    p = ws / rel
    return p.read_text() if p.exists() else None


def norm_lines(text: str) -> list:
    return [l.strip() for l in text.strip().splitlines() if l.strip()]
