"""One agent episode driven by a MODEL: it reads the task prompt, emits tool calls, sees real tool results, and answers.

An AgentRun is a small state machine so many episodes can be batched together (run_eval.py asks a generator for the next
assistant message of every active episode, then feeds each result back). The task, its environment and its checker are the
same ones used to build the training data (regenerated from family + seed), so the score is exact and automatic.
"""
from __future__ import annotations

import random
from typing import Optional

import envs.tasks  # noqa: F401  (registers families)
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, TOOLS
from envs.sandbox import ARG_NAME, Workspace
from envs.tasks.api_base import ApiTask
from envs.tasks.base import make_task
from envs.workspace_api import WorkspaceAPI, tool_schema
from evaluation.parse import Parsed, parse_assistant_output


class AgentRun:
    def __init__(self, family: str, seed: int, max_steps: int = 8):
        self.family, self.seed, self.max_steps = family, seed, max_steps
        self.task = make_task(family, seed)
        r = random.Random(f"sys:{family}:{seed}")                      # same system-prompt choice as the dataset builder
        sys_sandbox, sys_api = SYSTEM_PROMPTS[r.randrange(len(SYSTEM_PROMPTS))], API_SYSTEM_PROMPTS[r.randrange(len(API_SYSTEM_PROMPTS))]
        self.is_api = isinstance(self.task, ApiTask)
        if self.is_api:
            self.env = WorkspaceAPI(self.task.world)
            self.tools = [tool_schema(t) for t in self.task.tools]
            self.user_turns = [t.prompt for t in self.task.turns]
            system = sys_api
        else:
            self.env = Workspace(self.task.files)
            self.tools = TOOLS
            self.user_turns = [self.task.prompt] + [f.prompt for f in self.task.followups]
            system = sys_sandbox
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": self.user_turns[0]}]
        self.turn, self.steps_in_turn, self.replies = 0, 0, []
        self.done, self.fail_reason = False, None
        self.n_calls = self.format_errors = self.tool_errors = self.n_assistant = 0
        self.result: Optional[dict] = None

    # ---- interface for generators -------------------------------------------------------------------------------------
    def prompt_text(self, tok) -> str:
        return tok.apply_chat_template(self.messages, tools=self.tools, tokenize=False, add_generation_prompt=True)

    # ---- one model output -----------------------------------------------------------------------------------------------
    def feed(self, text: Optional[str]) -> None:
        if text is None:                                               # generator could not produce (e.g. prompt too long)
            return self._finish("generation_failed")
        self.n_assistant += 1
        p: Parsed = parse_assistant_output(text, self.tools)
        self.format_errors += p.malformed
        if p.calls:
            self._act(p)
            if self.steps_in_turn >= self.max_steps and not self.done:
                self._finish("step_cap")
            return
        reply = p.content if p.content else text.strip()[:2000]
        self.messages.append({"role": "assistant", "content": reply})
        self.replies.append(reply)
        if self.turn + 1 < len(self.user_turns):
            self.turn += 1
            self.steps_in_turn = 0
            self.messages.append({"role": "user", "content": self.user_turns[self.turn]})
        else:
            self._finish(None)

    def _act(self, p: Parsed) -> None:
        self.messages.append({"role": "assistant", "content": p.content,
                              "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in p.calls]})
        for c in p.calls:
            self.n_calls += 1
            out = self._execute(c["name"], c["arguments"])
            if out.startswith('{"error"') or "[exit code" in out:
                self.tool_errors += 1
            self.messages.append({"role": "tool", "content": out})
        self.steps_in_turn += 1

    def _execute(self, name: str, args: dict) -> str:
        if self.is_api:
            return self.env.call(name, args)
        if name not in ARG_NAME:
            return f"Error: unknown tool {name!r}. Available tools: run_python, run_bash."
        key = ARG_NAME[name]
        if not isinstance(args, dict) or not isinstance(args.get(key), str):
            return f"Error: {name} needs a string argument {key!r}."
        return self.env.run(name, args[key])

    # ---- scoring ----------------------------------------------------------------------------------------------------------
    def _finish(self, reason: Optional[str]) -> None:
        self.done = True
        if reason:
            ok, why = False, reason
        else:
            try:
                ok, why = self.task.check(self.env, self.replies) if self.is_api else self.task.check(self.env.root, self.replies[-1])
            except Exception as e:                                     # a broken answer must never crash the evaluation
                ok, why = False, f"checker_error:{type(e).__name__}"
        self.result = {"family": self.family, "seed": self.seed, "kind": "api" if self.is_api else "sandbox", "category": self.task.category,
                       "difficulty": self.task.difficulty, "ok": bool(ok), "reason": why if not ok else "ok", "n_calls": self.n_calls,
                       "n_assistant_msgs": self.n_assistant, "format_errors": self.format_errors, "tool_errors": self.tool_errors,
                       "turns": len(self.user_turns), "tags": self.task.tags, "messages": self.messages}
        if not self.is_api:
            self.env.close()
