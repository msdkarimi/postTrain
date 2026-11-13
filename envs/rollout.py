"""Run a scripted task in the sandbox and turn it into a canonical conversation (see etl/schema.py)."""
from __future__ import annotations

import json

from envs.sandbox import ARG_NAME, Workspace
from envs.tasks.base import Task
from envs.tasks.api_base import ApiTask
from envs.workspace_api import WorkspaceAPI, tool_schema

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run Python 3 code (standard library only, no internet) with the workspace directory /work as the "
                       "working directory. Returns stdout, stderr and the exit code.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code to execute."}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a bash command line with the workspace directory /work as the working directory (no internet). "
                       "Returns stdout, stderr and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                       "required": ["command"]}}},
]

SYSTEM_PROMPTS = [
    "You are a helpful workspace assistant. The user's files are in the current directory (/work). Use the tools to inspect "
    "the files and do the work instead of guessing. When you are done, reply with a short summary of the result.",
    "You are an assistant that completes office and data tasks on files in /work. Look at the files before acting, verify "
    "your results, and finish with a concise answer.",
    "You can run Python (standard library only) and bash in the user's workspace at /work. Never invent data: read it from "
    "the files. If something the user asks for does not exist, say so.",
    "Workspace assistant. Files live in /work. Think briefly, use tools step by step, check the output, then answer clearly.",
]


def run_scripted(task: Task, system: str) -> dict:
    """Execute task.steps for real. Returns messages, observations, and the checker verdict."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]
    obs: list = []
    with Workspace(task.files) as ws:
        for i, step in enumerate(task.steps):
            plan = step.plan(obs) if callable(step.plan) else step.plan
            arg = step.arg(obs) if callable(step.arg) else step.arg
            out = ws.run(step.tool, arg)
            if step.expect_error and ws.last_returncode == 0:
                return {"ok": False, "reason": "intended_bug_did_not_fail", "messages": messages, "observations": obs}
            if not step.expect_error and ws.last_returncode != 0:
                return {"ok": False, "reason": f"step_{i}_failed_unexpectedly", "messages": messages, "observations": obs + [out]}
            call = {"type": "function", "function": {"name": step.tool, "arguments": {ARG_NAME[step.tool]: arg}}}
            messages.append({"role": "assistant", "content": plan, "tool_calls": [call]})
            messages.append({"role": "tool", "content": out})
            obs.append(out)
        messages.append({"role": "assistant", "content": task.final})
        final = task.final
        for fu in task.followups:                              # multi-turn: the workspace state persists between turns
            messages.append({"role": "user", "content": fu.prompt})
            for step in fu.steps:
                plan = step.plan(obs) if callable(step.plan) else step.plan
                arg = step.arg(obs) if callable(step.arg) else step.arg
                out = ws.run(step.tool, arg)
                if ws.last_returncode != 0 and not step.expect_error:
                    return {"ok": False, "reason": "followup_step_failed", "messages": messages, "observations": obs + [out]}
                call = {"type": "function", "function": {"name": step.tool, "arguments": {ARG_NAME[step.tool]: arg}}}
                messages.append({"role": "assistant", "content": plan, "tool_calls": [call]})
                messages.append({"role": "tool", "content": out})
                obs.append(out)
            messages.append({"role": "assistant", "content": fu.final})
            final = fu.final
        ok, reason = task.check(ws.root, final)
    return {"ok": ok, "reason": reason, "messages": messages, "observations": obs}


API_SYSTEM_PROMPTS = [
    "You are a productivity assistant with access to workspace tools (calendar, contacts, email, tasks, files). Use the tools to get facts "
    "instead of guessing, make tool calls with exact arguments, and finish with a short, accurate reply.",
    "You help the user manage their calendar, email, tasks and files through the tools provided. Never send an email unless the user "
    "explicitly asks you to; drafts are fine. If a request is ambiguous, ask a short clarifying question before acting.",
    "Workspace assistant. Call tools when you need data or need to change something; answer directly when no tool is needed; say so "
    "plainly when a request is outside what the tools can do. Confirm before deleting anything.",
    "You are an assistant embedded in a work platform. Use the available functions step by step, check their results, and report what "
    "you actually did. Don't invent data that the tools did not return.",
]


def run_api(task: ApiTask, system: str) -> dict:
    """Execute an ApiTask against a fresh WorkspaceAPI; returns the same shape as run_scripted (+ tools)."""
    api = WorkspaceAPI(task.world)
    messages = [{"role": "system", "content": system}]
    obs: list = []
    replies: list = []
    tools = [tool_schema(t) for t in task.tools]

    def fail(reason):
        return {"ok": False, "reason": reason, "messages": messages, "observations": obs, "tools": tools}

    for turn in task.turns:
        messages.append({"role": "user", "content": turn.prompt})
        for step in turn.steps:
            snap = list(obs)                                   # parallel calls never see each other's results
            plan = step.plan(snap) if callable(step.plan) else step.plan
            tool_calls, outs = [], []
            for c in step.calls:
                args = c.args(snap) if callable(c.args) else c.args
                out = api.call(c.tool, args)
                if out.startswith('{"error"') != c.expect_error:
                    return fail(f"unexpected_{'error' if not c.expect_error else 'success'}:{c.tool}:{out[:80]}")
                tool_calls.append({"type": "function", "function": {"name": c.tool, "arguments": args}})
                outs.append(out)
            messages.append({"role": "assistant", "content": plan, "tool_calls": tool_calls})
            for o in outs:
                messages.append({"role": "tool", "content": o})
                obs.append(o)
        final = turn.final(obs) if callable(turn.final) else turn.final
        messages.append({"role": "assistant", "content": final})
        replies.append(final)
    ok, reason = task.check(api, replies)
    return {"ok": ok, "reason": reason, "messages": messages, "observations": obs, "tools": tools, "api": api}


def run_any(task, system_scripted: str, system_api: str) -> dict:
    return run_api(task, system_api) if isinstance(task, ApiTask) else {**run_scripted(task, system_scripted), "tools": TOOLS}
