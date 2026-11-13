"""Local sandbox for the two agent tools (run_python, run_bash).

Each task gets a throwaway workspace directory. A tool call runs as a fresh subprocess with cwd = the workspace,
a scrubbed environment (no secrets, LC_ALL=C, TZ=UTC), CPU/file-size limits and a wall-clock timeout. The temp path is
rewritten to /work in the output, so observations look the same wherever the workspace really lives.

NOT a security sandbox: there is no network or kernel isolation. It is fine for code we wrote ourselves (dataset
generation). For model-written code (evaluation / RL) run the same interface inside Docker or another isolated backend.

format_observation() defines what the model sees after a tool call; keep it identical in every environment.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_OBS_CHARS = 2000
TIMEOUT_S = 10
ARG_NAME = {"run_python": "code", "run_bash": "command"}


def format_observation(stdout: str, stderr: str, returncode: int, max_chars: int = MAX_OBS_CHARS) -> str:
    out = stdout
    if stderr.strip():
        out += ("\n" if out and not out.endswith("\n") else "") + "[stderr]\n" + stderr
    if returncode != 0:
        out += ("\n" if out and not out.endswith("\n") else "") + f"[exit code: {returncode}]"
    if not out.strip():
        out = "(no output)"
    out = out.rstrip("\n")
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n...[truncated {len(out) - max_chars} characters]"
    return out


def _limits(timeout: int):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 2, timeout + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))
    return apply


class Workspace:
    def __init__(self, files: dict[str, str]):
        self.root = Path(tempfile.mkdtemp(prefix="ws_")).resolve()
        self.last_returncode = 0
        for rel, content in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, tool: str, arg: str, timeout: int = TIMEOUT_S) -> str:
        if tool == "run_python":
            cmd = [sys.executable, "-I", "-c", arg]
        elif tool == "run_bash":
            cmd = ["bash", "-c", arg]
        else:
            raise ValueError(f"unknown tool {tool!r}")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "LC_ALL": "C", "LANG": "C", "TZ": "UTC",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PWD": str(self.root)}
        proc = subprocess.Popen(cmd, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, stdin=subprocess.DEVNULL, preexec_fn=_limits(timeout))
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            stderr += f"\nCommand timed out after {timeout} seconds"
            rc = 124
        self.last_returncode = rc
        text = format_observation(stdout, stderr, rc)
        # show /work instead of the real temp dir (macOS resolves /var -> /private/var)
        real = str(self.root)
        for variant in (real, real.replace("/private", "", 1)):      # ordered: longest path first
            text = text.replace(variant, "/work")
        return text
