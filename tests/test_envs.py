"""Tests for the sandbox, the task families and the trajectory filters (no network, no model needed)."""
import json
import random

import pytest

import envs.tasks  # noqa: F401  (registers the families)
from distill.filters import hygiene, normalize_prompt
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, run_any, run_scripted
from envs.sandbox import Workspace, format_observation
from envs.tasks.base import FAMILIES, Step, Task, make_task
from etl.schema import validate


# ---- sandbox ---------------------------------------------------------------------------------------------------
def test_sandbox_runs_python_and_bash_in_a_private_workspace():
    with Workspace({"a.txt": "hello\n"}) as ws:
        assert ws.run("run_bash", "cat a.txt") == "hello"
        assert ws.run("run_python", "print(open('a.txt').read().upper().strip())") == "HELLO"
        ws.run("run_bash", "echo more > b.txt")
        assert (ws.root / "b.txt").exists()
    assert not ws.root.exists()                                   # cleaned up


def test_sandbox_formats_errors_and_hides_the_real_path():
    with Workspace({}) as ws:
        out = ws.run("run_python", "open('missing.csv')")
        assert "FileNotFoundError" in out and "[exit code: 1]" in out and ws.last_returncode == 1
        assert str(ws.root) not in ws.run("run_bash", "pwd") and ws.run("run_bash", "pwd") == "/work"


def test_sandbox_timeout_and_truncation():
    with Workspace({}) as ws:
        out = ws.run("run_python", "import time; time.sleep(30)", timeout=1)
        assert "timed out" in out and ws.last_returncode == 124
        long = ws.run("run_python", "print('x' * 5000)")
        assert "[truncated" in long and len(long) < 2200
    assert format_observation("", "", 0) == "(no output)"


def test_sandbox_environment_is_scrubbed():
    with Workspace({}) as ws:
        assert ws.run("run_bash", "echo ${SECRET_TOKEN:-unset}") == "unset"
        assert ws.run("run_bash", "echo $TZ") == "UTC"


# ---- task families -----------------------------------------------------------------------------------------------
def test_there_are_many_families_and_they_are_deterministic():
    assert len(FAMILIES) >= 60
    for name in ("csv_filtered_sum", "common_free_slot", "missing_information"):
        a, b = make_task(name, 123), make_task(name, 123)
        assert a.prompt == b.prompt and a.files == b.files
        assert make_task(name, 124).prompt != a.prompt or make_task(name, 124).files != a.files


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_scripted_solution_passes_its_independent_checker(name):
    for seed in (7001, 7002, 7003):
        task = make_task(name, seed)
        res = run_any(task, SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
        assert res["ok"], f"{name} seed {seed}: {res['reason']}"
        assert hygiene(task, res) is None, f"{name} seed {seed}: {hygiene(task, res)}"
        assert validate({"messages": res["messages"], "tools": None}) is None


def test_a_wrong_answer_is_rejected_by_the_checker():
    task = make_task("csv_filtered_sum", 7010)
    bad = Task(**{**task.__dict__, "final": "The total is 1.00."})
    res = run_scripted(bad, SYSTEM_PROMPTS[0])
    assert not res["ok"]


def test_error_recovery_attempts_really_fail_first():
    found = 0
    for seed in range(8000, 8120):
        task = make_task("csv_filtered_sum", seed)
        if "error_recovery" in task.tags:
            res = run_scripted(task, SYSTEM_PROMPTS[0])
            tools = [m["content"] for m in res["messages"] if m["role"] == "tool"]
            assert any("[exit code: 1]" in t for t in tools) and "[exit code" not in tools[-1]
            found += 1
    assert found >= 3


def test_abstention_tasks_do_not_invent_data():
    task = make_task("missing_information", 7020)
    res = run_scripted(task, SYSTEM_PROMPTS[0])
    assert res["ok"] and not any(m.get("tool_calls") and "open(" in json.dumps(m["tool_calls"]) and "w'" in json.dumps(m["tool_calls"]) for m in res["messages"])


# ---- filters -----------------------------------------------------------------------------------------------------
def _fake_task(grounded, final="The answer is 4242."):
    return Task("f", "c", "easy", "p", {}, [], final, lambda ws, r: (True, "ok"), grounded)


def _res(messages, obs):
    return {"messages": messages, "observations": obs, "ok": True, "reason": "ok"}


def test_filters_reject_ungrounded_claims_and_leaks_and_trailing_errors():
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    call = [{"type": "function", "function": {"name": "run_bash", "arguments": {"command": "ls"}}}]
    good = base + [{"role": "assistant", "content": "Let me look.", "tool_calls": call}, {"role": "tool", "content": "value 4242"},
                   {"role": "assistant", "content": "The answer is 4242."}]
    assert hygiene(_fake_task(["4242"]), _res(good, ["value 4242"])) is None
    assert hygiene(_fake_task(["9999"]), _res(good, ["value 4242"])) == "ungrounded_claim"
    leak = base + [{"role": "assistant", "content": "I already know it is 4242.", "tool_calls": call}, {"role": "tool", "content": "value 4242"},
                   {"role": "assistant", "content": "The answer is 4242."}]
    assert hygiene(_fake_task(["4242"]), _res(leak, ["value 4242"])) == "answer_leak_in_plan"
    err = good[:-2] + [{"role": "tool", "content": "boom\n[exit code: 1]"}, good[-1]]
    assert hygiene(_fake_task([]), _res(err, ["boom\n[exit code: 1]"])) == "ends_with_error"


def test_prompt_normalisation_for_dedup():
    assert normalize_prompt("  Hello   WORLD ") == normalize_prompt("hello world")


# ---- dataset builder (tiny end-to-end run) -------------------------------------------------------------------------
def test_builder_end_to_end_small(tmp_path):
    import subprocess, sys
    out = tmp_path / "raw"
    subprocess.run([sys.executable, "-m", "distill.build_trajectories", "--n-train", "300", "--n-test-per-family", "2", "--out", str(out),
                    "--workers", "2"], check=True, capture_output=True)
    train = [json.loads(l) for l in (out / "agent_trajectories.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (out / "agent_test_tasks.jsonl").read_text().splitlines()]
    assert len(train) >= 150 and len(test) == 2 * len(FAMILIES)
    assert {r["meta"]["kind"] for r in train} == {"api", "sandbox"}
    assert {r["meta"]["seed"] for r in train}.isdisjoint({t["seed"] for t in test if t["seed"] >= 10_000})
    from distill.build_trajectories import task_signature
    assert not ({task_signature(make_task(r["meta"]["family"], r["meta"]["seed"])) for r in train} & {t["sig"] for t in test})
    assert all(validate({"messages": r["messages"], "tools": r["tools"]}) is None for r in train)
    assert (out / "DATASET_CARD.md").exists() and (out / "agent_stats.json").exists()
