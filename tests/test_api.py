"""Tests for the simulated workspace API and the function-calling task families."""
import json

import pytest

import envs.tasks  # noqa: F401
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, run_any
from envs.tasks.api_base import ApiTask, free_start, make_world, pick_tools
from envs.tasks.base import FAMILIES, make_task
from envs.workspace_api import TOOL_DEFS, WorkspaceAPI, tool_schema
import random


def world():
    return make_world(random.Random(5))


def err(out):
    d = json.loads(out)
    return d.get("code") if "error" in d else None


def test_every_tool_has_a_valid_schema():
    assert len(TOOL_DEFS) == 22
    for name in TOOL_DEFS:
        fn = tool_schema(name)["function"]
        props, req = fn["parameters"]["properties"], fn["parameters"]["required"]
        assert fn["name"] == name and fn["description"] and set(req) <= set(props)


def test_api_validates_arguments_and_returns_real_errors():
    api = WorkspaceAPI(world())
    assert err(api.call("no_such_tool", {})) == "unknown_tool"
    assert err(api.call("read_file", {})) == "invalid_arguments"                       # missing required
    assert err(api.call("read_file", {"path": 3})) == "invalid_arguments"              # wrong type
    assert err(api.call("tasks_list", {"status": "later"})) == "invalid_arguments"     # enum
    assert err(api.call("read_file", {"path": "nope.txt"})) == "not_found"
    name = api.contacts[0]["name"]
    assert err(api.call("calendar_list_events", {"person": name, "date": "10/03/2025"})) == "invalid_argument"
    assert err(api.call("calendar_create_event", {"title": "x", "date": "2025-03-10", "start": "9.30", "duration_minutes": 30, "attendees": [name]})) == "invalid_argument"
    assert err(api.call("calendar_list_events", {"person": "Nobody Here", "date": "2025-03-10"})) == "not_found"


def test_free_slot_matches_independent_brute_force():
    for seed in range(30):
        w = make_world(random.Random(seed))
        api = WorkspaceAPI(w)
        att = [c["name"] for c in w["contacts"][:3]]
        got = json.loads(api.call("calendar_find_free_slot", {"attendees": att, "date": w["day"], "duration_minutes": 45}))["slot"]
        exp = free_start(w["events"], att, w["day"], 45)
        assert (got["start"] if got else None) == (None if exp is None else f"{exp // 60:02d}:{exp % 60:02d}")


def test_calendar_email_task_state_changes():
    w = world()
    api = WorkspaceAPI(w)
    a, b = [c["name"] for c in w["contacts"][:2]]
    ev = json.loads(api.call("calendar_create_event", {"title": "Sync", "date": w["day"], "start": "09:00", "duration_minutes": 30, "attendees": [a, b]}))["event"]
    assert json.loads(api.call("calendar_update_event", {"event_id": ev["id"], "start": "10:15"}))["event"]["end"] == "10:45"
    assert err(api.call("calendar_delete_event", {"event_id": "evt_nope"})) == "not_found"
    d = json.loads(api.call("email_draft", {"to": ["x@y.com"], "subject": "s", "body": "b"}))["draft_id"]
    assert not api.sent and err(api.call("email_draft", {"to": ["not-an-email"], "subject": "s", "body": "b"})) == "invalid_argument"
    api.call("email_send", {"draft_id": d})
    assert len(api.sent) == 1 and err(api.call("email_send", {"draft_id": d})) == "not_found"
    t = json.loads(api.call("tasks_add", {"title": "T", "owner": a, "due": w["day"]}))["task"]
    api.call("tasks_complete", {"task_id": t["id"]})
    assert json.loads(api.call("tasks_list", {"status": "done"}))["tasks"][0]["id"] == t["id"]


def test_calculator_is_safe_and_exact():
    api = WorkspaceAPI(world())
    assert json.loads(api.call("calculator", {"expression": "(120 - 45) / 120 * 100"}))["result"] == 62.5
    assert err(api.call("calculator", {"expression": "__import__('os').system('echo hi')"})) == "invalid_expression"
    assert err(api.call("calculator", {"expression": "1 / 0"})) == "invalid_expression"


def test_table_aggregate_and_search():
    api = WorkspaceAPI({"files": {"d/s.csv": "region,amount\nN,10.5\nS,4\nN,5\n", "a.txt": "Hello World\nbye\n"}, "contacts": [], "events": []})
    assert json.loads(api.call("table_aggregate", {"path": "d/s.csv", "column": "amount", "op": "sum", "where_column": "region", "where_value": "N"}))["result"] == 15.5
    assert json.loads(api.call("table_aggregate", {"path": "d/s.csv", "column": "amount", "op": "avg", "group_by": "region"}))["groups"] == {"N": 7.75, "S": 4.0}
    assert err(api.call("table_aggregate", {"path": "d/s.csv", "column": "nope", "op": "sum"})) == "invalid_argument"
    assert err(api.call("table_aggregate", {"path": "d/s.csv", "column": "region", "op": "sum"})) == "invalid_argument"      # text column: an error for the model, not a crash
    assert json.loads(api.call("search_files", {"query": "hello world"}))["matches"][0]["path"] == "a.txt"
    assert json.loads(api.call("list_files", {"path": "."}))["entries"] == ["a.txt", "d/"]


def test_tool_sets_contain_required_tools_and_distractors():
    rng = random.Random(1)
    for _ in range(20):
        tools = pick_tools(rng, ["email_draft", "tasks_add"])
        assert {"email_draft", "tasks_add"} <= set(tools) and 5 <= len(tools) <= 12 and len(set(tools)) == len(tools)


API_FAMILIES = sorted(n for n, (fn, cat, w) in FAMILIES.items() if n.startswith("api_"))


def test_api_dataset_has_parallel_calls_errors_multi_turn_and_no_tool_cases():
    seen = {"parallel": 0, "error": 0, "multi": 0, "no_tool": 0}
    for name in API_FAMILIES:
        for seed in range(9000, 9015):
            res = run_any(make_task(name, seed), SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
            assert res["ok"], (name, seed, res["reason"])
            m = res["messages"]
            seen["parallel"] += any(x["role"] == "assistant" and len(x.get("tool_calls", [])) >= 2 for x in m)
            seen["error"] += any(x["role"] == "tool" and x["content"].startswith('{"error"') for x in m)
            seen["multi"] += sum(x["role"] == "user" for x in m) > 1
            seen["no_tool"] += not any(x.get("tool_calls") for x in m)
            for i, x in enumerate(m):                                         # parallel calls are answered by the same number of tool messages
                if x["role"] == "assistant" and x.get("tool_calls"):
                    k = len(x["tool_calls"])
                    assert [y["role"] for y in m[i + 1:i + 1 + k]] == ["tool"] * k
    assert all(v > 0 for v in seen.values()), seen


def test_clarification_family_asks_before_acting():
    task = make_task("api_ambiguous_contact", 123)
    assert isinstance(task, ApiTask) and len(task.turns) == 2
    res = run_any(task, SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
    first_reply = next(m for m in res["messages"] if m["role"] == "assistant" and not m.get("tool_calls"))
    assert "?" in first_reply["content"] and not res["api"].sent and len(res["api"].drafts) == 1


def test_draft_family_never_sends_unless_asked():
    sent_cases = drafted_cases = 0
    for seed in range(9100, 9140):
        task = make_task("api_draft_or_send", seed)
        res = run_any(task, SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
        wants_send = "email_send" in task.tools and any(c.tool == "email_send" for s in task.turns[0].steps for c in s.calls)
        assert bool(res["api"].sent) == wants_send
        sent_cases += wants_send
        drafted_cases += not wants_send
    assert sent_cases and drafted_cases
