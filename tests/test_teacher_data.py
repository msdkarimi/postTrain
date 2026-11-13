import json

import pytest

from distill import rewrite, teacher_tasks

yaml = pytest.importorskip("yaml")


def test_key_tokens_and_loss_detection():
    orig = "Hey, Show me Theo Silva's meetings on 2025-04-14 and save to out/report.md for 3 people."
    assert {"2025-04-14", "out/report.md", "3", "Theo", "Silva"} <= set(rewrite.key_tokens(orig))
    assert "Show" not in rewrite.key_tokens(orig)                      # sentence-initial verbs are not facts
    assert rewrite.missing(orig, "what does 2025-04-14 look like for Theo Silva? write out/report.md, 3 people") == []
    assert rewrite.missing(orig, "show Theo's meetings for 2025-04-14, save it somewhere, 3 people") == ["Silva", "out/report.md"]


def test_rewrite_apply_rejects_changed_facts_and_keeps_trajectory(tmp_path):
    src = tmp_path / "src.jsonl"
    rec = {"id": "agent/f/1", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Sum sales.csv for the West region in 2025-01."},
                                           {"role": "assistant", "content": "ok"}], "tools": [], "meta": {"family": "f", "tags": ["x"]}}
    src.write_text(json.dumps(rec) + "\n")
    (tmp_path / "todo.tsv").write_text("agent/f/1\tSum sales.csv for the West region in 2025-01.\n")
    (tmp_path / "rw.tsv").write_text("1 ||| how much did the West region bring in during 2025-01? it's in sales.csv\n1 ||| how much did we sell in 2025-01 in sales.csv?\n")
    import argparse
    out = tmp_path / "out.jsonl"
    rewrite.cmd_apply(argparse.Namespace(src=str(src), rewrites=str(tmp_path / "rw.tsv"), out=str(out), report=False, suffix="teacher", todo=str(tmp_path / "todo.tsv")))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["id"] == "agent/f/1/teacher"                 # the second rewrite lost "West" and was rejected
    assert rows[0]["messages"][1]["content"].startswith("how much did the West")
    assert rows[0]["messages"][2] == rec["messages"][2] and "teacher_rewrite" in rows[0]["meta"]["tags"]


def test_teacher_tasks_are_verified_by_execution(tmp_path):
    good = """
- id: ok1
  category: demo
  prompt: What is in a.txt?
  files: {a.txt: "hello"}
  steps:
    - plan: Let me read it.
      tool: run_bash
      arg: cat a.txt
  final: The file contains the word hello.
  grounded: [hello]
  check: {reply_contains: [hello]}
- id: bad_claim
  category: demo
  prompt: What is in a.txt?
  files: {a.txt: "hello"}
  steps:
    - plan: Let me read it.
      tool: run_bash
      arg: cat a.txt
  final: The file contains the word goodbye.
  grounded: [goodbye]
  check: {reply_contains: [goodbye]}
- id: bad_state
  category: demo
  prompt: Write out.txt with the word hi.
  files: {a.txt: "hello"}
  steps:
    - plan: I'll write it.
      tool: run_bash
      arg: echo hi > out.txt
  final: Wrote out.txt with hi.
  check: {files: {out.txt: {equals: "bye"}}}
"""
    p = tmp_path / "t.yaml"
    p.write_text(good)
    rows, drops = teacher_tasks.build(teacher_tasks.load_yaml(str(p)), verbose=False)
    assert [r["id"] for r in rows] == ["agent/teacher/ok1"]                         # ungrounded claim and wrong file content are both rejected
    assert sum(drops.values()) == 2
    assert rows[0]["messages"][3]["content"].strip() == "hello"                     # the observation is real execution output


def test_teacher_api_tasks_check_final_state_not_words(tmp_path):
    from distill import teacher_api_tasks as ta
    base = """
- id: good
  category: demo
  tools: [contacts_search, tasks_add]
  world: {contacts: [{name: Bo Chen, email: bo@x.com, team: Eng}]}
  prompt: Add a task for Bo to review the docs.
  steps:
    - plan: I need Bo's full name for the owner field.
      calls: [{tool: contacts_search, args: {query: Bo}}]
    - plan: Bo Chen it is, so I can add the task.
      calls: [{tool: tasks_add, args: {title: Review the docs, owner: Bo Chen}}]
  final: Added "Review the docs" for Bo Chen.
  grounded: [Bo Chen]
  check: {tasks: [{title: Review the docs, owner: Bo Chen}]}
- id: says_done_but_did_nothing
  category: demo
  tools: [contacts_search, tasks_add]
  world: {contacts: [{name: Bo Chen, email: bo@x.com, team: Eng}]}
  prompt: Add a task for Bo to update the wiki.
  steps:
    - plan: I need Bo's full name.
      calls: [{tool: contacts_search, args: {query: Bo}}]
  final: Added "Update the wiki" for Bo Chen.
  grounded: [Bo Chen]
  check: {tasks: [{title: Update the wiki, owner: Bo Chen}]}
- id: wrong_error_expectation
  category: demo
  tools: [contacts_search]
  world: {contacts: [{name: Bo Chen, email: bo@x.com, team: Eng}]}
  prompt: Find Bo.
  steps:
    - plan: Search.
      calls: [{tool: contacts_search, args: {query: Bo}, fails: true}]
  final: Found Bo Chen.
  check: {}
"""
    p = tmp_path / "a.yaml"
    p.write_text(base)
    rows, drops = ta.build(ta.load_yaml(str(p)), verbose=False)
    assert [r["id"] for r in rows] == ["agent/teacher_api/good"]
    assert sum(drops.values()) == 2                                           # no state change, and an "intended error" that did not happen
    assert rows[0]["meta"]["kind"] == "api"
