import csv
import io
import json
import re

from distill import code_gen, extract_gen, fc_gen, if_struct_gen, reasoning_multi, refusal_gen, teacher_tasks

REFUSAL_MARKERS = re.compile(r"guessing|Nobody can know|Nothing in my knowledge|ring a bell|no record|know nothing|not familiar|don't have|can't|cannot|no real|no solution|undefined|doesn't|does not|nobody|isn't|aren't|don't know|don't recogni|not mention|There is no|no last digit|no largest|contradict|no such|not able|no way|never|missing|insufficient|I'd rather|have no|have any reliable|so I can't|find .* in what I know", re.I)


def test_refusal_data_refuses_when_it_should_and_answers_the_twins():
    rows = refusal_gen.build(scale=0.04)
    assert len(rows) > 200
    seen_groups = set()
    for r in rows:
        last = r["messages"][-1]["content"]
        assert r["messages"][-1]["role"] == "assistant" and len(last) > 12
        seen_groups.add(r["meta"]["tags"][-1] if r["meta"]["category"] == "answerable" else r["meta"]["tags"][1])
        if r["meta"]["category"] == "refusal":
            assert REFUSAL_MARKERS.search(" ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")), last[:120]
        else:
            assert not re.search(r"\bnot familiar\b|\bcan't verify\b", last)
        if r["meta"]["tags"][1:2] == ["f"]:
            assert r["tools"] and "tool_calls" not in json.dumps(r["messages"])        # a capability refusal calls nothing
    assert seen_groups == {"a", "b", "c", "d", "e", "f", "answerable_twin"}


def test_fictional_names_are_never_stated_as_facts():
    rows = [r for r in refusal_gen.build(scale=0.04) if r["meta"]["tags"][1:2] == ["a"]]
    assert rows
    for r in rows:
        a = r["messages"][1]["content"]
        assert not re.search(r"\b(founded in|was born|published in|ceo is|population of .* is)\b", a, re.I)


def test_extraction_outputs_are_exact_in_every_format():
    seen = set()
    for seed in range(300):
        rec = extract_gen.build_one(70_000 + seed)
        first_user, first_ans = rec["messages"][0]["content"], rec["messages"][1]["content"]
        if "as a json" in first_user.lower():
            rows = json.loads(first_ans)
            assert rows and all(set(x) == set(rows[0]) for x in rows)
            seen.add("json")
        elif "as a csv" in first_user.lower():
            rows = list(csv.DictReader(io.StringIO(first_ans)))
            assert rows and len(rows[0]) >= 3
            seen.add("csv")
        elif "markdown table" in first_user.lower():
            assert first_ans.splitlines()[1].startswith("| ---")
            seen.add("md")
        else:
            assert all(line.startswith("- ") for line in first_ans.splitlines())
            seen.add("bullets")
    assert seen == {"json", "csv", "md", "bullets"}


def test_multi_turn_reasoning_keeps_state_and_ends_every_turn_with_a_final_answer():
    import random
    for name, fn in reasoning_multi.MULTI.items():
        for k in range(40):
            msgs = fn(random.Random(f"{name}{k}"))
            assert len(msgs) >= 4 and msgs[0][0] == "user"
            for role, text in msgs:
                if role == "assistant":
                    assert text.rstrip().splitlines()[-1].startswith("Final answer:") and "Step 1:" in text


def test_dependent_chain_pipes_the_first_result_into_the_second_call():
    for seed in range(120):
        rec = fc_gen.build_one("dep_chain", 80_000 + seed)
        assert fc_gen.check(rec) is None
        calls = [c for m in rec["messages"] if m.get("tool_calls") for c in m["tool_calls"]]
        assert len(calls) == 2
        first_result = next(m["content"] for m in rec["messages"] if m["role"] == "tool")
        second_args = json.dumps(calls[1]["function"]["arguments"])
        val = json.loads(first_result)
        if isinstance(val, dict):
            val = val["temperature"]
        assert str(val) in second_args, (first_result, second_args)


def test_code_tasks_pass_their_hidden_tests_and_a_bug_is_caught():
    entries = [code_gen.make_entry(n, 7) for n in list(code_gen.T)[:6]]
    rows, drops = teacher_tasks.build(entries, verbose=False)
    assert len(rows) == 6 and not drops
    bad = code_gen.make_entry("sum_of_squares", 3)
    bad["steps"][-2]["arg"] = bad["steps"][-2]["arg"].replace("x * x", "x + x")
    rows, drops = teacher_tasks.build([bad], verbose=False)
    assert rows == [] and sum(drops.values()) == 1


def test_system_rule_examples_keep_the_rules_out_of_the_user_turn():
    rows, stats = if_struct_gen.build_system(60, 5)
    assert len(rows) == 60
    for r in rows:
        assert r["messages"][0]["role"] == "system" and "- " in r["messages"][0]["content"]
        assert r["messages"][1]["role"] == "user" and "Requirements" not in r["messages"][1]["content"]


def test_template_caps_remove_near_duplicates_but_keep_distinct_examples():
    import random
    from etl.sources.local_jsonl import dedupe_templates, sentences_of, shape_of

    def row(user, asst):
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]}
    same = [("x", i, row(f"What is the capital of Country{i}?", f"I'm not familiar with Place{i} and I can't verify that it exists.")) for i in range(30)]
    other = [("x", 100 + i, row(f"Explain topic number {i} with words {'abc' * i}.", f"A different answer number {i} that talks about {'xyz ' * (i % 5)}and more.")) for i in range(5)]
    assert shape_of(same[0][2]) == shape_of(same[1][2])                       # names and numbers are masked, so these are one template
    kept, dropped = dedupe_templates(list(same + other), random.Random(0), 4, None)
    assert sum(1 for k in kept if k[1] < 100) == 4 and dropped == 26 and sum(1 for k in kept if k[1] >= 100) >= 4
    filler = [("x", i, row(f"Question {'q' * i}?", f"Here is a unique sentence number {'z' * i} about topic. Remember: keep it simple and stay calm today.")) for i in range(20)]
    kept, dropped = dedupe_templates(list(filler), random.Random(0), None, 5)
    assert len(kept) == 5 and dropped == 15                                   # the shared filler sentence may appear in at most 5 examples
    assert any("keep it simple" in s for s in sentences_of(filler[0][2]))


def test_diversity_report_runs_on_a_tiny_file(tmp_path):
    from distill import diversity_report
    p = tmp_path / "t.jsonl"
    rows = [{"source": "a/x", "messages": [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": f"This answer is about item number {i} and nothing else."}]} for i in range(10)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    rep = diversity_report.report(str(p))
    assert rep["a"]["examples"] == 10 and rep["a"]["distinct_sentence_ratio"] == 1.0 and rep["a"]["shape_ratio"] <= 1.0


def test_per_prompt_cap_and_rewrites_stay_with_their_original_in_the_split():
    import random
    from etl.sources.local_jsonl import dedupe_templates
    from etl.transforms import split_train_val

    def row(user, a):
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": a}]}
    rows = [("x", i, row("What time is it right now?", f"I can't see a clock, variant number {i} of the same reply.")) for i in range(9)] + [("x", 99, row("Another prompt entirely?", "A different answer."))]
    kept, dropped = dedupe_templates(rows, random.Random(0), None, None, 2)
    assert sum(1 for k in kept if k[1] < 99) == 2 and dropped == 7 and any(k[1] == 99 for k in kept)

    recs = []
    for i in range(400):
        base = {"id": f"agent/fam/{i}", "source": "a", "messages": [{"role": "user", "content": f"task {i}"}, {"role": "assistant", "content": "ok"}], "tools": None}
        rw = dict(base, id=f"agent/fam/{i}/teacher", messages=[{"role": "user", "content": f"please do task number {i} for me"}, {"role": "assistant", "content": "ok"}])
        recs += [base, rw]
    train, val = split_train_val(recs, 0.1, 42)
    val_ids, train_ids = {r["id"] for r in val}, {r["id"] for r in train}
    assert val_ids and all((i.replace("/teacher", "") in val_ids) == (i in val_ids) for i in val_ids | train_ids if i.endswith("/teacher") or i + "/teacher" in val_ids | train_ids)
    assert not any(i + "/teacher" in train_ids for i in val_ids if not i.endswith("/teacher"))
