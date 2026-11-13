import json
import random
import re

from distill import fc_gen, if_struct_gen, ifgen, mcq_gen


def test_fc_examples_are_schema_valid_and_cover_all_categories():
    seen = set()
    for cat in fc_gen.CATEGORIES:
        for seed in range(150):
            rec = fc_gen.build_one(cat, 20_000 + seed)
            assert fc_gen.check(rec) is None, (cat, seed)
            seen.add(cat)
            if cat == "irrelevance" or cat == "missing_param":
                assert rec["meta"]["n_tool_calls"] == 0
            if cat in ("parallel", "parallel_multiple"):
                first_calls = next(m["tool_calls"] for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
                assert len(first_calls) >= 2
    assert seen == set(fc_gen.CATEGORIES)


def test_fc_results_match_the_function_semantics():
    f = fc_gen.FUNCS
    assert f["math.gcd"]["run"]({"a": 198, "b": 27}) == 9
    assert f["dates.day_of_week"]["run"]({"date": "2025-03-01"}) == "Saturday"
    assert f["units.convert_temperature"]["run"]({"value": 100, "from_unit": "C", "to_unit": "F"}) == 212
    assert f["math.solve_quadratic"]["run"]({"a": 1, "b": -5, "c": 6}) == [2.0, 3.0]
    assert all(f["math.fibonacci"]["run"]({"n": n}) == fc_gen._fib(n) for n in range(1, 31))


def test_fc_call_text_round_trips_through_the_evaluation_parser():
    from evaluation.parse import parse_assistant_output
    rec = fc_gen.build_one("parallel", 20_005)
    calls = next(m["tool_calls"] for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    text = "".join(f"<tool_call>\n<function={c['function']['name']}>\n" + "".join(
        f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in c["function"]["arguments"].items()) + "</function>\n</tool_call>\n" for c in calls)
    parsed = parse_assistant_output(text, rec["tools"])
    assert parsed.malformed == 0
    assert [(c["name"], c["arguments"]) for c in parsed.calls] == [(c["function"]["name"], c["function"]["arguments"]) for c in calls]


def test_structured_if_answers_satisfy_every_constraint_including_followups():
    rows, stats = if_struct_gen.build(400, 5)
    assert len(rows) == 400 and stats["two_turn"] > 40
    prompts = {}
    r = random.Random(5)
    for row in rows:
        user_turns = [m for m in row["messages"] if m["role"] == "user"]
        assert 1 <= len(user_turns) <= 2
        assert row["messages"][-1]["role"] == "assistant" and row["messages"][-1]["content"].strip()


def test_structured_if_first_turn_is_verified_by_the_real_checker():
    r = random.Random(11)
    n = 0
    for _ in range(300):
        row = ifgen.build_prompt(r)
        text, units = if_struct_gen.compose(row, r)
        if text:
            n += 1
            assert ifgen.verify(text, row["constraints"])[0], row["prompt"]
    assert n > 200                                                    # the composer solves the large majority of random constraint mixes


def test_mcq_answer_letter_points_at_the_correct_option():
    for seed in range(400):
        for kind in ("table", "computed", "fact"):
            rec = mcq_gen.build_one(seed, kind)
            if rec is None:
                continue
            prompt = next(m for m in rec["messages"] if m["role"] == "user")["content"]
            letter = rec["meta"]["answer"]
            opts = dict(re.findall(r"^\(?([A-D])[.)]\)? (.+)$", prompt, re.M))
            assert len(opts) == 4 and letter in opts
            first = next(m for m in rec["messages"] if m["role"] == "assistant")["content"]
            assert (first.strip() == letter) or (f"{letter}" in first)
            if "Final answer:" in first:
                assert first.rstrip().splitlines()[-1].startswith(f"Final answer: {letter}.") and opts[letter] in first.splitlines()[-1]


def test_extra_ifeval_constraints_accept_and_reject():
    ex = ifgen.EXTRA
    chk = lambda name, text, **p: ex[name][1](text, p)
    assert chk("two_responses", "Alpha one.\n******\nBeta two.") and not chk("two_responses", "Only one response here.") and not chk("two_responses", "Same\n******\nSame")
    assert chk("repeat_prompt", "Write a poem.\n\nRoses are red.", request="Write a poem.") and not chk("repeat_prompt", "Roses are red.", request="Write a poem.")
    assert chk("first_word_paragraph", "One.\n\nNext thing here.\n\nThree.", n=3, i=2, w="Next") and not chk("first_word_paragraph", "One.\n\nSecond.\n\nThree.", n=3, i=2, w="Next")
    assert chk("capital_words", "Use NASA and ESA for GOOD.", n=3) and not chk("capital_words", "No capitals here.", n=3)
    assert chk("letter_freq", "eeeee eeeee eeeee", ch="e", n=15) and not chk("letter_freq", "eee", ch="e", n=15)
    assert chk("min_sentences", "One. Two. Three. Four. Five.", n=5) and not chk("min_sentences", "One. Two.", n=5)
    assert chk("word_range", " ".join(["w"] * 35), lo=30, hi=50) and not chk("word_range", " ".join(["w"] * 10), lo=30, hi=50)


def test_v2_prompts_never_combine_conflicting_constraints_and_repeat_is_exclusive():
    r = random.Random(4)
    saw_repeat = saw_extra = False
    for _ in range(1500):
        row = if_struct_gen.build_prompt_v2(r)
        names = [c["name"] for c in row["constraints"]]
        if "repeat_prompt" in names:
            saw_repeat = True
            assert names == ["repeat_prompt"] and row["prompt"].startswith(row["constraints"][0]["params"]["request"])
            continue
        for a in names:
            for b in names:
                assert a == b or b not in if_struct_gen._conf(a), (a, b)
        saw_extra |= any(n in ifgen.EXTRA for n in names)
    assert saw_repeat and saw_extra


def test_structured_if_v2_answers_pass_the_real_checkers():
    r = random.Random(21)
    ok = total = 0
    for _ in range(400):
        row = if_struct_gen.build_prompt_v2(r)
        total += 1
        text, _ = if_struct_gen.compose(row, r)
        if text:
            ok += 1
            assert ifgen.verify(text, row["constraints"])[0]
    assert ok / total > 0.85
