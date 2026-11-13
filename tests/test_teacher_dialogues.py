import glob
import json

from distill import teacher_dialogues as td


def _build(tmp_path, text):
    p = tmp_path / "d.txt"
    p.write_text(text)
    ds = td.parse(str(p))
    assert len(ds) == 1
    return td.build_one(ds[0])


def test_all_shipped_dialogues_verify():
    ds = [d for p in sorted(glob.glob(td.FILES)) for d in td.parse(p)]
    assert len(ds) >= 120 and len({d["id"] for d in ds}) == len(ds)
    cats = set()
    for d in ds:
        rec, errs = td.build_one(d)
        assert not errs, (d["id"], errs)
        assert rec["meta"]["n_turns"] >= 2 and rec["messages"][-1]["role"] == "assistant"
        cats.add(rec["meta"]["category"])
        for m in rec["messages"]:
            if m["role"] == "tool":
                json.loads(m["content"])
    assert len(cats) >= 8


def test_good_dialogue_and_meta(tmp_path):
    rec, errs = _build(tmp_path, "### mt/x1 | skills=cot_whatif\n@user\n2+2?\n@assistant\nStep 1: 2 + 2 = 4.\n\nFinal answer: 4\n@check has=4 ; max_words n=20\n"
                                 "@user\nAnd times 3?\n@assistant\n4 x 3 = 12.\n@check has=12\n")
    assert not errs
    assert rec["meta"]["family"] == "mt_cot_whatif" and rec["meta"]["category"] == "reasoning" and rec["meta"]["n_turns"] == 2
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "user", "assistant"]


def test_failed_check_single_turn_and_bad_code_are_rejected(tmp_path):
    _, errs = _build(tmp_path, "### mt/x2 | skills=if_persist\n@user\nhi\n@assistant\nHello there friend\n@check max_words n=2\n@user\nbye\n@assistant\nBye\n")
    assert errs
    _, errs = _build(tmp_path, "### mt/x3 | skills=advice\n@user\nhi\n@assistant\nHello\n")
    assert any("multi-turn" in e for e in errs)
    code = "### mt/x4 | skills=code\n@user\nsquare\n@assistant\n```python\ndef f(x):\n    return x + x\n```\n@pyassert f(3) == 9\n@user\nok\n@assistant\nDone.\n"
    _, errs = _build(tmp_path, code)
    assert any("pyassert" in e for e in errs)
    _, errs = _build(tmp_path, code.replace("x + x", "x * x"))
    assert not errs


def test_pure_tool_result_must_match_real_function(tmp_path):
    name = next(n for n in td.fc_gen.FUNCS if n == "lists.sort_numbers")
    good = td.fc_gen.FUNCS[name]["run"]({"numbers": [3, 1, 2]})
    tmpl = ("### mt/x5 | skills=tools_session | tools=lists.sort_numbers\n@user\nSort 3, 1, 2.\n@call lists.sort_numbers {\"numbers\": [3, 1, 2]}\n"
            "@result %s\n@assistant\nSorted: 1, 2, 3.\n@user\nThanks, which is largest?\n@assistant\nThe largest is 3.\n")
    rec, errs = _build(tmp_path, tmpl % json.dumps(good))
    assert not errs and rec["meta"]["n_tool_calls"] == 1 and rec["meta"]["kind"] == "api"
    wrong = json.loads(json.dumps(good).replace("1, 2, 3", "3, 2, 1"))
    assert wrong != good
    _, errs = _build(tmp_path, tmpl % json.dumps(wrong))
    assert errs
