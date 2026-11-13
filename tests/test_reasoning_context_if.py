import json
import random
import re

from distill import context_gen, ifgen, reasoning_gen, teacher_sample


def test_reasoning_families_are_clean_at_scale():
    assert len(reasoning_gen.FAMILIES) >= 20
    for name in reasoning_gen.FAMILIES:
        for k in range(150):
            rec = reasoning_gen.build_one(name, 10_000 + k)
            assert reasoning_gen.check_record(rec) is None, (name, k)
            text = rec["messages"][-1]["content"]
            assert text.index("Step 1:") < text.index("Final answer:")          # reasoning first, answer last
            assert rec["messages"][-2]["role"] == "user" and rec["tools"] == []


def test_reasoning_arithmetic_is_actually_right():
    """Independent re-computation for a few families (the generator is correct by construction; this guards the text)."""
    for k in range(60):
        rec = reasoning_gen.build_one("work_together", 10_000 + k)
        m = re.search(r"(\d+) hours to .* (\d+) hours|in (\d+) hours .* in (\d+) hours", rec["messages"][-2]["content"])
        a, b = [int(x) for x in m.groups() if x]
        assert abs(float(rec["meta"]["answer"]) - a * b / (a + b)) < 0.01


def test_context_families_quote_the_document_and_abstain():
    kinds = {"answerable": 0, "not_found": 0}
    for name in context_gen.FAMILIES:
        for k in range(200):
            rec = context_gen.build_one(name, 10_000 + k)
            assert context_gen.check_record(rec) is None, (name, k)
            kinds["not_found" if rec["meta"]["answer"] == "not_found" else "answerable"] += 1
    assert 0.15 < kinds["not_found"] / sum(kinds.values()) < 0.35


def test_if_verifiers_accept_and_reject():
    ok = lambda name, resp, **p: ifgen.CONSTRAINTS[name][1](resp, p)
    assert ok("max_words", "one two three", n=5) and not ok("max_words", "a b c d e f", n=5)
    assert ok("n_bullets", "- a\n- b\n- c", n=3) and not ok("n_bullets", "- a\n- b", n=3)
    assert ok("no_commas", "No commas here.") and not ok("no_commas", "a, b")
    assert ok("lowercase", "all lower") and not ok("lowercase", "Not lower")
    assert ok("title", "<<My Title>>\nbody") and not ok("title", "My Title")
    assert ok("json", '```json\n{"title": "t", "content": "c"}\n```') and not ok("json", '{"title": "t"}')
    assert ok("closer", "Hello. Thanks for reading.", c="Thanks for reading.") and not ok("closer", "Thanks for reading. Bye", c="Thanks for reading.")
    assert ok("sections", "SECTION 1\nx\nSECTION 2\ny", n=2) and not ok("sections", "SECTION 1\nx", n=2)
    assert ok("keywords", "Balance and trust.", ks=["balance", "trust"]) and not ok("keywords", "Balance only.", ks=["balance", "trust"])
    assert ok("banned", "A quiet day.", k="very") and not ok("banned", "A very quiet day.", k="very")


def test_if_prompts_never_combine_conflicting_constraints():
    r = random.Random(3)
    for _ in range(2000):
        row = ifgen.build_prompt(r)
        names = [c["name"] for c in row["constraints"]]
        for a in names:
            for b in names:
                assert a == b or b not in ifgen.CONSTRAINTS[a][2], (a, b)
        assert 1 <= len(names) <= 3


def test_teacher_sample_if_keeps_only_verified_answers():
    rows = [{"id": "if/0", "prompt": "Write about tea. Answer in at most 5 words.", "constraints": [{"name": "max_words", "params": {"n": 5}}], "form": "x", "topic": "t"},
            {"id": "if/1", "prompt": "Write about tea without commas.", "constraints": [{"name": "no_commas", "params": {}}], "form": "x", "topic": "t"}]
    calls = {"n": 0}

    def fake(chats):
        calls["n"] += 1
        return ["Tea is warm and nice today friend" if "at most 5" in c[-1]["content"] else "Tea, sweet" if calls["n"] == 1 else "Tea is sweet" for c in chats]
    kept, stats = teacher_sample.run_if(rows, fake, k=3)
    assert [r["id"] for r in kept] == ["ifverified/if/1"]                        # prompt 0's answer has 7 words every time -> never kept
    assert kept[0]["messages"][-1]["content"] == "Tea is sweet" and stats["failed_by"]["max_words"] == 3


def test_teacher_sample_paraphrase_requires_every_fact():
    recs = {"agent/f/1": {"messages": [{"role": "user", "content": "Move Planning on 2025-06-11 to 13:00 and email Bo."}], "meta": {"family": "f"}}}
    shots = [("Sum sales.csv.", "How much does sales.csv add up to?")]
    good = lambda chats: ["could you shift the Planning meeting on 2025-06-11 to 13:00 and drop Bo an email?"]
    bad = lambda chats: ["could you move the meeting to the afternoon and email Bo?"]
    lines, _ = teacher_sample.run_paraphrase(recs, shots, good, n_per_family=5, k=1)
    assert len(lines) == 1 and lines[0].startswith("agent/f/1 ||| ")
    lines, _ = teacher_sample.run_paraphrase(recs, shots, bad, n_per_family=5, k=2)
    assert lines == []


def test_teacher_loader_choice_and_thinking_switch(tmp_path):
    import json as _json
    from evaluation.generate import is_conditional_generation
    (tmp_path / "vlm").mkdir()
    (tmp_path / "vlm" / "config.json").write_text(_json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    (tmp_path / "lm").mkdir()
    (tmp_path / "lm" / "config.json").write_text(_json.dumps({"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"]}))
    assert is_conditional_generation(str(tmp_path / "vlm")) and not is_conditional_generation(str(tmp_path / "lm"))
    assert not is_conditional_generation("/does/not/exist")

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking=None):
            return f"thinking={enable_thinking}"
    p = teacher_sample.ChatPrompt([{"role": "user", "content": "x"}])
    assert p.prompt_text(Tok()) == "thinking=False"
    teacher_sample.ChatPrompt.thinking = True
    try:
        assert p.prompt_text(Tok()) == "thinking=True"
    finally:
        teacher_sample.ChatPrompt.thinking = False


def test_teacher_written_if_answers_are_verified(tmp_path):
    from distill import teacher_if
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "if/1", "prompt": "Write about tea. At most 5 words.", "constraints": [{"name": "max_words", "params": {"n": 5}}], "form": "x", "topic": "t"}) + "\n"
                       + json.dumps({"id": "if/2", "prompt": "Write about tea. No commas.", "constraints": [{"name": "no_commas", "params": {}}], "form": "x", "topic": "t"}) + "\n")
    ans = tmp_path / "a.txt"
    ans.write_text("### if/1\nTea is warm and lovely today friend\n### if/2\nTea is warm\nand sweet\n")
    got = teacher_if.load_answers(str(ans))
    assert got == {"if/1": "Tea is warm and lovely today friend", "if/2": "Tea is warm\nand sweet"}
    import argparse
    teacher_if.ANSWERS = str(ans)
    teacher_if.load_answers.__defaults__ = (str(ans),)
    out = tmp_path / "o.jsonl"
    teacher_if.cmd_build(argparse.Namespace(prompts=str(prompts), out=str(out)))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["ifverified/if/2"]                    # the 7-word answer fails max_words and is not kept
