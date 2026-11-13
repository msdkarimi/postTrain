"""Tests for the evaluation harness: parser, oracle (must be perfect), negative controls (must fail), batching, CLI smoke test."""
import json
import subprocess
import sys

import pytest
import torch

import envs.tasks  # noqa: F401
from envs.tasks.base import FAMILIES
from evaluation.agent import AgentRun
from evaluation.parse import cut_at_stop, parse_assistant_output
from evaluation.run_eval import summarize

TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {
    "s": {"type": "string"}, "n": {"type": "integer"}, "b": {"type": "boolean"}, "a": {"type": "array", "items": {"type": "string"}}}}}}]


# ---- parser -----------------------------------------------------------------------------------------------------------
def test_parser_reads_text_calls_and_types():
    text = ("Let me check.\n\n<tool_call>\n<function=t>\n<parameter=s>\n007\n</parameter>\n<parameter=n>\n30\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n"
            "<parameter=a>\n[\"x\", \"y\"]\n</parameter>\n</function>\n</tool_call><|im_end|>")
    p = parse_assistant_output(text, TOOLS)
    assert p.content == "Let me check." and p.malformed == 0
    assert p.calls == [{"name": "t", "arguments": {"s": "007", "n": 30, "b": True, "a": ["x", "y"]}}]       # strings stay strings


def test_parser_parallel_calls_multiline_strings_and_think_prefix():
    one = "<tool_call>\n<function=t>\n<parameter=s>\nline1\nline2\n</parameter>\n</function>\n</tool_call>"
    p = parse_assistant_output("<think>\n\n</think>\n\n" + one + "\n" + one, TOOLS)
    assert len(p.calls) == 2 and p.calls[0]["arguments"]["s"] == "line1\nline2" and p.content == ""


def test_parser_flags_malformed_and_plain_answers():
    cut = "<tool_call>\n<function=t>\n<parameter=s>\nabc\n</parameter>\n"            # truncated before </function>
    p = parse_assistant_output(cut, TOOLS)
    assert p.calls == [] and p.malformed == 1
    plain = parse_assistant_output("The answer is 4.<|im_end|>", TOOLS)
    assert plain.calls == [] and plain.content == "The answer is 4." and plain.malformed == 0
    assert cut_at_stop("hello<|im_end|>junk") == "hello"


def test_wrongly_typed_value_is_kept_so_the_api_can_reject_it():
    p = parse_assistant_output("<tool_call>\n<function=t>\n<parameter=n>\nthirty\n</parameter>\n</function>\n</tool_call>", TOOLS)
    assert p.calls[0]["arguments"]["n"] == "thirty"


# ---- harness vs. oracle and negative controls --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def drive(runs, gen):
    active = list(runs)
    while active:
        for r, t in zip(active, gen(active)):
            r.feed(t)
        active = [r for r in active if not r.done]
    return [r.result for r in runs]


def test_oracle_scores_100_percent_on_every_family(tok):
    from evaluation.generate import OracleGenerator
    runs = [AgentRun(f, s) for f in sorted(FAMILIES) for s in (0, 1)]
    results = drive(runs, OracleGenerator(tok))
    bad = [(r["family"], r["seed"], r["reason"]) for r in results if not r["ok"]]
    assert not bad, bad
    assert sum(r["format_errors"] for r in results) == 0 and len(results) == 2 * len(FAMILIES)


def test_lazy_model_that_never_uses_tools_fails(tok):
    runs = [AgentRun(f, 0) for f in sorted(FAMILIES)]
    results = drive(runs, lambda active: ["Sure, all done!"] * len(active))
    rate = sum(r["ok"] for r in results) / len(results)
    assert rate <= 0.05, f"a model that only says 'done' passed {rate:.0%}: the checkers are too lenient"


def test_truncated_tool_call_is_counted_as_format_error_and_fails(tok):
    from evaluation.generate import OracleGenerator
    oracle = OracleGenerator(tok)

    def cut(active):
        return [t[: max(10, len(t) - 25)] if "<tool_call>" in t else t for t in oracle(active)]

    results = drive([AgentRun("api_find_and_schedule", s) for s in range(4)], cut)
    assert sum(r["format_errors"] for r in results) >= 3 and not any(r["ok"] for r in results)


def test_step_cap_stops_a_model_that_loops_forever(tok):
    looping = "<tool_call>\n<function=get_current_date>\n</function>\n</tool_call>"
    results = drive([AgentRun("api_today_agenda", 0, max_steps=3)], lambda active: [looping] * len(active))
    assert results[0]["reason"] == "step_cap" and results[0]["n_calls"] == 3


def test_summary_groups_by_family_category_and_skill(tok):
    from evaluation.generate import OracleGenerator
    results = drive([AgentRun(f, 0) for f in ("api_draft_or_send", "api_no_tool_needed", "api_task_review", "api_inbox_digest", "api_cancel_event", "api_table_aggregate")], OracleGenerator(tok))
    s = summarize(results)
    assert s["overall"]["n"] == 6 and s["overall"]["pass_rate"] == 1.0 and s["by_kind"]["api"]["n"] == 6 and "api_behaviour" in s["by_category"]


# ---- batching (left padding) on the real architecture ----------------------------------------------------------------------------
def test_left_padding_does_not_change_next_token_logits(tmp_path, tok):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tests.tiny_model import build_tiny_model
    path = build_tiny_model(str(tmp_path / "tiny"))
    t = AutoTokenizer.from_pretrained(path)
    t.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    texts = ["short prompt", "a much longer prompt with quite a few more tokens in it so the first one needs padding", "medium sized prompt here"]

    @torch.no_grad()
    def logits(batch):
        enc = t(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        return model.generate(**enc, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores[0]

    together = logits(texts)
    for i, text in enumerate(texts):
        assert torch.allclose(together[i], logits([text])[0], atol=2e-3), f"left padding changed the logits of prompt {i}"


# ---- the CLI end to end with a (random) model ---------------------------------------------------------------------------------------
def test_cli_runs_end_to_end_with_a_tiny_random_model(tmp_path, tok):
    from tests.tiny_model import build_tiny_model
    model = build_tiny_model(str(tmp_path / "tiny"))
    out = tmp_path / "eval"
    subprocess.run([sys.executable, "-m", "evaluation.run_eval", "--model", model, "--out", str(out), "--families", "api_today_agenda,api_no_tool_needed",
                    "--per-family", "2", "--max-new-tokens", "8", "--batch-size", "2"], check=True, capture_output=True)
    s = json.loads((out / "summary.json").read_text())
    assert s["overall"]["n"] == 4 and (out / "results.jsonl").exists() and (out / "failures.md").exists() and (out / "summary.md").exists()
