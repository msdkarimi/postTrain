import json

from evaluation import plot
from evaluation.bench import PRESETS, extract_scores


def fake_lm_eval():
    return {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.4, "prompt_level_strict_acc_stderr,none": 0.02, "inst_level_strict_acc,none": 0.5, "inst_level_strict_acc_stderr,none": "N/A"},
                        "gsm8k": {"exact_match,flexible-extract": 0.3, "exact_match_stderr,flexible-extract": 0.03, "exact_match,strict-match": 0.2},
                        "mmlu": {"acc,none": 0.35, "acc_stderr,none": 0.004}},
            "n-samples": {"ifeval": {"effective": 541}, "gsm8k": {"effective": 1319}, "mmlu_a": {"effective": 100}, "mmlu_b": {"effective": 50}}}


def test_extract_scores():
    s = extract_scores(fake_lm_eval())
    assert s["ifeval_prompt_strict"] == {"value": 0.4, "stderr": 0.02, "n": 541}
    assert s["ifeval_inst_strict"]["stderr"] is None
    assert s["gsm8k_flexible"]["value"] == 0.3 and s["gsm8k_strict"]["stderr"] is None
    assert s["mmlu"]["n"] == 150


def test_presets():
    assert PRESETS["full"]["ifeval"] is None and PRESETS["quick"]["mmlu"] == 10


def _eval(rate):
    blk = lambda n=8: {"n": n, "pass": round(rate * n), "pass_rate": rate}
    return {"overall": {**blk(40), "format_error_rate": 0.01, "tool_error_rate": 0.02}, "by_category": {"csv": blk(), "api": blk()},
            "by_skill": {"tool_use": blk(), "math": blk()}, "by_family": {f"fam{i}": {"n": 8, "pass": i, "pass_rate": i / 8 * rate} for i in range(8)}}


def _bench(v):
    return {"scores": {k: {"value": v, "stderr": 0.02, "n": 100} for k, _ in plot.BENCH_LABELS}, "settings": {}, "preset": "quick"}


def test_plot_end_to_end(tmp_path, monkeypatch):
    for name, rate in (("base", 0.2), ("sft", 0.7)):
        (tmp_path / "eval" / name).mkdir(parents=True)
        (tmp_path / "eval" / name / "summary.json").write_text(json.dumps(_eval(rate)))
        (tmp_path / "bench" / name).mkdir(parents=True)
        (tmp_path / "bench" / name / "bench.json").write_text(json.dumps(_bench(rate)))
    monkeypatch.setattr("sys.argv", ["plot", "--eval-root", str(tmp_path / "eval"), "--bench-root", str(tmp_path / "bench"), "--out", str(tmp_path / "plots")])
    plot.main()
    for f in ("heldout_by_category", "heldout_skills", "heldout_family_delta", "benchmarks", "forgetting_delta"):
        assert (tmp_path / "plots" / f"{f}.png").stat().st_size > 5000
    rep = (tmp_path / "plots" / "report.md").read_text()
    assert "| pass rate |" in rep and "sft vs base" in rep


def test_model_colors_are_stable():
    allm = plot.order_models(["sft", "base", "qwen-instruct"])
    assert allm[:2] == ["base", "qwen-instruct"]
    assert plot.colors_for(allm, ["sft"])["sft"] == plot.colors_for(allm, allm)["sft"]


# ---- BFCL adapter (skipped where bfcl-eval is not installed) -------------------------------------------------------------------------
import pytest  # noqa: E402

bfcl_mod = pytest.importorskip("bfcl_eval")
from evaluation import bfcl  # noqa: E402


def _oracle_text(item):
    calls = []
    for gt in item["truth"]:
        for name, params in gt.items():
            args = "".join(f"<parameter={k}>\n{json.dumps(v[0]) if not isinstance(v[0], str) else v[0]}\n</parameter>" for k, v in params.items() if v and v[0] != "")
            calls.append(f"<tool_call>\n<function={name}>\n{args}\n</function>\n</tool_call>")
    return "\n".join(calls) + "<|im_end|>"


def test_bfcl_oracle_scores_high_and_silence_scores_only_irrelevance():
    cats = ["simple_python", "parallel", "irrelevance"]
    items = {c: bfcl.load_category(c, 25) for c in cats}
    state = {"i": 0}

    def gen_oracle(prompts):
        c = cats[state["i"]]; state["i"] += 1
        return ["" if c == "irrelevance" else _oracle_text(it) for it in items[c]]

    res = bfcl.evaluate(gen_oracle, categories=cats, limit=25)["scores"]
    assert res["bfcl_simple_python"]["value"] >= 0.9 and res["bfcl_parallel"]["value"] >= 0.9 and res["bfcl_irrelevance"]["value"] == 1.0

    res0 = bfcl.evaluate(lambda ps: ["" for _ in ps], categories=cats, limit=25)["scores"]
    assert res0["bfcl_simple_python"]["value"] == 0 and res0["bfcl_parallel"]["value"] == 0 and res0["bfcl_irrelevance"]["value"] == 1.0
    assert abs(res0["bfcl_ast"]["value"] - 1 / 3) < 1e-9


def test_bfcl_tools_are_openai_style():
    it = bfcl.load_category("simple_python", 1)[0]
    t = bfcl.to_tools(it["functions"])[0]
    assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object"
