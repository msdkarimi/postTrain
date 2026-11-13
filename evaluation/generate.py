"""Generators: given active AgentRuns, return the next raw assistant text for each (greedy decoding).

HFGenerator runs a Hugging Face causal LM (a saved model directory or a Hub id), optionally with a LoRA adapter, in batches with
left padding. Batching is only a speed-up; tests/test_eval.py checks that padded and unpadded logits agree on this architecture,
and --batch-size 1 is always available as the reference.
OracleGenerator replays the scripted teacher's answers (a perfect model) and exists to validate the harness itself.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import envs.tasks  # noqa: F401
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, run_any
from envs.tasks.base import make_task
from evaluation.parse import cut_at_stop


def is_conditional_generation(model_path: str) -> bool:
    """Vision-language checkpoints (e.g. Qwen3.8-27B: Qwen3_5ForConditionalGeneration) must be loaded with the image-text class so every weight
    lands where it belongs; text-only checkpoints (our 0.8B student) keep the plain causal-LM class. Text-only prompts work with both."""
    try:
        return any(a.endswith("ForConditionalGeneration") for a in (AutoConfig.from_pretrained(model_path).architectures or []))
    except Exception:
        return False


class HFGenerator:
    def __init__(self, model_path: str, adapter: str | None = None, dtype: str = "bfloat16", batch_size: int = 4,
                 max_new_tokens: int = 512, max_prompt_tokens: int = 8192, temperature: float = 0.0, top_p: float = 1.0):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype if self.device == "cuda" else "float32"]
        if is_conditional_generation(model_path):
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dt, device_map={"": self.device} if self.device == "cuda" else None).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt).to(self.device).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]
        self.batch_size, self.max_new, self.max_prompt = batch_size, max_new_tokens, max_prompt_tokens
        self.temperature, self.top_p = temperature, top_p            # 0 = greedy (evaluation); > 0 = sampling (teacher data generation)

    @torch.no_grad()
    def __call__(self, runs: list) -> list:
        texts = [r.prompt_text(self.tok) for r in runs]
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        out: list = [None] * len(runs)
        order = sorted((i for i in range(len(runs)) if lens[i] <= self.max_prompt), key=lambda i: lens[i])   # similar lengths per batch
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            if self.temperature > 0:
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=None,
                                          eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
            else:
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                          eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
            for j, i in enumerate(idx):
                out[i] = cut_at_stop(self.tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=False))
        return out


class OracleGenerator:
    """A perfect model: returns exactly what the scripted teacher would say next, rendered by the real chat template."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def _script(self, run):
        key = (run.family, run.seed)
        if key not in self.cache:
            res = run_any(make_task(*key), SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
            assert res["ok"], f"scripted solution failed for {key}"
            self.cache[key] = (res["messages"], res["tools"])
        return self.cache[key]

    def __call__(self, runs: list) -> list:
        outs = []
        for r in runs:
            msgs, tools = self._script(r)
            idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][r.n_assistant]
            prefix = self.tok.apply_chat_template(msgs[:idx], tools=tools, tokenize=False, add_generation_prompt=True)
            upto = self.tok.apply_chat_template(msgs[:idx + 1], tools=tools, tokenize=False)
            assert upto.startswith(prefix)
            outs.append(upto[len(prefix):])
        return outs
