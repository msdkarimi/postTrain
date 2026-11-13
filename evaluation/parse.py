"""Parse a model's raw output (Qwen3.5 chat template format) into plain text + structured tool calls.

The template renders a tool call as
    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>
String arguments are written raw, other types (numbers, booleans, arrays, objects) as JSON, so the parser uses each tool's
JSON schema to decide how to read a value. Anything that does not parse is reported as MALFORMED (a real model error).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass
class Parsed:
    content: str
    calls: list = field(default_factory=list)       # [{"name": str, "arguments": dict}]
    malformed: int = 0                                # <tool_call> openers that could not be parsed into a complete call


def cut_at_stop(text: str) -> str:
    for s in STOP_TOKENS:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text


def _convert(value: str, ptype):
    if ptype == "string" or ptype is None:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value                                   # wrong type -> the API will reject it, like it should


def parse_assistant_output(text: str, tools: list) -> Parsed:
    text = _THINK_RE.sub("", cut_at_stop(text), count=1)
    schemas = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    calls = []
    for name, body in _CALL_RE.findall(text):
        name = name.strip()
        props = schemas.get(name, {})
        args = {k.strip(): _convert(v, props.get(k.strip(), {}).get("type")) for k, v in _PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    first = text.find("<tool_call>")
    content = (text if first == -1 else text[:first]).strip()
    return Parsed(content=content, calls=calls, malformed=max(0, text.count("<tool_call>") - len(calls)))
