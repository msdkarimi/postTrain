"""Writes DATASET_CARD.md next to the generated data (numbers come from sandbox_stats.json)."""
from __future__ import annotations

from pathlib import Path

TECHNIQUES = """\
| # | Technique | What it does here |
|---|---|---|
| 1 | Verifiable task synthesis | {n_fam} task families generated procedurally from seeds; every task has an independent checker |
| 2 | Two environments | a real sandbox (`run_python` / `run_bash`) and a simulated workspace API with 22 typed tools (calendar, contacts, email, tasks, files, calculator, table aggregation) |
| 3 | Execution-grounded trajectories | tool results and API errors are real outputs of the sandbox / API, never invented |
| 4 | Rejection sampling | a trajectory is kept only if the checker confirms the final state (files, calendar events, drafts, tasks) and the reply |
| 5 | Short ReAct-style reasoning | one or two sentences of plan before each action, reacting to what the tool returned (kept under 450 characters) |
| 6 | Look-then-act chains | search -> read, list -> read, find slot -> create, read -> compute -> write; later arguments are built from earlier tool results |
| 7 | Parallel tool calls | {parallel} assistant messages issue several independent calls at once |
| 8 | Tool selection among distractors | 5-10 tools are offered per task, only some are needed (2-3 for most), in random order |
| 9 | Typed arguments | enums, arrays, ISO dates, HH:MM times, optional parameters; the API validates and rejects bad arguments |
| 10 | API error recovery | {errors} real error responses (invalid arguments, not found) followed by a corrected call |
| 11 | Multi-turn | {multi} conversations with follow-ups that change earlier results, plus clarification questions when a request is ambiguous |
| 12 | Tool restraint | {no_tool} trajectories use no tool: direct answers, unsupported requests (honest limitation + alternatives) |
| 13 | Safety and precision | draft vs send (never send unless asked), confirm before deleting, only touch what was asked |
| 14 | Sandbox error recovery | ~{rec_pct}% of sandbox tasks start with a plausible mistake that really fails (traceback), then a fix quoting the error |
| 15 | Strategy and data diversity | Python vs bash / awk / grep; header names, delimiters, date formats, currency strings, blanks, duplicates, distractor files, decoy lines |
| 16 | Prompt expansion | composable phrasings, intros / outros, casual typing variation, 4 + 4 system prompts |
| 17 | Abstention | tasks asking for data that does not exist; the right behaviour is to look, then say so |
| 18 | Grounding + answer-leak filters | every value in the final reply must appear in a tool output; plans may not state a result before it was observed |
| 19 | Dedup + decontamination | a task is a duplicate only if prompt AND data are identical; train seeds (>= 10000) never overlap held-out test seeds (0..N-1); any train task identical to a test task is dropped |
| 20 | Stratified sampling hooks | `max_per_family` / `max_examples` in the SFT mix keep every family represented when you can only train on a subset |
| 21 | Replay mixing (in the SFT config) | combined with Hermes function-calling data and a little SmolTalk chat |
"""


def write_card(path: Path, stats: dict) -> None:
    tags = stats["tags"]
    kept = max(stats["kept"], 1)
    rec_pct = round(100 * tags.get("error_recovery", 0) / kept)
    abstain = tags.get("abstention", 0)
    fam_rows = "\n".join(f"| {n} | {v['attempted']} | {v['kept']} |" for n, v in stats["per_family"].items())
    drops = "\n".join(f"| {k} | {v} |" for k, v in sorted(stats["drop_reasons"].items(), key=lambda kv: -kv[1])) or "| (none) | 0 |"
    text = f"""# Agent trajectories v2: sandbox + function calling (scripted-teacher distillation)

Verified tool-use trajectories for a workspace assistant, in two environments: a sandbox with `run_python` / `run_bash`,
and a simulated workspace API with typed function-calling tools. Built for supervised fine-tuning of Qwen3.5-0.8B / 2B
and as the task suite for later RL (the checkers are the reward).

## How it was made (read this)
The "teacher" is a set of solution programs **written by hand**, interactively by an LLM (in this repository), several strategies per
task family, with short ReAct-style plans. It is *not* a language model sampled once per example. Every step is
executed for real, and a trajectory is kept only if an independent checker verifies it. Consequence: observations
and verification are real, but code style and reasoning wording are more templated than a live teacher's would be.
A live-teacher run can reuse the same sandbox, tasks and checkers (`envs/`, `distill/`).

## Size
* task attempts: {stats['attempted']}, kept: **{stats['kept']}** ({100 * stats['keep_rate']:.1f}%)
* held-out test tasks: {stats['test_tasks']} (`agent_test_tasks.jsonl`, solutions not included)
* distinct prompt texts: {stats['unique_prompts']} (several families use a generic instruction and vary the data they act on)
* kind: {stats['kind']}; multi-turn conversations: {stats['turns']['multi_turn']}; parallel-call messages: {stats['parallel_call_messages']}; API error responses: {stats['api_error_responses']}
* tool calls per trajectory: mean {stats['tool_calls']['mean']}, max {stats['tool_calls']['max']}; by tool: {stats['tool_calls']['by_tool']}
* difficulty: {stats['difficulty']}
* category: {stats['category']}

## Techniques used
{TECHNIQUES.format(n_fam=len(stats['per_family']), rec_pct=rec_pct, parallel=stats['parallel_call_messages'], errors=stats['api_error_responses'], multi=stats['turns']['multi_turn'], no_tool=stats['no_tool_trajectories'])}
## Per-family counts
| family | attempted | kept |
|---|---|---|
{fam_rows}

## Rejected / dropped
| reason | count |
|---|---|
{drops}

## Known limitations
* Templated reasoning and code (see above); no live-LLM sampling, no LLM paraphrasing of prompts.
* Mostly short conversations (at most two user turns); non-thinking mode (short visible plans, no `<think>` traces).
* The workspace API is a simulation: it models the shape of calendar / email / task tools, not any real product.
* Observations were produced on macOS with a restricted, portable subset of shell tools (no `wc`, GNU-only flags) so they
  match Linux output; the sandbox is not isolated from the network (fine for code we wrote; use Docker for model code).
* Synthetic data: names, companies and numbers are random.

## Files and reproduction
`python -m distill.build_trajectories --n-train 40000 --out data/raw` regenerates everything deterministically.
Record format: canonical conversations (`etl/schema.py`): `messages` (system, user, assistant with structured `tool_calls`,
`tool`), `tools`, and `meta` (family, category, difficulty, tags, seed).
"""
    path.write_text(text)
