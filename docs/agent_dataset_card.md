# Agent trajectories v2: sandbox + function calling (scripted-teacher distillation)

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
* task attempts: 38997, kept: **38948** (99.9%)
* held-out test tasks: 504 (`agent_test_tasks.jsonl`, solutions not included)
* distinct prompt texts: 30988 (several families use a generic instruction and vary the data they act on)
* kind: {'sandbox': 20850, 'api': 18098}; multi-turn conversations: 4432; parallel-call messages: 10088; API error responses: 928
* tool calls per trajectory: mean 2.73, max 10; by tool: {'run_python': 26723, 'run_bash': 24818, 'calendar_list_events': 7312, 'read_file': 6019, 'calendar_find_free_slot': 4864, 'contacts_lookup': 4162, 'email_draft': 3743, 'list_files': 3679, 'tasks_add': 3599, 'calendar_create_event': 3208, 'email_read': 3005, 'delete_file': 2754, 'calculator': 1745, 'email_list_inbox': 1518, 'calendar_update_event': 1518, 'write_file': 1366, 'tasks_complete': 1164, 'tasks_list': 962, 'table_aggregate': 835, 'contacts_search': 759, 'search_files': 759, 'calendar_delete_event': 683, 'get_current_date': 607, 'email_send': 380}
* difficulty: {'easy': 14241, 'medium': 16359, 'hard': 8348}
* category: {'abstain': 718, 'api_comms': 4553, 'multi_turn': 2079, 'files': 4209, 'api_files': 4857, 'api_calendar': 6146, 'text': 4247, 'api_behaviour': 2542, 'cross_app': 2011, 'data': 5803, 'calendar': 1783}

## Techniques used
| # | Technique | What it does here |
|---|---|---|
| 1 | Verifiable task synthesis | 63 task families generated procedurally from seeds; every task has an independent checker |
| 2 | Two environments | a real sandbox (`run_python` / `run_bash`) and a simulated workspace API with 22 typed tools (calendar, contacts, email, tasks, files, calculator, table aggregation) |
| 3 | Execution-grounded trajectories | tool results and API errors are real outputs of the sandbox / API, never invented |
| 4 | Rejection sampling | a trajectory is kept only if the checker confirms the final state (files, calendar events, drafts, tasks) and the reply |
| 5 | Short ReAct-style reasoning | one or two sentences of plan before each action, reacting to what the tool returned (kept under 450 characters) |
| 6 | Look-then-act chains | search -> read, list -> read, find slot -> create, read -> compute -> write; later arguments are built from earlier tool results |
| 7 | Parallel tool calls | 10088 assistant messages issue several independent calls at once |
| 8 | Tool selection among distractors | 5-10 tools are offered per task, only some are needed (2-3 for most), in random order |
| 9 | Typed arguments | enums, arrays, ISO dates, HH:MM times, optional parameters; the API validates and rejects bad arguments |
| 10 | API error recovery | 928 real error responses (invalid arguments, not found) followed by a corrected call |
| 11 | Multi-turn | 4432 conversations with follow-ups that change earlier results, plus clarification questions when a request is ambiguous |
| 12 | Tool restraint | 1707 trajectories use no tool: direct answers, unsupported requests (honest limitation + alternatives) |
| 13 | Safety and precision | draft vs send (never send unless asked), confirm before deleting, only touch what was asked |
| 14 | Sandbox error recovery | ~4% of sandbox tasks start with a plausible mistake that really fails (traceback), then a fix quoting the error |
| 15 | Strategy and data diversity | Python vs bash / awk / grep; header names, delimiters, date formats, currency strings, blanks, duplicates, distractor files, decoy lines |
| 16 | Prompt expansion | composable phrasings, intros / outros, casual typing variation, 4 + 4 system prompts |
| 17 | Abstention | tasks asking for data that does not exist; the right behaviour is to look, then say so |
| 18 | Grounding + answer-leak filters | every value in the final reply must appear in a tool output; plans may not state a result before it was observed |
| 19 | Dedup + decontamination | a task is a duplicate only if prompt AND data are identical; train seeds (>= 10000) never overlap held-out test seeds (0..N-1); any train task identical to a test task is dropped |
| 20 | Stratified sampling hooks | `max_per_family` / `max_examples` in the SFT mix keep every family represented when you can only train on a subset |
| 21 | Replay mixing (in the SFT config) | combined with Hermes function-calling data and a little SmolTalk chat |

## Per-family counts
| family | attempted | kept |
|---|---|---|
| api_agenda_summary | 683 | 683 |
| api_ambiguous_contact | 759 | 759 |
| api_calculator_basic | 531 | 531 |
| api_calculator_chain | 607 | 607 |
| api_cancel_event | 683 | 683 |
| api_confirm_destructive | 835 | 835 |
| api_conflict_check | 759 | 759 |
| api_draft_or_send | 948 | 948 |
| api_file_not_found_recover | 759 | 759 |
| api_find_and_schedule | 948 | 948 |
| api_followup_adjust | 759 | 759 |
| api_inbox_digest | 759 | 759 |
| api_meeting_prep | 948 | 948 |
| api_no_tool_needed | 948 | 948 |
| api_reply_confirm_meeting | 759 | 759 |
| api_reschedule | 759 | 759 |
| api_search_and_answer | 759 | 759 |
| api_table_aggregate | 835 | 835 |
| api_task_capture | 759 | 759 |
| api_task_review | 569 | 569 |
| api_today_agenda | 607 | 607 |
| api_unsupported_request | 759 | 759 |
| api_weekly_digest | 683 | 683 |
| api_write_summary | 683 | 683 |
| bash_file_counts | 455 | 455 |
| common_free_slot | 759 | 759 |
| config_update | 455 | 455 |
| csv_clean_dedupe | 569 | 569 |
| csv_count_by_month | 379 | 379 |
| csv_filter_export | 493 | 493 |
| csv_filtered_sum | 759 | 759 |
| csv_group_followup | 759 | 759 |
| csv_group_totals | 569 | 569 |
| csv_join_city_totals | 569 | 569 |
| csv_overdue_days | 531 | 531 |
| csv_percent_of_total | 493 | 493 |
| csv_pivot_month_category | 493 | 493 |
| csv_stats | 379 | 379 |
| csv_sum_followup | 759 | 759 |
| csv_top_n | 569 | 569 |
| extract_emails | 569 | 569 |
| grep_keyword_count | 379 | 379 |
| inbox_triage | 493 | 493 |
| ini_to_json | 417 | 417 |
| invite_draft | 759 | 759 |
| json_filter_transform | 569 | 569 |
| json_merge_dedupe | 493 | 493 |
| json_to_markdown | 455 | 455 |
| largest_file | 379 | 379 |
| log_slowest_requests | 493 | 493 |
| log_summary | 569 | 569 |
| markdown_toc | 417 | 417 |
| merge_csv_parts | 493 | 493 |
| missing_information | 759 | 718 |
| notes_edit_followup | 569 | 561 |
| organize_by_ext | 569 | 569 |
| overdue_reminders | 759 | 759 |
| rename_extension | 379 | 379 |
| schedule_conflicts | 569 | 569 |
| text_replace_files | 455 | 455 |
| text_word_stats | 455 | 455 |
| timesheet_hours | 455 | 455 |
| todo_extract | 455 | 455 |

## Rejected / dropped
| reason | count |
|---|---|
| duplicate_task | 48 |
| decontaminated_vs_test | 1 |

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
