"""Developer check: run every task family over many seeds and report solutions that fail their own checker.

    python -m distill.selfcheck --seeds 40
    python -m distill.selfcheck --family csv_filtered_sum --seeds 200 --show-failures 3
"""
from __future__ import annotations

import argparse
import random
import traceback
from collections import Counter, defaultdict

import envs.tasks  # noqa: F401  (registers families)
from envs.rollout import API_SYSTEM_PROMPTS, SYSTEM_PROMPTS, run_any
from envs.tasks.base import FAMILIES, make_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--family", default=None)
    ap.add_argument("--show-failures", type=int, default=1)
    args = ap.parse_args()
    names = [args.family] if args.family else sorted(FAMILIES)
    total_ok = total = 0
    for name in names:
        reasons, shown, ok_n, tags = Counter(), 0, 0, Counter()
        for seed in range(args.start, args.start + args.seeds):
            try:
                task = make_task(name, seed)
                res = run_any(task, SYSTEM_PROMPTS[0], API_SYSTEM_PROMPTS[0])
                ok, reason = res["ok"], res["reason"]
                if ok:
                    for tok in task.grounded:
                        if not any(tok in o for o in res["observations"]):
                            ok, reason = False, f"ungrounded:{tok[:30]}"
                            break
            except Exception as e:
                ok, reason = False, f"EXC {type(e).__name__}: {e}"
                if shown < args.show_failures:
                    traceback.print_exc()
            if ok:
                ok_n += 1
                tags.update(task.tags)
            else:
                reasons[reason] += 1
                if shown < args.show_failures and not reason.startswith("EXC"):
                    shown += 1
                    print(f"--- {name} seed {seed} failed: {reason}\nPROMPT: {task.prompt}")
                    for m in res["messages"][2:]:
                        print(f"[{m['role']}]", (m.get('content') or '')[:300].replace("\n", "\\n"),
                              "| CALL:" + str(m['tool_calls'][0]['function']['arguments'])[:400].replace("\\n", "\n") if m.get("tool_calls") else "")
        total_ok += ok_n
        total += args.seeds
        print(f"{name:28s} {ok_n:3d}/{args.seeds}  {dict(reasons) if reasons else ''}  recovery={tags.get('error_recovery', 0)}")
    print(f"\nTOTAL {total_ok}/{total} ({100 * total_ok / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
