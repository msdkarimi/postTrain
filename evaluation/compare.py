"""Side-by-side table of several evaluation runs.

    python -m evaluation.compare outputs/eval/base outputs/eval/instruct outputs/eval/sft
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        raise SystemExit("usage: python -m evaluation.compare <eval_dir> [<eval_dir> ...]")
    sums = [json.loads((d / "summary.json").read_text()) for d in dirs]
    names = [d.name for d in dirs]

    def row(label, get):
        cells = []
        for s in sums:
            v = get(s)
            cells.append("-" if v is None else f"{100 * v:.1f}%")
        return f"| {label} | " + " | ".join(cells) + " |"

    head = "| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    print("## Overall\n\n" + head)
    print(row("pass rate", lambda s: s["overall"]["pass_rate"]))
    print(row("tool-call format errors / assistant msg", lambda s: s["overall"]["format_error_rate"]))
    print(row("tool / API errors / call", lambda s: s["overall"]["tool_error_rate"]))
    for title, key in (("By kind", "by_kind"), ("By category", "by_category"), ("By skill (tag)", "by_skill"), ("By family", "by_family")):
        keys = sorted({k for s in sums for k in s[key]})
        print(f"\n## {title}\n\n{head}")
        for k in keys:
            print(row(k, lambda s, k=k: s[key].get(k, {}).get("pass_rate")))


if __name__ == "__main__":
    main()
