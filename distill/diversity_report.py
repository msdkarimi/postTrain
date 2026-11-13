"""How repetitive is a built training set?  python -m distill.diversity_report data/processed/sft_v2/canonical/train.jsonl

Per source: examples, share of the set, how many assistant sentences (25+ characters) are distinct, the most repeated sentence, and how many distinct
"shapes" (first user + first assistant message with numbers / quotes / capitalised words masked) the examples fall into. A high repetition count or a low
distinct-shape ratio means the model will mostly see the same few templates."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from etl.sources.local_jsonl import sentences_of, shape_of


def report(path: str) -> dict:
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by[r["source"].split("/")[0]].append(r)
    total = sum(len(v) for v in by.values())
    out = {}
    for src, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        sent, shapes, n_sent = Counter(), Counter(), 0
        for r in rows:
            shapes[shape_of(r)] += 1
            for x in sentences_of(r):
                sent[x] += 1
                n_sent += 1
        top = sent.most_common(1)
        out[src] = {"examples": len(rows), "share": round(len(rows) / total, 3), "distinct_sentence_ratio": round(len(sent) / max(n_sent, 1), 3),
                    "top_sentence_count": top[0][1] if top else 0, "distinct_shapes": len(shapes), "shape_ratio": round(len(shapes) / len(rows), 3), "top_shape_count": shapes.most_common(1)[0][1]}
    return out


if __name__ == "__main__":
    rep = report(sys.argv[1])
    print(f"{'source':14s} {'examples':>8s} {'share':>6s} {'distinct sent':>13s} {'top sent x':>10s} {'shape ratio':>11s} {'top shape x':>11s}")
    for k, v in rep.items():
        print(f"{k:14s} {v['examples']:8d} {v['share']:6.1%} {v['distinct_sentence_ratio']:13.2f} {v['top_sentence_count']:10d} {v['shape_ratio']:11.2f} {v['top_shape_count']:11d}")
