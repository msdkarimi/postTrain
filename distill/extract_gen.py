"""Structured information extraction from a supplied text, with exact outputs and follow-up edits.

    python -m distill.extract_gen --n 3200 --out data/raw/extraction.jsonl

Six generated document types (action items, an expense log, contacts, a schedule, a product list, release notes) are turned into JSON, CSV, a markdown
table or a bullet list as asked. The output is built from the same records that produced the text, so it is exact; the user asks for ONLY the structure
(no extra words), which is what instruction-following and function-calling benchmarks need. About 35% continue with a second turn: filter the rows,
sort them, or convert the same data to another format.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from datetime import date, timedelta
from pathlib import Path

from envs.tasks.common import FIRST, PROJECTS

FORMATS = ["json", "csv", "markdown table", "bullet list"]


def pick(r, xs):
    return r.choice(xs)


def render(rows: list, keys: list, fmt: str, r=None, indent=False) -> str:
    if fmt == "json":
        return json.dumps(rows, ensure_ascii=False, indent=2 if indent else None)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(keys)
        for row in rows:
            w.writerow([row[k] for k in keys])
        return buf.getvalue().rstrip("\n")
    if fmt == "markdown table":
        return "\n".join(["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"] + ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows])
    return "\n".join("- " + "; ".join(f"{k}: {row[k]}" for k in keys) for row in rows)


def fmt_hint(fmt: str, keys: list) -> str:
    return {"json": f" Return a JSON array of objects with the keys {', '.join(keys)}.", "csv": f" Return CSV with the header {','.join(keys)}.",
            "markdown table": f" Return a markdown table with the columns {', '.join(keys)}.", "bullet list": " Return one bullet per item in the form \"- " + "; ".join(f"{k}: ..." for k in keys) + "\"."}[fmt]


# ---- document types: (noun, doc text, rows, keys, categorical field, numeric field)
def d_actions(r):
    owners = r.sample(FIRST, 4)
    tasks = r.sample(["send the budget draft", "update the roadmap", "book the venue", "review the contract", "prepare the demo", "fix the login bug", "write the release notes", "contact the vendor"], 4)
    d0 = date(2025, 3, 3) + timedelta(days=r.randrange(0, 300))
    rows = [{"owner": o, "task": t, "due": (d0 + timedelta(days=r.randrange(1, 25))).isoformat()} for o, t in zip(owners, tasks)]
    doc = f"Meeting notes ({pick(r, PROJECTS)} sync)\n" + "\n".join(f"- {x['owner']} will {x['task']} by {x['due']}." for x in rows)
    return "action items", doc, rows, ["owner", "task", "due"], "owner", None


def d_ledger(r):
    cats = r.sample(["Travel", "Software", "Meals", "Office", "Training"], 3)
    d0 = date(2025, 1, 1)
    rows = sorted(({"date": (d0 + timedelta(days=r.randrange(0, 90))).isoformat(), "category": pick(r, cats), "description": pick(r, ["train ticket", "licence", "team lunch", "chairs", "course fee", "taxi", "subscription"]),
                    "amount": r.randrange(8, 400)} for _ in range(r.randrange(4, 8))), key=lambda x: x["date"])
    doc = "Expense log:\n" + "\n".join(f"{x['date']} | {x['category']} | {x['description']} | {x['amount']} EUR" for x in rows)
    return "expenses", doc, rows, ["date", "category", "description", "amount"], "category", "amount"


def d_contacts(r):
    names = r.sample(FIRST, 4)
    last = r.sample(["Rossi", "Berg", "Kim", "Silva", "Novak", "Haddad", "Larsen"], 4)
    teams = [pick(r, ["Sales", "Support", "Finance", "Design", "Engineering"]) for _ in names]
    rows = [{"name": f"{n} {l}", "team": t, "email": f"{n.lower()}.{l.lower()}@acme.com"} for n, l, t in zip(names, last, teams)]
    doc = "Team contacts: " + " ".join(f"{x['name']} ({x['team']}) can be reached at {x['email']}." for x in rows)
    return "contacts", doc, rows, ["name", "team", "email"], "team", None


def d_schedule(r):
    t, rows = 8 * 60 + r.randrange(0, 4) * 15, []
    for title in r.sample(["Opening talk", "Workshop A", "Coffee break", "Panel", "Lunch", "Workshop B", "Demo session", "Closing remarks"], r.randrange(4, 7)):
        dur = pick(r, [30, 45, 60, 90])
        rows.append({"start": f"{t // 60:02d}:{t % 60:02d}", "end": f"{(t + dur) // 60:02d}:{(t + dur) % 60:02d}", "title": title, "minutes": dur})
        t += dur + pick(r, [0, 0, 15])
    doc = "Conference schedule:\n" + "\n".join(f"{x['start']}-{x['end']} {x['title']}" for x in rows)
    return "sessions", doc, rows, ["start", "end", "title"], None, None


def d_products(r):
    items = r.sample(["Laptop", "Monitor", "Keyboard", "Webcam", "Headset", "Dock", "Mouse", "Desk lamp"], r.randrange(4, 7))
    rows = [{"product": p, "price": r.randrange(10, 400), "stock": r.randrange(0, 60)} for p in items]
    doc = "Product list:\n" + "\n".join(f"- {x['product']}: {x['price']} EUR, {x['stock']} in stock" for x in rows)
    return "products", doc, rows, ["product", "price", "stock"], None, "price"


def d_releases(r):
    feats = r.sample(["dark mode", "CSV export", "offline sync", "two-factor login", "keyboard shortcuts", "team folders", "API tokens", "voice notes"], r.randrange(4, 7))
    v, d0, rows = (1, r.randrange(0, 3)), date(2024, 9, 1), []
    for f in feats:
        d0 += timedelta(days=r.randrange(15, 80))
        rows.append({"version": f"{v[0]}.{v[1]}", "date": d0.isoformat(), "feature": f})
        v = (v[0], v[1] + 1) if r.random() < 0.7 else (v[0] + 1, 0)
    doc = "Release notes:\n" + "\n".join(f"- {x['version']} ({x['date']}): added {x['feature']}." for x in rows)
    return "releases", doc, rows, ["version", "date", "feature"], None, None


DOCS = [d_actions, d_ledger, d_contacts, d_schedule, d_products, d_releases]
INTROS = ["", "", "", "Here is a text. ", "I have these notes. ", "Quick task: "]


def build_one(seed: int):
    r = random.Random(f"extract:{seed}")
    noun, doc, rows, keys, cat, num = DOCS[seed % len(DOCS)](r)
    fmt = pick(r, FORMATS)
    indent = fmt == "json" and r.random() < 0.4
    ask = f"{pick(r, INTROS)}Extract the {noun} from the text below as {'a ' if fmt != 'bullet list' else 'a '}{fmt}.{fmt_hint(fmt, keys)} Return only the {fmt} and nothing else."
    prompt = f"{ask}\n\n{doc}" if r.random() < 0.7 else f"{doc}\n\n{ask}"
    msgs = [("user", prompt), ("assistant", render(rows, keys, fmt, indent=indent))]
    tags = ["extraction", fmt.replace(" ", "_")]
    if r.random() < 0.35:
        ops = ["convert", "sort"] + (["filter_cat"] if cat else []) + (["filter_num"] if num else [])
        op = pick(r, ops)
        if op == "convert":
            fmt2 = pick(r, [f for f in FORMATS if f != fmt])
            msgs += [("user", f"Now give the same data as a {fmt2}. Return only the {fmt2}."), ("assistant", render(rows, keys, fmt2, indent=False))]
        elif op == "sort":
            k = pick(r, [x for x in keys if x not in ("task", "description")][:3])
            desc = r.random() < 0.5 and any(isinstance(x[k], int) for x in rows)
            out = sorted(rows, key=lambda x: x[k], reverse=desc)
            msgs += [("user", f"Sort it by {k}{' from highest to lowest' if desc else ' in ascending order'}, same format. Return only the result."), ("assistant", render(out, keys, fmt, indent=indent))]
        elif op == "filter_cat":
            val = pick(r, [x[cat] for x in rows])
            out = [x for x in rows if x[cat] == val]
            msgs += [("user", f"Keep only the rows where {cat} is {val}, same format. Return only the result."), ("assistant", render(out, keys, fmt, indent=indent))]
        else:
            thr = sorted(x[num] for x in rows)[len(rows) // 2]
            out = [x for x in rows if x[num] >= thr]
            msgs += [("user", f"Keep only the rows where {num} is at least {thr}, same format. Return only the result."), ("assistant", render(out, keys, fmt, indent=indent))]
        tags.append("two_turn")
    return {"id": f"extract/{seed}", "messages": [{"role": ro, "content": c} for ro, c in msgs], "tools": [],
            "meta": {"family": f"extract_{noun.replace(' ', '_')}", "category": "extraction", "difficulty": "medium", "kind": "extraction", "seed": seed, "tags": tags,
                     "n_tool_calls": 0, "n_turns": sum(ro == "user" for ro, _ in msgs), "n_tools_offered": 0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--out", default="data/raw/extraction.jsonl")
    a = ap.parse_args()
    rows, seen, k = [], set(), 0
    while len(rows) < a.n and k < a.n * 4:
        rec = build_one(70_000 + k)
        k += 1
        key = rec["messages"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(rec)
    random.Random(6).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(rows)} examples -> {a.out}; two-turn {sum(x['meta']['n_turns'] > 1 for x in rows)}")


if __name__ == "__main__":
    main()
