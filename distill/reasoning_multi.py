"""Multi-turn step-by-step reasoning: a word problem, then follow-ups that CHANGE a number or ASK for a related quantity ("what if ...", "how much of that ...").

    python -m distill.reasoning_multi --n 1000 --out data/raw/reasoning_multiturn.jsonl

Each turn is solved in the same numbered-steps format as distill/reasoning_gen.py (answer only at the end), computed in code, and the later turns reuse
the earlier numbers (so the model must carry state through the conversation instead of solving isolated problems). Seven problem types.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from distill.reasoning_gen import INTROS, ITEMS, OUTROS, Sol, n2
from envs.tasks.common import FIRST

MULTI: dict = {}


def mt(name):
    def deco(fn):
        MULTI[name] = fn
        return fn
    return deco


def pick(r, xs):
    return r.choice(xs)


def ans(r, sol, final):
    return sol.text(final)


def U(r, text, first=False):
    return f"{pick(r, INTROS)}{text}{pick(r, OUTROS)}".strip() if first else text


@mt("discount_whatif")
def m_discount(r):
    item, P, p = pick(r, ITEMS), r.randrange(40, 401, 10), pick(r, [10, 15, 20, 25, 30])
    q = pick(r, [x for x in (10, 15, 20, 25, 30, 40, 50) if x != p])
    s1, s2 = Sol(r), Sol(r)
    d1, d2 = P * p / 100, P * q / 100
    s1.step("Find the discount.", f"{p}% of {P} EUR = {n2(d1)} EUR.")
    s1.step("Subtract it.", f"{P} - {n2(d1)} = {n2(P - d1)} EUR.")
    s2.step(f"Redo the discount with {q}% instead of {p}%.", f"{q}% of {P} EUR = {n2(d2)} EUR.")
    s2.step("Subtract it from the same original price.", f"{P} - {n2(d2)} = {n2(P - d2)} EUR.")
    msgs = [("user", U(r, f"A {item} costs {P} EUR and is {p}% off. What is the sale price?", True)), ("assistant", ans(r, s1, f"{n2(P - d1)} EUR")),
            ("user", f"What if the discount were {q}% instead?"), ("assistant", ans(r, s2, f"{n2(P - d2)} EUR"))]
    if r.random() < 0.5:
        s3 = Sol(r)
        s3.step("Compare the two discount amounts from the earlier steps.", f"{n2(d2)} - {n2(d1)} = {n2(d2 - d1)} EUR.")
        msgs += [("user", "How much more would I save with that second discount?"), ("assistant", ans(r, s3, f"{n2(abs(d2 - d1))} EUR {'more' if d2 > d1 else 'less'}"))]
    return msgs


@mt("tip_whatif")
def m_tip(r):
    total, tip, n, m = r.randrange(60, 300, 10), pick(r, [10, 15, 20]), r.randrange(2, 6), r.randrange(1, 4)
    g = total * (100 + tip) / 100
    s1, s2 = Sol(r), Sol(r)
    s1.step("Add the tip.", f"{total} x {(100 + tip) / 100:g} = {n2(g)} EUR.")
    s1.step(f"Split between {n} people.", f"{n2(g)} / {n} = {n2(round(g / n, 2))} EUR each.")
    s2.step(f"The bill with tip is unchanged: {n2(g)} EUR. The group is now {n} + {m} = {n + m} people.", "")
    s2.step("Divide again.", f"{n2(g)} / {n + m} = {n2(round(g / (n + m), 2))} EUR each.")
    return [("user", U(r, f"Dinner for {n} people costs {total} EUR before a {tip}% tip. What does each person pay if we split it evenly with the tip?", True)), ("assistant", ans(r, s1, f"{n2(round(g / n, 2))} EUR each")),
            ("user", f"What if {m} more {'person' if m == 1 else 'people'} joined and we split the same bill?"), ("assistant", ans(r, s2, f"{n2(round(g / (n + m), 2))} EUR each"))]


@mt("travel_whatif")
def m_travel(r):
    sp, t = pick(r, [40, 60, 80, 100]), pick(r, [45, 75, 90, 120, 150])
    d = sp * t // 60
    sp2 = pick(r, [x for x in (30, 50, 60, 75, 90, 120) if x != sp and (d * 60) % x == 0] or [sp * 2])
    t2 = d * 60 / sp2
    from distill.reasoning_gen import hm
    s1, s2 = Sol(r), Sol(r)
    s1.step("Use time = distance / speed.", f"{d} / {sp} = {d / sp:g} hours = {t} minutes.")
    s2.step(f"Same distance, new speed. Use time = distance / speed with the {d} km from before.", f"{d} / {sp2} = {d / sp2:g} hours.")
    s2.step("Convert to minutes.", f"{d / sp2:g} h x 60 = {n2(t2)} minutes.")
    return [("user", U(r, f"A trip of {d} km at a steady {sp} km/h - how long does it take?", True)), ("assistant", ans(r, s1, hm(t))),
            ("user", f"And at {sp2} km/h?"), ("assistant", ans(r, s2, hm(int(round(t2)))))]


@mt("compound_more")
def m_compound(r):
    P, rate, y1, extra = r.randrange(1000, 5001, 500), pick(r, [2, 3, 4, 5, 6]), pick(r, [1, 2]), pick(r, [1, 2])
    f = (100 + rate) / 100
    s1, s2, bal = Sol(r), Sol(r), float(P)
    for y in range(1, y1 + 1):
        nb = round(bal * f, 2)
        s1.step(f"Year {y}.", f"{n2(bal)} x {f:g} = {n2(nb)} EUR.")
        bal = nb
    first = bal
    for y in range(y1 + 1, y1 + extra + 1):
        nb = round(bal * f, 2)
        s2.step(f"Year {y} (start from the balance after year {y - 1}, {n2(bal)} EUR).", f"{n2(bal)} x {f:g} = {n2(nb)} EUR.")
        bal = nb
    return [("user", U(r, f"I put {P} EUR in an account paying {rate}% a year, compounded yearly. What is the balance after {y1} {'year' if y1 == 1 else 'years'}?", True)), ("assistant", ans(r, s1, f"{n2(first)} EUR")),
            ("user", f"And after {extra} more {'year' if extra == 1 else 'years'}?"), ("assistant", ans(r, s2, f"{n2(bal)} EUR"))]


@mt("vat_breakdown")
def m_vat(r):
    net, t, t2 = r.randrange(20, 400, 5), pick(r, [7, 19, 21]), None
    t2 = pick(r, [x for x in (7, 10, 19, 21, 23) if x != t])
    g1, g2 = net * (100 + t) / 100, net * (100 + t2) / 100
    s1, s2, s3 = Sol(r), Sol(r), Sol(r)
    s1.step("Add the VAT to the net price.", f"{net} x {(100 + t) / 100:g} = {n2(g1)} EUR.")
    s2.step("The VAT is the gross price minus the net price from the first answer.", f"{n2(g1)} - {net} = {n2(g1 - net)} EUR.")
    s3.step(f"Recompute with {t2}% on the same net price of {net} EUR.", f"{net} x {(100 + t2) / 100:g} = {n2(g2)} EUR.")
    return [("user", U(r, f"An item costs {net} EUR before VAT. What is the price with {t}% VAT?", True)), ("assistant", ans(r, s1, f"{n2(g1)} EUR")),
            ("user", "How much of that is VAT?"), ("assistant", ans(r, s2, f"{n2(g1 - net)} EUR")),
            ("user", f"What would the price be with {t2}% VAT instead?"), ("assistant", ans(r, s3, f"{n2(g2)} EUR"))]


@mt("ratio_total")
def m_ratio(r):
    parts = pick(r, [(2, 3), (1, 4), (3, 5), (2, 3, 5), (1, 2, 3)])
    names = r.sample(FIRST, len(parts))
    u1, u2 = r.randrange(5, 30) * 5, r.randrange(5, 30) * 5
    while u2 == u1:
        u2 = r.randrange(5, 30) * 5
    t1, t2 = u1 * sum(parts), u2 * sum(parts)
    ratio = ":".join(map(str, parts))

    def solve(total, unit, reuse):
        s = Sol(r)
        s.step("Total number of parts." if not reuse else f"The ratio is unchanged, so there are still {sum(parts)} parts.", f"{' + '.join(map(str, parts))} = {sum(parts)}.")
        s.step("Value of one part.", f"{total} / {sum(parts)} = {unit} EUR.")
        s.step("Each share.", "; ".join(f"{n}: {p} x {unit} = {p * unit}" for n, p in zip(names, parts)) + ".")
        return s, ", ".join(f"{n} {p * unit} EUR" for n, p in zip(names, parts))
    s1, a1 = solve(t1, u1, False)
    s2, a2 = solve(t2, u2, True)
    return [("user", U(r, f"{', '.join(names[:-1])} and {names[-1]} share {t1} EUR in the ratio {ratio}. How much does each get?", True)), ("assistant", ans(r, s1, a1)),
            ("user", f"What if the total were {t2} EUR with the same ratio?"), ("assistant", ans(r, s2, a2))]


@mt("average_next")
def m_avg(r):
    n = r.randrange(3, 6)
    xs = [r.randrange(55, 100) for _ in range(n)]
    nxt = r.randrange(40, 101)
    s1, s2 = Sol(r), Sol(r)
    tot = sum(xs)
    s1.step("Add the scores.", f"{' + '.join(map(str, xs))} = {tot}.")
    s1.step(f"Divide by {n}.", f"{tot} / {n} = {n2(round(tot / n, 2))}.")
    s2.step(f"The total so far is {tot}. Add the new score.", f"{tot} + {nxt} = {tot + nxt}.")
    s2.step(f"Now there are {n + 1} scores, so divide by {n + 1}.", f"{tot + nxt} / {n + 1} = {n2(round((tot + nxt) / (n + 1), 2))}.")
    return [("user", U(r, f"My test scores are {', '.join(map(str, xs))}. What is my average?", True)), ("assistant", ans(r, s1, n2(round(tot / n, 2)))),
            ("user", f"What will my average be if I score {nxt} on the next test?"), ("assistant", ans(r, s2, n2(round((tot + nxt) / (n + 1), 2))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/raw/reasoning_multiturn.jsonl")
    a = ap.parse_args()
    rows, seen, k = [], set(), 0
    while len(rows) < a.n and k < a.n * 20:
        name = list(MULTI)[k % len(MULTI)]
        r = random.Random(f"rm:{name}:{k}")
        k += 1
        msgs = [{"role": ro, "content": c} for ro, c in MULTI[name](r)]
        key = json.dumps(msgs, sort_keys=True)
        if key in seen or any(not m["content"].strip() for m in msgs):
            continue
        assert all(m["content"].rstrip().splitlines()[-1].startswith("Final answer:") for m in msgs if m["role"] == "assistant"), name
        seen.add(key)
        rows.append({"id": f"rmulti/{name}/{k}", "messages": msgs, "tools": [],
                     "meta": {"family": f"rmulti_{name}", "category": "reasoning", "difficulty": "medium", "kind": "reasoning", "seed": k, "tags": ["cot", "step_by_step", "multi_turn", name],
                              "n_tool_calls": 0, "n_turns": sum(m["role"] == "user" for m in msgs), "n_tools_offered": 0}})
    random.Random(4).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(MULTI)} problem types -> {len(rows)} conversations in {a.out}")


if __name__ == "__main__":
    main()
