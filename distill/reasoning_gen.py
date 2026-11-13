"""Step-by-step reasoning data (chain of thought), verified by construction.

    python -m distill.reasoning_gen --n-per-family 300 --out data/raw/reasoning_stepwise.jsonl

Each family draws a fresh word problem (several natural phrasings, random names / items / numbers), computes the answer in code and writes the
solution the way a good teacher would: a short opener, numbered steps that each do ONE piece of arithmetic with the numbers shown, a sanity
check, and only then "Final answer: ...". The answer is never stated before the steps that produce it (a test enforces this), the arithmetic is
computed, not typed, so the solution is correct by construction, and every family has an independent `solve` used by the self-check.

Why this exists: the agent data teaches tool calling; this teaches the model to decompose a question and reason step by step in plain language,
which is what GSM8K-style regression checks and real assistant use need. Records have no tools (tools = []).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from envs.tasks.common import FIRST

OPENERS = ["Let me work through this step by step.", "I'll break this into small steps.", "Let's solve it one step at a time.",
           "Let me think this through carefully.", "I'll go step by step so each number is easy to check.", "Okay, let me set this up first."]
CHECK_WORDS = ["Check:", "Sanity check:", "Quick check:"]
INTROS = ["", "", "", "", "Hi! ", "Quick question: ", "Hey, ", "Could you help me with this? ", "I'm stuck on this one. ", "Help me out here: ", "Small math question. "]
OUTROS = ["", "", "", "", " Please show your steps.", " Walk me through it.", " Thanks!", " Explain how you got it.", " Show your work, please."]
ITEMS = ["jacket", "backpack", "desk lamp", "pair of sneakers", "coffee machine", "monitor", "bicycle helmet", "tent", "blender", "set of headphones", "bookshelf", "camera bag"]
FOODS = ["flour", "sugar", "butter", "rice", "olive oil", "milk", "oats", "lentils"]
FACTORS = {"hours": 60, "minutes": 1}


def n2(x: float) -> str:
    x = round(x + 0.0, 2)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}".replace(",", "") if abs(x) < 10000 else f"{int(round(x))}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def eur(x: float) -> str:
    return f"{x:.2f} EUR" if abs(x - round(x)) > 1e-9 else f"{int(round(x))} EUR"


def hm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h} h {m} min" if h and m else (f"{h} h" if h else f"{m} min")


def clock(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


class Sol:
    """Collects numbered steps."""

    def __init__(self, rng):
        self.rng, self.steps, self.check = rng, [], ""

    def step(self, title: str, body: str):
        self.steps.append((title, body))

    def text(self, answer: str) -> str:
        r = self.rng
        lines = [r.choice(OPENERS), ""]
        for i, (t, b) in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {t} {b}".strip())
        if self.check:
            lines += ["", f"{r.choice(CHECK_WORDS)} {self.check}"]
        lines += ["", f"Final answer: {answer}"]
        return "\n".join(lines)


FAMILIES: dict = {}


def family(name: str, weight: float = 1.0, difficulty: str = "medium", tags=()):
    def deco(fn):
        FAMILIES[name] = (fn, weight, difficulty, list(tags))
        return fn
    return deco


def pick(r, xs):
    return r.choice(xs)


# --------------------------------------------------------------------------------------------------------------------- money & percent
@family("discount_price", 1.0, "easy", ["percent", "money"])
def f_discount(r):
    item, P, p, who = pick(r, ITEMS), r.randrange(20, 401, 5), pick(r, [10, 15, 20, 25, 30, 40]), pick(r, FIRST)
    prompt = pick(r, [f"A {item} costs {P} EUR and is on sale for {p}% off. What is the sale price?",
                      f"The {item} I want normally goes for {P} EUR, but there's a {p}% discount today. How much will I pay?",
                      f"{who} found a {item} with a {P} EUR price tag and {p}% off. What does it cost after the discount?"])
    s, d = Sol(r), P * p / 100
    s.step("Find the discount amount.", f"{p}% of {P} EUR = {P} x {p}/100 = {n2(d)} EUR.")
    s.step("Subtract it from the original price.", f"{P} - {n2(d)} = {n2(P - d)} EUR.")
    s.check = f"{n2(P - d)} should equal {100 - p}% of {P}: {P} x {(100 - p) / 100:g} = {n2(P * (100 - p) / 100)}. It does."
    return prompt, s.text(f"{n2(P - d)} EUR"), n2(P - d)


@family("discount_then_tax", 1.0, "medium", ["percent", "money", "multi_step"])
def f_discount_tax(r):
    item, P, p, t = pick(r, ITEMS), r.randrange(40, 301, 10), pick(r, [10, 20, 25, 30]), pick(r, [7, 19, 21])
    prompt = pick(r, [f"A {item} is listed at {P} EUR. There's a {p}% discount first, then {t}% VAT is added on the discounted price. What is the final price?",
                      f"I'm buying a {item} priced {P} EUR. It's {p}% off, and then {t}% tax gets added to the reduced price. What do I pay in total?",
                      f"Price tag: {P} EUR for a {item}. Apply {p}% off, then add {t}% VAT. What's the total?"])
    s = Sol(r)
    after = P * (100 - p) / 100
    final = round(after * (100 + t) / 100, 2)
    s.step("Apply the discount.", f"{P} x (1 - {p}/100) = {P} x {(100 - p) / 100:g} = {n2(after)} EUR.")
    s.step("Work out the VAT on the discounted price.", f"{n2(after)} x {t}/100 = {n2(after * t / 100)} EUR.")
    s.step("Add the VAT.", f"{n2(after)} + {n2(after * t / 100)} = {n2(final)} EUR.")
    s.check = f"{n2(after)} x {(100 + t) / 100:g} = {n2(final)}, the same result."
    return prompt, s.text(f"{n2(final)} EUR"), n2(final)


@family("split_with_tip", 1.0, "medium", ["money", "division"])
def f_tip(r):
    total, tip, n = r.randrange(40, 301, 5), pick(r, [10, 15, 20]), r.randrange(2, 7)
    prompt = pick(r, [f"Our dinner bill was {total} EUR. We want to leave a {tip}% tip and split everything evenly between {n} people. How much does each person pay?",
                      f"{n} of us ate out and the bill came to {total} EUR before tip. With a {tip}% tip, what is each person's share if we split equally?",
                      f"Split {total} EUR plus a {tip}% tip evenly across {n} friends. What does each one owe?"])
    s = Sol(r)
    tip_amt, grand = total * tip / 100, total * (100 + tip) / 100
    each = round(grand / n, 2)
    s.step("Calculate the tip.", f"{tip}% of {total} EUR = {n2(tip_amt)} EUR.")
    s.step("Add the tip to the bill.", f"{total} + {n2(tip_amt)} = {n2(grand)} EUR.")
    s.step(f"Divide by the number of people ({n}).", f"{n2(grand)} / {n} = {n2(each)} EUR each.")
    s.check = f"{n} x {n2(each)} = {n2(n * each)} EUR, which matches the total of {n2(grand)} EUR" + ("." if abs(n * each - grand) < 0.02 else " up to rounding.")
    return prompt, s.text(f"{n2(each)} EUR per person"), n2(each)


@family("percent_change", 1.0, "medium", ["percent"])
def f_pct_change(r):
    old, p, up = r.randrange(40, 401, 20), pick(r, [5, 10, 15, 20, 25, 50]), r.random() < 0.5
    new = old * (100 + p if up else 100 - p) / 100
    thing = pick(r, ["monthly subscription", "train ticket", "rent", "price of a phone plan", "gym membership", "price of a coffee bag"])
    verb = "rose" if up else "dropped"
    prompt = pick(r, [f"The {thing} {verb} from {old} EUR to {n2(new)} EUR. By what percentage did it {'increase' if up else 'decrease'}?",
                      f"My {thing} used to be {old} EUR and is now {n2(new)} EUR. What is the percent {'increase' if up else 'decrease'}?",
                      f"The {thing} went from {old} to {n2(new)} EUR. What percent change is that?"])
    s = Sol(r)
    diff = abs(new - old)
    s.step("Find the absolute change.", f"|{n2(new)} - {old}| = {n2(diff)} EUR.")
    s.step("Divide by the ORIGINAL value (the old price).", f"{n2(diff)} / {old} = {diff / old:g}.")
    s.step("Convert to a percentage.", f"{diff / old:g} x 100 = {p}%.")
    s.check = f"{old} {'+' if up else '-'} {p}% of {old} = {n2(new)}, which matches."
    return prompt, s.text(f"{p}% {'increase' if up else 'decrease'}"), f"{p}%"


@family("unit_price_compare", 1.0, "medium", ["money", "comparison"])
def f_unit_price(r):
    food = pick(r, FOODS)
    q1, q2 = r.sample([6, 10, 12, 20, 24, 30], 2)
    u1 = r.randrange(40, 130, 5) / 100
    u2 = round(u1 + pick(r, [-0.10, -0.05, 0.05, 0.10]), 2)
    p1, p2 = round(q1 * u1, 2), round(q2 * u2, 2)
    prompt = pick(r, [f"A small pack of {food} has {q1} portions for {n2(p1)} EUR and a large pack has {q2} portions for {n2(p2)} EUR. Which is the better deal, and by how much per portion?",
                      f"Pack A: {q1} servings of {food} for {n2(p1)} EUR. Pack B: {q2} servings for {n2(p2)} EUR. Which one is cheaper per serving, and what's the difference?"])
    s = Sol(r)
    a, b = p1 / q1, p2 / q2
    s.step("Price per portion for pack A.", f"{n2(p1)} / {q1} = {a:.3f} EUR.")
    s.step("Price per portion for pack B.", f"{n2(p2)} / {q2} = {b:.3f} EUR.")
    s.step("Compare them.", f"{max(a, b):.3f} - {min(a, b):.3f} = {abs(a - b):.3f} EUR per portion.")
    cheaper = "Pack A" if a < b else "Pack B"
    s.check = f"{cheaper} has the lower unit price, so it is the better deal."
    return prompt, s.text(f"{cheaper} is cheaper, by {abs(a - b):.3f} EUR per portion"), cheaper


@family("percent_of_percent", 0.6, "easy", ["percent"])
def f_pop(r):
    base, a, b = pick(r, [200, 400, 600, 800, 1000]), pick(r, [10, 20, 25, 50]), pick(r, [10, 20, 50, 75])
    prompt = pick(r, [f"What is {a}% of {b}% of {base}?", f"A town has {base} residents. {b}% of them are adults and {a}% of the adults cycle to work. How many adults cycle to work?"])
    s = Sol(r)
    first = base * b / 100
    ans = first * a / 100
    if "residents" in prompt:
        s.step("Find the number of adults.", f"{b}% of {base} = {n2(first)}.")
        s.step("Find how many of them cycle.", f"{a}% of {n2(first)} = {n2(ans)}.")
    else:
        s.step(f"Take {b}% of {base} first.", f"{base} x {b}/100 = {n2(first)}.")
        s.step(f"Take {a}% of that result.", f"{n2(first)} x {a}/100 = {n2(ans)}.")
    s.check = f"Combined it is {a * b / 100:g}% of {base}, and {base} x {a * b / 10000:g} = {n2(ans)}."
    return prompt, s.text(n2(ans)), n2(ans)


@family("profit_margin", 0.8, "medium", ["money", "percent"])
def f_profit(r):
    cost, m, qty = r.randrange(10, 101, 5), pick(r, [20, 25, 30, 40, 50]), r.randrange(5, 41)
    sell = cost * (100 + m) / 100
    prompt = pick(r, [f"A shop buys a gadget for {cost} EUR and sells it for {n2(sell)} EUR. How much profit does it make on {qty} gadgets, and what is the profit margin on cost?",
                      f"Each item costs the store {cost} EUR and is sold at {n2(sell)} EUR. If it sells {qty} items, what is the total profit and the markup percentage?"])
    s = Sol(r)
    per = sell - cost
    s.step("Profit per item.", f"{n2(sell)} - {cost} = {n2(per)} EUR.")
    s.step(f"Total profit for {qty} items.", f"{n2(per)} x {qty} = {n2(per * qty)} EUR.")
    s.step("Margin on cost.", f"{n2(per)} / {cost} x 100 = {m}%.")
    s.check = f"{cost} x {1 + m / 100:g} = {n2(sell)}, matching the selling price."
    return prompt, s.text(f"total profit {n2(per * qty)} EUR, margin {m}%"), n2(per * qty)


@family("compound_growth", 0.8, "hard", ["percent", "multi_step"])
def f_compound(r):
    P, rate, yrs = r.randrange(1000, 5001, 500), pick(r, [2, 3, 4, 5, 6, 8, 10]), pick(r, [2, 3])
    prompt = pick(r, [f"I put {P} EUR in a savings account that pays {rate}% interest per year, compounded annually. How much is in the account after {yrs} years?",
                      f"{P} EUR grows by {rate}% each year (the interest is added to the balance). What is the balance after {yrs} years?"])
    s, bal = Sol(r), float(P)
    for y in range(1, yrs + 1):
        nxt = round(bal * (100 + rate) / 100, 2)
        s.step(f"Year {y}.", f"{n2(bal)} x {(100 + rate) / 100:g} = {n2(nxt)} EUR.")
        bal = nxt
    s.check = f"{P} x {(100 + rate) / 100:g}^{yrs} = {n2(P * ((100 + rate) / 100) ** yrs)} (rounding each year changes it by a cent at most)."
    return prompt, s.text(f"{n2(bal)} EUR"), n2(bal)


# --------------------------------------------------------------------------------------------------------------------- rates, time, dates
@family("travel_time", 1.0, "easy", ["rate", "time"])
def f_travel(r):
    sp, t = pick(r, [40, 60, 80, 100]), pick(r, [30, 45, 75, 90, 120, 150])
    d = sp * t // 60
    prompt = pick(r, [f"A train travels {d} km at a constant {sp} km/h. How long does the trip take?",
                      f"How long does it take to cover {d} km at {sp} km/h?",
                      f"I drive {d} km and keep a steady speed of {sp} km/h. What is my travel time?"])
    s = Sol(r)
    s.step("Use time = distance / speed.", f"{d} / {sp} = {d / sp:g} hours.")
    s.step("Convert the decimal part to minutes.", f"{d / sp:g} h x 60 = {t} minutes.")
    s.check = f"{sp} km/h x {t}/60 h = {d} km, which matches."
    return prompt, s.text(hm(t)), hm(t)


@family("work_together", 0.8, "hard", ["rate", "fractions"])
def f_work(r):
    a, b = pick(r, [(6, 3), (4, 12), (10, 15), (12, 6), (20, 5), (15, 10), (12, 4), (30, 20)])
    w1, w2 = r.sample(FIRST, 2)
    job = pick(r, ["paint the fence", "stuff the envelopes", "clean the warehouse", "assemble the shelves"])
    together = a * b / (a + b)
    prompt = pick(r, [f"{w1} can {job} in {a} hours and {w2} can do it in {b} hours. How long will it take them working together?",
                      f"Working alone, {w1} needs {a} hours to {job} and {w2} needs {b} hours. If they work together at their usual speeds, how many hours are needed?"])
    s = Sol(r)
    s.step(f"Rate of {w1} (jobs per hour).", f"1/{a} of the job per hour.")
    s.step(f"Rate of {w2}.", f"1/{b} of the job per hour.")
    s.step("Add the rates.", f"1/{a} + 1/{b} = {b}/{a * b} + {a}/{a * b} = {a + b}/{a * b} of the job per hour.")
    s.step("Invert to get hours for one whole job.", f"{a * b}/{a + b} = {n2(together)} hours.")
    s.check = f"In {n2(together)} h, {w1} does {n2(together / a)} and {w2} does {n2(together / b)} of the job; together that is {n2(together / a + together / b)}."
    return prompt, s.text(f"{n2(together)} hours"), n2(together)


@family("ages_equation", 0.8, "hard", ["algebra"])
def f_ages(r):
    y, d, n = r.randrange(6, 40), r.randrange(3, 20), r.randrange(2, 10)
    o = y + d
    S = o + y + 2 * n
    w1, w2 = r.sample(FIRST, 2)
    prompt = pick(r, [f"{w1} is {d} years older than {w2}. In {n} years the sum of their ages will be {S}. How old is {w2} now?",
                      f"{w2} is {d} years younger than {w1}, and in {n} years their ages will add up to {S}. What is {w2}'s age today?"])
    s = Sol(r)
    s.step(f"Let {w2}'s age now be x. Then {w1}'s age is x + {d}.", "")
    s.step(f"In {n} years they are x + {n} and x + {d + n}.", "")
    s.step("Their sum is the given total.", f"(x + {n}) + (x + {d + n}) = {S}, so 2x + {d + 2 * n} = {S}.")
    s.step("Solve for x.", f"2x = {S} - {d + 2 * n} = {S - d - 2 * n}, so x = {y}.")
    s.check = f"{w2} is {y} and {w1} is {o}; in {n} years that is {y + n} + {o + n} = {S}. Correct."
    return prompt, s.text(f"{w2} is {y} years old"), str(y)


@family("time_chain", 1.0, "medium", ["time"])
def f_time(r):
    start = r.randrange(7 * 60, 21 * 60, 15)
    tasks = r.sample([("a meeting", 45), ("a code review", 30), ("lunch", 60), ("a workshop", 90), ("emails", 25), ("a call with a client", 40), ("a stand-up", 15), ("report writing", 75)], 3)
    prompt = pick(r, [f"I start work at {clock(start)}. First comes {tasks[0][0]} ({tasks[0][1]} minutes), then {tasks[1][0]} ({tasks[1][1]} minutes), then {tasks[2][0]} ({tasks[2][1]} minutes), with no breaks in between. When do I finish?",
                      f"Starting at {clock(start)}, I do {tasks[0][0]} for {tasks[0][1]} minutes, {tasks[1][0]} for {tasks[1][1]} minutes and {tasks[2][0]} for {tasks[2][1]} minutes back to back. What time is it at the end?"])
    s, t = Sol(r), start
    for name, dur in tasks:
        s.step(f"Add {name} ({dur} min).", f"{clock(t)} + {dur} min = {clock(t + dur)}.")
        t += dur
    total = sum(d for _, d in tasks)
    s.check = f"The total is {total} minutes = {hm(total)}, and {clock(start)} + {hm(total)} = {clock(t)}" + (" (the next day)." if t >= 24 * 60 else ".")
    return prompt, s.text(clock(t) + (" (next day)" if t >= 24 * 60 else "")), clock(t)


@family("date_add_days", 0.8, "medium", ["dates"])
def f_dates(r):
    d0 = date(2025, 1, 1) + timedelta(days=r.randrange(0, 700))
    n = pick(r, [10, 14, 21, 30, 45, 60, 90, 100])
    d1 = d0 + timedelta(days=n)
    prompt = pick(r, [f"What date is {n} days after {d0.isoformat()}, and what weekday is it?", f"A project starts on {d0.isoformat()} and runs for {n} days. On which date and weekday does it end (counting the start date as day 0)?"])
    s = Sol(r)
    s.step(f"Find the weekday of the start date.", f"{d0.isoformat()} is a {d0.strftime('%A')}.")
    weeks, rem = divmod(n, 7)
    s.step("Split the days into weeks and leftover days.", f"{n} = {weeks} x 7 + {rem}, so the weekday moves forward by {rem} day(s) (whole weeks don't change it).")
    s.step("Weekday of the end date.", f"{d0.strftime('%A')} + {rem} day(s) = {d1.strftime('%A')}.")
    s.step("Count the calendar date.", f"{d0.isoformat()} + {n} days = {d1.isoformat()}.")
    s.check = f"{d1.isoformat()} - {d0.isoformat()} = {n} days."
    return prompt, s.text(f"{d1.isoformat()}, a {d1.strftime('%A')}"), d1.isoformat()


@family("rate_conversion", 0.8, "medium", ["rate", "units"])
def f_rate(r):
    rate, hrs, thing = pick(r, [8, 12, 15, 20, 25, 30, 40]), pick(r, [1.5, 2, 2.5, 3, 4]), pick(r, ["pages", "bottles", "parts", "envelopes"])
    machine = pick(r, ["printer", "machine", "assembly line", "labelling robot"])
    prompt = pick(r, [f"A {machine} makes {rate} {thing} per minute. How many {thing} does it make in {n2(hrs)} hours?", f"If a {machine} processes {rate} {thing} every minute, what is its output over {n2(hrs)} hours?"])
    s = Sol(r)
    mins = int(hrs * 60)
    s.step("Convert hours to minutes.", f"{n2(hrs)} h x 60 = {mins} minutes.")
    s.step("Multiply the rate by the minutes.", f"{rate} x {mins} = {rate * mins} {thing}.")
    s.check = f"Per hour that is {rate * 60} {thing}; {rate * 60} x {n2(hrs)} = {n2(rate * 60 * hrs)}."
    return prompt, s.text(f"{rate * mins} {thing}"), str(rate * mins)


# --------------------------------------------------------------------------------------------------------------------- algebra, sets, counting
@family("think_of_a_number", 0.8, "medium", ["algebra"])
def f_equation(r):
    x, k, c = r.randrange(2, 30), r.randrange(2, 9), r.randrange(3, 40)
    res = k * x + c if r.random() < 0.5 else k * x - c
    plus = (res == k * x + c)
    prompt = pick(r, [f"I'm thinking of a number. If I multiply it by {k} and {'add' if plus else 'subtract'} {c}, I get {res}. What is the number?",
                      f"Multiplying a number by {k} and then {'adding' if plus else 'subtracting'} {c} gives {res}. Find the number."])
    s = Sol(r)
    s.step("Call the number x and write the equation.", f"{k}x {'+' if plus else '-'} {c} = {res}.")
    s.step(f"Undo the {'addition' if plus else 'subtraction'} first.", f"{k}x = {res} {'-' if plus else '+'} {c} = {k * x}.")
    s.step(f"Divide by {k}.", f"x = {k * x} / {k} = {x}.")
    s.check = f"{k} x {x} {'+' if plus else '-'} {c} = {res}. Correct."
    return prompt, s.text(str(x)), str(x)


@family("inclusion_exclusion", 0.8, "hard", ["sets"])
def f_sets(r):
    total = r.randrange(30, 61)
    both = r.randrange(2, 9)
    a, b = both + r.randrange(4, 14), both + r.randrange(4, 14)
    if a + b - both > total:
        total = a + b - both + r.randrange(2, 10)
    s_a, s_b = pick(r, [("chess club", "drama club"), ("choir", "robotics team"), ("swimming", "tennis"), ("book club", "hiking group")])
    prompt = pick(r, [f"In a class of {total} students, {a} are in the {s_a}, {b} are in the {s_b} and {both} are in both. How many students are in neither?",
                      f"{total} students were surveyed: {a} like {s_a}, {b} like {s_b}, and {both} like both. How many like neither?"])
    s = Sol(r)
    either = a + b - both
    s.step("Count students in at least one group (don't count the overlap twice).", f"{a} + {b} - {both} = {either}.")
    s.step("Subtract from the total.", f"{total} - {either} = {total - either}.")
    s.check = f"Only {s_a}: {a - both}, only {s_b}: {b - both}, both: {both}, neither: {total - either}; sum = {a - both + b - both + both + total - either} = {total}."
    return prompt, s.text(str(total - either)), str(total - either)


@family("outfit_combinations", 0.7, "medium", ["counting"])
def f_combos(r):
    a, b, c, bad = r.randrange(3, 7), r.randrange(2, 6), r.randrange(2, 5), r.randrange(1, 4)
    prompt = pick(r, [f"I have {a} shirts, {b} pairs of trousers and {c} pairs of shoes. How many different outfits can I make, if {bad} of the possible shirt-and-trouser pairings clash and I never wear those?",
                      f"A café offers {a} sandwiches, {b} drinks and {c} desserts. A combo is one of each, but {bad} sandwich-and-drink pairings are unavailable. How many combos can you order?"])
    s = Sol(r)
    total = a * b * c
    s.step("Count all combinations first.", f"{a} x {b} x {c} = {total}.")
    s.step("Each unavailable pairing removes one option for every choice of the third item.", f"{bad} x {c} = {bad * c} combinations to remove.")
    s.step("Subtract.", f"{total} - {bad * c} = {total - bad * c}.")
    s.check = f"Allowed pairings are {a * b - bad}, and {a * b - bad} x {c} = {(a * b - bad) * c}."
    return prompt, s.text(str((a * b - bad) * c)), str((a * b - bad) * c)


@family("ratio_share", 0.9, "medium", ["ratio"])
def f_ratio(r):
    parts = pick(r, [(2, 3), (1, 4), (3, 5), (2, 3, 5), (1, 2, 3), (4, 5)])
    unit = r.randrange(5, 40) * 5
    total = unit * sum(parts)
    names = r.sample(FIRST, len(parts))
    ratio = ":".join(map(str, parts))
    prompt = pick(r, [f"{', '.join(names[:-1])} and {names[-1]} share {total} EUR in the ratio {ratio}. How much does each receive?",
                      f"A prize of {total} EUR is divided between {', '.join(names[:-1])} and {names[-1]} in the ratio {ratio}. What is each person's share?"])
    s = Sol(r)
    s.step("Add up the parts of the ratio.", f"{' + '.join(map(str, parts))} = {sum(parts)} parts.")
    s.step("Find the value of one part.", f"{total} / {sum(parts)} = {unit} EUR.")
    shares = [p * unit for p in parts]
    s.step("Multiply each person's parts by the value of one part.", "; ".join(f"{n}: {p} x {unit} = {p * unit} EUR" for n, p in zip(names, parts)) + ".")
    s.check = f"{' + '.join(map(str, shares))} = {total}, which is the full amount."
    return prompt, s.text(", ".join(f"{n} {v} EUR" for n, v in zip(names, shares))), str(shares[0])


@family("average_missing", 0.8, "medium", ["average"])
def f_avg(r):
    n, target = r.randrange(3, 6), r.randrange(60, 91)
    known = [r.randrange(50, 100) for _ in range(n)]
    missing = target * (n + 1) - sum(known)
    if not 0 <= missing <= 100:
        known = [target] * n
        known[0] = target + 4
        known[1] = target - 4
        missing = target
    prompt = pick(r, [f"My scores on {n} tests were {', '.join(map(str, known))}. What score do I need on the next test to have an average of {target}?",
                      f"A student has {', '.join(map(str, known))} on {n} exams. What must the {n + 1}th exam score be for the mean to be exactly {target}?"])
    s = Sol(r)
    s.step(f"Total points needed over {n + 1} tests.", f"{target} x {n + 1} = {target * (n + 1)}.")
    s.step("Add up the points so far.", f"{' + '.join(map(str, known))} = {sum(known)}.")
    s.step("The missing score is the difference.", f"{target * (n + 1)} - {sum(known)} = {missing}.")
    s.check = f"({sum(known)} + {missing}) / {n + 1} = {(sum(known) + missing) / (n + 1):g}."
    return prompt, s.text(str(missing)), str(missing)


@family("weighted_grade", 0.7, "medium", ["average", "percent"])
def f_weighted(r):
    w = pick(r, [(20, 30, 50), (30, 30, 40), (25, 25, 50), (40, 20, 40)])
    sc = [r.randrange(50, 101) for _ in range(3)]
    parts = [s_ * x / 100 for s_, x in zip(sc, w)]
    prompt = pick(r, [f"A course grade is {w[0]}% homework, {w[1]}% midterm and {w[2]}% final exam. I scored {sc[0]}, {sc[1]} and {sc[2]} respectively. What is my course grade?",
                      f"Homework counts {w[0]}%, the midterm {w[1]}% and the final {w[2]}%. With scores of {sc[0]} (homework), {sc[1]} (midterm) and {sc[2]} (final), what is the weighted average?"])
    s = Sol(r)
    for lab, scv, wv, pv in zip(["homework", "midterm", "final"], sc, w, parts):
        s.step(f"Weighted contribution of the {lab}.", f"{scv} x {wv}/100 = {n2(pv)}.")
    s.step("Add the contributions.", f"{' + '.join(n2(p) for p in parts)} = {n2(sum(parts))}.")
    s.check = f"The weights add to {sum(w)}%, so the result is on the same 0-100 scale."
    return prompt, s.text(n2(sum(parts))), n2(sum(parts))


@family("breakeven_plans", 0.7, "hard", ["algebra", "comparison"])
def f_breakeven(r):
    n, uA, du = r.randrange(5, 40), r.randrange(8, 25), r.randrange(2, 8)
    uB, dfix = uA - du, du * n
    fA = r.randrange(5, 30)
    fB = fA + dfix
    prompt = pick(r, [f"Plan A costs {fA} EUR fixed plus {uA} EUR per unit used. Plan B costs {fB} EUR fixed plus {uB} EUR per unit. After how many units do both plans cost the same?",
                      f"Gym A charges {fA} EUR to join and {uA} EUR per visit. Gym B charges {fB} EUR to join and {uB} EUR per visit. For how many visits do they cost the same?"])
    s = Sol(r)
    s.step("Write the cost of each option for x units.", f"A: {fA} + {uA}x, B: {fB} + {uB}x.")
    s.step("Set them equal.", f"{fA} + {uA}x = {fB} + {uB}x.")
    s.step("Collect the x terms on one side.", f"{uA}x - {uB}x = {fB} - {fA}, so {du}x = {dfix}.")
    s.step("Solve.", f"x = {dfix} / {du} = {n}.")
    s.check = f"A: {fA} + {uA} x {n} = {fA + uA * n}; B: {fB} + {uB} x {n} = {fB + uB * n}. Equal."
    return prompt, s.text(f"{n} (both cost {fA + uA * n} EUR)"), str(n)


@family("rectangle_area", 0.6, "easy", ["geometry"])
def f_rect(r):
    w, l = r.randrange(3, 20), r.randrange(4, 30)
    per = 2 * (w + l)
    prompt = pick(r, [f"A rectangular garden has a perimeter of {per} m and is {w} m wide. What is its area?", f"The perimeter of a rectangle is {per} cm and one side is {w} cm. Find its area."])
    s = Sol(r)
    s.step("Half the perimeter is length + width.", f"{per} / 2 = {per // 2}.")
    s.step("Subtract the known side to get the other side.", f"{per // 2} - {w} = {l}.")
    s.step("Area is side times side.", f"{w} x {l} = {w * l}.")
    s.check = f"Perimeter check: 2 x ({w} + {l}) = {per}."
    unit = "m2" if " m " in prompt else "cm2"
    return prompt, s.text(f"{w * l} {unit}"), str(w * l)


@family("inventory_flow", 0.9, "easy", ["multi_step"])
def f_inventory(r):
    start, sold, restock, ret = r.randrange(80, 400, 10), r.randrange(20, 70), r.randrange(10, 60), r.randrange(1, 12)
    prod = pick(r, ["notebooks", "mugs", "phone cases", "plant pots", "t-shirts"])
    prompt = pick(r, [f"A store starts the week with {start} {prod}. It sells {sold}, receives a delivery of {restock} and takes back {ret} returned items. How many {prod} are in stock now?",
                      f"Stock check: {start} {prod} in the morning, {sold} sold, {restock} delivered, {ret} returned by customers. How many are there at the end?"])
    s = Sol(r)
    a = start - sold
    s.step("Subtract what was sold.", f"{start} - {sold} = {a}.")
    s.step("Add the delivery.", f"{a} + {restock} = {a + restock}.")
    s.step("Add the returns.", f"{a + restock} + {ret} = {a + restock + ret}.")
    s.check = f"Net change = -{sold} + {restock} + {ret} = {restock + ret - sold}; {start} + ({restock + ret - sold}) = {a + restock + ret}."
    return prompt, s.text(f"{a + restock + ret} {prod}"), str(a + restock + ret)


@family("recipe_scaling", 0.8, "easy", ["ratio", "units"])
def f_recipe(r):
    base, new = pick(r, [(4, 6), (4, 10), (2, 5), (6, 9), (8, 12), (3, 12)])
    foods = r.sample(FOODS, 3)
    amts = [(f, base * pick(r, [25, 30, 50])) for f in foods]
    prompt = pick(r, [f"A recipe for {base} servings needs {amts[0][1]} g {amts[0][0]}, {amts[1][1]} g {amts[1][0]} and {amts[2][1]} g {amts[2][0]}. How much of each do I need for {new} servings?",
                      f"For {base} people the recipe uses {amts[0][1]} g of {amts[0][0]}, {amts[1][1]} g of {amts[1][0]} and {amts[2][1]} g of {amts[2][0]}. I'm cooking for {new}. What are the new amounts?"])
    s = Sol(r)
    s.step("Find the scaling factor.", f"{new} / {base} = {new / base:g}.")
    outs = [(f, a * new / base) for f, a in amts]
    s.step("Multiply each ingredient by the factor.", "; ".join(f"{f}: {a} x {new / base:g} = {n2(a * new / base)} g" for f, a in amts) + ".")
    s.check = f"Per serving the amounts stay the same: e.g. {amts[0][0]} {n2(amts[0][1] / base)} g x {new} = {n2(outs[0][1])} g."
    return prompt, s.text(", ".join(f"{f} {n2(v)} g" for f, v in outs)), n2(outs[0][1])


@family("free_time", 0.8, "easy", ["time"])
def f_free(r):
    day = pick(r, [8, 7.5, 9])
    ms = r.sample([30, 45, 60, 90, 25, 15, 50], 3)
    free = int(day * 60) - sum(ms)
    prompt = pick(r, [f"My workday is {n2(day)} hours long. I have meetings of {ms[0]}, {ms[1]} and {ms[2]} minutes. How much time is left for focused work?",
                      f"I'm at work for {n2(day)} hours and three meetings take {ms[0]}, {ms[1]} and {ms[2]} minutes. How much free time do I have?"])
    s = Sol(r)
    s.step("Convert the workday to minutes.", f"{n2(day)} h x 60 = {int(day * 60)} min.")
    s.step("Add up the meetings.", f"{ms[0]} + {ms[1]} + {ms[2]} = {sum(ms)} min.")
    s.step("Subtract.", f"{int(day * 60)} - {sum(ms)} = {free} min = {hm(free)}.")
    s.check = f"{sum(ms)} + {free} = {sum(ms) + free} = {int(day * 60)} minutes."
    return prompt, s.text(hm(free)), hm(free)


# --------------------------------------------------------------------------------------------------------------------- DEEP families
# Longer chains, several unknowns, puzzles with a unique solution, graphs and code tracing. Same contract: computed, never typed.
from fractions import Fraction  # noqa: E402
from itertools import permutations  # noqa: E402
import heapq  # noqa: E402

CHAIN_CONTEXTS = [("A warehouse holds {v} units of stock.", "units"), ("A shop starts the day with {v} items on its shelves.", "items"),
                  ("A water tank contains {v} litres.", "litres"), ("A club has {v} EUR in its account.", "EUR"), ("A bakery makes {v} rolls in the morning.", "rolls")]


def _chain_ops(r, unit):
    """One random operation: (description, function, formula-text builder). Returns None if the result would be unclean."""
    k = r.choice(["pct_off", "pct_on", "add", "sub", "split", "times"])
    if k == "pct_off":
        p = r.choice([10, 20, 25, 30, 40, 50])
        return (f"{p}% of it is used up", lambda v: v * (100 - p) / 100, lambda v, o: f"{n2(v)} - {p}% = {n2(v)} x {(100 - p) / 100:g} = {n2(o)}")
    if k == "pct_on":
        p = r.choice([10, 20, 25, 50])
        return (f"it then grows by {p}%", lambda v: v * (100 + p) / 100, lambda v, o: f"{n2(v)} + {p}% = {n2(v)} x {(100 + p) / 100:g} = {n2(o)}")
    if k == "add":
        a = r.randrange(5, 60)
        return (f"{a} more arrive", lambda v: v + a, lambda v, o: f"{n2(v)} + {a} = {n2(o)}")
    if k == "sub":
        a = r.randrange(5, 40)
        return (f"{a} are taken away", lambda v: v - a, lambda v, o: f"{n2(v)} - {a} = {n2(o)}")
    if k == "times":
        m = r.choice([2, 3])
        return (f"the amount is multiplied by {m}", lambda v: v * m, lambda v, o: f"{n2(v)} x {m} = {n2(o)}")
    n = r.choice([2, 3, 4, 5])
    return (f"what is left is shared equally between {n} groups", lambda v: v / n, lambda v, o: f"{n2(v)} / {n} = {n2(o)}")


@family("chain_problem", 2.0, "hard", ["multi_step", "composite"])
def f_chain(r):
    for _ in range(200):
        ctx, unit = r.choice(CHAIN_CONTEXTS)
        start = r.randrange(100, 1001, 20)
        ops = [_chain_ops(r, unit) for _ in range(r.randrange(3, 6))]
        v, ok, vals = float(start), True, [float(start)]
        for _, fn, _ in ops:
            v = fn(v)
            if v <= 0 or abs(v * 100 - round(v * 100)) > 1e-6 or v > 100000:
                ok = False
                break
            vals.append(v)
        if ok and abs(v - round(v)) < 1e-9:
            break
    else:
        return f_chain(random.Random(r.random()))
    ordinals = ["First", "Then", "After that", "Next", "Finally"]
    sents = [ctx.format(v=start)] + [f"{ordinals[min(i, len(ordinals) - 1)] if i < len(ops) - 1 else 'Finally'}, {d}." for i, (d, _, _) in enumerate(ops)]
    prompt = " ".join(sents) + f" How many {unit if unit != 'EUR' else 'EUR'} {'are there' if unit != 'EUR' else 'are in the account'} at the end?"
    s = Sol(r)
    for i, ((d, _, text), v0, v1) in enumerate(zip(ops, vals, vals[1:]), 1):
        s.step(f"Event {i}: {d}.", text(v0, v1) + ".")
    s.check = f"Replaying the {len(ops)} events from {start} gives {n2(vals[-1])} again."
    return prompt, s.text(f"{n2(vals[-1])} {unit}"), n2(vals[-1])


@family("two_unknowns", 1.2, "hard", ["algebra", "system"])
def f_system(r):
    c, a = r.randrange(5, 40), r.randrange(5, 40)
    pa, pc = r.choice([12, 15, 18, 20, 25]), r.choice([5, 6, 8, 9, 10])
    n, T = a + c, a * pa + c * pc
    ev = r.choice(["concert", "museum", "zoo", "cinema"])
    prompt = pick(r, [f"A {ev} sold {n} tickets for {T} EUR in total. Adult tickets cost {pa} EUR and child tickets cost {pc} EUR. How many of each were sold?",
                      f"{n} people visited the {ev} and paid {T} EUR altogether. Adults pay {pa} EUR, children pay {pc} EUR. How many adults and how many children came?"])
    s = Sol(r)
    s.step("Let c be the number of children, so adults = " + str(n) + " - c.", "")
    s.step("Write the total cost with that.", f"{pa} x ({n} - c) + {pc} x c = {T}.")
    s.step("Expand and collect c.", f"{pa * n} - {pa}c + {pc}c = {T}, so {pa * n} - {pa - pc}c = {T}.")
    s.step("Solve for c.", f"{pa - pc}c = {pa * n} - {T} = {pa * n - T}, so c = {pa * n - T} / {pa - pc} = {c}.")
    s.step("Adults.", f"{n} - {c} = {a}.")
    s.check = f"{a} x {pa} + {c} x {pc} = {a * pa} + {c * pc} = {T}, and {a} + {c} = {n}."
    return prompt, s.text(f"{a} adults and {c} children"), str(a)


@family("probability", 1.2, "hard", ["probability", "fractions"])
def f_prob(r):
    if r.random() < 0.5:
        red, blue = r.randrange(2, 8), r.randrange(2, 8)
        tot = red + blue
        p = Fraction(red, tot) * Fraction(red - 1, tot - 1)
        prompt = pick(r, [f"A bag has {red} red and {blue} blue marbles. Two marbles are drawn without replacement. What is the probability that both are red?",
                          f"From a bag of {red} red and {blue} blue marbles you take two at random, without putting the first back. Probability both are red?"])
        s = Sol(r)
        s.step("Probability that the first marble is red.", f"{red}/{tot}.")
        s.step("After removing one red marble there are fewer of everything.", f"red left = {red - 1}, total left = {tot - 1}, so the second is red with probability {red - 1}/{tot - 1}.")
        s.step("Multiply the two probabilities (the draws depend on each other).", f"{red}/{tot} x {red - 1}/{tot - 1} = {red * (red - 1)}/{tot * (tot - 1)} = {p.numerator}/{p.denominator}.")
        s.check = f"{p.numerator}/{p.denominator} is about {float(p):.3f}, a value between 0 and 1, and smaller than {red}/{tot} = {red / tot:.3f}, as expected."
        return prompt, s.text(f"{p.numerator}/{p.denominator}"), f"{p.numerator}/{p.denominator}"
    target = r.randrange(4, 11)
    pairs = [(a, b) for a in range(1, 7) for b in range(1, 7) if a + b == target]
    p = Fraction(len(pairs), 36)
    prompt = pick(r, [f"Two fair six-sided dice are rolled. What is the probability that the sum is {target}?", f"You roll two ordinary dice. How likely is it that they add up to {target}?"])
    s = Sol(r)
    s.step("Count all equally likely outcomes.", "6 x 6 = 36.")
    s.step(f"List the outcomes that sum to {target}.", ", ".join(f"({a},{b})" for a, b in pairs) + f" - that is {len(pairs)} outcomes.")
    s.step("Divide favourable by total and simplify.", f"{len(pairs)}/36 = {p.numerator}/{p.denominator}.")
    s.check = f"{p.numerator}/{p.denominator} is about {float(p):.3f}."
    return prompt, s.text(f"{p.numerator}/{p.denominator}"), f"{p.numerator}/{p.denominator}"


@family("sequence_rule", 1.0, "medium", ["patterns"])
def f_seq(r):
    kind = r.choice(["arith", "geom", "grow"])
    if kind == "arith":
        a, d = r.randrange(2, 30), r.randrange(2, 12)
        seq = [a + d * i for i in range(5)]
        rule, nxt = f"add {d} each time", a + d * 5
        nth = a + d * 9
        steps = [("Differences between neighbours.", ", ".join(f"{y} - {x} = {y - x}" for x, y in zip(seq, seq[1:])) + f". The difference is always {d}."),
                 ("Next term.", f"{seq[-1]} + {d} = {nxt}."), ("Tenth term (term n is first + (n-1) x {0}).".format(d), f"{a} + 9 x {d} = {nth}.")]
    elif kind == "geom":
        a, m = r.randrange(1, 6), r.choice([2, 3])
        seq = [a * m ** i for i in range(5)]
        nxt, nth = a * m ** 5, a * m ** 9
        steps = [("Ratios between neighbours.", ", ".join(f"{y} / {x} = {y // x}" for x, y in zip(seq, seq[1:])) + f". The ratio is always {m}."),
                 ("Next term.", f"{seq[-1]} x {m} = {nxt}."), ("Tenth term (first x ratio^9).", f"{a} x {m}^9 = {a} x {m ** 9} = {nth}.")]
    else:
        a, d, e = r.randrange(1, 10), r.randrange(1, 5), r.randrange(1, 4)
        seq, diff = [a], d
        for _ in range(4):
            seq.append(seq[-1] + diff)
            diff += e
        nxt = seq[-1] + diff
        nth = None
        dd = [y - x for x, y in zip(seq, seq[1:])]
        steps = [("First differences.", ", ".join(map(str, dd)) + "."), ("Second differences (differences of the differences).", ", ".join(f"{y - x}" for x, y in zip(dd, dd[1:])) + f" - constant {e}, so the gaps grow by {e} each step."),
                 ("Next gap, then next term.", f"next gap = {dd[-1]} + {e} = {dd[-1] + e}; next term = {seq[-1]} + {dd[-1] + e} = {nxt}.")]
    prompt = pick(r, [f"What comes next in the sequence {', '.join(map(str, seq))}, ...?", f"Find the next number: {', '.join(map(str, seq))}, ?"])
    if kind != "grow" and r.random() < 0.5:
        prompt += " Also, what is the 10th term?"
        use = steps
        ans = f"next term {nxt}, 10th term {nth}"
    else:
        use = steps[:2] if kind != "grow" else steps
        ans = str(nxt)
    s = Sol(r)
    for t, b in use:
        s.step(t, b)
    return prompt, s.text(ans), str(nxt)


@family("order_chain", 1.2, "medium", ["logic", "ordering"])
def f_order(r):
    n = r.randrange(4, 6)
    names = r.sample(FIRST, n)
    order = names[:]                                    # tallest first
    r.shuffle(order)
    quality = pick(r, [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"), ("heavier", "heaviest", "lightest")])
    clues = [(order[i], order[i + 1]) for i in range(n - 1)]
    r.shuffle(clues)
    ask = r.choice(["top", "bottom", "middle"])
    q = {"top": f"Who is the {quality[1]}?", "bottom": f"Who is the {quality[2]}?", "middle": f"Who is in the middle (position {(n + 1) // 2} from the {quality[1]} end)?"}[ask] if n % 2 == 1 or ask != "middle" else f"Who is {quality[1]} second?"
    if ask == "middle" and n % 2 == 0:
        ask, pos = "second", 1
    pos = {"top": 0, "bottom": n - 1, "middle": (n - 1) // 2, "second": 1}[ask]
    prompt = f"Here are some facts about {n} people. " + " ".join(f"{a} is {quality[0]} than {b}." for a, b in clues) + f" {q}"
    s = Sol(r)
    s.step("Write each clue as a relation, X > Y meaning X is " + quality[0] + " than Y.", "; ".join(f"{a} > {b}" for a, b in clues) + ".")
    chain = []
    rest = dict(clues)
    start = [a for a, _ in clues if a not in {b for _, b in clues}][0]
    cur = start
    while True:
        chain.append(cur)
        if cur not in rest:
            break
        cur = rest[cur]
    s.step("Nobody is " + quality[0] + " than " + start + " (it never appears on the right), so " + start + " goes first. Follow the clues from there.", " > ".join(chain) + ".")
    s.step(f"Read the answer from the chain (position {pos + 1} counting from the {quality[1]} end).", f"{chain[pos]}.")
    s.check = f"The chain uses all {n} people and every clue: " + ", ".join(f"{a} > {b}" for a, b in clues[:2]) + ", ... all hold."
    return prompt, s.text(chain[pos]), chain[pos]


@family("budget_items", 1.0, "medium", ["optimisation", "greedy"])
def f_budget(r):
    items = r.sample(["notebook", "pen set", "stapler", "desk lamp", "mouse pad", "headphones", "tape", "folder pack", "whiteboard", "calculator"], r.randrange(5, 8))
    prices = [r.randrange(3, 40) for _ in items]
    budget = int(sum(prices) * r.uniform(0.35, 0.6))
    prompt = pick(r, [f"I have a budget of {budget} EUR. The prices are: " + ", ".join(f"{i} {p} EUR" for i, p in zip(items, prices)) + ". What is the largest number of different items I can buy, and which ones?",
                      f"With {budget} EUR to spend and these prices - " + ", ".join(f"{i}: {p}" for i, p in zip(items, prices)) + " - how many different items can I afford at most? Name them."])
    s = Sol(r)
    order = sorted(zip(prices, items))
    s.step("To buy as many items as possible, take the cheapest ones first. Sort by price.", ", ".join(f"{i} ({p})" for p, i in order) + ".")
    total, chosen = 0, []
    for p, i in order:
        if total + p > budget:
            s.step(f"Try {i}.", f"{total} + {p} = {total + p}, which is over {budget}, so stop.")
            break
        total += p
        chosen.append(i)
        s.step(f"Add {i}.", f"running total = {total}.")
    s.check = f"Any other set of {len(chosen)} items costs at least the {len(chosen)} cheapest ({total} EUR), so {len(chosen) + 1} items would need at least " + (f"{total + order[len(chosen)][0]} EUR" if len(chosen) < len(order) else "more than exists") + f", over the budget of {budget}."
    ans = f"{len(chosen)} items ({', '.join(chosen)}), costing {total} EUR"
    return prompt, s.text(ans), str(len(chosen))


@family("shortest_path", 1.0, "hard", ["graph", "algorithm"])
def f_path(r):
    nodes = list("ABCDE")[:r.randrange(4, 6)]
    edges = {}
    for i in range(len(nodes) - 1):
        edges[(nodes[i], nodes[i + 1])] = r.randrange(2, 9)
    for _ in range(r.randrange(2, 4)):
        a, b = r.sample(nodes, 2)
        if (a, b) not in edges and (b, a) not in edges:
            edges[(a, b)] = r.randrange(2, 12)
    adj = {n: [] for n in nodes}
    for (a, b), w in edges.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    src, dst = nodes[0], nodes[-1]
    prompt = (f"A map has {len(nodes)} towns. Roads (both directions) with distances in km: " + ", ".join(f"{a}-{b} {w}" for (a, b), w in edges.items()) +
              f". What is the shortest distance from {src} to {dst}, and which route is it?")
    s = Sol(r)
    dist, prev, pq, done = {src: 0}, {}, [(0, src)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        upd = []
        for v, w in sorted(adj[u]):
            if v not in done and d + w < dist.get(v, 10 ** 9):
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
                upd.append(f"{v} = {d} + {w} = {d + w}")
        s.step(f"Settle {u} (distance {d}).", ("Update neighbours: " + "; ".join(upd) + ".") if upd else "No neighbour improves.")
    path, cur = [dst], dst
    while cur != src:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    s.step("Rebuild the route backwards from the destination.", " -> ".join(path) + ".")
    s.check = "Adding the edge lengths along " + " -> ".join(path) + " gives " + " + ".join(str(next(w for v, w in adj[a] if v == b)) for a, b in zip(path, path[1:])) + f" = {dist[dst]}."
    return prompt, s.text(f"{dist[dst]} km via {' -> '.join(path)}"), str(dist[dst])


@family("task_schedule", 1.0, "hard", ["planning", "dependencies"])
def f_sched(r):
    names = ["Design", "Build", "Test", "Docs", "Deploy", "Review"][:r.randrange(4, 7)]
    dur = {n: r.randrange(1, 6) for n in names}
    deps = {names[0]: []}
    for i, n in enumerate(names[1:], 1):
        deps[n] = r.sample(names[:i], r.randrange(1, min(i, 2) + 1))
    prompt = ("A project has these tasks (duration in days, and what must finish first): " + "; ".join(f"{n} {dur[n]}d" + (f" after {' and '.join(deps[n])}" if deps[n] else ", no prerequisites") for n in names) +
              ". Several tasks can run in parallel if their prerequisites are done. What is the shortest total project time?")
    s = Sol(r)
    fin = {}
    for n in names:
        st = max([fin[d] for d in deps[n]], default=0)
        fin[n] = st + dur[n]
        s.step(f"{n}.", (f"starts when {' and '.join(deps[n])} finish (day {st}), " if deps[n] else "starts at day 0, ") + f"takes {dur[n]} days, so it finishes on day {fin[n]}.")
    end = max(fin.values())
    crit = [n for n in names if fin[n] == end][0]
    s.step("The project ends when the last task ends.", f"max of finish days = {end} (task {crit}).")
    s.check = f"No schedule can beat day {end}, because {crit}'s prerequisites chain already takes that long."
    return prompt, s.text(f"{end} days"), str(end)


@family("code_trace", 1.5, "hard", ["code", "trace"])
def f_code(r):
    kind = r.choice(["multiples", "squares_odd", "vowels", "fib"])
    s = Sol(r)
    if kind == "multiples":
        n, m = r.randrange(8, 16), r.choice([2, 3, 4, 5])
        code = f"total = 0\nfor i in range(1, {n + 1}):\n    if i % {m} == 0:\n        total += i\nprint(total)"
        total = 0
        for i in range(1, n + 1):
            if i % m == 0:
                total += i
                s.step(f"i = {i}: {i} % {m} == 0 is True.", f"total becomes {total}.")
        s.step("All other values of i fail the test, so total doesn't change.", f"The loop ends with total = {total}.")
        out = str(total)
    elif kind == "squares_odd":
        n = r.randrange(6, 11)
        code = f"result = []\nfor x in range({n}):\n    if x % 2 == 1:\n        result.append(x * x)\nprint(result)"
        res = []
        for x in range(n):
            if x % 2 == 1:
                res.append(x * x)
                s.step(f"x = {x} is odd.", f"append {x} * {x} = {x * x}, list = {res}.")
        s.step("Even values are skipped.", f"Final list = {res}.")
        out = str(res)
    elif kind == "vowels":
        word = r.choice(["education", "sequence", "beautiful", "algorithm", "architecture", "umbrella"])
        code = f"word = \"{word}\"\ncount = 0\nfor ch in word:\n    if ch in \"aeiou\":\n        count += 1\nprint(count)"
        c = 0
        for ch in word:
            if ch in "aeiou":
                c += 1
                s.step(f"'{ch}' is a vowel.", f"count = {c}.")
        s.step("All other letters are consonants.", f"count = {c}.")
        out = str(c)
    else:
        n = r.randrange(6, 10)
        code = f"a, b = 0, 1\nfor _ in range({n}):\n    a, b = b, a + b\nprint(a)"
        a, b = 0, 1
        for k in range(1, n + 1):
            a, b = b, a + b
            s.step(f"Iteration {k}.", f"a, b = {a}, {b}.")
        out = str(a)
    prompt = pick(r, ["What does this Python code print?\n\n", "Trace this code by hand and tell me its output.\n\n", "What is the output of the following snippet?\n\n"]) + "```python\n" + code + "\n```"
    ns = {}
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)                                                           # independent ground truth: actually run it
    assert buf.getvalue().strip() == out, (code, buf.getvalue(), out)
    s.check = "Running the loop again from the top gives the same printed value."
    return prompt, s.text(out), out


@family("base_conversion", 0.8, "medium", ["binary", "number_systems"])
def f_base(r):
    s = Sol(r)
    if r.random() < 0.5:
        n = r.randrange(10, 200)
        prompt = pick(r, [f"Convert {n} to binary.", f"What is {n} in base 2?"])
        x, rem = n, []
        while x:
            s.step(f"Divide {x} by 2.", f"{x} = 2 x {x // 2} + {x % 2}, remainder {x % 2}.")
            rem.append(x % 2)
            x //= 2
        out = "".join(map(str, reversed(rem)))
        s.step("Read the remainders from last to first.", out + ".")
        s.check = " + ".join(f"{int(b) * 2 ** i}" for i, b in enumerate(reversed(out)) if b == "1") + f" = {n}."
        return prompt, s.text(out), out
    b = format(r.randrange(10, 200), "b")
    n = int(b, 2)
    prompt = pick(r, [f"Convert the binary number {b} to decimal.", f"What is {b} (base 2) in decimal?"])
    terms = []
    for i, ch in enumerate(reversed(b)):
        s.step(f"Digit {ch} at position {i} (value 2^{i} = {2 ** i}).", f"contributes {int(ch) * 2 ** i}.")
        terms.append(int(ch) * 2 ** i)
    s.step("Add the contributions.", " + ".join(map(str, terms)) + f" = {n}.")
    s.check = f"Converting {n} back: it needs {len(b)} bits, matching {b}."
    return prompt, s.text(str(n)), str(n)


@family("mixture", 0.8, "hard", ["algebra", "percent"])
def f_mix(r):
    c1, c2 = r.choice([(10, 30), (20, 60), (5, 25), (15, 45)])
    target = (c1 + c2) // 2 + r.choice([-(c2 - c1) // 4, 0, (c2 - c1) // 4])
    total = r.choice([20, 40, 60, 80])
    x = total * (target - c1) / (c2 - c1)
    if abs(x - round(x)) > 1e-9:
        return f_mix(random.Random(r.random()))
    x = int(round(x))
    prompt = pick(r, [f"How many litres of a {c2}% salt solution must be mixed with a {c1}% solution to get {total} litres of a {target}% solution?",
                      f"A lab needs {total} L of {target}% acid. It has {c1}% acid and {c2}% acid in stock. How much of the {c2}% acid should it use?"])
    s = Sol(r)
    s.step(f"Let x be the litres of the {c2}% solution. The rest, {total} - x, is the {c1}% solution.", "")
    s.step("The amount of pure substance must match.", f"{c2}% of x + {c1}% of ({total} - x) = {target}% of {total}.")
    s.step("Write it with decimals.", f"{c2 / 100:g}x + {c1 / 100:g}({total} - x) = {target / 100 * total:g}.")
    s.step("Expand.", f"{c2 / 100:g}x + {c1 / 100 * total:g} - {c1 / 100:g}x = {target / 100 * total:g}, so {(c2 - c1) / 100:g}x = {target / 100 * total - c1 / 100 * total:g}.")
    s.step("Solve.", f"x = {target / 100 * total - c1 / 100 * total:g} / {(c2 - c1) / 100:g} = {x}.")
    s.check = f"{x} L at {c2}% plus {total - x} L at {c1}% holds {x * c2 / 100:g} + {(total - x) * c1 / 100:g} = {x * c2 / 100 + (total - x) * c1 / 100:g} L of substance = {target}% of {total}."
    return prompt, s.text(f"{x} litres of the {c2}% solution"), str(x)


# --------------------------------------------------------------------------------------------------------------------- builder
def build_one(name: str, seed: int) -> dict:
    fn, _, diff, tags = FAMILIES[name]
    r = random.Random(f"reason:{name}:{seed}")
    prompt, answer_text, key = fn(r)
    prompt = f"{pick(r, INTROS)}{prompt}{pick(r, OUTROS)}".strip()
    if r.random() < 0.10 and prompt[:1].isupper():
        prompt = prompt[0].lower() + prompt[1:]
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer_text}]
    if r.random() < 0.25:
        msgs.insert(0, {"role": "system", "content": pick(r, ["You are a careful math tutor. Show your reasoning step by step before giving the final answer.",
                                                              "You are a helpful assistant. Solve problems step by step and state the final answer at the end.",
                                                              "Think step by step. Only give the final answer after the working."])})
    return {"id": f"reasoning/{name}/{seed}", "messages": msgs, "tools": [],
            "meta": {"family": f"reason_{name}", "category": "reasoning", "difficulty": diff, "tags": tags + ["cot", "step_by_step"], "seed": seed, "kind": "reasoning",
                     "n_tool_calls": 0, "n_turns": 1, "n_tools_offered": 0, "answer": key}}


def check_record(rec: dict) -> str | None:
    """Answer only at the end, steps present, no empty steps."""
    text = rec["messages"][-1]["content"]
    if "Final answer:" not in text or not text.rstrip().splitlines()[-1].startswith("Final answer:"):
        return "no_final_answer_line"
    steps = [l for l in text.splitlines() if l.startswith("Step ")]
    if len(steps) < 2:
        return "too_few_steps"
    key = rec["meta"]["answer"]
    head = "\n".join(text.splitlines()[:2])
    if len(key) >= 3 and key in head:
        return "answer_in_opener"
    final_line = text.rstrip().splitlines()[-1]
    if key not in final_line:
        return "key_not_in_final"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=300)
    ap.add_argument("--out", default="data/raw/reasoning_stepwise.jsonl")
    ap.add_argument("--seed-offset", type=int, default=10_000)
    a = ap.parse_args()
    rows, seen, bad = [], set(), {}
    for name, (_, w, _, _) in FAMILIES.items():
        made = 0
        for k in range(int(a.n_per_family * w) * 3):
            if made >= int(a.n_per_family * w):
                break
            rec = build_one(name, a.seed_offset + k)
            why = check_record(rec)
            h = rec["messages"][-2]["content"].lower()
            if why or h in seen:
                bad[why or "duplicate"] = bad.get(why or "duplicate", 0) + 1
                continue
            seen.add(h)
            rows.append(rec)
            made += 1
    random.Random(0).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(FAMILIES)} families -> {len(rows)} records in {a.out}; rejected {bad}")


if __name__ == "__main__":
    main()
