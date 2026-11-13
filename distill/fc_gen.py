"""BFCL-style function-calling data: arbitrary typed functions, chosen and called correctly (the skill the BFCL benchmark measures).

    python -m distill.fc_gen --n 8000 --out data/raw/fc_bfcl_style.jsonl

Our own function library (≈65 functions over math, geometry, finance, units, text, dates, lists, statistics, weather, travel, shopping, maps,
dev tools ...; none copied from BFCL) with hand-written request templates. Every example is built FROM the arguments, so the expected call is
known exactly. Categories, mirroring what BFCL scores:
  simple              one function offered, one call
  multiple            2-5 functions offered, pick the right one
  parallel            same function, several calls in ONE assistant message
  parallel_multiple   different functions, several calls in one message
  irrelevance         no offered function fits -> answer directly / say so, call nothing
  missing_param       a required argument is not given -> ask for it instead of guessing
  loop                call -> real result (computed for math/text/date functions, deterministic simulation for lookups) -> grounded final answer
  two_turn            two requests in one conversation (tools stay available), each answered from its result
Half of the single-call examples end at the call itself (exactly what the benchmark asks the model to produce); the rest continue with the result.
Records carry the tool schemas in the `tools` field, so the model's own chat template renders them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from datetime import date, timedelta
from pathlib import Path
from statistics import median, pstdev

FUNCS: dict = {}
S, I, N, B, A = "string", "integer", "number", "boolean", "array"


def h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def fmt(v) -> str:
    if isinstance(v, list):
        items = [fmt(x) for x in v]
        return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


O = "object"


def P(name, typ, desc, req=True, enum=None, items=None, default=None, props=None, view=None):
    return {"name": name, "type": typ, "desc": desc, "req": req, "enum": enum, "items": items, "default": default, "props": props, "view": view}


def fn(name, desc, params, asks, sample, run=None, say=None, sim=None, domain=""):
    used = [set(re.findall(r"{(\w+)}", t)) for t in asks]
    req = {p["name"] for p in params if p["req"]}
    assert any(req <= u for u in used), name
    FUNCS[name] = dict(name=name, desc=desc, params=params, asks=asks, used=used, sample=sample, run=run, say=say, sim=sim, domain=domain or name.split(".")[0])


def schema(f) -> dict:
    props, req = {}, []
    for p in f["params"]:
        d = {"type": p["type"], "description": p["desc"] + (f" Default: {p['default']}." if p["default"] is not None else "")}
        if p["enum"]:
            d["enum"] = p["enum"]
        if p["type"] == A:
            d["items"] = {"type": p["items"] or S}
        if p["type"] == O:
            d["properties"] = {k: {"type": v} for k, v in p["props"].items()}
            d["required"] = list(p["props"])
        props[p["name"]] = d
        if p["req"]:
            req.append(p["name"])
    return {"type": "function", "function": {"name": f["name"], "description": f["desc"], "parameters": {"type": "object", "properties": props, "required": req}}}


CITIES = ["Oslo", "Lisbon", "Turin", "Zurich", "Lyon", "Porto", "Graz", "Ghent", "Malmo", "Utrecht", "Bilbao", "Krakow", "Vienna", "Dublin", "Athens", "Prague"]
FIRST = ["Ana", "Marco", "Priya", "Jonas", "Sofia", "Liam", "Aisha", "Diego", "Mei", "Oscar"]


def R(a, b):
    return lambda r: r.randrange(a, b + 1)


# ------------------------------------------------------------------------------------------------------------------------------- math
fn("math.factorial", "Compute the factorial of a non-negative integer.", [P("number", I, "The integer whose factorial is wanted.")],
   ["What is {number} factorial?", "Compute the factorial of {number}.", "Can you work out {number}! for me?"], lambda r: {"number": r.randrange(3, 13)},
   run=lambda a: math.factorial(a["number"]), say=lambda a, x: f"{a['number']}! = {x}.")
fn("math.gcd", "Return the greatest common divisor of two integers.", [P("a", I, "First integer."), P("b", I, "Second integer.")],
   ["What is the greatest common divisor of {a} and {b}?", "Find the GCD of {a} and {b}.", "gcd of {a} and {b}?"], lambda r: {"a": r.randrange(12, 200), "b": r.randrange(12, 200)},
   run=lambda a: math.gcd(a["a"], a["b"]), say=lambda a, x: f"The greatest common divisor of {a['a']} and {a['b']} is {x}.")
fn("math.lcm", "Return the least common multiple of two integers.", [P("a", I, "First integer."), P("b", I, "Second integer.")],
   ["What is the least common multiple of {a} and {b}?", "Find the LCM of {a} and {b}.", "lcm({a}, {b})?"], lambda r: {"a": r.randrange(2, 40), "b": r.randrange(2, 40)},
   run=lambda a: math.lcm(a["a"], a["b"]), say=lambda a, x: f"The least common multiple of {a['a']} and {a['b']} is {x}.")
fn("math.is_prime", "Check whether an integer is a prime number.", [P("number", I, "The integer to test.")],
   ["Is {number} a prime number?", "Check if {number} is prime.", "Tell me whether {number} is prime."], lambda r: {"number": r.randrange(10, 300)},
   run=lambda a: all(a["number"] % d for d in range(2, int(a["number"] ** 0.5) + 1)),
   say=lambda a, x: f"Yes, {a['number']} is prime." if x else f"No, {a['number']} is not prime.")
fn("math.power", "Raise a base to an integer exponent.", [P("base", N, "The base."), P("exponent", I, "The exponent.")],
   ["What is {base} to the power of {exponent}?", "Calculate {base} raised to {exponent}.", "Compute {base}^{exponent}."], lambda r: {"base": r.randrange(2, 12), "exponent": r.randrange(2, 7)},
   run=lambda a: a["base"] ** a["exponent"], say=lambda a, x: f"{a['base']} to the power of {a['exponent']} is {x}.")
fn("math.percentage", "Compute what percentage one number is of another.", [P("part", N, "The part."), P("whole", N, "The whole.")],
   ["What percentage is {part} of {whole}?", "{part} is what percent of {whole}?", "Work out {part} as a percentage of {whole}."], lambda r: {"part": r.randrange(1, 50) * 2, "whole": r.choice([100, 200, 250, 400, 500])},
   run=lambda a: round(a["part"] / a["whole"] * 100, 2), say=lambda a, x: f"{fmt(a['part'])} is {x:g}% of {fmt(a['whole'])}.")
fn("math.average", "Compute the arithmetic mean of a list of numbers.", [P("numbers", A, "The numbers to average.", items=N)],
   ["What is the average of {numbers}?", "Compute the mean of {numbers}.", "Find the average value of these numbers: {numbers}."], lambda r: {"numbers": [r.randrange(1, 100) for _ in range(r.randrange(3, 7))]},
   run=lambda a: round(sum(a["numbers"]) / len(a["numbers"]), 2), say=lambda a, x: f"The average of {fmt(a['numbers'])} is {x:g}.")
fn("math.median", "Compute the median of a list of numbers.", [P("numbers", A, "The numbers.", items=N)],
   ["What is the median of {numbers}?", "Find the median value of {numbers}."], lambda r: {"numbers": [r.randrange(1, 100) for _ in range(r.randrange(3, 8))]},
   run=lambda a: median(a["numbers"]), say=lambda a, x: f"The median of {fmt(a['numbers'])} is {x:g}.")
fn("math.solve_quadratic", "Solve a x^2 + b x + c = 0 and return the real roots in ascending order.", [P("a", N, "Coefficient of x^2."), P("b", N, "Coefficient of x."), P("c", N, "Constant term.")],
   ["Solve the quadratic equation {a}x^2 + {b}x + {c} = 0.", "Find the roots of {a}x^2 + {b}x + {c}."], lambda r: (lambda p, q: {"a": 1, "b": -(p + q), "c": p * q})(r.randrange(-6, 7), r.randrange(-6, 7)),
   run=lambda a: sorted({(-a["b"] + s * math.sqrt(a["b"] ** 2 - 4 * a["a"] * a["c"])) / (2 * a["a"]) for s in (1, -1)}), say=lambda a, x: "The roots are " + " and ".join(f"{v:g}" for v in x) + ".")
fn("math.fibonacci", "Return the n-th Fibonacci number (F1 = 1, F2 = 1).", [P("n", I, "Position in the sequence (at least 1).")],
   ["What is the {n}th Fibonacci number?", "Give me Fibonacci number {n}."], lambda r: {"n": r.randrange(5, 31)},
   run=lambda a: (lambda f: f(f, a["n"]))(lambda f, n, m={}: 1 if n <= 2 else (m.setdefault(n, f(f, n - 1) + f(f, n - 2)))) if a["n"] <= 22 else _fib(a["n"]),
   say=lambda a, x: f"The {a['n']}th Fibonacci number is {x}.")


def _fib(n):
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


fn("math.to_binary", "Convert a non-negative integer to its binary representation.", [P("number", I, "The integer to convert.")],
   ["Convert {number} to binary.", "What is {number} in base 2?"], lambda r: {"number": r.randrange(5, 500)},
   run=lambda a: format(a["number"], "b"), say=lambda a, x: f"{a['number']} in binary is {x}.")

# ------------------------------------------------------------------------------------------------------------------------------- geometry
fn("geometry.circle_area", "Calculate the area of a circle from its radius.", [P("radius", N, "Radius of the circle.")],
   ["What is the area of a circle with radius {radius}?", "Calculate the area of a circle whose radius is {radius}.", "Find the area of a circle of radius {radius}."], lambda r: {"radius": r.randrange(1, 20)},
   run=lambda a: round(math.pi * a["radius"] ** 2, 2), say=lambda a, x: f"A circle with radius {fmt(a['radius'])} has an area of about {x:g}.")
fn("geometry.rectangle_perimeter", "Calculate the perimeter of a rectangle.", [P("width", N, "Width."), P("height", N, "Height.")],
   ["What is the perimeter of a rectangle that is {width} wide and {height} high?", "Find the perimeter of a {width} by {height} rectangle."], lambda r: {"width": r.randrange(2, 30), "height": r.randrange(2, 30)},
   run=lambda a: 2 * (a["width"] + a["height"]), say=lambda a, x: f"The perimeter of a {fmt(a['width'])} by {fmt(a['height'])} rectangle is {x:g}.")
fn("geometry.triangle_area", "Calculate the area of a triangle from its base and height.", [P("base", N, "Base length."), P("height", N, "Height."), P("unit", S, "Unit of length.", req=False, enum=["cm", "m", "in"], default="m")],
   ["Find the area of a triangle with base {base} and height {height}.", "What's the area of a triangle with a base of {base} {unit} and a height of {height} {unit}?"], lambda r: {"base": r.randrange(2, 30), "height": r.randrange(2, 30), "unit": r.choice(["cm", "m", "in"])},
   run=lambda a: a["base"] * a["height"] / 2, say=lambda a, x: f"The area is {x:g} square {a.get('unit', 'units')}.")
fn("geometry.sphere_volume", "Calculate the volume of a sphere from its radius.", [P("radius", N, "Radius of the sphere.")],
   ["What is the volume of a sphere with radius {radius}?", "Compute the volume of a ball of radius {radius}."], lambda r: {"radius": r.randrange(1, 15)},
   run=lambda a: round(4 / 3 * math.pi * a["radius"] ** 3, 2), say=lambda a, x: f"The volume is about {x:g}.")
fn("geometry.hypotenuse", "Calculate the hypotenuse of a right triangle from its two legs.", [P("a", N, "First leg."), P("b", N, "Second leg.")],
   ["What is the hypotenuse of a right triangle with legs {a} and {b}?", "Find the longest side of a right triangle whose other sides are {a} and {b}."], lambda r: {"a": r.choice([3, 5, 6, 8, 9, 12]), "b": r.choice([4, 12, 8, 15, 12, 16])},
   run=lambda a: round(math.hypot(a["a"], a["b"]), 2), say=lambda a, x: f"The hypotenuse is {x:g}.")

# ------------------------------------------------------------------------------------------------------------------------------- finance
fn("finance.compound_interest", "Compute the final amount of an investment with compound interest.", [P("principal", N, "Initial amount."), P("rate_percent", N, "Annual interest rate in percent."), P("years", I, "Number of years."),
                                                                                                    P("compounds_per_year", I, "Compounding periods per year.", req=False, default=1)],
   ["How much will {principal} be worth after {years} years at {rate_percent}% annual interest?", "I invest {principal} at {rate_percent}% for {years} years compounded {compounds_per_year} times a year. What is the final amount?"],
   lambda r: {"principal": r.randrange(1, 20) * 500, "rate_percent": r.choice([2, 3, 4, 5, 6]), "years": r.randrange(2, 15), "compounds_per_year": r.choice([1, 2, 4, 12])},
   run=lambda a: round(a["principal"] * (1 + a["rate_percent"] / 100 / a.get("compounds_per_year", 1)) ** (a["years"] * a.get("compounds_per_year", 1)), 2),
   say=lambda a, x: f"After {a['years']} years the investment would be worth about {x:,.2f}.")
fn("finance.loan_payment", "Compute the fixed monthly payment of a loan.", [P("principal", N, "Loan amount."), P("annual_rate_percent", N, "Annual interest rate in percent."), P("months", I, "Number of monthly payments.")],
   ["What is the monthly payment on a loan of {principal} at {annual_rate_percent}% annual interest over {months} months?", "Calculate the monthly installment for borrowing {principal} for {months} months at {annual_rate_percent}% a year."],
   lambda r: {"principal": r.randrange(5, 60) * 1000, "annual_rate_percent": r.choice([3, 4.5, 5, 6, 7.5]), "months": r.choice([12, 24, 36, 48, 60])},
   run=lambda a: round(a["principal"] * (a["annual_rate_percent"] / 1200) / (1 - (1 + a["annual_rate_percent"] / 1200) ** -a["months"]), 2), say=lambda a, x: f"The monthly payment would be about {x:,.2f}.")
fn("finance.vat_price", "Add VAT to a net price.", [P("net_price", N, "Price without VAT."), P("vat_percent", N, "VAT rate in percent.")],
   ["What is {net_price} with {vat_percent}% VAT added?", "A product costs {net_price} before tax. What is the price with {vat_percent}% VAT?"], lambda r: {"net_price": r.randrange(10, 500), "vat_percent": r.choice([7, 19, 21, 23])},
   run=lambda a: round(a["net_price"] * (1 + a["vat_percent"] / 100), 2), say=lambda a, x: f"With {fmt(a['vat_percent'])}% VAT the price is {x:,.2f}.")
fn("finance.tip_split", "Split a restaurant bill with tip between several people.", [P("bill", N, "Bill amount before tip."), P("tip_percent", N, "Tip in percent."), P("people", I, "Number of people.")],
   ["Split a {bill} bill with a {tip_percent}% tip between {people} people.", "Our bill is {bill}. With a {tip_percent}% tip, what does each of the {people} of us pay?"], lambda r: {"bill": r.randrange(40, 300), "tip_percent": r.choice([10, 15, 20]), "people": r.randrange(2, 8)},
   run=lambda a: round(a["bill"] * (1 + a["tip_percent"] / 100) / a["people"], 2), say=lambda a, x: f"Each person pays {x:,.2f}.")
RATES = {("EUR", "USD"): 1.10, ("USD", "EUR"): 0.91, ("EUR", "GBP"): 0.85, ("GBP", "EUR"): 1.18, ("USD", "GBP"): 0.78, ("GBP", "USD"): 1.28, ("EUR", "JPY"): 160.0, ("JPY", "EUR"): 0.00625}
fn("finance.currency_convert", "Convert an amount between two currencies at the bank's fixed rate.", [P("amount", N, "Amount to convert."), P("from_currency", S, "Source currency code.", enum=["EUR", "USD", "GBP", "JPY"]), P("to_currency", S, "Target currency code.", enum=["EUR", "USD", "GBP", "JPY"])],
   ["Convert {amount} {from_currency} to {to_currency}.", "How many {to_currency} is {amount} {from_currency}?"], lambda r: dict(zip(("from_currency", "to_currency"), r.choice(list(RATES)))) | {"amount": r.randrange(10, 1000, 10)},
   run=lambda a: round(a["amount"] * RATES[(a["from_currency"], a["to_currency"])], 2), say=lambda a, x: f"{fmt(a['amount'])} {a['from_currency']} is {x:,.2f} {a['to_currency']}.")

# ------------------------------------------------------------------------------------------------------------------------------- units
LEN = {"m": 1.0, "km": 1000.0, "mile": 1609.344, "ft": 0.3048, "inch": 0.0254}
fn("units.convert_length", "Convert a length between units.", [P("value", N, "The length."), P("from_unit", S, "Unit to convert from.", enum=list(LEN)), P("to_unit", S, "Unit to convert to.", enum=list(LEN))],
   ["Convert {value} {from_unit} to {to_unit}.", "How many {to_unit} are in {value} {from_unit}?"], lambda r: {"value": r.randrange(1, 100), "from_unit": (u := r.sample(list(LEN), 2))[0], "to_unit": u[1]},
   run=lambda a: round(a["value"] * LEN[a["from_unit"]] / LEN[a["to_unit"]], 3), say=lambda a, x: f"{fmt(a['value'])} {a['from_unit']} is {x:g} {a['to_unit']}.")


def _temp(a):
    v, f, t = a["value"], a["from_unit"], a["to_unit"]
    c = v if f == "C" else (v - 32) * 5 / 9 if f == "F" else v - 273.15
    return round(c if t == "C" else c * 9 / 5 + 32 if t == "F" else c + 273.15, 2)


fn("units.convert_temperature", "Convert a temperature between Celsius, Fahrenheit and Kelvin.", [P("value", N, "The temperature."), P("from_unit", S, "Source unit.", enum=["C", "F", "K"]), P("to_unit", S, "Target unit.", enum=["C", "F", "K"])],
   ["Convert {value} degrees {from_unit} to {to_unit}.", "What is {value}{from_unit} in {to_unit}?"], lambda r: {"value": r.randrange(-20, 110), "from_unit": (u := r.sample(["C", "F", "K"], 2))[0], "to_unit": u[1]},
   run=_temp, say=lambda a, x: f"{fmt(a['value'])} {a['from_unit']} is {x:g} {a['to_unit']}.")
WT = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523}
fn("units.convert_weight", "Convert a weight between units.", [P("value", N, "The weight."), P("from_unit", S, "Unit to convert from.", enum=list(WT)), P("to_unit", S, "Unit to convert to.", enum=list(WT))],
   ["Convert {value} {from_unit} to {to_unit}.", "How much is {value} {from_unit} in {to_unit}?"], lambda r: {"value": r.randrange(1, 200), "from_unit": (u := r.sample(list(WT), 2))[0], "to_unit": u[1]},
   run=lambda a: round(a["value"] * WT[a["from_unit"]] / WT[a["to_unit"]], 3), say=lambda a, x: f"{fmt(a['value'])} {a['from_unit']} is {x:g} {a['to_unit']}.")

# ------------------------------------------------------------------------------------------------------------------------------- text
WORDS = ["river", "lantern", "orchard", "compass", "marble", "violin", "harvest", "meadow", "anchor", "blanket", "candle", "mirror"]
SENT = ["The quick brown fox jumps over the lazy dog", "Learning to code takes patience and practice", "Rain fell softly on the old stone bridge", "A good map is worth a thousand words", "We met at the harbour before sunrise"]
fn("text.word_count", "Count the words in a text.", [P("text", S, "The text to analyse.")],
   ["How many words are in this sentence: \"{text}\"?", "Count the words in \"{text}\"."], lambda r: {"text": r.choice(SENT)},
   run=lambda a: len(a["text"].split()), say=lambda a, x: f"The text has {x} words.")
fn("text.reverse", "Reverse the characters of a text.", [P("text", S, "The text to reverse.")],
   ["Reverse the string \"{text}\".", "Write \"{text}\" backwards."], lambda r: {"text": r.choice(WORDS) + " " + r.choice(WORDS)},
   run=lambda a: a["text"][::-1], say=lambda a, x: f"Reversed: {x}")
fn("text.change_case", "Change the case of a text.", [P("text", S, "The text."), P("case", S, "Target case.", enum=["upper", "lower", "title"])],
   ["Convert \"{text}\" to {case} case.", "Make \"{text}\" {case}case."], lambda r: {"text": r.choice(SENT)[:30], "case": r.choice(["upper", "lower", "title"])},
   run=lambda a: {"upper": a["text"].upper(), "lower": a["text"].lower(), "title": a["text"].title()}[a["case"]], say=lambda a, x: f"Result: {x}")
fn("text.is_palindrome", "Check whether a word or phrase is a palindrome (ignoring case and spaces).", [P("text", S, "The text to test.")],
   ["Is \"{text}\" a palindrome?", "Check if \"{text}\" reads the same backwards."], lambda r: {"text": r.choice(["level", "radar", "A man a plan a canal Panama", "hello", "world", "Was it a car or a cat I saw", "banana", "noon", "garden"])},
   run=lambda a: (lambda t: t == t[::-1])("".join(c for c in a["text"].lower() if c.isalnum())), say=lambda a, x: f"Yes, \"{a['text']}\" is a palindrome." if x else f"No, \"{a['text']}\" is not a palindrome.")
fn("text.count_letter", "Count how many times a letter appears in a text (case-insensitive).", [P("text", S, "The text."), P("letter", S, "A single letter.")],
   ["How many times does the letter {letter} appear in \"{text}\"?", "Count the {letter}'s in \"{text}\"."], lambda r: {"text": r.choice(SENT), "letter": r.choice("aeiotr")},
   run=lambda a: a["text"].lower().count(a["letter"]), say=lambda a, x: f"The letter {a['letter']} appears {x} times.")
fn("text.truncate", "Shorten a text to a maximum number of characters, adding '...' if it was cut.", [P("text", S, "The text."), P("max_length", I, "Maximum number of characters.")],
   ["Truncate \"{text}\" to {max_length} characters.", "Cut \"{text}\" down to at most {max_length} characters."], lambda r: {"text": r.choice(SENT), "max_length": r.randrange(8, 25)},
   run=lambda a: a["text"] if len(a["text"]) <= a["max_length"] else a["text"][:a["max_length"]] + "...", say=lambda a, x: f"Truncated: {x}")

# ------------------------------------------------------------------------------------------------------------------------------- dates
def rdate(r, y0=2023, y1=2026):
    return (date(y0, 1, 1) + timedelta(days=r.randrange(0, (date(y1, 12, 31) - date(y0, 1, 1)).days))).isoformat()


fn("dates.add_days", "Add (or subtract) days to a date given as YYYY-MM-DD.", [P("date", S, "Start date, YYYY-MM-DD."), P("days", I, "Days to add; negative to subtract.")],
   ["What date is {days} days after {date}?", "Add {days} days to {date}."], lambda r: {"date": rdate(r), "days": r.randrange(5, 200)},
   run=lambda a: (date.fromisoformat(a["date"]) + timedelta(days=a["days"])).isoformat(), say=lambda a, x: f"{a['days']} days after {a['date']} is {x}.")
fn("dates.days_between", "Count the days between two dates (YYYY-MM-DD).", [P("start_date", S, "Start date."), P("end_date", S, "End date.")],
   ["How many days are there between {start_date} and {end_date}?", "Count the days from {start_date} to {end_date}."], lambda r: (lambda d: {"start_date": d, "end_date": (date.fromisoformat(d) + timedelta(days=r.randrange(10, 400))).isoformat()})(rdate(r)),
   run=lambda a: (date.fromisoformat(a["end_date"]) - date.fromisoformat(a["start_date"])).days, say=lambda a, x: f"There are {x} days between {a['start_date']} and {a['end_date']}.")
fn("dates.day_of_week", "Return the weekday of a date (YYYY-MM-DD).", [P("date", S, "The date.")],
   ["What day of the week is {date}?", "Which weekday was {date}?"], lambda r: {"date": rdate(r)},
   run=lambda a: date.fromisoformat(a["date"]).strftime("%A"), say=lambda a, x: f"{a['date']} is a {x}.")
fn("dates.is_leap_year", "Check whether a year is a leap year.", [P("year", I, "The year.")],
   ["Is {year} a leap year?", "Was {year} a leap year?"], lambda r: {"year": r.randrange(1896, 2101)},
   run=lambda a: (a["year"] % 4 == 0 and a["year"] % 100 != 0) or a["year"] % 400 == 0, say=lambda a, x: f"Yes, {a['year']} is a leap year." if x else f"No, {a['year']} is not a leap year.")
fn("dates.age_on", "Compute a person's age in full years on a given date.", [P("birth_date", S, "Birth date, YYYY-MM-DD."), P("on_date", S, "Reference date, YYYY-MM-DD.")],
   ["How old is someone born on {birth_date} as of {on_date}?", "What is the age on {on_date} of a person born {birth_date}?"], lambda r: {"birth_date": rdate(r, 1960, 2010), "on_date": rdate(r, 2024, 2026)},
   run=lambda a: (lambda b, o: o.year - b.year - ((o.month, o.day) < (b.month, b.day)))(date.fromisoformat(a["birth_date"]), date.fromisoformat(a["on_date"])), say=lambda a, x: f"That person is {x} years old.")

# ------------------------------------------------------------------------------------------------------------------------------- lists & stats
fn("lists.sort_numbers", "Sort a list of numbers.", [P("numbers", A, "Numbers to sort.", items=N), P("descending", B, "Sort from largest to smallest.", req=False, default=False)],
   ["Sort these numbers: {numbers}.", "Sort {numbers} from smallest to largest."], lambda r: {"numbers": [r.randrange(1, 100) for _ in range(r.randrange(4, 8))]},
   run=lambda a: sorted(a["numbers"], reverse=bool(a.get("descending"))), say=lambda a, x: "Sorted: " + ", ".join(map(str, x)) + ".")
fn("lists.sum_numbers", "Add up a list of numbers.", [P("numbers", A, "Numbers to add.", items=N)],
   ["What is the sum of {numbers}?", "Add up {numbers}."], lambda r: {"numbers": [r.randrange(1, 200) for _ in range(r.randrange(3, 8))]},
   run=lambda a: sum(a["numbers"]), say=lambda a, x: f"The sum is {x}.")
fn("lists.max_value", "Return the largest number in a list.", [P("numbers", A, "The numbers.", items=N)],
   ["What is the largest of {numbers}?", "Find the maximum of {numbers}."], lambda r: {"numbers": [r.randrange(1, 500) for _ in range(r.randrange(3, 8))]},
   run=lambda a: max(a["numbers"]), say=lambda a, x: f"The largest number is {x}.")
fn("lists.count_occurrences", "Count how many times an item appears in a list of strings.", [P("items", A, "The list.", items=S), P("target", S, "The item to count.")],
   ["How many times does {target} appear in {items}?", "Count the occurrences of {target} in the list {items}."], lambda r: (lambda base, t: {"items": [r.choice(base + [t, t]) for _ in range(r.randrange(5, 9))], "target": t})(r.sample(WORDS, 3), r.choice(WORDS)),
   run=lambda a: a["items"].count(a["target"]), say=lambda a, x: f"\"{a['target']}\" appears {x} times.")
fn("stats.std_dev", "Compute the population standard deviation of a list of numbers.", [P("numbers", A, "The numbers.", items=N)],
   ["What is the standard deviation of {numbers}?", "Compute the population standard deviation of {numbers}."], lambda r: {"numbers": [r.randrange(1, 60) for _ in range(r.randrange(4, 8))]},
   run=lambda a: round(pstdev(a["numbers"]), 3), say=lambda a, x: f"The standard deviation is {x:g}.")

# ------------------------------------------------------------------------------------------------------------------------------- lookups (deterministic simulation)
CONDS = ["sunny", "cloudy", "light rain", "windy", "clear", "foggy", "overcast"]


def _weather(a):
    t = 4 + h("w", a["city"]) % 22
    t = round(t * 9 / 5 + 32) if a.get("units") == "fahrenheit" else t
    return {"city": a["city"], "temperature": t, "units": a.get("units", "celsius"), "condition": CONDS[h("c", a["city"]) % len(CONDS)]}


fn("weather.get_current", "Get the current weather for a city.", [P("city", S, "City name."), P("units", S, "Temperature units.", req=False, enum=["celsius", "fahrenheit"], default="celsius")],
   ["What's the weather like in {city} right now?", "Tell me the current weather in {city} in {units}.", "How is the weather in {city}?"], lambda r: {"city": r.choice(CITIES), "units": r.choice(["celsius", "fahrenheit"])},
   run=_weather, say=lambda a, x: f"It is {x['condition']} in {x['city']} at {x['temperature']} degrees {x['units']}.")
fn("weather.get_forecast", "Get a multi-day weather forecast for a city.", [P("city", S, "City name."), P("days", I, "Number of days (1-7).")],
   ["Give me the {days}-day forecast for {city}.", "What's the weather forecast for {city} over the next {days} days?"], lambda r: {"city": r.choice(CITIES), "days": r.randrange(2, 7)},
   run=lambda a: [{"day": i + 1, "high": 8 + h("f", a["city"], i) % 15, "condition": CONDS[h("fc", a["city"], i) % len(CONDS)]} for i in range(a["days"])],
   say=lambda a, x: f"Forecast for {a['city']}: " + "; ".join(f"day {d['day']}: {d['condition']}, high {d['high']}" for d in x) + ".")
fn("travel.search_flights", "Search flights between two cities on a date.", [P("origin", S, "Departure city."), P("destination", S, "Arrival city."), P("date", S, "Travel date, YYYY-MM-DD."), P("passengers", I, "Number of passengers.", req=False, default=1)],
   ["Find flights from {origin} to {destination} on {date}.", "I need {passengers} seats from {origin} to {destination} on {date}."], lambda r: {"origin": (c := r.sample(CITIES, 2))[0], "destination": c[1], "date": rdate(r, 2026, 2026), "passengers": r.randrange(1, 5)},
   run=lambda a: [{"flight": f"FL{100 + h('fl', a['origin'], a['destination'], i) % 800}", "departs": f"{6 + i * 4:02d}:{(h('m', i) % 4) * 15:02d}", "price": 80 + h("p", a["origin"], a["destination"], i) % 220} for i in range(3)],
   say=lambda a, x: f"I found {len(x)} flights from {a['origin']} to {a['destination']}: " + "; ".join(f"{f['flight']} at {f['departs']} for {f['price']} EUR" for f in x) + ".")
fn("travel.book_hotel", "Book a hotel room in a city.", [P("city", S, "City."), P("check_in", S, "Check-in date, YYYY-MM-DD."), P("nights", I, "Number of nights."), P("guests", I, "Number of guests.", req=False, default=1)],
   ["Book a hotel in {city} from {check_in} for {nights} nights.", "Reserve a room for {guests} guests in {city} starting {check_in} for {nights} nights."], lambda r: {"city": r.choice(CITIES), "check_in": rdate(r, 2026, 2026), "nights": r.randrange(1, 8), "guests": r.randrange(1, 4)},
   run=lambda a: {"confirmation": f"HT{h('ht', a['city'], a['check_in']) % 90000 + 10000}", "total": (70 + h("tot", a["city"]) % 90) * a["nights"]}, say=lambda a, x: f"Booked! Your confirmation number is {x['confirmation']} and the total is {x['total']} EUR.")
CATS = ["electronics", "books", "garden", "kitchen", "sports"]
fn("shop.search_products", "Search the product catalogue.", [P("query", S, "Search words."), P("max_price", N, "Maximum price.", req=False), P("category", S, "Category filter.", req=False, enum=CATS)],
   ["Search for {query}.", "Find {query} under {max_price} euros.", "Look for {query} in {category}."], lambda r: {"query": r.choice(["wireless headphones", "garden hose", "cookbook", "running shoes", "desk lamp", "coffee grinder"]), "max_price": r.choice([30, 50, 100, 200]), "category": r.choice(CATS)},
   run=lambda a: [{"id": f"P{h('pid', a['query'], i) % 9000 + 1000}", "name": f"{a['query'].title()} {chr(65 + i)}", "price": 10 + h("pp", a["query"], i) % 80} for i in range(3)],
   say=lambda a, x: "Here are the top results: " + "; ".join(f"{p['name']} ({p['id']}) for {p['price']} EUR" for p in x) + ".")
fn("shop.add_to_cart", "Add a product to the shopping cart.", [P("product_id", S, "Product id, e.g. P1234."), P("quantity", I, "How many to add.")],
   ["Add {quantity} of product {product_id} to my cart.", "Put {quantity} x {product_id} in the cart."], lambda r: {"product_id": f"P{r.randrange(1000, 9999)}", "quantity": r.randrange(1, 6)},
   run=lambda a: {"cart_items": a["quantity"], "added": a["product_id"]}, say=lambda a, x: f"Added {a['quantity']} of {a['product_id']} to your cart.")
CUIS = ["italian", "thai", "mexican", "indian", "japanese", "vegan"]
fn("food.find_restaurants", "Find restaurants in a city.", [P("city", S, "City."), P("cuisine", S, "Cuisine.", enum=CUIS), P("price_level", I, "Price level 1 (cheap) to 4 (expensive).", req=False)],
   ["Find {cuisine} restaurants in {city}.", "Where can I eat {cuisine} food in {city} at price level {price_level}?"], lambda r: {"city": r.choice(CITIES), "cuisine": r.choice(CUIS), "price_level": r.randrange(1, 5)},
   run=lambda a: [{"name": f"{r_} {a['cuisine'].title()} House", "rating": round(3.5 + h("rt", a["city"], r_) % 15 / 10, 1)} for r_ in ("Casa", "Luna", "Nova")],
   say=lambda a, x: f"Top {a['cuisine']} places in {a['city']}: " + "; ".join(f"{p['name']} (rated {p['rating']})" for p in x) + ".")
fn("maps.get_distance", "Get the distance and travel time between two places.", [P("origin", S, "Start."), P("destination", S, "End."), P("mode", S, "Travel mode.", req=False, enum=["driving", "walking", "cycling"], default="driving")],
   ["How far is it from {origin} to {destination}?", "How long does it take to go from {origin} to {destination} by {mode}?"], lambda r: {"origin": (c := r.sample(CITIES, 2))[0], "destination": c[1], "mode": r.choice(["driving", "walking", "cycling"])},
   run=lambda a: {"km": 20 + h("d", *sorted([a["origin"], a["destination"]])) % 600, "mode": a.get("mode", "driving")},
   say=lambda a, x: f"It is about {x['km']} km from {a['origin']} to {a['destination']} ({x['mode']}).")
fn("music.play_song", "Play a song.", [P("title", S, "Song title."), P("artist", S, "Artist name.", req=False)],
   ["Play {title}.", "Put on {title} by {artist}."], lambda r: {"title": r.choice(["Blue Horizon", "Night Train", "Paper Boats", "Golden Hour", "Slow River"]), "artist": r.choice(["The Lanterns", "Mira Vale", "North Coast", "Echo Park"])},
   run=lambda a: {"now_playing": a["title"]}, say=lambda a, x: f"Now playing {a['title']}.")
fn("movies.get_showtimes", "Get movie showtimes in a city on a date.", [P("movie", S, "Movie title."), P("city", S, "City."), P("date", S, "Date, YYYY-MM-DD.")],
   ["When is {movie} showing in {city} on {date}?", "Get showtimes for {movie} in {city} on {date}."], lambda r: {"movie": r.choice(["Midnight Express", "The Long Orchard", "Ocean Letters", "Iron Garden"]), "city": r.choice(CITIES), "date": rdate(r, 2026, 2026)},
   run=lambda a: ["14:30", "17:15", "20:00"] if h("mv", a["movie"], a["city"]) % 2 else ["13:00", "16:45", "21:15"], say=lambda a, x: f"{a['movie']} is showing in {a['city']} at " + ", ".join(x) + ".")
fn("news.get_headlines", "Get recent news headlines on a topic.", [P("topic", S, "Topic."), P("count", I, "Number of headlines.", req=False, default=5)],
   ["Give me {count} headlines about {topic}.", "What's in the news about {topic}?"], lambda r: {"topic": r.choice(["technology", "climate", "sports", "economy", "science"]), "count": r.randrange(2, 6)},
   run=lambda a: [f"{a['topic'].title()} update #{i + 1}: report {h('n', a['topic'], i) % 900 + 100}" for i in range(a.get("count", 3))], say=lambda a, x: "Headlines: " + "; ".join(x) + ".")
fn("support.create_ticket", "Create a support ticket.", [P("subject", S, "Short subject."), P("priority", S, "Priority.", enum=["low", "medium", "high"]), P("description", S, "What happened.")],
   ["Open a {priority} priority ticket titled \"{subject}\": {description}.", "Create a support ticket with subject {subject}, priority {priority}. Details: {description}."],
   lambda r: {"subject": r.choice(["Login fails", "Invoice missing", "App crashes on start", "Cannot reset password"]), "priority": r.choice(["low", "medium", "high"]), "description": r.choice(["it started this morning", "it happens on every attempt", "the error code is E42", "it only affects my phone"])},
   run=lambda a: {"ticket_id": f"T-{h('t', a['subject']) % 9000 + 1000}", "status": "open"}, say=lambda a, x: f"Ticket {x['ticket_id']} is open with {a['priority']} priority.")
fn("hr.get_employee", "Look up an employee by id.", [P("employee_id", S, "Employee id like E1234.")],
   ["Look up employee {employee_id}.", "Who is employee {employee_id}?"], lambda r: {"employee_id": f"E{r.randrange(1000, 9999)}"},
   run=lambda a: {"name": f"{FIRST[h('e', a['employee_id']) % 10]} {['Rossi', 'Berg', 'Kim', 'Silva', 'Novak'][h('l', a['employee_id']) % 5]}", "department": ["Sales", "Support", "Finance", "Design"][h("d", a["employee_id"]) % 4]},
   say=lambda a, x: f"{a['employee_id']} is {x['name']} from {x['department']}.")
fn("devtools.run_tests", "Run the test suite of a project.", [P("project", S, "Project name."), P("verbose", B, "Show every test.", req=False, default=False)],
   ["Run the tests for {project}.", "Please run {project}'s test suite."], lambda r: {"project": r.choice(["billing-api", "web-client", "data-pipeline", "auth-service"]), "verbose": r.random() < 0.3},
   run=lambda a: {"passed": 40 + h("tp", a["project"]) % 60, "failed": h("tf", a["project"]) % 3}, say=lambda a, x: f"{a['project']}: {x['passed']} tests passed, {x['failed']} failed.")
fn("devtools.create_branch", "Create a git branch in a repository.", [P("repo", S, "Repository name."), P("branch_name", S, "New branch name."), P("from_branch", S, "Branch to start from.", req=False, default="main")],
   ["Create a branch called {branch_name} in {repo}.", "In {repo}, create {branch_name} from {from_branch}."], lambda r: {"repo": r.choice(["billing-api", "web-client", "data-pipeline"]), "branch_name": r.choice(["feature/export", "fix/login-bug", "chore/deps", "feature/dark-mode"]), "from_branch": r.choice(["main", "develop"])},
   run=lambda a: {"created": a["branch_name"], "from": a.get("from_branch", "main")}, say=lambda a, x: f"Created branch {x['created']} from {x['from']} in {a['repo']}.")
fn("fitness.log_workout", "Log a workout.", [P("activity", S, "Activity name."), P("minutes", I, "Duration in minutes."), P("calories", I, "Calories burned.", req=False)],
   ["Log a {minutes} minute {activity} workout.", "I did {activity} for {minutes} minutes and burned {calories} calories - log it."], lambda r: {"activity": r.choice(["running", "cycling", "swimming", "yoga", "rowing"]), "minutes": r.randrange(15, 90, 5), "calories": r.randrange(100, 600, 25)},
   run=lambda a: {"logged": True, "weekly_minutes": 60 + h("wm", a["activity"]) % 180}, say=lambda a, x: f"Logged {a['minutes']} minutes of {a['activity']}. This week you are at {x['weekly_minutes']} minutes.")
fn("library.search_books", "Search a library catalogue.", [P("title", S, "Title words."), P("author", S, "Author name.", req=False)],
   ["Search the library for books titled {title}.", "Does the library have \"{title}\" by {author}?"], lambda r: {"title": r.choice(["The Quiet Orchard", "River Atlas", "Night Gardens", "Letters from Porto"]), "author": r.choice(["I. Marsh", "L. Okafor", "T. Brandt"])},
   run=lambda a: {"available": h("lb", a["title"]) % 3 != 0, "copies": 1 + h("lc", a["title"]) % 4}, say=lambda a, x: f"\"{a['title']}\" is {'available' if x['available'] else 'on loan'} ({x['copies']} copies).")
fn("planner.add_reminder", "Add a reminder.", [P("text", S, "What to be reminded of."), P("time", S, "When, e.g. 2026-03-04 09:30.")],
   ["Remind me to {text} at {time}.", "Set a reminder for {time}: {text}."], lambda r: {"text": r.choice(["call the dentist", "pay the rent", "water the plants", "send the report", "renew the passport"]), "time": rdate(r, 2026, 2026) + f" {r.randrange(7, 20):02d}:{r.choice(['00', '30'])}"},
   run=lambda a: {"reminder_id": f"R{h('r', a['text']) % 9000 + 1000}"}, say=lambda a, x: f"Reminder {x['reminder_id']} is set for {a['time']}.")
fn("messaging.send_message", "Send a text message to a contact.", [P("recipient", S, "Contact name."), P("message", S, "Message text.")],
   ["Text {recipient}: {message}", "Send {recipient} a message saying \"{message}\"."], lambda r: {"recipient": r.choice(FIRST), "message": r.choice(["I'm running ten minutes late", "Dinner at eight?", "The files are ready", "Thanks for your help today"])},
   run=lambda a: {"sent": True}, say=lambda a, x: f"Message sent to {a['recipient']}.")


# ------------------------------------------------------------------------------------------------------------------------------- second library (science, health, home, transport, time, social, dev, commerce, ...)
MOLAR = {"H2O": 18.015, "CO2": 44.01, "NaCl": 58.44, "CH4": 16.04, "O2": 32.0, "NH3": 17.03, "C6H12O6": 180.16}
fn("science.molar_mass", "Return the molar mass in g/mol of a common compound.", [P("formula", S, "Chemical formula.", enum=list(MOLAR))],
   ["What is the molar mass of {formula}?", "How many grams per mole is {formula}?"], lambda r: {"formula": r.choice(list(MOLAR))},
   run=lambda a: MOLAR[a["formula"]], say=lambda a, x: f"The molar mass of {a['formula']} is {x:g} g/mol.")
fn("health.bmi", "Compute the body mass index from weight and height.", [P("weight_kg", N, "Weight in kilograms."), P("height_cm", N, "Height in centimetres.")],
   ["What is the BMI of someone who weighs {weight_kg} kg and is {height_cm} cm tall?", "Calculate the body mass index for {weight_kg} kg at {height_cm} cm."], lambda r: {"weight_kg": r.randrange(50, 110), "height_cm": r.randrange(150, 200)},
   run=lambda a: round(a["weight_kg"] / (a["height_cm"] / 100) ** 2, 1), say=lambda a, x: f"The BMI is {x:g}.")
fn("health.dosage_by_weight", "Compute a total dose in mg from body weight and a per-kilogram dose.", [P("weight_kg", N, "Weight in kilograms."), P("mg_per_kg", N, "Dose in mg per kilogram.")],
   ["What total dose is needed for {weight_kg} kg at {mg_per_kg} mg per kg?", "Work out the dose for a patient of {weight_kg} kg using {mg_per_kg} mg/kg."], lambda r: {"weight_kg": r.randrange(10, 100), "mg_per_kg": r.choice([2, 5, 7.5, 10, 15])},
   run=lambda a: a["weight_kg"] * a["mg_per_kg"], say=lambda a, x: f"The total dose is {x:g} mg.")
fn("physics.kinetic_energy", "Compute kinetic energy in joules.", [P("mass_kg", N, "Mass in kilograms."), P("velocity_ms", N, "Velocity in metres per second.")],
   ["What is the kinetic energy of a {mass_kg} kg object moving at {velocity_ms} m/s?", "Compute the kinetic energy for mass {mass_kg} kg and speed {velocity_ms} m/s."], lambda r: {"mass_kg": r.randrange(1, 100), "velocity_ms": r.randrange(2, 40)},
   run=lambda a: 0.5 * a["mass_kg"] * a["velocity_ms"] ** 2, say=lambda a, x: f"The kinetic energy is {x:g} joules.")
fn("physics.free_fall_time", "Compute how long an object takes to fall from a height (no air resistance, g = 9.81).", [P("height_m", N, "Height in metres.")],
   ["How long does it take an object to fall {height_m} metres?", "Free fall time from a height of {height_m} m?"], lambda r: {"height_m": r.choice([5, 10, 20, 45, 80, 100, 125])},
   run=lambda a: round(math.sqrt(2 * a["height_m"] / 9.81), 2), say=lambda a, x: f"It takes about {x:g} seconds.")
fn("science.half_life_remaining", "Compute how much of a radioactive sample remains after some time.", [P("initial", N, "Initial amount."), P("half_life", N, "Half-life."), P("elapsed", N, "Elapsed time in the same unit.")],
   ["{initial} grams of a substance with a half-life of {half_life} days: how much is left after {elapsed} days?", "How much of {initial} g remains after {elapsed} days if the half-life is {half_life} days?"],
   lambda r: (lambda hl: {"initial": r.choice([100, 200, 80, 64]), "half_life": hl, "elapsed": hl * r.randrange(1, 5)})(r.choice([2, 5, 10, 30])),
   run=lambda a: round(a["initial"] * 0.5 ** (a["elapsed"] / a["half_life"]), 3), say=lambda a, x: f"About {x:g} grams remain.")
fn("education.grade_average", "Compute a weighted grade average.", [P("scores", A, "Scores.", items=N), P("weights", A, "Weights in percent, same order as the scores.", items=N)],
   ["My scores are {scores} with weights {weights} percent. What is my weighted average?", "Compute the weighted grade for scores {scores} and weights {weights}."],
   lambda r: (lambda w: {"scores": [r.randrange(50, 100) for _ in w], "weights": w})(r.choice([[30, 30, 40], [50, 50], [20, 30, 50], [25, 25, 25, 25]])),
   run=lambda a: round(sum(s * w for s, w in zip(a["scores"], a["weights"])) / sum(a["weights"]), 2), say=lambda a, x: f"Your weighted average is {x:g}.")
fn("education.reading_time", "Estimate reading time in minutes.", [P("word_count", I, "Number of words."), P("wpm", I, "Reading speed in words per minute.", req=False, default=200)],
   ["How long does it take to read {word_count} words at {wpm} words per minute?", "Estimate the reading time of a {word_count}-word article."], lambda r: {"word_count": r.randrange(300, 6000, 100), "wpm": r.choice([150, 200, 250, 300])},
   run=lambda a: round(a["word_count"] / a.get("wpm", 200), 1), say=lambda a, x: f"It takes about {x:g} minutes.")
fn("home.paint_needed", "Estimate litres of paint needed for a wall area.", [P("area_m2", N, "Wall area in square metres."), P("coats", I, "Number of coats.", req=False, default=2)],
   ["How much paint do I need for {area_m2} square metres with {coats} coats?", "Estimate the litres of paint for {area_m2} m2 of wall."], lambda r: {"area_m2": r.randrange(10, 80, 2), "coats": r.choice([1, 2, 3])},
   run=lambda a: round(a["area_m2"] * a.get("coats", 2) / 10, 1), say=lambda a, x: f"You need about {x:g} litres of paint.")
fn("home.energy_cost", "Compute an electricity cost.", [P("kwh", N, "Energy used in kWh."), P("price_per_kwh", N, "Price per kWh.")],
   ["What does {kwh} kWh cost at {price_per_kwh} per kWh?", "My bill is {kwh} kWh at a rate of {price_per_kwh} per kWh. What is the cost?"], lambda r: {"kwh": r.randrange(50, 600, 10), "price_per_kwh": r.choice([0.18, 0.22, 0.25, 0.3, 0.35])},
   run=lambda a: round(a["kwh"] * a["price_per_kwh"], 2), say=lambda a, x: f"The cost is {x:,.2f}.")
fn("home.tile_count", "Compute how many square tiles are needed to cover a floor (rounded up).", [P("room_length_m", N, "Room length in metres."), P("room_width_m", N, "Room width in metres."), P("tile_size_cm", I, "Tile side in centimetres.")],
   ["How many {tile_size_cm} cm tiles cover a {room_length_m} by {room_width_m} metre room?", "Tiles needed for a floor of {room_length_m} m x {room_width_m} m with {tile_size_cm} cm tiles?"], lambda r: {"room_length_m": r.choice([3, 4, 5, 6]), "room_width_m": r.choice([2, 3, 4]), "tile_size_cm": r.choice([20, 25, 30, 50])},
   run=lambda a: math.ceil(a["room_length_m"] * a["room_width_m"] * 10000 / a["tile_size_cm"] ** 2), say=lambda a, x: f"You need {x} tiles.")
fn("transport.fuel_cost", "Compute the fuel cost of a trip.", [P("distance_km", N, "Distance in km."), P("consumption_l_per_100km", N, "Fuel consumption in litres per 100 km."), P("price_per_litre", N, "Fuel price per litre.")],
   ["What does a {distance_km} km trip cost if the car uses {consumption_l_per_100km} litres per 100 km and fuel is {price_per_litre} per litre?", "Fuel cost for {distance_km} km at {consumption_l_per_100km} L/100km and {price_per_litre} per litre?"],
   lambda r: {"distance_km": r.randrange(50, 800, 10), "consumption_l_per_100km": r.choice([4.5, 5.5, 6, 7.5, 8]), "price_per_litre": r.choice([1.6, 1.75, 1.9, 2.0])},
   run=lambda a: round(a["distance_km"] * a["consumption_l_per_100km"] / 100 * a["price_per_litre"], 2), say=lambda a, x: f"The trip costs about {x:,.2f} in fuel.")
fn("transport.train_search", "Search train connections between two cities on a date.", [P("origin", S, "Departure city."), P("destination", S, "Arrival city."), P("date", S, "Date, YYYY-MM-DD.")],
   ["Find trains from {origin} to {destination} on {date}.", "What trains run from {origin} to {destination} on {date}?"], lambda r: {"origin": (c := r.sample(CITIES, 2))[0], "destination": c[1], "date": rdate(r, 2026, 2026)},
   run=lambda a: [{"train": f"T{h('tr', a['origin'], a['destination'], i) % 900 + 100}", "departs": f"{7 + 3 * i:02d}:{(h('tm', i) % 4) * 15:02d}", "duration_min": 60 + h("td", a["origin"], a["destination"]) % 240} for i in range(3)],
   say=lambda a, x: "Trains: " + "; ".join(f"{t['train']} at {t['departs']} ({t['duration_min']} min)" for t in x) + ".")
fn("transport.car_rental", "Get a car rental quote.", [P("city", S, "Pickup city."), P("pickup_date", S, "Pickup date, YYYY-MM-DD."), P("days", I, "Rental length in days."), P("car_class", S, "Car class.", enum=["economy", "compact", "suv"])],
   ["Get a quote for a {car_class} car in {city} from {pickup_date} for {days} days.", "How much is an {car_class} rental in {city} for {days} days starting {pickup_date}?"],
   lambda r: {"city": r.choice(CITIES), "pickup_date": rdate(r, 2026, 2026), "days": r.randrange(1, 15), "car_class": r.choice(["economy", "compact", "suv"])},
   run=lambda a: {"daily_rate": {"economy": 29, "compact": 39, "suv": 69}[a["car_class"]], "total": {"economy": 29, "compact": 39, "suv": 69}[a["car_class"]] * a["days"]}, say=lambda a, x: f"The quote is {x['total']} EUR ({x['daily_rate']} EUR per day).")
fn("transport.parking_fee", "Compute a parking fee, capped at a daily maximum if given.", [P("hours", N, "Hours parked."), P("rate_per_hour", N, "Hourly rate."), P("daily_cap", N, "Maximum fee per day.", req=False)],
   ["What is the parking fee for {hours} hours at {rate_per_hour} per hour?", "Parking for {hours} hours costs {rate_per_hour} an hour with a daily cap of {daily_cap}. What do I pay?"], lambda r: {"hours": r.randrange(1, 12), "rate_per_hour": r.choice([1.5, 2, 2.5, 3]), "daily_cap": r.choice([10, 12, 15])},
   run=lambda a: min(a["hours"] * a["rate_per_hour"], a.get("daily_cap", 1e9)), say=lambda a, x: f"The parking fee is {x:,.2f}.")
OFFSET = {"Lisbon": 0, "London": 0, "Paris": 1, "Berlin": 1, "Athens": 2, "New York": -5, "Tokyo": 9, "Sydney": 10, "Los Angeles": -8}
fn("time.timezone_convert", "Convert a clock time between two cities (standard time, no daylight saving).", [P("time", S, "Time as HH:MM."), P("from_city", S, "Source city.", enum=list(OFFSET)), P("to_city", S, "Target city.", enum=list(OFFSET))],
   ["What is {time} in {from_city} in {to_city} time?", "When it is {time} in {from_city}, what time is it in {to_city}?"], lambda r: {"time": f"{r.randrange(0, 24):02d}:{r.choice(['00', '15', '30', '45'])}", "from_city": (c := r.sample(list(OFFSET), 2))[0], "to_city": c[1]},
   run=lambda a: (lambda m: f"{(m // 60) % 24:02d}:{m % 60:02d}")(int(a["time"][:2]) * 60 + int(a["time"][3:]) + (OFFSET[a["to_city"]] - OFFSET[a["from_city"]]) * 60), say=lambda a, x: f"It is {x} in {a['to_city']}.")
fn("time.duration_between", "Compute the minutes between two clock times on the same day.", [P("start", S, "Start time HH:MM."), P("end", S, "End time HH:MM.")],
   ["How many minutes are there from {start} to {end}?", "How long is it from {start} until {end}?"], lambda r: (lambda s: {"start": f"{s // 60:02d}:{s % 60:02d}", "end": f"{(s + r.randrange(15, 400, 5)) // 60 % 24:02d}:{(s + 0) % 60:02d}"})(r.randrange(6 * 60, 14 * 60, 15)),
   run=lambda a: (int(a["end"][:2]) * 60 + int(a["end"][3:])) - (int(a["start"][:2]) * 60 + int(a["start"][3:])), say=lambda a, x: f"That is {x} minutes.")
fn("time.add_minutes", "Add minutes to a clock time (wrapping past midnight).", [P("time", S, "Time HH:MM."), P("minutes", I, "Minutes to add.")],
   ["What time is it {minutes} minutes after {time}?", "Add {minutes} minutes to {time}."], lambda r: {"time": f"{r.randrange(0, 24):02d}:{r.choice(['00', '10', '30', '45'])}", "minutes": r.randrange(10, 300, 5)},
   run=lambda a: (lambda m: f"{(m // 60) % 24:02d}:{m % 60:02d}")(int(a["time"][:2]) * 60 + int(a["time"][3:]) + a["minutes"]), say=lambda a, x: f"That is {x}.")
fn("social.schedule_post", "Schedule a social media post.", [P("platform", S, "Platform.", enum=["twitter", "linkedin", "instagram"]), P("text", S, "Post text."), P("publish_at", S, "Publish time, e.g. 2026-04-01 09:00.")],
   ["Schedule this {platform} post for {publish_at}: {text}", "Post \"{text}\" on {platform} at {publish_at}."], lambda r: {"platform": r.choice(["twitter", "linkedin", "instagram"]), "text": r.choice(["Our new release is live!", "We are hiring engineers.", "Thanks for a great event.", "Big news coming soon."]), "publish_at": rdate(r, 2026, 2026) + f" {r.randrange(7, 20):02d}:00"},
   run=lambda a: {"post_id": f"SP{h('sp', a['text']) % 9000 + 1000}", "status": "scheduled"}, say=lambda a, x: f"Scheduled as {x['post_id']} for {a['publish_at']}.")
fn("mailer.send_email", "Queue an email for sending.", [P("to", S, "Recipient email address."), P("subject", S, "Subject."), P("body", S, "Body text.")],
   ["Email {to} with the subject \"{subject}\" and the text: {body}", "Send a mail to {to}: subject {subject}, message {body}"], lambda r: {"to": f"{r.choice(['ana', 'marco', 'priya', 'jonas'])}@example.com", "subject": r.choice(["Meeting notes", "Invoice 204", "Welcome aboard", "Schedule change"]), "body": r.choice(["Please find the details attached.", "Let me know if Thursday works.", "Thanks for your quick reply.", "The files are in the shared folder."])},
   run=lambda a: {"queued": True, "to": a["to"]}, say=lambda a, x: f"Your email to {a['to']} is queued.")
fn("mailer.search_inbox", "Search the mailbox.", [P("query", S, "Search words."), P("folder", S, "Folder to search.", req=False, enum=["inbox", "sent", "archive"])],
   ["Search my mail for {query}.", "Find emails about {query} in the {folder} folder."], lambda r: {"query": r.choice(["invoice", "meeting", "contract", "flight confirmation"]), "folder": r.choice(["inbox", "sent", "archive"])},
   run=lambda a: {"count": 1 + h("ms", a["query"]) % 9}, say=lambda a, x: f"I found {x['count']} matching emails.")
fn("devtools.search_code", "Search a repository for a pattern.", [P("repo", S, "Repository."), P("pattern", S, "Text or regex to search."), P("file_type", S, "Restrict to a file extension.", req=False)],
   ["Search {repo} for {pattern}.", "Find {pattern} in the {file_type} files of {repo}."], lambda r: {"repo": r.choice(["billing-api", "web-client", "data-pipeline"]), "pattern": r.choice(["TODO", "deprecated_call", "api_key", "retry_count"]), "file_type": r.choice(["py", "js", "go"])},
   run=lambda a: {"matches": 1 + h("sc", a["repo"], a["pattern"]) % 14}, say=lambda a, x: f"There are {x['matches']} matches for {a['pattern']} in {a['repo']}.")
fn("devtools.get_build_status", "Get the latest CI build status of a project branch.", [P("project", S, "Project."), P("branch", S, "Branch.", req=False, default="main")],
   ["What is the build status of {project}?", "Is the latest build of {project} on {branch} green?"], lambda r: {"project": r.choice(["billing-api", "web-client", "auth-service"]), "branch": r.choice(["main", "develop", "release"])},
   run=lambda a: {"status": ["passing", "failing", "passing"][h("b", a["project"], a.get("branch", "main")) % 3]}, say=lambda a, x: f"The latest build of {a['project']} is {x['status']}.")
fn("devtools.deploy", "Deploy a service version to an environment.", [P("service", S, "Service name."), P("environment", S, "Target environment.", enum=["staging", "production"]), P("version", S, "Version tag.")],
   ["Deploy {service} version {version} to {environment}.", "Roll out {version} of {service} to {environment}."], lambda r: {"service": r.choice(["billing-api", "web-client", "auth-service"]), "environment": r.choice(["staging", "production"]), "version": f"v{r.randrange(1, 5)}.{r.randrange(0, 12)}.{r.randrange(0, 9)}"},
   run=lambda a: {"deployment_id": f"D{h('d', a['service'], a['version']) % 9000 + 1000}", "status": "started"}, say=lambda a, x: f"Deployment {x['deployment_id']} started for {a['service']} {a['version']} on {a['environment']}.")
fn("devtools.format_json", "Pretty-print a JSON string with a given indent.", [P("json_text", S, "Compact JSON text."), P("indent", I, "Spaces per indent level.", req=False, default=2)],
   ["Pretty-print this JSON: {json_text}", "Format {json_text} with an indent of {indent}."], lambda r: {"json_text": r.choice(['{"a":1,"b":[1,2]}', '{"name":"Ana","age":30}', '{"ok":true,"items":[]}', '{"x":{"y":2}}']), "indent": r.choice([2, 4])},
   run=lambda a: json.dumps(json.loads(a["json_text"]), indent=a.get("indent", 2)), say=lambda a, x: "Formatted JSON:\n" + x)
fn("devtools.regex_test", "Find all matches of a regular expression in a text.", [P("pattern", S, "Regular expression."), P("text", S, "Text to search.")],
   ["Find all matches of the regex {pattern} in \"{text}\".", "Which parts of \"{text}\" match {pattern}?"], lambda r: r.choice([{"pattern": "[0-9]+", "text": "Order 66 shipped 3 items on day 12"}, {"pattern": "[A-Z][a-z]+", "text": "Ana met Bo in Lisbon last Friday"}, {"pattern": "[a-z]+@[a-z]+[.]com", "text": "write to ana@mail.com or bo@site.com today"}]),
   run=lambda a: re.findall(a["pattern"], a["text"]), say=lambda a, x: "Matches: " + ", ".join(x) + ".")
fn("database.run_query", "Run a read-only SQL query against a named database.", [P("database", S, "Database name."), P("sql", S, "A SELECT statement.")],
   ["Run this on the {database} database: {sql}", "Query {database} with {sql}"], lambda r: {"database": r.choice(["sales", "inventory", "hr"]), "sql": r.choice(["SELECT COUNT(*) FROM orders", "SELECT name FROM products LIMIT 5", "SELECT AVG(salary) FROM staff", "SELECT * FROM customers WHERE country = 'PT'"])},
   run=lambda a: {"rows": 1 + h("q", a["sql"]) % 40, "elapsed_ms": 5 + h("e", a["sql"]) % 90}, say=lambda a, x: f"The query returned {x['rows']} rows in {x['elapsed_ms']} ms.")
fn("filesystem.compress", "Compress a file or folder.", [P("path", S, "Path to compress."), P("format", S, "Archive format.", enum=["zip", "tar.gz"])],
   ["Compress {path} as {format}.", "Make a {format} archive of {path}."], lambda r: {"path": r.choice(["reports/2025", "photos", "backup/db", "docs"]), "format": r.choice(["zip", "tar.gz"])},
   run=lambda a: {"archive": a["path"].replace("/", "_") + "." + a["format"], "size_mb": 5 + h("z", a["path"]) % 300}, say=lambda a, x: f"Created {x['archive']} ({x['size_mb']} MB).")
fn("filesystem.disk_usage", "Report the disk usage of a folder in megabytes.", [P("path", S, "Folder path.")],
   ["How much disk space does {path} use?", "Disk usage of {path}?"], lambda r: {"path": r.choice(["/var/log", "/home/ana/projects", "/data/exports", "/tmp/cache"])},
   run=lambda a: {"used_mb": 50 + h("du", a["path"]) % 4000}, say=lambda a, x: f"{a['path']} uses {x['used_mb']} MB.")
fn("shop.track_order", "Track an order by id.", [P("order_id", S, "Order id like O-1234.")],
   ["Where is my order {order_id}?", "Track order {order_id}."], lambda r: {"order_id": f"O-{r.randrange(1000, 9999)}"},
   run=lambda a: {"status": ["processing", "shipped", "out for delivery", "delivered"][h("o", a["order_id"]) % 4]}, say=lambda a, x: f"Order {a['order_id']} is {x['status']}.")
COUPONS = {"SAVE10": 10, "SAVE20": 20, "WELCOME5": 5, "SPRING15": 15}
fn("shop.apply_coupon", "Apply a coupon code to a cart total.", [P("code", S, "Coupon code.", enum=list(COUPONS)), P("cart_total", N, "Cart total before the coupon.")],
   ["Apply coupon {code} to my {cart_total} cart.", "My cart is {cart_total}. What do I pay with the code {code}?"], lambda r: {"code": r.choice(list(COUPONS)), "cart_total": r.randrange(30, 400, 5)},
   run=lambda a: round(a["cart_total"] * (1 - COUPONS[a["code"]] / 100), 2), say=lambda a, x: f"With {a['code']} you pay {x:,.2f}.")
fn("shop.create_order", "Create an order for product ids with a shipping address.", [P("items", A, "Product ids.", items=S), P("shipping", O, "Shipping details.", props={"address": S, "express": B}, view=lambda v: v["address"] + (" with express delivery" if v["express"] else ""))],
   ["Order {items} and ship to {shipping}.", "Please order products {items}, delivery to {shipping}."], lambda r: {"items": [f"P{r.randrange(1000, 9999)}" for _ in range(r.randrange(1, 4))], "shipping": {"address": r.choice(["12 Oak Street Oslo", "5 Harbour Road Lisbon", "30 Mill Lane Graz", "8 Rue Victor Lyon"]), "express": r.random() < 0.4}},
   run=lambda a: {"order_id": f"O-{h('co', *a['items']) % 9000 + 1000}", "items": len(a["items"])}, say=lambda a, x: f"Order {x['order_id']} was created with {x['items']} item(s).")
fn("finance.roi", "Compute return on investment in percent.", [P("gain", N, "Final value of the investment."), P("cost", N, "Amount invested.")],
   ["What is the ROI if I invested {cost} and it is now worth {gain}?", "Compute the return on investment for a cost of {cost} and a final value of {gain}."], lambda r: (lambda c: {"cost": c, "gain": c + r.choice([-1, 1]) * r.randrange(1, 9) * c // 10})(r.randrange(10, 100) * 100),
   run=lambda a: round((a["gain"] - a["cost"]) / a["cost"] * 100, 2), say=lambda a, x: f"The ROI is {x:g}%.")
fn("finance.break_even", "Compute the break-even number of units.", [P("fixed_costs", N, "Fixed costs."), P("price_per_unit", N, "Selling price per unit."), P("cost_per_unit", N, "Variable cost per unit.")],
   ["How many units must I sell to break even with fixed costs of {fixed_costs}, a price of {price_per_unit} and a unit cost of {cost_per_unit}?", "Break-even point: fixed {fixed_costs}, price {price_per_unit}, variable cost {cost_per_unit}."],
   lambda r: (lambda p, c: {"fixed_costs": r.randrange(5, 50) * 100, "price_per_unit": p, "cost_per_unit": c})(*r.choice([(20, 12), (15, 9), (50, 30), (8, 5)])),
   run=lambda a: math.ceil(a["fixed_costs"] / (a["price_per_unit"] - a["cost_per_unit"])), say=lambda a, x: f"You need to sell {x} units to break even.")
fn("finance.salary_after_tax", "Compute net salary after a flat tax rate.", [P("gross", N, "Gross salary."), P("tax_percent", N, "Flat tax rate in percent.")],
   ["What is {gross} after {tax_percent}% tax?", "My gross pay is {gross}; with {tax_percent}% tax what is my net?"], lambda r: {"gross": r.randrange(20, 120) * 100, "tax_percent": r.choice([15, 20, 25, 30, 35])},
   run=lambda a: round(a["gross"] * (1 - a["tax_percent"] / 100), 2), say=lambda a, x: f"Your net pay is {x:,.2f}.")
fn("finance.stock_quote", "Get the latest price of a stock.", [P("symbol", S, "Ticker symbol.", enum=["ACME", "GLBX", "INIT", "HOOL"])],
   ["What is the current price of {symbol}?", "Get a quote for {symbol}."], lambda r: {"symbol": r.choice(["ACME", "GLBX", "INIT", "HOOL"])},
   run=lambda a: {"symbol": a["symbol"], "price": 20 + h("s", a["symbol"]) % 180 + 0.25 * (h("s2", a["symbol"]) % 4)}, say=lambda a, x: f"{x['symbol']} is trading at {x['price']:.2f}.")
SYN = {"happy": ["joyful", "cheerful", "glad"], "sad": ["unhappy", "sorrowful", "gloomy"], "fast": ["quick", "rapid", "swift"], "big": ["large", "huge", "enormous"], "small": ["little", "tiny", "compact"], "smart": ["clever", "bright", "intelligent"]}
fn("language.synonyms", "Return synonyms for a common English word.", [P("word", S, "The word.", enum=list(SYN))],
   ["Give me synonyms for {word}.", "What are some other words for {word}?"], lambda r: {"word": r.choice(list(SYN))},
   run=lambda a: SYN[a["word"]], say=lambda a, x: f"Synonyms for {a['word']}: " + ", ".join(x) + ".")
fn("events.find_concerts", "Find concerts in a city.", [P("city", S, "City."), P("genre", S, "Music genre.", enum=["rock", "jazz", "classical", "pop"]), P("month", S, "Month name.", req=False)],
   ["Find {genre} concerts in {city}.", "Are there any {genre} concerts in {city} in {month}?"], lambda r: {"city": r.choice(CITIES), "genre": r.choice(["rock", "jazz", "classical", "pop"]), "month": r.choice(["April", "June", "September", "November"])},
   run=lambda a: [{"artist": f"{n} {a['genre'].title()} Ensemble", "date": rdate(r_ := random.Random(h("cc", a["city"], n)), 2026, 2026)} for n in ("North", "Silver")], say=lambda a, x: "Upcoming: " + "; ".join(f"{c['artist']} on {c['date']}" for c in x) + ".")
fn("sports.get_score", "Get the latest score of a team's match.", [P("team", S, "Team name."), P("date", S, "Match date, YYYY-MM-DD.")],
   ["What was the score of {team} on {date}?", "How did {team} do on {date}?"], lambda r: {"team": r.choice(["Harbor FC", "Lakeside United", "Valley Rovers", "North City"]), "date": rdate(r, 2025, 2026)},
   run=lambda a: {"score": f"{h('g1', a['team'], a['date']) % 5}-{h('g2', a['team'], a['date']) % 4}"}, say=lambda a, x: f"{a['team']} finished {x['score']}.")
fn("iot.set_thermostat", "Set the thermostat of a room.", [P("room", S, "Room name."), P("temperature_c", N, "Target temperature in Celsius."), P("mode", S, "Operating mode.", req=False, enum=["heat", "cool", "auto"])],
   ["Set the {room} to {temperature_c} degrees.", "Put the {room} thermostat to {temperature_c} C in {mode} mode."], lambda r: {"room": r.choice(["living room", "bedroom", "office", "kitchen"]), "temperature_c": r.randrange(16, 26), "mode": r.choice(["heat", "cool", "auto"])},
   run=lambda a: {"ok": True}, say=lambda a, x: f"The {a['room']} is now set to {a['temperature_c']} degrees.")
fn("iot.get_sensor_reading", "Read the current value of a sensor.", [P("sensor_id", S, "Sensor id like S-12.")],
   ["What does sensor {sensor_id} read?", "Read sensor {sensor_id}."], lambda r: {"sensor_id": f"S-{r.randrange(1, 60)}"},
   run=lambda a: {"value": round(15 + h("sn", a["sensor_id"]) % 150 / 10, 1), "unit": "C"}, say=lambda a, x: f"Sensor {a['sensor_id']} reads {x['value']} {x['unit']}.")
WATER = {("tomato", "summer"): 1, ("tomato", "winter"): 3, ("basil", "summer"): 2, ("basil", "winter"): 4, ("cactus", "summer"): 10, ("cactus", "winter"): 21, ("fern", "summer"): 2, ("fern", "winter"): 4}
fn("plants.watering_interval", "Return how many days to wait between waterings.", [P("plant", S, "Plant.", enum=["tomato", "basil", "cactus", "fern"]), P("season", S, "Season.", enum=["summer", "winter"])],
   ["How often should I water my {plant} in {season}?", "Watering interval for a {plant} during {season}?"], lambda r: {"plant": r.choice(["tomato", "basil", "cactus", "fern"]), "season": r.choice(["summer", "winter"])},
   run=lambda a: WATER[(a["plant"], a["season"])], say=lambda a, x: f"Water your {a['plant']} every {x} days in {a['season']}.")
fn("util.random_number", "Return a reproducible random integer between two bounds using a seed.", [P("minimum", I, "Lower bound."), P("maximum", I, "Upper bound."), P("seed", I, "Random seed.")],
   ["Pick a random number between {minimum} and {maximum} with seed {seed}.", "Random integer from {minimum} to {maximum}, seed {seed}."], lambda r: {"minimum": r.randrange(1, 20), "maximum": r.randrange(30, 500), "seed": r.randrange(1, 9999)},
   run=lambda a: random.Random(a["seed"]).randint(a["minimum"], a["maximum"]), say=lambda a, x: f"The number is {x}.")
fn("util.generate_password", "Generate a password of a given length.", [P("length", I, "Number of characters."), P("include_symbols", B, "Include symbols.", req=False, default=True)],
   ["Generate a {length} character password.", "Make me a password with {length} characters and symbols: {include_symbols}."], lambda r: {"length": r.randrange(8, 33), "include_symbols": r.random() < 0.5},
   run=lambda a: "".join(("abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789" + ("!#$%&*" if a.get("include_symbols", True) else ""))[h("pw", a["length"], i) % (54 + (6 if a.get("include_symbols", True) else 0))] for i in range(a["length"])),
   say=lambda a, x: f"Here is your password: {x}")
fn("util.qr_code", "Create a QR code image for a text or link.", [P("text", S, "Content to encode."), P("size", I, "Image size in pixels.", req=False, default=200)],
   ["Make a QR code for {text}.", "Create a {size} pixel QR code that links to {text}."], lambda r: {"text": r.choice(["https://example.com/menu", "https://example.com/wifi", "Hello world", "https://example.com/tickets"]), "size": r.choice([128, 200, 256, 512])},
   run=lambda a: {"image_url": f"https://img.example/qr/{h('qr', a['text']) % 99999}.png"}, say=lambda a, x: f"Your QR code is ready: {x['image_url']}")

DOMAINS = sorted({f["domain"] for f in FUNCS.values()})
# ------------------------------------------------------------------------------------------------------------------------------- general Q&A for irrelevance
QA = [("What is the capital of France?", "The capital of France is Paris."), ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
      ("How many days are in a week?", "There are seven days in a week."), ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
      ("What colour do you get by mixing blue and yellow?", "Mixing blue and yellow gives green."), ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
      ("How many continents are there?", "There are seven continents."), ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
      ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."), ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit."),
      ("What language is spoken in Brazil?", "Portuguese is the main language spoken in Brazil."), ("What is the tallest mountain above sea level?", "Mount Everest is the tallest mountain above sea level."),
      ("How many sides does a hexagon have?", "A hexagon has six sides."), ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
      ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."), ("What is the opposite of 'ancient'?", "The opposite of ancient is modern."),
      ("Who developed the theory of general relativity?", "Albert Einstein developed the theory of general relativity."), ("What is a group of lions called?", "A group of lions is called a pride."),
      ("How many minutes are in an hour?", "There are 60 minutes in an hour."), ("What does CPU stand for?", "CPU stands for central processing unit."),
      ("What is the main ingredient of guacamole?", "The main ingredient of guacamole is avocado."), ("Which is the smallest prime number?", "The smallest prime number is 2."),
      ("What do bees make?", "Bees make honey and beeswax."), ("In which country is the Colosseum?", "The Colosseum is in Rome, Italy."),
      ("What is the past tense of 'go'?", "The past tense of go is went."), ("Name a primary colour.", "Red is a primary colour; blue and yellow are too."),
      ("What is 7 times 8?", "7 times 8 is 56."), ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon."),
      ("What is the currency of Japan?", "The currency of Japan is the yen."), ("What is the longest bone in the human body?", "The femur, or thigh bone, is the longest bone in the human body.")]
UNSUPPORTED = ["Order me a pizza.", "Turn off the living room lights.", "Write me a poem about autumn.", "Translate 'good morning' into Spanish.", "Tell me a joke.", "Recommend a good book for the weekend.",
               "What should I name my new puppy?", "Draft a thank-you note to my neighbour.", "Explain what machine learning is.", "Give me three tips for a job interview."]
UNSUPPORTED_ANS = {
    "Order me a pizza.": "I can't order food with the tools I have here, but I'm happy to help you compare toppings or find a recipe to make at home.",
    "Turn off the living room lights.": "I don't have a tool to control your lights, so I can't do that from here. You could use your smart-home app or the wall switch.",
    "Write me a poem about autumn.": "Gold leaves drift down in the quiet air,\nThe evening light is soft and rare,\nThe season turns with a gentle sigh,\nAnd autumn paints the open sky.",
    "Translate 'good morning' into Spanish.": "'Good morning' in Spanish is 'buenos dias'.",
    "Tell me a joke.": "Why did the scarecrow win an award? Because he was outstanding in his field.",
    "Recommend a good book for the weekend.": "Try a short novel you can finish in a couple of sittings, such as 'The Old Man and the Sea' by Ernest Hemingway.",
    "What should I name my new puppy?": "Some friendly ideas: Biscuit, Juniper, Pixel, Mocha or Scout. Pick a short name that is easy to call out.",
    "Draft a thank-you note to my neighbour.": "Dear neighbour, thank you for keeping an eye on our house last week. It meant a lot to us, and we are grateful to have you next door. Warm regards.",
    "Explain what machine learning is.": "Machine learning is a way of building software that learns patterns from examples instead of following hand-written rules, so it can make predictions on new data.",
    "Give me three tips for a job interview.": "1) Research the company beforehand. 2) Practise answers to common questions out loud. 3) Prepare two or three questions to ask them.",
}


QA += [("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."), ("How many legs does a spider have?", "A spider has eight legs."),
       ("What is the square root of 144?", "The square root of 144 is 12."), ("Which animal is known as the king of the jungle?", "The lion is often called the king of the jungle."),
       ("What is the capital of Australia?", "The capital of Australia is Canberra."), ("How many players are on a football (soccer) team on the pitch?", "Each side has eleven players on the pitch."),
       ("What is the hardest natural substance?", "Diamond is the hardest natural substance."), ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone."),
       ("What is the longest river in Africa?", "The Nile is the longest river in Africa."), ("How many strings does a standard violin have?", "A standard violin has four strings."),
       ("What do caterpillars turn into?", "Caterpillars turn into butterflies or moths."), ("What is the chemical formula of carbon dioxide?", "Carbon dioxide is CO2."),
       ("Which metal is liquid at room temperature?", "Mercury is the metal that is liquid at room temperature."), ("What is the main language spoken in Egypt?", "Arabic is the main language spoken in Egypt."),
       ("How many months have 31 days?", "Seven months have 31 days."), ("What is the plural of 'mouse'?", "The plural of mouse is mice."), ("What is 15% of 200?", "15% of 200 is 30."),
       ("Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun."), ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
       ("Who painted The Starry Night?", "The Starry Night was painted by Vincent van Gogh."), ("What is the largest mammal?", "The blue whale is the largest mammal."),
       ("What shape has three sides?", "A triangle has three sides."), ("How many hours are in a day?", "There are 24 hours in a day."), ("What is the currency of the United Kingdom?", "The currency of the United Kingdom is the pound sterling."),
       ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."), ("What does HTML stand for?", "HTML stands for HyperText Markup Language."),
       ("What is the speed of sound in air, roughly?", "Sound travels at about 343 metres per second in air at room temperature."), ("Which ocean is the smallest?", "The Arctic Ocean is the smallest ocean."),
       ("What is a synonym for 'big'?", "A synonym for big is large."), ("How many degrees are in a right angle?", "A right angle is 90 degrees."), ("Which organ pumps blood around the body?", "The heart pumps blood around the body."),
       ("What is the capital of Canada?", "The capital of Canada is Ottawa."), ("What is the main gas in the air we breathe?", "Nitrogen is the main gas in air, making up about 78 percent."),
       ("What is the opposite of 'increase'?", "The opposite of increase is decrease."), ("Who discovered penicillin?", "Alexander Fleming discovered penicillin."), ("What is the boiling point of water in Fahrenheit?", "Water boils at 212 degrees Fahrenheit at sea level."),
       ("What is the largest desert in the world by area?", "Antarctica is the largest desert by area, and the Sahara is the largest hot desert."), ("How many bones are in the adult human body?", "An adult human has 206 bones."),
       ("What is the next prime number after 7?", "The next prime after 7 is 11."), ("In which year did World War II end?", "World War II ended in 1945."), ("What is the capital of Spain?", "The capital of Spain is Madrid."),
       ("What is a baby kangaroo called?", "A baby kangaroo is called a joey."), ("Which part of a plant absorbs water from the soil?", "The roots absorb water from the soil."), ("What is the Roman numeral for 50?", "The Roman numeral for 50 is L."),
       ("What does GPS stand for?", "GPS stands for Global Positioning System."), ("What is the tallest animal?", "The giraffe is the tallest animal."), ("Which season comes after winter?", "Spring comes after winter."),
       ("How many sides does an octagon have?", "An octagon has eight sides."), ("What is the symbol for the element oxygen?", "The symbol for oxygen is O."), ("Which instrument has black and white keys?", "The piano has black and white keys."),
       ("What is 9 squared?", "9 squared is 81."), ("What is the capital of Italy?", "The capital of Italy is Rome."), ("Who wrote 'Pride and Prejudice'?", "Pride and Prejudice was written by Jane Austen."),
       ("What is the study of the weather called?", "The study of weather is called meteorology."), ("How many continents does the Earth have?", "The Earth has seven continents by the common convention."),
       ("What do pandas mainly eat?", "Giant pandas mainly eat bamboo."), ("What is the largest bone in the human body?", "The femur is the largest bone in the human body."), ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
       ("Which vitamin helps the body absorb calcium?", "Vitamin D helps the body absorb calcium."), ("What is the unit of electric current?", "The unit of electric current is the ampere."), ("What is the first month of the year?", "January is the first month of the year."),
       ("How many centimetres are in a metre?", "There are 100 centimetres in a metre."), ("What is the process of water turning into vapour called?", "That process is called evaporation."), ("Who is the author of '1984'?", "1984 was written by George Orwell."),
       ("What is a triangle with all sides equal called?", "A triangle with all sides equal is called an equilateral triangle."), ("What is the capital of Portugal?", "The capital of Portugal is Lisbon.")]
UNSUPPORTED += ["Pick me a good name for my bakery.", "Write a haiku about the sea.", "Give me a recipe for pancakes.", "How do I politely decline a wedding invitation?", "Suggest a workout for beginners.",
                "Rewrite this sentence to sound more formal: 'Hey, send me the file.'", "What are the pros and cons of working from home?", "Explain recursion in simple terms.", "Give me a packing list for a beach holiday.",
                "Write a short bedtime story about a fox.", "How can I improve my public speaking?", "What is a good way to learn vocabulary?", "Write a limerick about a cat.", "Suggest three names for a hiking club.",
                "Help me write a birthday message for my sister.", "What should I consider when adopting a dog?", "Summarise the plot of Cinderella in two sentences.", "Describe the colour blue to someone who has never seen it.",
                "Give me ideas for a rainy-day activity with kids.", "How do I start a vegetable garden?", "Write a motivational quote for Monday.", "Explain the difference between 'affect' and 'effect'.", "What is a healthy breakfast?",
                "Draft a polite out-of-office message.", "How do I write a good thank-you email?", "Tell me an interesting fact about octopuses."]
UNSUPPORTED_ANS.update({
    "Pick me a good name for my bakery.": "A few ideas: Golden Crumb, The Rising Loaf, Flour & Fable, Little Oven or Sunday Bread. Choose one that is short, easy to say and tells people what you sell.",
    "Write a haiku about the sea.": "Grey waves fold and roll,\nsalt wind carries gulls aloft,\nthe tide keeps its time.",
    "Give me a recipe for pancakes.": "Whisk 200 g flour, 2 eggs, 300 ml milk and a pinch of salt into a smooth batter, rest it for 10 minutes, then fry a ladle at a time in a lightly oiled pan for about a minute per side.",
    "How do I politely decline a wedding invitation?": "Reply promptly, thank them for the invitation, say you are sorry you cannot attend and wish them a wonderful day. A short handwritten note or message is enough; you do not need a detailed reason.",
    "Suggest a workout for beginners.": "Try three days a week: a 5 minute walk to warm up, then 2 rounds of 10 squats, 8 knee push-ups, 20 seconds of plank and 10 glute bridges, and finish with gentle stretching. Add a little each week.",
    "Rewrite this sentence to sound more formal: 'Hey, send me the file.'": "Could you please send me the file at your earliest convenience?",
    "What are the pros and cons of working from home?": "Pros: no commute, flexible hours and fewer interruptions. Cons: it can feel isolating, the line between work and home blurs, and collaboration can be slower.",
    "Explain recursion in simple terms.": "Recursion is when a function solves a problem by calling itself on a smaller version of the same problem until it reaches a case simple enough to answer directly.",
    "Give me a packing list for a beach holiday.": "Swimwear, sunscreen, a hat and sunglasses, a light cover-up, sandals, a towel, a refillable water bottle, a book, a phone charger and a small first-aid kit.",
    "Write a short bedtime story about a fox.": "A little fox named Pip could not sleep, so she counted the stars from her burrow. When she reached forty, the moon smiled, the wind hushed the trees, and Pip curled up warm and safe and slept until morning.",
    "How can I improve my public speaking?": "Practise out loud, record yourself, start with a clear opening and a single main message, slow down, and use pauses. Speaking in front of a few friends first helps build confidence.",
    "What is a good way to learn vocabulary?": "Learn words in context from reading, review them with spaced-repetition flashcards, and use each new word in a sentence of your own within a day.",
    "Write a limerick about a cat.": "A cat with a talent for naps\nStretched out on the warm window gaps;\nShe purred with delight\nFrom morning till night,\nAnd dreamed of the sun on her laps.",
    "Suggest three names for a hiking club.": "Trail Blazers, Summit Seekers and Boots & Views are three friendly options.",
    "Help me write a birthday message for my sister.": "Happy birthday! Thank you for always being there for me. I hope your day is full of laughter, cake and everything you love. Here is to another great year!",
    "What should I consider when adopting a dog?": "Think about your time, space and budget, the dog's size and energy level, the cost of food and vet care, and whether everyone at home is ready for a long-term commitment.",
    "Summarise the plot of Cinderella in two sentences.": "Cinderella, mistreated by her stepfamily, goes to the royal ball with the help of her fairy godmother but must leave at midnight, losing a glass slipper. The prince finds her by the slipper and they marry.",
    "Describe the colour blue to someone who has never seen it.": "Blue feels like the calm of cool water or a clear sky: a quiet, open colour that many people associate with peace, distance and the sea.",
    "Give me ideas for a rainy-day activity with kids.": "Build a blanket fort, bake simple cookies, hold an indoor treasure hunt, make a paper-plate craft or put on a puppet show.",
    "How do I start a vegetable garden?": "Pick a sunny spot, prepare good soil or pots with drainage, begin with easy crops like lettuce, radishes and herbs, water regularly and add a little compost as plants grow.",
    "Write a motivational quote for Monday.": "A new week is a blank page, so write the first line with intention.",
    "Explain the difference between 'affect' and 'effect'.": "'Affect' is usually a verb meaning to influence something, while 'effect' is usually a noun meaning the result. For example: the rain affected our plans, and the effect was a cancelled picnic.",
    "What is a healthy breakfast?": "Something with protein, fibre and some fruit, for example oats with yoghurt and berries, or wholegrain toast with egg and tomato.",
    "Draft a polite out-of-office message.": "Thank you for your message. I am out of the office until Monday and will reply when I return. For urgent matters please contact a colleague at the main office.",
    "How do I write a good thank-you email?": "Open with a clear thank you, mention exactly what you appreciated, say what it made possible, and close warmly. Keep it short and send it promptly.",
    "Tell me an interesting fact about octopuses.": "An octopus has three hearts: two pump blood through the gills and one pumps it to the rest of the body."})

PLANS = ["I'll call the function for that.", "Let me look that up.", "I'll run that now.", "I'll use the right tool for this.", ""]


def _call(f, args):
    return {"type": "function", "function": {"name": f["name"], "arguments": args}}


def _ask(f, r, args=None, bare=False):
    """-> (prompt text, args) ; the prompt only mentions the arguments that end up in the call."""
    args = args or f["sample"](r)
    req = {p["name"] for p in f["params"] if p["req"]}
    idx = [i for i, u in enumerate(f["used"]) if req <= u and all(k in args for k in u)]
    if not idx:                                             # fall back to a template whose placeholders are all available
        idx = [i for i, u in enumerate(f["used"]) if all(k in args for k in u)]
    t = f["asks"][r.choice(idx)]
    used = set(re.findall(r"{(\w+)}", t))
    views = {p["name"]: p["view"] for p in f["params"] if p["view"]}
    vals = {k: (views[k](args[k]) if k in views else fmt(args[k])) for k in used}
    text = t.format(**vals)
    return text, {p["name"]: args[p["name"]] for p in f["params"] if p["name"] in used}


def _result(f, args, r):
    out = f["run"](args)
    return out, json.dumps(out if not isinstance(out, float) else round(out, 4), ensure_ascii=False)


def _norm(args, f):
    """JSON-native argument values (ints stay ints, floats that are whole become ints) for stable rendering."""
    return args


def ex_simple(r, ending="call"):
    f = r.choice(list(FUNCS.values()))
    text, args = _ask(f, r)
    tools = [schema(f)]
    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": r.choice(PLANS) if ending == "loop" else "", "tool_calls": [_call(f, args)]}]
    if ending == "loop":
        out, js = _result(f, args, r)
        msgs += [{"role": "tool", "content": js}, {"role": "assistant", "content": f["say"](args, out)}]
    return msgs, tools, [f["name"]]


def pick_distractors(r, f, k):
    others = [g for g in FUNCS.values() if g["name"] != f["name"]]
    same = [g for g in others if g["domain"] == f["domain"]]
    pool = r.sample(same, min(len(same), r.randrange(0, 3))) + r.sample(others, k)
    seen, out = set(), []
    for g in pool:
        if g["name"] not in seen:
            seen.add(g["name"])
            out.append(g)
    return out[:k]


def ex_multiple(r, ending="call"):
    f = r.choice(list(FUNCS.values()))
    text, args = _ask(f, r)
    offered = [f] + pick_distractors(r, f, r.randrange(1, 5))
    r.shuffle(offered)
    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "", "tool_calls": [_call(f, args)]}]
    if ending == "loop":
        out, js = _result(f, args, r)
        msgs += [{"role": "tool", "content": js}, {"role": "assistant", "content": f["say"](args, out)}]
    return msgs, [schema(g) for g in offered], [f["name"]]


def ex_parallel(r, ending="call"):
    f = r.choice([g for g in FUNCS.values() if len(g["asks"]) >= 2])
    n = r.randrange(2, 5)
    asks, calls, seen = [], [], set()
    for _ in range(40):                                    # functions with few distinct samples may not reach n unique calls
        if len(calls) >= n:
            break
        text, args = _ask(f, r)
        key = json.dumps(args, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        asks.append(text)
        calls.append((args))
    prompt = asks[0] + "".join(r.choice([" Also, ", " And ", " Then ", " Additionally, ", " After that, ", " Also: "]) + a[0].lower() + a[1:] for a in asks[1:])
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": r.choice(["", "", "I'll run these calls together."]), "tool_calls": [_call(f, a) for a in calls]}]
    if ending == "loop":
        outs = [_result(f, a, r) for a in calls]
        msgs += [{"role": "tool", "content": js} for _, js in outs]
        msgs.append({"role": "assistant", "content": " ".join(f["say"](a, o) for a, (o, _) in zip(calls, outs))})
    offered = [f] + pick_distractors(r, f, r.randrange(0, 3))
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [f["name"]] * n


def ex_parallel_multiple(r, ending="call"):
    fs = r.sample(list(FUNCS.values()), r.randrange(2, 4))
    parts = [_ask(g, r) for g in fs]
    prompt = parts[0][0] + "".join(r.choice([" Also, ", " And ", " Then ", " Additionally, ", " After that, ", " Also: "]) + t[0].lower() + t[1:] for t, _ in parts[1:])
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "", "tool_calls": [_call(g, a) for g, (_, a) in zip(fs, parts)]}]
    if ending == "loop":
        outs = [_result(g, a, r) for g, (_, a) in zip(fs, parts)]
        msgs += [{"role": "tool", "content": js} for _, js in outs]
        msgs.append({"role": "assistant", "content": " ".join(g["say"](a, o) for g, (_, a), (o, _) in zip(fs, parts, outs))})
    offered = fs + pick_distractors(r, fs[0], r.randrange(0, 3))
    offered = [g for i, g in enumerate(offered) if g["name"] not in {x["name"] for x in offered[:i]}]
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [g["name"] for g in fs]


def ex_irrelevance(r, ending="call"):
    offered = r.sample(list(FUNCS.values()), r.randrange(1, 5))
    if r.random() < 0.65:
        q, a = r.choice(QA)
    else:
        q = r.choice(UNSUPPORTED)
        a = UNSUPPORTED_ANS[q]
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}], [schema(g) for g in offered], []


def ex_missing(r, ending="call"):
    f = r.choice([g for g in FUNCS.values() if any(p["req"] for p in g["params"])])
    req = [p for p in f["params"] if p["req"]]
    base = f["desc"].rstrip(".")
    base = base[0].lower() + base[1:]
    user = r.choice([f"Can you {base}?", f"I need you to {base}.", f"Please {base} for me."])
    need = "; ".join(f"{p['name'].replace('_', ' ')} ({p['desc'].rstrip('.').lower()})" for p in req)
    q = r.choice([f"Sure - I just need a few details first: {need}. What should I use?", f"I can do that, but I'm missing some required information: {need}. Could you provide it?",
                  f"Happy to help. Before I call the function I need: {need}."])
    offered = [f] + pick_distractors(r, f, r.randrange(0, 3))
    r.shuffle(offered)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": q}], [schema(g) for g in offered], []


def ex_two_turn(r, ending="call"):
    f1 = r.choice(list(FUNCS.values()))
    f2 = f1 if r.random() < 0.4 else r.choice(list(FUNCS.values()))
    t1, a1 = _ask(f1, r)
    t2, a2 = _ask(f2, r)
    o1, j1 = _result(f1, a1, r)
    o2, j2 = _result(f2, a2, r)
    follow = r.choice(["Thanks. Next: ", "Great. Now ", "Okay, and ", "One more thing: "])
    msgs = [{"role": "user", "content": t1}, {"role": "assistant", "content": "", "tool_calls": [_call(f1, a1)]}, {"role": "tool", "content": j1}, {"role": "assistant", "content": f1["say"](a1, o1)},
            {"role": "user", "content": follow + t2[0].lower() + t2[1:]}, {"role": "assistant", "content": "", "tool_calls": [_call(f2, a2)]}]
    if ending == "loop":
        msgs += [{"role": "tool", "content": j2}, {"role": "assistant", "content": f2["say"](a2, o2)}]
    offered = [f1, f2] + pick_distractors(r, f1, r.randrange(0, 3))
    offered = [g for i, g in enumerate(offered) if g["name"] not in {x["name"] for x in offered[:i]}]
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [f1["name"], f2["name"]]



def ex_clarify_call(r, ending="loop"):
    """User first asks without the details, the assistant asks for them, the user answers, then the call (and result) follows."""
    f = r.choice([g for g in FUNCS.values() if any(p["req"] for p in g["params"]) and g["run"] is not None])
    req = [p for p in f["params"] if p["req"]]
    base = f["desc"].rstrip(".")
    base = base[0].lower() + base[1:]
    user1 = r.choice([f"Can you {base}?", f"I need you to {base}.", f"Please {base} for me."])
    need = "; ".join(f"{p['name'].replace('_', ' ')} ({p['desc'].rstrip('.').lower()})" for p in req)
    ask = r.choice([f"Sure - I just need a few details first: {need}. What should I use?", f"Happy to help. Before I call the function I need: {need}."])
    args = f["sample"](r)
    args = {p["name"]: args[p["name"]] for p in req}
    views = {p["name"]: p["view"] for p in f["params"] if p["view"]}
    parts = [f"the {p['name'].replace('_', ' ')} is {views[p['name']](args[p['name']]) if p['name'] in views else fmt(args[p['name']])}" for p in req]
    user2 = r.choice(["", "Sure: ", "Okay - ", "Here you go: "]) + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
    user2 = user2[0].upper() + user2[1:] if not user2.startswith(("Sure", "Okay", "Here")) else user2
    msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": ask}, {"role": "user", "content": user2}, {"role": "assistant", "content": "", "tool_calls": [_call(f, args)]}]
    if ending == "loop":
        out, js = _result(f, args, r)
        msgs += [{"role": "tool", "content": js}, {"role": "assistant", "content": f["say"](args, out)}]
    offered = [f] + pick_distractors(r, f, r.randrange(0, 3))
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [f["name"]]


NOT_FOUND_FUNCS = {"hr.get_employee": ("employee_id", "No employee with id {v} exists.", "I couldn't find an employee with id {v}: the lookup says no such employee exists. Could you double-check the id?"),
                   "shop.track_order": ("order_id", "No order with id {v} was found.", "I couldn't find order {v}: the system reports that no such order exists. Please check the order number and I'll try again.")}
EMPTY_FUNCS = {"shop.search_products": ("[]", "The search for {q} returned no products. Try broader words, remove the price limit or pick another category and I'll search again."),
               "mailer.search_inbox": ('{"count": 0}', "No emails match {q}. Try a different keyword or another folder and I'll search again."),
               "devtools.search_code": ('{"matches": 0}', "There are no matches for {q}. You could try a looser pattern or another file type."),
               "food.find_restaurants": ("[]", "I didn't find any matching restaurants. Try another cuisine or a different price level and I'll look again.")}


def ex_tool_error(r, ending="loop"):
    kind = r.choices(["transient", "not_found", "empty"], [0.55, 0.25, 0.20])[0]
    if kind == "transient":
        f = r.choice([g for g in FUNCS.values() if g["run"] is not None])
        text, args = _ask(f, r)
        out, js = _result(f, args, r)
        msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "", "tool_calls": [_call(f, args)]},
                {"role": "tool", "content": json.dumps({"error": "The service is temporarily unavailable. Please try again.", "code": 503})},
                {"role": "assistant", "content": r.choice(["That failed with a temporary error, so I'll try the same call again.", "The service reported a temporary problem. Retrying the same request."]), "tool_calls": [_call(f, args)]},
                {"role": "tool", "content": js}, {"role": "assistant", "content": f["say"](args, out)}]
    elif kind == "not_found":
        name = r.choice(list(NOT_FOUND_FUNCS))
        f = FUNCS[name]
        pname, err, say = NOT_FOUND_FUNCS[name]
        text, args = _ask(f, r)
        v = args[pname]
        msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "", "tool_calls": [_call(f, args)]},
                {"role": "tool", "content": json.dumps({"error": err.format(v=v), "code": 404})}, {"role": "assistant", "content": say.format(v=v)}]
    else:
        name = r.choice(list(EMPTY_FUNCS))
        f = FUNCS[name]
        raw, say = EMPTY_FUNCS[name]
        text, args = _ask(f, r)
        q = args.get("query") or args.get("pattern") or args.get("cuisine", "that")
        msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "", "tool_calls": [_call(f, args)]}, {"role": "tool", "content": raw},
                {"role": "assistant", "content": say.format(q=f"\"{q}\"")}]
    offered = [f] + pick_distractors(r, f, r.randrange(0, 3))
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [f["name"]]



def _numeric_sources():
    """Every function whose result is a positive number for all of a few sampled inputs (so its result can feed another call)."""
    out = []
    for name, f in FUNCS.items():
        if f["run"] is None:
            continue
        try:
            outs = [f["run"](f["sample"](random.Random(i))) for i in range(8)]
        except Exception:
            continue
        if all(isinstance(o, (int, float)) and not isinstance(o, bool) and 0 < o < 10 ** 7 for o in outs):
            out.append(name)
    return out


NUM_SRC = _numeric_sources()


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def ex_dep_chain(r, ending="loop"):
    """Two dependent calls: the result of the first becomes an argument of the second ("... now convert THAT to binary")."""
    kind = r.choices(["num", "date", "weather"], [0.7, 0.15, 0.15])[0]
    if kind == "date":
        f1, f2 = FUNCS["dates.add_days"], FUNCS["dates.day_of_week"]
        t1, a1 = _ask(f1, r)
        o1, j1 = _result(f1, a1, r)
        a2 = {"date": o1}
        o2, j2 = _result(f2, a2, r)
        follow = r.choice([f"Thanks. What weekday is that date?", "And which day of the week does that fall on?", "What day of the week is that?"])
    elif kind == "weather":
        f1, f2 = FUNCS["weather.get_current"], FUNCS["units.convert_temperature"]
        a1 = {"city": r.choice(CITIES), "units": "celsius"}
        t1 = r.choice([f"What's the weather like in {a1['city']} right now in celsius?", f"Tell me the current weather in {a1['city']} in celsius."])
        o1, j1 = _result(f1, a1, r)
        a2 = {"value": o1["temperature"], "from_unit": "C", "to_unit": "F"}
        o2, j2 = _result(f2, a2, r)
        follow = r.choice(["Please convert that temperature to Fahrenheit.", "What is that in Fahrenheit?", "Convert that to F."])
    else:
        for _ in range(50):
            f1 = FUNCS[r.choice(NUM_SRC)]
            t1, a1 = _ask(f1, r)
            o1, j1 = _result(f1, a1, r)
            if _isnum(o1) and 0 < o1 < 10 ** 7:
                break
        else:
            return ex_dep_chain(random.Random(r.random()), ending)
        sinks = []
        if isinstance(o1, int) and o1 >= 1:
            sinks.append(("math.to_binary", lambda: ({"number": o1}, "Now convert that result to binary.")))
        if isinstance(o1, int) and 2 <= o1 <= 5000:
            sinks.append(("math.is_prime", lambda: ({"number": o1}, "Is that number prime?")))
        sinks.append(("finance.vat_price", lambda: (lambda v: ({"net_price": o1, "vat_percent": v}, f"Treat that as a price and add {v}% VAT."))(r.choice([7, 19, 21]))))
        sinks.append(("finance.tip_split", lambda: (lambda tp, n: ({"bill": o1, "tip_percent": tp, "people": n}, f"Treat that as a bill and split it between {n} people with a {tp}% tip."))(r.choice([10, 15, 20]), r.randrange(2, 6))))
        if 0 < o1 <= 60:
            sinks.append(("geometry.circle_area", lambda: ({"radius": o1}, "What is the area of a circle with that radius?")))
        if isinstance(o1, int) and 1 <= o1 <= 900:
            sinks.append(("time.add_minutes", lambda: (lambda tm: ({"time": tm, "minutes": o1}, f"What time is it that many minutes after {tm}?"))(f"{r.randrange(6, 20):02d}:{r.choice(['00', '30'])}")))
        if 0 < o1 <= 3650 and isinstance(o1, int):
            sinks.append(("dates.add_days", lambda: (lambda d: ({"date": d, "days": o1}, r.choice([f"Add that many days to {d}.", f"What date is that many days after {d}?"])))(rdate(r, 2026, 2026))))
        sinks.append(("math.percentage", lambda: (lambda w: ({"part": o1, "whole": w}, r.choice([f"What percentage is that of {w}?", f"Express that as a percent of {w}."])))(r.choice([100, 200, 250, 500, 1000]) if o1 <= 100 else o1 * r.choice([2, 4, 5, 10]))))
        sinks.append(("units.convert_length", lambda: (lambda u: ({"value": o1, "from_unit": u[0], "to_unit": u[1]}, r.choice([f"Treat that as a length in {u[0]} and convert it to {u[1]}.", f"Convert that many {u[0]} to {u[1]}."])))(r.sample(list(LEN), 2))))
        sinks.append(("finance.currency_convert", lambda: (lambda c: ({"amount": o1, "from_currency": c[0], "to_currency": c[1]}, r.choice([f"Convert that amount from {c[0]} to {c[1]}.", f"Treat that as {c[0]} and tell me the value in {c[1]}."])))(r.choice(list(RATES)))))
        name, mk = r.choice(sinks)
        f2 = FUNCS[name]
        a2, follow = mk()
        o2, j2 = _result(f2, a2, r)
    msgs = [{"role": "user", "content": t1}, {"role": "assistant", "content": "", "tool_calls": [_call(f1, a1)]}, {"role": "tool", "content": j1}, {"role": "assistant", "content": f1["say"](a1, o1)},
            {"role": "user", "content": follow}, {"role": "assistant", "content": r.choice(["", "", "I'll use the result from before."]), "tool_calls": [_call(f2, a2)]}, {"role": "tool", "content": j2},
            {"role": "assistant", "content": f2["say"](a2, o2)}]
    offered = [f1, f2] + pick_distractors(r, f1, r.randrange(0, 3))
    offered = [g for i, g in enumerate(offered) if g["name"] not in {x["name"] for x in offered[:i]}]
    r.shuffle(offered)
    return msgs, [schema(g) for g in offered], [f1["name"], f2["name"]]


EXTRA_CATEGORIES = {"clarify_call": (ex_clarify_call, 1.0), "tool_error": (ex_tool_error, 1.0), "dep_chain": (ex_dep_chain, 1.0)}

CATEGORIES = {"simple": (ex_simple, 1.0), "multiple": (ex_multiple, 1.4), "parallel": (ex_parallel, 1.0), "parallel_multiple": (ex_parallel_multiple, 1.0),
              "irrelevance": (ex_irrelevance, 0.8), "missing_param": (ex_missing, 0.5), "two_turn": (ex_two_turn, 0.8)}
SYSTEM = [None, None, None, "You are a helpful assistant with access to functions. Call a function only when it is needed and use exact argument values.",
          "You can call the provided functions. If no function fits the request, answer directly. If a required argument is missing, ask for it."]


def build_one(cat: str, seed: int) -> dict:
    r = random.Random(f"fc:{cat}:{seed}")
    fnc, _ = (CATEGORIES.get(cat) or EXTRA_CATEGORIES[cat])
    ending = "loop" if (r.random() < 0.45 or cat in EXTRA_CATEGORIES) else "call"
    msgs, tools, names = fnc(r, ending)
    sys_msg = r.choice(SYSTEM)
    if sys_msg:
        msgs = [{"role": "system", "content": sys_msg}] + msgs
    return {"id": f"fc/{cat}/{seed}", "messages": msgs, "tools": tools,
            "meta": {"family": f"fc_{cat}", "category": "function_calling", "difficulty": "medium", "tags": ["bfcl_style", cat, ending], "seed": seed, "kind": "api",
                     "n_tool_calls": sum(len(m.get("tool_calls", [])) for m in msgs), "n_turns": sum(m["role"] == "user" for m in msgs), "n_tools_offered": len(tools), "called": names}}


def check(rec: dict) -> str | None:
    """Every call names an offered function and passes the schema (required args present, types right, enums respected)."""
    offered = {t["function"]["name"]: t["function"]["parameters"] for t in rec["tools"]}
    for m in rec["messages"]:
        for c in m.get("tool_calls", []):
            name, args = c["function"]["name"], c["function"]["arguments"]
            if name not in offered:
                return "called_unoffered_function"
            sch = offered[name]
            if not set(sch["required"]) <= set(args):
                return "missing_required_arg"
            for k, v in args.items():
                spec = sch["properties"].get(k)
                if spec is None:
                    return "unknown_arg"
                t = spec["type"]
                ok = {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool), "number": isinstance(v, (int, float)) and not isinstance(v, bool),
                      "boolean": isinstance(v, bool), "array": isinstance(v, list), "object": isinstance(v, dict) and set(v) == set(spec.get("properties", v))}[t]
                if not ok:
                    return f"bad_type:{k}"
                if spec.get("enum") and v not in spec["enum"]:
                    return f"bad_enum:{k}"
    if rec["messages"][-1]["role"] not in ("assistant", "tool"):
        return "bad_end"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--out", default="data/raw/fc_bfcl_style.jsonl")
    ap.add_argument("--seed-offset", type=int, default=20_000)
    ap.add_argument("--categories", default="", help="comma list (default: the original seven); add clarify_call,tool_error for the multi-turn / error-handling set")
    a = ap.parse_args()
    cats = {c: (CATEGORIES.get(c) or EXTRA_CATEGORIES[c]) for c in (a.categories.split(",") if a.categories else CATEGORIES)}
    tot = sum(w for _, w in cats.values())
    rows, seen, bad = [], set(), {}
    for cat, (_, w) in cats.items():
        want, made, k = int(a.n * w / tot), 0, 0
        while made < want and k < want * 4:
            rec = build_one(cat, a.seed_offset + k)
            k += 1
            why = check(rec)
            key = json.dumps([rec["messages"], [t["function"]["name"] for t in rec["tools"]]], sort_keys=True)
            if why or key in seen:
                bad[why or "duplicate"] = bad.get(why or "duplicate", 0) + 1
                continue
            seen.add(key)
            rows.append(rec)
            made += 1
    random.Random(2).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(FUNCS)} functions ({len(DOMAINS)} domains) -> {len(rows)} records in {a.out}; rejected {bad}")


if __name__ == "__main__":
    main()
