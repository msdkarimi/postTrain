"""Honest "I don't know / I can't" data: refusing to invent answers, in the cases where there is no answer to give.

    python -m distill.refusal_gen --out data/raw/refusal_data.jsonl

Seven families (all built from templates, so each refusal is correct for its prompt by construction):
  fictional_entity   books, authors, companies, cities, theorems, libraries, films, battles, elements, festivals with INVENTED names -> not familiar, won't guess
  live_personal      weather right now, stock prices, today's news, the current time, account balances, parcels, 'what did I say yesterday' -> no live / private access
  unknowable_future  who will win, next week's price, lottery numbers, will it rain on a date, what will happen in an election -> cannot predict
  ill_posed          division by zero, the largest prime, x + 3 = x + 5, a problem whose data does not contain the asked quantity -> explain why there is no answer
  not_in_text        questions a supplied document does not answer (from distill/context_gen.py) -> says the text does not say
  no_capability      a request that none of the OFFERED functions can do -> says which capability is missing and what the tools CAN do
  answerable_twin    ordinary questions about REAL things (from the fact tables of distill/mcq_gen.py) that are simply answered -> keeps the model from refusing everything
About 20% of the first three families continue with a user who pushes ("just guess"); the assistant stays honest and offers something useful instead.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYL = ["bre", "lva", "ne", "qui", "ll", "fea", "ther", "zor", "vath", "ken", "dro", "mal", "ix", "ust", "ova", "ley", "tarn", "wick", "sol", "umb", "nar", "fel", "gra", "hol", "pry", "vesk", "olm", "dun", "ire", "kas"]
ADJ = ["Silent", "Crimson", "Hollow", "Distant", "Gilded", "Paper", "Last", "Wandering", "Midnight", "Amber"]
NOUN = ["Orchard", "Compass", "Lantern", "Archive", "Harbour", "Garden", "Mirror", "Cartographer", "Winter", "Symphony"]
FIRST = ["Ilsa", "Torvald", "Marisol", "Edric", "Nadiya", "Corbin", "Ysolde", "Anselm", "Petra", "Lucan"]
SYSTEMS = ["If you do not know something, say so instead of guessing.", "You are an honest assistant. Never invent facts.", "Be accurate. Admit uncertainty when you have no reliable information."]


def fake(r, n=None):
    return "".join(r.choice(SYL) for _ in range(n or r.randrange(2, 4))).capitalize()


def pick(r, xs):
    return r.choice(xs)


# ---------------------------------------------------------------------------------------------------------------------------------- A: fictional entities
def a_item(r):
    kind = pick(r, ["book", "author", "company", "city", "theorem", "library", "film", "battle", "element", "festival", "band", "recipe"])
    f1, f2 = fake(r), fake(r)
    if kind == "book":
        t = f"The {pick(r, ADJ)} {pick(r, NOUN)} of {f1}"
        return kind, t, pick(r, [f"Who wrote \"{t}\"?", f"Can you summarise the book \"{t}\"?", f"What is the main theme of \"{t}\"?"]), "an author you like or a well-known novel"
    if kind == "author":
        n = f"{pick(r, FIRST)} {f1}{pick(r, ['son', 'berg', 'ova', 'ides', 'wood'])}"
        return kind, n, pick(r, [f"What are the best-known works of {n}?", f"When was the novelist {n} born?", f"Tell me about the author {n}."]), "authors in a particular genre"
    if kind == "company":
        n = f"{f1} {pick(r, ['Dynamics', 'Labs', 'Industries', 'Holdings', 'Robotics'])}"
        return kind, n, pick(r, [f"In what year was {n} founded?", f"Who is the CEO of {n}?", f"How many people does {n} employ?"]), "how to research a company"
    if kind == "city":
        n = f"{f1}{pick(r, ['burg', 'haven', 'ford', 'stad', 'ville'])}"
        return kind, n, pick(r, [f"What is the population of {n}?", f"What is the main industry of the town of {n}?", f"Which country is {n} in?"]), "large cities or regions you can name"
    if kind == "theorem":
        n = f"{f1}-{f2}"
        return kind, f"the {n} theorem", pick(r, [f"Can you explain the {n} theorem?", f"State and prove the {n} theorem.", f"Who proved the {n} theorem?"]), "related theorems such as the Pythagorean theorem or the central limit theorem"
    if kind == "library":
        n = f"{f1.lower()}{pick(r, ['kit', 'flow', 'py', 'tools'])}"
        return kind, f"the Python library {n}", pick(r, [f"How do I install the Python library {n} and what is its main function?", f"What does the Python package {n} do?", f"Show me an example using {n}."]), "popular libraries in the same area"
    if kind == "film":
        n = f"{pick(r, ADJ)} {f1}"
        return kind, f"the film \"{n}\"", pick(r, [f"Summarise the plot of the film \"{n}\".", f"Who directed \"{n}\"?", f"Which actors star in the movie \"{n}\"?"]), "films in a similar genre"
    if kind == "battle":
        n = f"the Battle of {f1}"
        return kind, n, pick(r, [f"Who won {n}?", f"In what year was {n} fought?", f"What were the consequences of {n}?"]), "well-documented battles of the period"
    if kind == "element":
        n = f"{f1}ium"
        return kind, f"the element {n}", pick(r, [f"What is the atomic number of {n}?", f"What are the properties of {n}?", f"Where is {n} found?"]), "elements on the periodic table"
    if kind == "festival":
        n = f"the {f1} Festival"
        return kind, n, pick(r, [f"When does {n} take place?", f"What is the history of {n}?", f"Where is {n} held?"]), "well-known festivals"
    n = f"{f1} {pick(r, ['Stew', 'Pie', 'Salad', 'Bread'])}"
    return "recipe", n, pick(r, [f"What are the ingredients of the traditional dish {n}?", f"How do I cook {n}?"]), "similar dishes you could cook"


A_OPEN = ["I'm not familiar with {x}", "I don't have any reliable information about {x}", "I can't find {x} in what I know", "I don't recognise {x}", "Nothing in my knowledge matches {x}", "{x} doesn't ring a bell for me",
          "I have no record of {x}", "I'm afraid I know nothing about {x}"]
A_MID = [", and I can't verify that it exists", ", so I can't tell whether it is real or a different spelling of something I know", ". It may be fictional or very obscure", ", and I don't want to guess",
         ". It could be a very local or very new thing that I have never seen", ", and I would rather say so than bluff", ". Perhaps the name is spelled differently or comes from a private source"]
A_CLOSE = [" Any details I gave would be made up, and I'd rather not do that.", " If I answered anyway I would be inventing facts.", " A confident-sounding guess would only mislead you.", " I'd rather admit that than invent an answer.",
           " Anything more specific would be fabricated.", " Making up details would not help you.", " I prefer to be honest about the gap.", " A plausible-sounding answer could easily be wrong."]
A_HELP = [" If you can share where you saw it (a link, an excerpt or the exact spelling), I can work from that. I can also help with {alt}.", " Could you give me some context or the source? Otherwise I'm glad to help with {alt}.",
          " Tell me where you came across it and I'll help from that text, or I can talk about {alt}.", " Paste a passage about it and I will happily summarise it, or ask me about {alt}.",
          " Check the spelling and let me know what you find - or I can start from {alt}.", " If you have a document that mentions it I can read it with you; otherwise I can cover {alt}."]
A_PUSH = [("Just give me your best guess.", "I understand, but for something I don't recognise a guess would simply be invented, and it could mislead you. If you paste what you know about it, I'll happily help from that."),
          ("Come on, you must know something about it.", "I really don't have reliable information about it, and I'd rather say so than make something up. If you share a source or a description I can help you work with it."),
          ("Make something plausible up then.", "I can write clearly labelled fiction if you want that, but I can't present invented details as facts about a real thing. Would you like a fictional story or a general overview of the topic instead?")]


def gen_a(r):
    kind, x, q, alt = a_item(r)
    ans = pick(r, A_OPEN).format(x=x) + pick(r, A_MID) + "." + pick(r, A_CLOSE) + pick(r, A_HELP).format(alt=alt)
    msgs = [("user", q), ("assistant", ans)]
    if r.random() < 0.2:
        msgs += list(zip(("user", "assistant"), pick(r, A_PUSH)))
    return f"fictional_{kind}", msgs


# ---------------------------------------------------------------------------------------------------------------------------------- B: live / personal data
CITIES = ["Oslo", "Lisbon", "Turin", "Zurich", "Lyon", "Porto", "Graz", "Ghent", "Malmo", "Utrecht", "Bilbao", "Krakow", "Vienna", "Dublin", "Athens", "Prague"]
LIVE = [("weather", ["What's the weather like in {c} right now?", "Is it raining in {c} at the moment?", "How warm is it in {c} today?"], "the current weather in {c}", "a weather app or your national weather service"),
        ("stock", ["What is {s} trading at right now?", "Give me the current share price of {s}.", "Is {s} up or down today?"], "live stock prices", "a finance site or your broker's app"),
        ("news", ["What are today's top news headlines?", "What happened in the news this morning?", "Any breaking news right now?"], "today's news", "a news site or app"),
        ("time", ["What time is it right now?", "What's today's date and the current time?", "Can you tell me the exact time in {c} at this moment?"], "the current time or date", "your device's clock"),
        ("bank", ["What's my bank balance?", "How much money is in my account?", "Did my salary arrive yet?"], "your accounts", "your bank's app or website"),
        ("parcel", ["Where is my package right now?", "Has my parcel been delivered yet?", "When exactly will my order arrive?"], "your orders or parcels", "the carrier's tracking page"),
        ("social", ["What's trending on social media right now?", "What did my friend post this morning?", "How many likes does my latest post have?"], "live social media data", "the app itself"),
        ("traffic", ["How bad is the traffic in {c} right now?", "Is the main road in {c} blocked at the moment?"], "live traffic information", "a navigation app"),
        ("shop", ["Is the supermarket in {c} open right now?", "Is the pharmacy near me open at this moment?"], "current opening hours or live status", "the shop's website or a maps app"),
        ("memory", ["What did I tell you yesterday?", "What was the topic of our last conversation?", "Do you remember my name from last week?"], "earlier conversations", "your notes or the chat history, if you have it"),
        ("device", ["What's my IP address?", "How much battery does my phone have left?", "Is my wifi working?"], "your device or network", "your device's settings"),
        ("rates", ["What is the euro to dollar exchange rate this minute?", "How much is one bitcoin worth right now?"], "live exchange rates", "a currency converter or an exchange"),
        ("sports", ["What's the score of the match that is on right now?", "Who is winning the game tonight?"], "live sports scores", "a sports app or website"),
        ("flight", ["Is my flight to {c} on time?", "Which gate is my flight to {c} leaving from?"], "live flight information", "the airline's app or the airport board"),
        ("calendar", ["What is on my calendar tomorrow?", "When is my next meeting?"], "your calendar", "your calendar app"),
        ("inbox", ["Do I have any new emails?", "Did my boss reply to my message?"], "your inbox", "your email app"),
        ("location", ["Where am I right now?", "How far am I from the nearest station?"], "your current location", "your phone's maps app"),
        ("price", ["How much does a flight to {c} cost today?", "What is the cheapest hotel in {c} tonight?"], "live prices and availability", "a booking site"),
        ("queue", ["How long is the queue at the post office right now?", "Is the museum in {c} crowded today?"], "live crowd levels", "the venue's website or a maps app")]
STOCKS = ["ACME", "GLBX", "INIT", "HOOL", "a major tech company", "my favourite airline"]
B_FORM = ["I don't have access to {what}, so I can't tell you that. For an accurate answer, check {where}.", "That needs {what}, which I can't see from here. {where_cap} is the right place to look.", "I can't reach {what}, so any answer from me would be a guess. Try {where}.", "I can't see {what} - I have no live or personal data connection here. {where_cap} will have the real answer.",
          "I'm not able to look up {what}, and I won't guess because a made-up answer could be wrong. Try {where}."]
B_PUSH = [("Just estimate it.", "Any number I gave would be invented, not an estimate based on data, so it could easily be wrong. {where_cap} will give you the real value in seconds."),
          ("Can't you just check?", "I don't have a way to check that from here. If you paste the information into the chat, I'm happy to help you interpret or summarise it.")]


def gen_b(r):
    key, qs, what, where = pick(r, LIVE)
    c, s = pick(r, CITIES), pick(r, STOCKS)
    q = pick(r, qs).format(c=c, s=s)
    w = what.format(c=c)
    cap = where[0].upper() + where[1:]
    ans = pick(r, B_FORM).format(what=w, where=where, where_cap=cap)
    msgs = [("user", q), ("assistant", ans)]
    if r.random() < 0.2:
        u, a = pick(r, B_PUSH)
        msgs += [("user", u), ("assistant", a.format(where_cap=cap))]
    return f"live_{key}", msgs


# ---------------------------------------------------------------------------------------------------------------------------------- C: unknowable future
FUT = [("sports", ["Who will win the {y} {e}?", "Which team will win the {e} in {y}?"], "who wins depends on many uncertain events", "I can go through the strengths and recent form of the teams if you like"),
       ("price", ["What will the price of {a} be next {p}?", "Will {a} go up next {p}? Give me a number."], "markets move on information nobody has yet", "I can explain what typically drives the price of {a} or how to think about risk"),
       ("lottery", ["Which lottery numbers will win this week?", "Give me the winning numbers for the next draw."], "lottery draws are random, so no number is more likely than another", "I can explain the odds, or pick numbers for fun - they have the same chance as any others"),
       ("weather", ["Will it rain in {c} on {d}?", "What will the exact temperature in {c} be on {d}?"], "a forecast that far ahead is not reliable and I have no forecast data", "I can explain typical weather for that time of year or how to read a forecast"),
       ("personal", ["When will I get promoted?", "Will I pass my exam next month?", "Will my business succeed?"], "that depends on things nobody can know in advance", "I can help you plan concrete steps that improve your chances"),
       ("politics", ["Who will win the next election in {n}?", "What will the new government do in its first year?"], "outcomes like that depend on events that have not happened yet", "I can explain how the electoral system works or summarise the main positions that are publicly known"),
       ("tech", ["What will be the most popular programming language in {y}?", "Which company will make the best phone in {y}?"], "nobody can predict that reliably", "I can describe current trends and what they might mean"),
       ("health", ["How long will I live?", "Will I get the flu this winter?"], "that depends on many things nobody can foresee", "I can share what is known about healthy habits and how to lower common risks"),
       ("dating", ["Will my crush say yes if I ask?", "Will my friends like my gift?"], "it depends on other people's feelings that I can't see", "I can suggest ways to ask kindly and read the situation"),
       ("science", ["When exactly will we find life on other planets?", "When will a cure for all cancers be found?"], "no one knows when or whether that will happen", "I can summarise what current research says")]
EVENTS = ["World Cup", "Champions League", "Tour de France", "national championship"]
ASSETS = ["gold", "oil", "a major stock index", "bitcoin"]
PERIODS = ["week", "month", "year"]
COUNTRIES = ["France", "Germany", "Brazil", "Japan", "Canada", "Spain"]
C_FORM = ["I can't predict that: {why}. {help}.", "Nobody can know that yet - {why}. {help}.", "I'd be guessing, because {why}. {help}.", "I don't know, and nobody can say for certain - {why}. {help}.", "That isn't something I can know in advance because {why}. {help}."]
C_PUSH = [("Just guess.", "I'd rather not: any specific answer I gave would carry no real information, and it could mislead you. {help}."),
          ("Come on, give me one answer.", "I can't honestly give you a single answer to that, because nobody knows yet. {help}.")]


def gen_c(r):
    key, qs, why, help_ = pick(r, FUT)
    y = pick(r, [2027, 2028, 2030, 2032])
    dates = pick(r, ["next Tuesday", "the 14th of next month", "this coming weekend", "New Year's Day next year"])
    q = pick(r, qs).format(y=y, e=pick(r, EVENTS), a=(a := pick(r, ASSETS)), p=pick(r, PERIODS), c=pick(r, CITIES), d=dates, n=pick(r, COUNTRIES))
    help_ = help_.format(a=a)
    ans = pick(r, C_FORM).format(why=why, help=help_)
    msgs = [("user", q), ("assistant", ans)]
    if r.random() < 0.2:
        u, a2 = pick(r, C_PUSH)
        msgs += [("user", u), ("assistant", a2.format(help=help_))]
    return f"future_{key}", msgs


# ---------------------------------------------------------------------------------------------------------------------------------- D: ill-posed
def gen_d(r):
    a, b, t, v = r.randrange(2, 90), r.randrange(2, 90), r.randrange(6, 20), r.choice([40, 50, 60, 80, 100])
    while b == a:
        b = r.randrange(2, 90)
    items = [
        ("div_zero", f"Divide {a} by 0.", f"That has no answer: division by zero is undefined. Dividing {a} by a number asks 'what times that number gives {a}?', and nothing multiplied by 0 gives {a}. If you meant {a} divided by something small, tell me the number."),
        ("div_zero", f"What is {a}/0?", f"It is undefined. A number {a} divided by 0 would need a value x with x * 0 = {a}, but x * 0 is always 0. Did you mean a different divisor?"),
        ("largest_prime", "What is the largest prime number?", "There is no largest prime. Euclid showed that if you multiply any list of primes and add 1, you get a number with a prime factor outside the list, so the primes never end. I can tell you the largest one that is currently known if that is what you mean, but I can't give you a final largest prime."),
        ("no_solution", f"Solve for x: x + {a} = x + {b}.", f"There is no solution. Subtracting x from both sides gives {a} = {b}, which is false, so no value of x works."),
        ("no_solution", f"Solve for x: 0x = {a}.", f"There is no solution: 0 times any number is 0, so it can never equal {a}."),
        ("neg_root", f"What is the square root of -{a} in the real numbers?", f"There is no real square root of -{a}, because a real number squared is never negative. In the complex numbers it is i times the square root of {a}."),
        ("last_digit_pi", "What is the last digit of pi?", "Pi has no last digit. It is irrational, so its decimal expansion goes on forever without repeating, and there is no final digit to name."),
        ("underdetermined", f"A train leaves at {t}:00 and travels at {v} km/h. How old is the driver?", "The problem doesn't give any information about the driver's age, and the train's time and speed have no connection to it, so it can't be worked out. If there is more to the problem, share it and I'll solve it."),
        ("underdetermined", f"A box holds {a} red balls. How many blue balls does it hold?", f"The problem only tells me about the {a} red balls, so there is no way to know the number of blue ones. I'd need more information, such as the total or the ratio."),
        ("underdetermined", f"Tom is {t} years old. How tall is he?", "His age doesn't determine his height, and the problem gives nothing else, so I can't answer. If you tell me more, such as a measurement or a growth chart, I can help."),
        ("contradiction", f"Find a number that is both greater than {a + 10} and less than {a}.", f"There is no such number: anything greater than {a + 10} is also greater than {a}, so it cannot be less than {a}. The two conditions contradict each other."),
    ]
    key, q, ans = pick(r, items)
    return f"illposed_{key}", [("user", q), ("assistant", ans)]


# ---------------------------------------------------------------------------------------------------------------------------------- E: not in the text
def gen_e(r, seed):
    from distill import context_gen
    for k in range(60):
        name = r.choice(list(context_gen.FAMILIES))
        rec = context_gen.build_one(name, 500_000 + seed * 61 + k)
        if rec["meta"]["answer"] == "not_found":
            return f"notintext_{name}", [(m["role"], m["content"]) for m in rec["messages"]], rec["messages"][0]["role"] == "system"
    return None


# ---------------------------------------------------------------------------------------------------------------------------------- F: no matching capability
NEEDS = [("Book a table for two at an Italian restaurant tonight.", "make restaurant reservations", {"food", "travel"}), ("Send a text message to my mother saying I'll be late.", "send text messages", {"messaging", "mailer"}),
         ("Turn on the heating in the living room.", "control home devices", {"iot"}), ("Order me a taxi to the airport.", "book taxis", {"transport", "maps"}), ("Call the dentist and move my appointment.", "make phone calls", {"messaging"}),
         ("Pay my electricity bill.", "make payments", {"finance", "shop"}), ("Set an alarm for 7 am tomorrow.", "set alarms", {"planner"}), ("Post this photo on Instagram.", "post to social media", {"social"}),
         ("Print the document on the office printer.", "control printers", {"filesystem"}), ("Diagnose the rash on my arm.", "diagnose medical conditions", {"health"}), ("Water my plants while I'm away.", "operate devices at your home", {"plants", "iot"}),
         ("Buy me two tickets for the concert on Friday.", "buy tickets", {"events", "shop", "travel"}), ("Play my workout playlist.", "play music", {"music"}), ("Translate this contract into German.", "translate text", {"language"}),
         ("Delete all the files on my laptop.", "access your laptop or its files", {"filesystem"})]


def gen_f(r):
    from distill import fc_gen
    need, cap, block = r.choice(NEEDS)
    pool = [f for f in fc_gen.FUNCS.values() if f["domain"] not in block]
    offered = r.sample(pool, r.randrange(1, 4))
    can = "; ".join(f["desc"].rstrip(".")[0].lower() + f["desc"].rstrip(".")[1:] for f in offered)
    ans = r.choice([f"I can't {cap} with the functions I have here. What I can do with them: {can}. Let me know if any of that would help.",
                    f"Sorry, I don't have a function that can {cap}, so I can't do that. The tools available to me can only do the following: {can}.",
                    f"That isn't something I can do - none of my available functions {cap.replace('make ', 'make ').replace('send ', 'send ')}. I can: {can}."])
    msgs = [("user", need), ("assistant", ans)]
    return f"nocap_{cap.split()[0]}", msgs, [fc_gen.schema(f) for f in offered]


# ---------------------------------------------------------------------------------------------------------------------------------- twins
def gen_twin(r):
    from distill import mcq_gen
    tabs = [t for t in mcq_gen.TABLES if t["subject"] in ("geography", "literature", "history", "physics", "chemistry", "art", "music", "economics", "history of science")]
    t = r.choice(tabs)
    k, v = r.choice(t["rows"])
    q = r.choice(t["fwd"]).format(k=k)
    fact = t["fact"].format(k=k, v=v)
    return f"twin_{t['subject'].replace(' ', '_')}", [("user", q), ("assistant", fact)]


GENS = [("a", 1900), ("b", 1400), ("c", 900), ("d", 750), ("e", 1600), ("f", 1400), ("twin", 1000)]


def build(scale: float = 1.0):
    rows, seen = [], set()
    for name, want in GENS:
        want = int(want * scale)
        made = k = 0
        while made < want and k < want * 12:
            r = random.Random(f"refusal:{name}:{k}")
            k += 1
            sysflag, tools = r.random() < 0.15, None
            if name == "e":
                out = gen_e(r, k)
                if not out:
                    continue
                fam, msgs, sysflag = out
            elif name == "f":
                fam, msgs, tools = gen_f(r)
            else:
                fam, msgs = {"a": gen_a, "b": gen_b, "c": gen_c, "d": gen_d, "twin": gen_twin}[name](r)
            conv = [{"role": ro, "content": c} for ro, c in msgs]
            if sysflag and conv[0]["role"] != "system":
                conv.insert(0, {"role": "system", "content": r.choice(SYSTEMS)})
            key = json.dumps([conv, [t["function"]["name"] for t in tools or []]], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            rows.append({"id": f"refusal/{name}/{k}", "messages": conv, "tools": tools or [],
                         "meta": {"family": fam if name != "twin" else fam, "category": "refusal" if name != "twin" else "answerable", "difficulty": "easy", "kind": "refusal", "seed": k,
                                  "tags": ["refusal", name] if name != "twin" else ["answerable_twin"], "n_tool_calls": 0, "n_turns": sum(m["role"] == "user" for m in conv), "n_tools_offered": len(tools or [])}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/refusal_data.jsonl")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    rows = build(a.scale)
    random.Random(8).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    from collections import Counter
    c = Counter(x["meta"]["tags"][-1] if x["meta"]["category"] == "answerable" else x["meta"]["tags"][1] for x in rows)
    print(f"{len(rows)} examples -> {a.out}; by group {dict(c)}; multi-turn {sum(x['meta']['n_turns'] > 1 for x in rows)}")


if __name__ == "__main__":
    main()
