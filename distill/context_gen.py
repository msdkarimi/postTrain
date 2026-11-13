"""Context-following data: answer ONLY from a supplied text, step by step, with quoted evidence (and say so when the text doesn't contain the answer).

    python -m distill.context_gen --n-per-family 300 --out data/raw/context_qa.jsonl

Covers the survey's "context following" skill (reading comprehension / retrieval-augmented answering): policies, meeting notes, price plans,
schedules, changelogs and incident reports. Each example has a generated document, a question, and a solution that (1) locates the evidence
and QUOTES it (a test checks every quote is a real substring of the document), (2) does any arithmetic in a visible step, (3) ends with
"Final answer:". About a quarter of the questions are unanswerable from the text; the right answer is then "the text doesn't say", which
teaches the model not to invent facts. Answers are computed from the generated facts, so they are correct by construction.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from distill.reasoning_gen import OPENERS, clock
from envs.tasks.common import FIRST, LAST, PROJECTS

WRAP_DOC_FIRST = [
    "Read the text below and answer the question using only the text.\n\nText:\n{doc}\n\nQuestion: {q}",
    "Here is a note from my team:\n\n{doc}\n\n{q}",
    "Use only the information below.\n\n{doc}\n\nMy question: {q}",
    "{doc}\n\nBased on the text above: {q}",
    "Context:\n{doc}\n\nQuestion: {q}\nAnswer from the context only.",
]
WRAP_Q_FIRST = ["{q}\n\nHere is the text:\n{doc}", "Question: {q}\n\nSource text:\n{doc}"]
NOT_FOUND = ["The text doesn't say.", "The text does not mention this, so I can't tell from it.", "That isn't stated in the text."]

FAMILIES: dict = {}


def family(name, weight=1.0, difficulty="medium", tags=()):
    def deco(fn):
        FAMILIES[name] = (fn, weight, difficulty, list(tags))
        return fn
    return deco


def pick(r, xs):
    return r.choice(xs)


class Ans:
    """Solution builder with quotes that must exist in the document."""

    def __init__(self, r, doc):
        self.r, self.doc, self.lines, self.quotes = r, doc, [], []

    def find(self, quote: str, what: str):
        assert quote in self.doc, quote
        self.quotes.append(quote)
        self.lines.append(f'{what} The text says: "{quote}"')

    def say(self, text: str):
        self.lines.append(text)

    def text(self, final: str) -> str:
        body = [pick(self.r, ["Let me find the relevant part of the text first.", "I'll check what the text actually says before answering.", *OPENERS[:2]]), ""]
        body += [f"Step {i}: {l}" for i, l in enumerate(self.lines, 1)]
        return "\n".join(body + ["", f"Final answer: {final}"])


# -------------------------------------------------------------------------------------------------------------------------------- policy
@family("policy", 1.2, "medium", ["lookup", "arithmetic", "abstain"])
def c_policy(r):
    co = pick(r, ["Northwind", "Acme", "Globex", "Initech", "Hooli"])
    vac, carry, notice, limit, prob = pick(r, [22, 25, 28, 30]), pick(r, [3, 5, 8, 10]), pick(r, [2, 4, 6, 8]), pick(r, [60, 75, 90, 120]), pick(r, [3, 4, 6])
    lines = {"vac": f"Every full-time employee receives {vac} vacation days per calendar year.",
             "carry": f"Up to {carry} unused vacation days may be carried over into the next year.",
             "notice": f"After the probation period, the notice period for resignation is {notice} weeks.",
             "limit": f"Travel expenses are reimbursed up to {limit} EUR per day, with receipts.",
             "prob": f"The probation period lasts {prob} months."}
    order = list(lines)
    r.shuffle(order)
    doc = f"{co} HR policy summary\n" + "\n".join(f"- {lines[k]}" for k in order)
    kind = r.random()
    a = Ans(r, doc)
    if kind < 0.28:                                                          # unanswerable
        topic, kw = pick(r, [("how many remote-work days are allowed per week", "remote"), ("what the annual training budget is", "training"), ("how many sick days employees get", "sick")])
        a.say(f'The question is about {topic}. I look for any sentence on "{kw}" in the text and find none; the text covers vacation days, carry-over, notice period, travel expenses and probation.')
        return doc, f"According to the policy, {topic}?", a.text(pick(r, NOT_FOUND)), "not_found"
    if kind < 0.60:
        k, q, ans = pick(r, [("vac", "How many vacation days does a full-time employee get per year?", f"{vac} days"),
                             ("carry", "How many unused vacation days can be carried over?", f"{carry} days"),
                             ("notice", "How long is the notice period after probation?", f"{notice} weeks"),
                             ("limit", "What is the daily travel expense limit?", f"{limit} EUR"),
                             ("prob", "How long is the probation period?", f"{prob} months")])
        a.find(lines[k][:-1], "The relevant sentence:")
        a.say(f"So the value asked for is {ans}.")
        return doc, q, a.text(ans), ans.split()[0]
    if kind < 0.80:
        used = r.randrange(1, vac)
        a.find(lines["vac"][:-1], "First the yearly allowance.")
        a.say(f"Days left = allowance - days already taken = {vac} - {used} = {vac - used}.")
        return doc, f"I've already taken {used} vacation days this year. How many do I have left?", a.text(f"{vac - used} days"), str(vac - used)
    days = r.randrange(2, 6)
    a.find(lines["limit"][:-1], "The daily limit.")
    a.say(f"For {days} days the maximum is {limit} x {days} = {limit * days} EUR.")
    return doc, f"What is the most I can be reimbursed for a {days}-day business trip?", a.text(f"{limit * days} EUR"), str(limit * days)


# -------------------------------------------------------------------------------------------------------------------------------- meeting notes
@family("meeting_notes", 1.2, "medium", ["lookup", "counting", "abstain"])
def c_notes(r):
    proj = pick(r, PROJECTS)
    owners = r.sample(FIRST, 4)
    d0 = date(2025, 3, 3) + timedelta(days=r.randrange(0, 200))
    acts = [(pick(r, ["send the budget draft", "update the roadmap", "book the venue", "review the contract", "prepare the demo", "fix the login bug", "write the release notes", "contact the vendor"]),
             owners[i], d0 + timedelta(days=r.randrange(2, 20))) for i in range(4)]
    seen, uniq = set(), []
    for t in acts:
        if t[0] not in seen:
            seen.add(t[0])
            uniq.append(t)
    acts = uniq
    decision = pick(r, ["launch on the first Monday of next month", "keep the current pricing", "postpone the migration by two weeks", "hire one more engineer"])
    items = [f"- {o} will {t} by {d.isoformat()}." for t, o, d in acts]
    doc = f"Meeting notes - {proj} project sync ({d0.isoformat()})\nDecision: the team agreed to {decision}.\nAction items:\n" + "\n".join(items)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.25:
        topic = pick(r, ["the total budget approved", "which room the meeting took place in", "how many people attended"])
        a.say(f"The question asks for {topic}. I scan the decision line and every action item; none of them gives that information.")
        return doc, f"What does the text say about {topic}?", a.text(pick(r, NOT_FOUND)), "not_found"
    if kind < 0.55:
        t, o, d = pick(r, acts)
        a.find(f"{o} will {t} by {d.isoformat()}", "Find the action item.")
        a.say(f"The owner is {o}.")
        return doc, f"Who is responsible for the task to {t}?", a.text(o), o
    if kind < 0.75:
        t, o, d = pick(r, acts)
        a.find(f"{o} will {t} by {d.isoformat()}", "Find the item for that person.")
        a.say(f"The deadline is {d.isoformat()}.")
        return doc, f"By when does {o} need to {t}?", a.text(d.isoformat()), d.isoformat()
    cut = d0 + timedelta(days=10)
    hits = [(t, o, d) for t, o, d in acts if d <= cut]
    a.say(f"I list each deadline and compare it with {cut.isoformat()}: " + "; ".join(f"{o}: {d.isoformat()} ({'on or before' if d <= cut else 'after'})" for t, o, d in acts) + ".")
    a.say(f"Items due on or before {cut.isoformat()}: {len(hits)}.")
    return doc, f"How many action items are due on or before {cut.isoformat()}?", a.text(str(len(hits))), str(len(hits))


# -------------------------------------------------------------------------------------------------------------------------------- price plans
@family("price_plans", 1.2, "medium", ["comparison", "arithmetic", "abstain"])
def c_plans(r):
    names = r.sample(["Basic", "Plus", "Pro", "Team", "Business", "Starter"], 3)
    price = sorted(r.sample(range(5, 80, 5), 3))
    seats = sorted(r.sample([1, 3, 5, 10, 25, 50], 3))
    storage = sorted(r.sample([10, 50, 100, 250, 500, 1000], 3))
    doc = "Our plans:\n" + "\n".join(f"- {n}: {p} EUR per month, {s} user(s), {g} GB storage" for n, p, s, g in zip(names, price, seats, storage))
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.25:
        topic = pick(r, ["phone support", "an annual discount", "a free trial"])
        a.say(f"The question is about {topic}. The text lists only price, users and storage for each plan, so I check whether any line mentions it: none does.")
        return doc, f"Does the {pick(r, names)} plan include {topic}?", a.text(pick(r, NOT_FOUND)), "not_found"
    i = r.randrange(3)
    if kind < 0.50:
        months = pick(r, [3, 6, 12])
        a.find(f"{names[i]}: {price[i]} EUR per month", f"Find the {names[i]} plan.")
        a.say(f"{months} months cost {price[i]} x {months} = {price[i] * months} EUR.")
        return doc, f"How much does the {names[i]} plan cost for {months} months?", a.text(f"{price[i] * months} EUR"), str(price[i] * months)
    if kind < 0.75:
        need = pick(r, [s for s in seats if s > 1])
        ok = [(n, p) for n, p, s in zip(names, price, seats) if s >= need]
        a.say(f"I need a plan with at least {need} users: " + "; ".join(f"{n} has {s}" for n, s in zip(names, seats)) + ".")
        best = min(ok, key=lambda t: t[1])
        a.say(f"Among the plans that fit ({', '.join(n for n, _ in ok)}), the cheapest is {best[0]} at {best[1]} EUR per month.")
        return doc, f"Which is the cheapest plan that supports at least {need} users?", a.text(f"{best[0]} ({best[1]} EUR per month)"), best[0]
    need = pick(r, storage)
    cand = [n for n, g in zip(names, storage) if g >= need]
    a.find(f"{names[storage.index(need)]}: {price[storage.index(need)]} EUR per month, {seats[storage.index(need)]} user(s), {need} GB storage", f"Find the plan with {need} GB.")
    a.say(f"Plans with at least {need} GB: {', '.join(cand)}. The smallest such plan is {names[storage.index(need)]}.")
    return doc, f"Which plan has exactly {need} GB of storage?", a.text(names[storage.index(need)]), names[storage.index(need)]


# -------------------------------------------------------------------------------------------------------------------------------- schedule
@family("schedule", 1.0, "medium", ["lookup", "time", "abstain"])
def c_schedule(r):
    start = r.randrange(8, 10) * 60
    ev, t = [], start
    for name in r.sample(["Opening talk", "Workshop A", "Coffee break", "Panel discussion", "Lunch", "Workshop B", "Demo session", "Closing remarks"], 6):
        dur = pick(r, [30, 45, 60, 90])
        ev.append((name, t, t + dur))
        t += dur + pick(r, [0, 0, 15])
    doc = "Conference schedule:\n" + "\n".join(f"{clock(s)}-{clock(e)} {n}" for n, s, e in ev)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.22:
        a.say("The question asks about the speaker or room, but each line only has a time range and a title, so the text has no such detail.")
        return doc, f"Who is the speaker at {pick(r, ev)[0]}?", a.text(pick(r, NOT_FOUND)), "not_found"
    i = r.randrange(len(ev))
    n, s, e = ev[i]
    if kind < 0.50:
        a.find(f"{clock(s)}-{clock(e)} {n}", f"Find the {n} line.")
        a.say(f"It starts at {clock(s)}.")
        return doc, f"At what time does {n} start?", a.text(clock(s)), clock(s)
    if kind < 0.75:
        a.find(f"{clock(s)}-{clock(e)} {n}", "Find the line.")
        a.say(f"Duration = {clock(e)} - {clock(s)} = {e - s} minutes.")
        return doc, f"How long does {n} last?", a.text(f"{e - s} minutes"), str(e - s)
    i = r.randrange(len(ev) - 1)
    n, nx = ev[i][0], ev[i + 1]
    a.find(f"{clock(ev[i][1])}-{clock(ev[i][2])} {n}", f"Find {n}.")
    a.say(f"The next line in time order is {clock(nx[1])}-{clock(nx[2])} {nx[0]}.")
    return doc, f"Which session comes right after {n}?", a.text(nx[0]), nx[0]


# -------------------------------------------------------------------------------------------------------------------------------- changelog
@family("changelog", 1.0, "medium", ["lookup", "counting", "abstain"])
def c_changelog(r):
    feats = r.sample(["dark mode", "CSV export", "offline sync", "two-factor login", "keyboard shortcuts", "team folders", "API tokens", "voice notes"], 5)
    ver = [(1, r.randrange(0, 3))]
    for _ in range(4):
        ma, mi = ver[-1]
        ver.append((ma, mi + 1) if r.random() < 0.7 else (ma + 1, 0))
    d0 = date(2024, 9, 1)
    rel = []
    for (ma, mi), f in zip(ver, feats):
        d0 += timedelta(days=r.randrange(15, 80))
        rel.append((f"{ma}.{mi}", d0, f))
    doc = "Release notes:\n" + "\n".join(f"- {v} ({d.isoformat()}): added {f}." for v, d, f in rel)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.22:
        v = f"{ver[-1][0] + 2}.0"
        a.say(f"The question is about version {v}. I read every entry: the versions listed are {', '.join(x[0] for x in rel)}, so {v} does not appear.")
        return doc, f"What changed in version {v}?", a.text(pick(r, NOT_FOUND)), "not_found"
    v, d, f = pick(r, rel)
    if kind < 0.50:
        a.find(f"{v} ({d.isoformat()}): added {f}", f"Search for the entry about {f}.")
        a.say(f"The version is {v}.")
        return doc, f"Which version added {f}?", a.text(v), v
    if kind < 0.72:
        a.find(f"{v} ({d.isoformat()}): added {f}", "Find the entry.")
        a.say(f"It was released on {d.isoformat()}.")
        return doc, f"When was version {v} released?", a.text(d.isoformat()), d.isoformat()
    last = rel[-1]
    a.say("The versions in order are: " + ", ".join(x[0] for x in rel) + ". Releases are listed oldest to newest.")
    a.find(f"{last[0]} ({last[1].isoformat()}): added {last[2]}", "The last entry is the newest release.")
    return doc, "What is the latest version and what feature did it add?", a.text(f"{last[0]}, which added {last[2]}"), last[0]


# -------------------------------------------------------------------------------------------------------------------------------- incident report
@family("incident", 1.0, "hard", ["time", "arithmetic", "abstain"])
def c_incident(r):
    svc = pick(r, ["checkout", "login", "search", "payments", "notifications"])
    t0 = r.randrange(6 * 60, 20 * 60, 5)
    det, fix, rec = t0 + r.randrange(5, 25), t0 + r.randrange(30, 70), t0 + r.randrange(75, 120)
    cause = pick(r, ["an expired TLS certificate", "a misconfigured load balancer rule", "a database connection pool exhaustion", "a bad feature flag rollout"])
    who = r.sample(FIRST, 2)
    doc = (f"Incident report - {svc} service\n{clock(t0)} Errors on the {svc} service begin.\n{clock(det)} {who[0]} receives the first alert and starts investigating.\n"
           f"{clock(fix)} {who[1]} identifies the root cause: {cause}.\n{clock(rec)} Service fully recovered.")
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.25:
        a.say("The question asks for the number of affected customers. I read the four timeline entries: they give times, names and the cause, but no customer numbers.")
        return doc, "How many customers were affected by the outage?", a.text(pick(r, NOT_FOUND)), "not_found"
    if kind < 0.50:
        a.find(f"{clock(t0)} Errors on the {svc} service begin", "Find when the errors began.")
        a.find(f"{clock(rec)} Service fully recovered", "Find when the service recovered.")
        a.say(f"Duration = {clock(rec)} - {clock(t0)} = {rec - t0} minutes.")
        return doc, "How long did the outage last, from the first errors until full recovery?", a.text(f"{rec - t0} minutes"), str(rec - t0)
    if kind < 0.72:
        a.find(f"{clock(fix)} {who[1]} identifies the root cause: {cause}", "Find the root cause entry.")
        a.say(f"The cause is {cause}, found by {who[1]}.")
        return doc, "What was the root cause, and who found it?", a.text(f"{cause}, found by {who[1]}"), who[1]
    a.find(f"{clock(t0)} Errors on the {svc} service begin", "Find the start.")
    a.find(f"{clock(det)} {who[0]} receives the first alert", "Find the first alert.")
    a.say(f"Time to detection = {clock(det)} - {clock(t0)} = {det - t0} minutes.")
    return doc, "How many minutes passed between the errors starting and the first alert?", a.text(f"{det - t0} minutes"), str(det - t0)


# -------------------------------------------------------------------------------------------------------------------------------- DEEP families
# Multi-hop questions, longer documents with distractors, conflicting sources, and tables that need filtering + arithmetic.
@family("org_chart", 1.2, "hard", ["multi_hop", "lookup", "abstain"])
def c_org(r):
    names = r.sample(FIRST, 7)
    ceo, vps, ics = names[0], names[1:3], names[3:]
    boss = {vps[0]: ceo, vps[1]: ceo}
    for i, p in enumerate(ics):
        boss[p] = vps[i % 2]
    sents = [f"{p} reports to {b}." for p, b in boss.items()]
    r.shuffle(sents)
    doc = f"Org chart notes ({pick(r, ['Engineering', 'Sales', 'Operations'])}):\n" + "\n".join(sents) + f"\n{ceo} is the head of the department."
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.2:
        who = pick(r, [n for n in FIRST if n not in names])
        a.say(f"The question is about {who}. I check every sentence: {who} is not mentioned anywhere.")
        return doc, f"Who does {who} report to?", a.text(pick(r, NOT_FOUND)), "not_found"
    p = pick(r, ics)
    if kind < 0.55:
        a.find(f"{p} reports to {boss[p]}", f"Find {p}'s manager.")
        a.find(f"{boss[p]} reports to {boss[boss[p]]}", f"Now find who {boss[p]} reports to.")
        a.say(f"So {p}'s manager's manager is {boss[boss[p]]}.")
        return doc, f"Who is {p}'s manager's manager?", a.text(boss[boss[p]]), boss[boss[p]]
    if kind < 0.8:
        vp = pick(r, vps)
        reports = [x for x, b in boss.items() if b == vp]
        a.say(f"I look for every sentence ending in \"reports to {vp}\":")
        for x in reports:
            a.find(f"{x} reports to {vp}", "Found one.")
        a.say(f"That is {len(reports)} people: {', '.join(reports)}.")
        return doc, f"How many people report directly to {vp}, and who are they?", a.text(f"{len(reports)}: {', '.join(reports)}"), str(len(reports))
    vp = pick(r, vps)
    direct = [x for x, b in boss.items() if b == vp]
    a.say(f"First the people who report directly to {vp}: {', '.join(direct)}.")
    for x in direct:
        a.find(f"{x} reports to {vp}", "Evidence.")
    a.say(f"The people under {vp} are all in this group; nobody reports to them in the text, so the total below {vp} is {len(direct)}.")
    return doc, f"How many people work under {vp} in total (directly or indirectly)?", a.text(str(len(direct))), str(len(direct))


@family("ledger", 1.3, "hard", ["table", "arithmetic", "multi_hop"])
def c_ledger(r):
    cats = r.sample(["Travel", "Software", "Meals", "Office", "Training"], 3)
    rows = []
    for _ in range(r.randrange(7, 11)):
        c = pick(r, cats)
        rows.append((date(2025, 1, 1) + timedelta(days=r.randrange(0, 90)), c, pick(r, {"Travel": ["train ticket", "taxi", "flight"], "Software": ["licence", "subscription", "plugin"], "Meals": ["team lunch", "client dinner", "coffee"],
                                                                                   "Office": ["printer paper", "chairs", "monitor"], "Training": ["course fee", "workshop", "book"]}[c]), r.randrange(8, 400)))
    rows.sort()
    lines = [f"{d.isoformat()} | {c} | {n} | {a} EUR" for d, c, n, a in rows]
    doc = "Expense log:\n" + "\n".join(lines)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.2:
        a.say("The question is about the person who approved the expenses. Each log line has only a date, a category, a description and an amount, so that is not in the text.")
        return doc, "Who approved these expenses?", a.text(pick(r, NOT_FOUND)), "not_found"
    if kind < 0.65:
        cat = pick(r, cats)
        mine = [(l, x) for l, x in zip(lines, rows) if x[1] == cat]
        if not mine:
            return c_ledger(random.Random(r.random()))
        a.say(f"I pick out every line whose category is {cat}:")
        for l, _ in mine:
            a.find(l, "Line.")
        amounts = [x[3] for _, x in mine]
        a.say(f"Total = {' + '.join(map(str, amounts))} = {sum(amounts)} EUR.")
        return doc, f"How much was spent on {cat} in total?", a.text(f"{sum(amounts)} EUR"), str(sum(amounts))
    big = max(rows, key=lambda x: (x[3], x[0]))
    l = lines[rows.index(big)]
    a.say("I compare the amounts on all lines: " + ", ".join(str(x[3]) for x in rows) + f". The largest is {big[3]}.")
    a.find(l, "That line is:")
    return doc, "Which single expense was the largest, and what was it for?", a.text(f"{big[2]} on {big[0].isoformat()} ({big[3]} EUR)"), str(big[3])


@family("conflicting_sources", 1.2, "hard", ["reasoning", "dates", "abstain"])
def c_conflict(r):
    ev = pick(r, ["the quarterly review", "the product demo", "the team lunch", "the budget workshop"])
    t1, t2 = r.sample([f"{h:02d}:{m:02d}" for h in range(9, 17) for m in (0, 30)], 2)
    d1 = date(2025, 4, 1) + timedelta(days=r.randrange(0, 20))
    d2 = d1 + timedelta(days=r.randrange(1, 6))
    dated = r.random() < 0.6
    if dated:
        doc = f"Message A ({d1.isoformat()}): {ev.capitalize()} will start at {t1}.\nMessage B ({d2.isoformat()}): Update - {ev} now starts at {t2}."
    else:
        doc = f"Message A: {ev.capitalize()} will start at {t1}.\nMessage B: {ev.capitalize()} will start at {t2}."
    a = Ans(r, doc)
    a.find(f"will start at {t1}", "Read the first source.")
    a.find(f"starts at {t2}" if dated else f"will start at {t2}", "Read the second source.")
    if dated:
        a.say(f"The two sources disagree ({t1} vs {t2}). Message B is dated {d2.isoformat()}, which is later than {d1.isoformat()}, and it says \"Update\", so it replaces the earlier time.")
        return doc, f"When does {ev} start?", a.text(t2), t2
    a.say(f"The sources disagree ({t1} vs {t2}) and neither has a date or says it replaces the other, so I can't tell which one is right.")
    return doc, f"When does {ev} start?", a.text("The text doesn't say for certain: Message A says " + f"{t1} and Message B says {t2}, and nothing shows which is current."), "not_found"


@family("long_doc_multihop", 1.3, "hard", ["long_context", "multi_hop", "distractors"])
def c_long(r):
    place = pick(r, ["Ostrava Works", "the Lakeside Cafe", "Pinewood Library", "the Harbor Gym", "Maple Street Market"])
    price, qty, fee = r.randrange(4, 25), r.randrange(3, 12), r.choice([5, 10, 15])
    noise = ["The building was renovated two years ago.", "Parking is available behind the main entrance.", "Opening hours are shown on the door.", "The team celebrated its anniversary in May.",
             "Several local artists exhibit their work in the lobby.", "Free water is available at the front desk.", "A mobile app was announced last quarter.",
             "Staff wear green badges so visitors can recognise them.", "The newsletter is sent on the first Friday of every month.", "Quiet hours start at 20:00.",
             "There is a bicycle rack near the south door.", "The roof garden is open to members on weekends."]
    facts = [f"A single ticket costs {price} EUR.", f"Every booking adds a flat service fee of {fee} EUR.", f"The group discount does not apply to bookings below {qty + 5} tickets."]
    body = noise[:]
    r.shuffle(body)
    for f in facts:
        body.insert(r.randrange(2, len(body)), f)
    doc = f"About {place}\n" + " ".join(body)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.2:
        a.say("The question asks about the refund policy. I read through the whole text, including the sentences about tickets and fees, and none of them describes refunds.")
        return doc, f"What is the refund policy at {place}?", a.text(pick(r, NOT_FOUND)), "not_found"
    a.find(facts[0][:-1], "First fact needed (the ticket price).")
    a.find(facts[1][:-1], "Second fact needed (the fee).")
    a.say(f"{qty} tickets cost {qty} x {price} = {qty * price} EUR; adding the fee gives {qty * price} + {fee} = {qty * price + fee} EUR.")
    a.say("The discount sentence only matters for bookings above its threshold, and I'm not asked about a discount, so I ignore it.")
    return doc, f"What do I pay in total for a booking of {qty} tickets at {place}?", a.text(f"{qty * price + fee} EUR"), str(qty * price + fee)


@family("table_reasoning", 1.4, "hard", ["table", "filter", "arithmetic"])
def c_table(r):
    regions = r.sample(["North", "South", "East", "West"], 3)
    prods = r.sample(["Laptop", "Monitor", "Dock", "Webcam", "Headset"], 3)
    rows = [(pick(r, regions), pick(r, prods), r.randrange(2, 40)) for _ in range(r.randrange(6, 10))]
    lines = [f"| {g} | {p} | {u} |" for g, p, u in rows]
    doc = "Units sold last month:\n| Region | Product | Units |\n|---|---|---|\n" + "\n".join(lines)
    a = Ans(r, doc)
    kind = r.random()
    if kind < 0.2:
        a.say("The question asks about revenue. The table only has region, product and units, with no prices, so revenue cannot be computed from it.")
        return doc, "What was the total revenue?", a.text(pick(r, NOT_FOUND)), "not_found"
    if kind < 0.6:
        g = pick(r, [x[0] for x in rows])
        mine = [(l, x) for l, x in zip(lines, rows) if x[0] == g]
        a.say(f"I keep only the rows where Region is {g}:")
        for l, _ in mine:
            a.find(l, "Row.")
        tot = sum(x[2] for _, x in mine)
        a.say(f"Units: {' + '.join(str(x[2]) for _, x in mine)} = {tot}.")
        return doc, f"How many units were sold in the {g} region in total?", a.text(str(tot)), str(tot)
    pr = pick(r, [x[1] for x in rows])
    mine = [(l, x) for l, x in zip(lines, rows) if x[1] == pr]
    a.say(f"I keep the rows for {pr}:")
    for l, _ in mine:
        a.find(l, "Row.")
    tot = sum(x[2] for _, x in mine)
    a.say(f"{len(mine)} row(s) with {' + '.join(str(x[2]) for _, x in mine)} = {tot} units; average per row = {tot} / {len(mine)} = {tot / len(mine):.1f}.")
    return doc, f"How many {pr} units were sold in total, and what was the average per table row?", a.text(f"{tot} units in total, {tot / len(mine):.1f} on average per row"), str(tot)


def build_one(name: str, seed: int) -> dict:
    fn, _, diff, tags = FAMILIES[name]
    r = random.Random(f"ctx:{name}:{seed}")
    doc, q, answer, key = fn(r)
    wrap = pick(r, WRAP_Q_FIRST if r.random() < 0.2 else WRAP_DOC_FIRST)
    prompt = wrap.format(doc=doc, q=q)
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    if r.random() < 0.2:
        msgs.insert(0, {"role": "system", "content": pick(r, ["Answer only from the provided text. If the text does not contain the answer, say so.", "You are a careful reader. Quote evidence, then answer. Do not guess."])})
    return {"id": f"context/{name}/{seed}", "messages": msgs, "tools": [],
            "meta": {"family": f"ctx_{name}", "category": "context_following", "difficulty": diff, "tags": tags + ["cot", "step_by_step", "grounded"], "seed": seed, "kind": "context",
                     "n_tool_calls": 0, "n_turns": 1, "n_tools_offered": 0, "answer": key, "doc": doc}}


def check_record(rec: dict) -> str | None:
    text, doc, key = rec["messages"][-1]["content"], rec["meta"]["doc"], rec["meta"]["answer"]
    lines = text.rstrip().splitlines()
    if not lines[-1].startswith("Final answer:"):
        return "no_final_answer_line"
    if sum(l.startswith("Step ") for l in lines) < 1:
        return "no_steps"
    import re
    for q in re.findall(r'The text says: "([^"]+)"', text):
        if q not in doc:
            return "quote_not_in_doc"
    if key != "not_found" and key not in lines[-1] and key not in text:
        return "key_missing"
    if key == "not_found" and not any(p in lines[-1].lower() for p in ("doesn't say", "does not mention", "isn't stated")):
        return "unanswerable_without_refusal"
    head = "\n".join(lines[:2])
    if len(key) >= 4 and key != "not_found" and key in head:
        return "answer_in_opener"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=300)
    ap.add_argument("--out", default="data/raw/context_qa.jsonl")
    ap.add_argument("--seed-offset", type=int, default=10_000)
    a = ap.parse_args()
    rows, seen, bad = [], set(), {}
    for name, (_, w, _, _) in FAMILIES.items():
        made, k = 0, 0
        while made < int(a.n_per_family * w) and k < int(a.n_per_family * w) * 4:
            rec = build_one(name, a.seed_offset + k)
            k += 1
            why = check_record(rec)
            h = rec["messages"][-2]["content"].lower()
            if why or h in seen:
                bad[why or "duplicate"] = bad.get(why or "duplicate", 0) + 1
                continue
            seen.add(h)
            rec["meta"].pop("doc")
            rows.append(rec)
            made += 1
    random.Random(1).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(FAMILIES)} families -> {len(rows)} records in {a.out}; rejected {bad}")


if __name__ == "__main__":
    main()
