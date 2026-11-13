"""IFEval-style instruction following at scale: prompts from distill/ifgen.py answered with structure-exact responses, plus multi-turn follow-ups.

    python -m distill.if_struct_gen --n 4500 --out data/raw/if_structured.jsonl

The answers are composed from the teacher-written content bank (distill/if_topics.py: ten comma-free sentences per topic) and arranged so that every
constraint of the prompt holds: exact bullet / numbered-item / paragraph / section counts, word and sentence limits, required and banned words,
casing, no commas, a <<title>>, highlights, placeholders, a P.S., a closing phrase, JSON output, quotation marks. Each candidate is checked with the
SAME verifiers used for evaluation (distill.ifgen.verify); only passing ones are kept, so a response is never a near miss.

About a quarter of the examples are TWO-TURN: after the first answer the user asks for a transformation (shorten to N words, all lowercase, JSON,
add a P.S., exactly N bullets) which is built from the same content and verified against its own constraint, so the model learns to edit its own
previous answer under a new rule, not only to follow a rule on the first try.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from distill import ifgen
from distill.if_topics import BANK, BANK2, FILL
from distill.ifgen import KEYWORDS, verify

LEADS = {"email": ["I am writing to share a few notes about {t}.", "I wanted to put a few thoughts about {t} in writing.", "Here is a short update on {t}.", "A quick note on {t} before the week gets busy."],
         "explain": ["Let me explain {t} in a simple way.", "Here is {t} explained step by step.", "Think of {t} like this.", "{T} is easier to understand when you break it into pieces."],
         "advice": ["Here is some practical advice about {t}.", "If you are dealing with {t} these tips should help.", "A few tried and tested pointers on {t} follow.", "When it comes to {t} start with the basics."],
         "blog": ["Welcome back to the blog and today we look at {t}.", "In this post we take a closer look at {t}.", "Few topics come up as often as {t} so let us dig in.", "Today the topic is {t}."],
         "summary": ["Here is a short summary of {t}.", "The key ideas behind {t} are these.", "In brief this is what to know about {t}.", "Summing up {t} in a few lines follows."],
         "message": ["Hi team and I wanted to share a quick note about {t}.", "Quick heads-up from me about {t}.", "Sharing a short message on {t} for everyone.", "Just a note on {t} so we are all on the same page."],
         "story": ["Let me tell you a short story about {t}.", "It began on an ordinary morning with {t}.", "This story is about {t} and how it changed one small day.", "Once upon a time {t} became part of someone's week."],
         "list": ["Here are the main points about {t}.", "The essentials of {t} come down to this.", "These are the points worth remembering about {t}.", "A compact list on {t} follows."],
         "compare": ["There are two ways to look at {t}.", "Two perspectives on {t} are worth setting side by side.", "People tend to see {t} in one of two ways.", "Looking at {t} from two angles helps."],
         "description": ["Let me describe {t} briefly.", "Here is a short description of {t}.", "Picture {t} in a few clear lines.", "In a few words {t} can be described as follows."],
         "checklist": ["Here is a short checklist about {t}.", "Use this quick checklist for {t}.", "Before you start with {t} run through this list.", "A practical checklist on {t} follows."],
         "persuade": ["Consider why {t} deserves your attention.", "There is a strong case for taking {t} seriously.", "Here is why {t} is worth your time.", "Let me make the case for {t}."],
         "mistake": ["A common mistake about {t} is to skip the basics.", "Many people go wrong with {t} in the same way.", "The most frequent error with {t} is easy to avoid.", "One slip about {t} comes up again and again."],
         "dialogue": ["Here is a short chat about {t}.", "Two friends talk about {t} over coffee.", "Overheard in the office a short talk about {t}.", "A brief conversation about {t} follows."],
         "speech": ["Friends and colleagues let me speak for a moment about {t}.", "Thank you all for being here to talk about {t}.", "Good evening everyone and let us turn to {t}.", "I would like to say a few words on {t}."],
         "faq": ["Here are a few questions and answers about {t}.", "Frequently asked questions about {t} are answered below.", "Readers often ask about {t} so here are quick answers.", "A short FAQ on {t} follows."]}
CLOSES = ["Thank you for reading.", "I hope this helps.", "Feel free to ask if anything is unclear.", "That is the short version.", "Let me know how it goes.", "Good luck with it.", "Take it one step at a time.", "That covers the basics.",
          "Best of luck.", "I am happy to go into more detail.", "More details are easy to add if you need them.", "Hope that clears things up.", "Do come back with questions.", "That should get you started."]
SPEAKERS = [("Ana", "Ben"), ("Mia", "Leo"), ("Sofia", "Omar"), ("Priya", "Jonas"), ("Nora", "Kai"), ("Lena", "Tom"), ("Yara", "Hugo"), ("Iris", "Felix"), ("Maya", "Noah"), ("Elena", "Sam")]
FAQ_Q = ["What should I know first?", "Why does it matter?", "How do I get started?", "What helps most?", "What should I avoid?", "Where can I learn more?", "How long does it take?", "Who can help?",
         "Is it difficult for a beginner?", "What is the most common question?", "What does a good result look like?", "Can I do it on my own?", "What do experts suggest?", "Is there a quick tip?", "What are the first signs of progress?", "What would you do differently?"]
PS_TEXT = ["Remember to take it one step at a time.", "Ask me if you want more detail.", "Small steps add up.", "Share this with a friend.", "Keep a note of what works for you.", "Come back to this when you need a reminder.",
           "Do not hesitate to ask a follow-up question.", "A little patience goes a long way.", "I am glad to help again any time.", "Try it today and see how it feels.", "Print this out if it helps.", "Feel free to adapt this to your situation.",
           "One more thing: celebrate the small wins.", "Pass it on to anyone who might need it.", "Check back in a week and see what changed.", "Good luck and enjoy the process."]


def wcount(s: str) -> int:
    return len(ifgen.words(s))


def lcfirst(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


def make_units(row: dict, r: random.Random) -> list:
    bank = list(BANK[row["topic"]])
    r.shuffle(bank)
    fill = list(FILL)
    r.shuffle(fill)
    banned = next((c["params"]["k"] for c in row["constraints"] if c["name"] == "banned"), None)
    units = [u for u in bank + fill if not banned or not re.search(rf"\b{banned}\b", u, re.I)]
    if row["form"] == "dialogue":
        a, b = r.choice(SPEAKERS)
        units = [f"{[a, b][i % 2]}: {u}" for i, u in enumerate(units)]
    elif row["form"] == "faq":
        units = [f"Q: {FAQ_Q[i % len(FAQ_Q)]} A: {u}" for i, u in enumerate(units)]
    return units


def split_groups(items: list, n: int) -> list:
    """Distribute items into n non-empty groups in order."""
    if len(items) < n:
        return []
    size, extra = divmod(len(items), n)
    out, i = [], 0
    for g in range(n):
        take = size + (1 if g < extra else 0)
        out.append(items[i:i + take])
        i += take
    return out


def highlight(unit: str) -> str:
    ws = unit.split(" ")
    k = min(3, len(ws))
    if ws[0] in ("Q:", "A:") or ws[0].endswith(":"):
        return unit
    return "*" + " ".join(ws[:k]) + "* " + " ".join(ws[k:])


def title_of(topic: str) -> str:
    t = topic.strip()
    return t[0].upper() + t[1:]


def assemble(row: dict, units: list, k: int, r: random.Random) -> str | None:
    cons = {c["name"]: c["params"] for c in row["constraints"]}
    form, topic = row["form"], row["topic"]
    extras = []
    if "keywords" in cons:
        k1, k2 = cons["keywords"]["ks"]
        extras.append(r.choice(["Keep {a} and {b} in mind.", "Two words worth remembering are {a} and {b}.", "It helps to think about {a} as well as {b}.", "Let {a} and {b} guide your choices.", "Bring a sense of {a} and {b} to it.", "Both {a} and {b} matter here.", "Never forget {a} and {b}.", "Aim for {a} together with {b}."]).format(a=k1, b=k2))
    if "placeholders" in cons:
        n = cons["placeholders"]["n"]
        extras.append(r.choice({2: ["Write to [name] at [address] for details.", "Send questions to [name] at [address].", "Contact [name] using [address] for more."],
                                 3: ["Write to [name] at [address] before [date] for details.", "Please reach [name] at [address] by [date].", "Send your notes to [name] at [address] before [date]."]}.get(n, ["Write to [name] at [address] before [date] or ask [friend] for help."])))
    if "capital_words" in cons:
        extras.append(r.choice(["Remember: KEEP IT SIMPLE AND STAY CALM.", "NOTE: TAKE YOUR TIME AND CHECK EACH STEP.", "TIP: START SMALL AND BUILD UP SLOWLY.", "REMEMBER: BE KIND TO YOURSELF TODAY.", "KEY POINT: PLAN FIRST AND THEN ACT.", "IMPORTANT: ASK FOR HELP EARLY.", "ALWAYS: DOUBLE CHECK THE DETAILS.", "GOAL: STEADY PROGRESS EVERY WEEK.", "REMINDER: REST IS PART OF THE PLAN.", "WARNING: DO NOT RUSH THE BASICS.", "BONUS: SHARE WHAT YOU LEARN.", "FOCUS: ONE TASK AT A TIME."]))
    n_struct = next((cons[x]["n"] for x in ("n_bullets", "numbered", "sections", "n_paragraphs") if x in cons), None)
    kind = next((x for x in ("n_bullets", "numbered", "sections", "n_paragraphs", "json", "first_word_paragraph", "two_responses", "repeat_prompt") if x in cons), "plain")
    lead = r.choice(LEADS[form]).format(t=topic, T=topic[:1].upper() + topic[1:])
    body = list(units[:k])
    for e in extras:
        body.insert(r.randrange(len(body) + 1), e)
    if "highlights" in cons:
        idx = [i for i, u in enumerate(body) if not u.startswith(("Q:", "A:")) and ":" not in u.split(" ")[0]]
        if len(idx) < cons["highlights"]["n"]:
            return None
        for i in r.sample(idx, cons["highlights"]["n"]):
            body[i] = highlight(body[i])
    if "starts_with" in cons:
        lead = f"{cons['starts_with']['w']} {lcfirst(lead)}"
    if kind == "n_bullets" or kind == "numbered":
        n = n_struct
        if len(body) != n:
            return None
        mark = (lambda i: "- ") if kind == "n_bullets" else (lambda i: f"{i + 1}. ")
        lines = [f"{mark(i)}{u}" for i, u in enumerate(body)]
        text = lead + "\n" + "\n".join(lines)
    elif kind == "sections":
        groups = split_groups(body, n_struct)
        if not groups:
            return None
        groups[0][0] = lead + " " + groups[0][0]
        text = "\n\n".join(f"SECTION {i + 1}\n" + " ".join(g) for i, g in enumerate(groups))
    elif kind == "n_paragraphs":
        groups = split_groups(body, n_struct)
        if not groups:
            return None
        groups[0][0] = lead + " " + groups[0][0]
        text = "\n\n".join(" ".join(g) for g in groups)
    elif kind == "first_word_paragraph":
        p = cons["first_word_paragraph"]
        groups = split_groups(body, p["n"])
        if not groups:
            return None
        groups[0][0] = lead + " " + groups[0][0]
        paras = [" ".join(g) for g in groups]
        paras[p["i"] - 1] = f"{p['w']} " + lcfirst(paras[p["i"] - 1])
        text = "\n\n".join(paras)
    elif kind == "two_responses":
        half = max(len(body) // 2, 1)
        a_part, b_part = " ".join(body[:half]), " ".join(body[half:])
        if not b_part:
            return None
        text = lead + " " + a_part + "\n******\n" + b_part
    elif kind == "repeat_prompt":
        text = cons["repeat_prompt"]["request"] + "\n\n" + lead + " " + " ".join(body)
    elif kind == "json":
        text = json.dumps({"title": title_of(topic), "content": lead + " " + " ".join(body)}, ensure_ascii=False)
    else:
        txt = lead + " " + " ".join(body)
        if "max_words" not in cons and "max_sentences" not in cons and r.random() < 0.4:
            txt += " " + r.choice(CLOSES)
        text = txt
    if "title" in cons and kind != "json":
        text = f"<<{title_of(topic)}>>\n" + text
    if "postscript" in cons:
        text += "\nP.S. " + r.choice(PS_TEXT)
    if "closer" in cons:
        text += "\n" + cons["closer"]["c"]
    if "quoted" in cons:
        text = '"' + text + '"'
    if "lowercase" in cons:
        text = text.lower()
    if "uppercase" in cons:
        text = text.upper()
    return text


def candidate_counts(cons: dict) -> list:
    n_struct = next((cons[x]["n"] for x in ("n_bullets", "numbered") if x in cons), None)
    extra_n = ("keywords" in cons) + ("placeholders" in cons) + ("capital_words" in cons)
    if n_struct:
        k = n_struct - extra_n
        return [k] if k >= 1 else []
    if "first_word_paragraph" in cons:
        n = cons["first_word_paragraph"]["n"]
        return list(range(max(n - extra_n, 1), n + 5))
    if "sections" in cons or "n_paragraphs" in cons:
        n = (cons.get("sections") or cons.get("n_paragraphs"))["n"]
        return list(range(max(n - extra_n, 1), n + 5))
    if "two_responses" in cons:
        return [4, 5, 6, 7]
    if "repeat_prompt" in cons:
        return [3, 4, 5]
    if "min_sentences" in cons:
        n = cons["min_sentences"]["n"]
        return list(range(max(n - 3, 3), n + 6))
    if "letter_freq" in cons:
        return list(range(5, 23))
    if "word_range" in cons:
        return [2, 3, 4, 5, 6, 7, 8]
    if "max_words" in cons or "max_sentences" in cons:
        return [1, 2, 3, 4, 5, 6]
    if "min_words" in cons:
        return list(range(6, 24))
    return [3, 4, 5, 6, 7]


def compose(row: dict, r: random.Random, tries: int = 12):
    cons = {c["name"]: c["params"] for c in row["constraints"]}
    for _ in range(tries):
        units = make_units(row, r)
        if "min_words" in cons or "max_words" in cons:
            pass
        for k in sorted(candidate_counts(cons), key=lambda x: r.random()) if "max_words" not in cons and "max_sentences" not in cons else candidate_counts(cons)[::-1]:
            if k > len(units):
                continue
            text = assemble(row, units, k, r)
            if text and verify(text, row["constraints"])[0]:
                return text, units[:k]
    return None, None


# ------------------------------------------------------------------------------------------------------------------------------- follow-ups
def followup(row: dict, text: str, units: list, r: random.Random):
    """-> (user2, assistant2, constraints2) or None"""
    cons1 = {c["name"] for c in row["constraints"]}
    ops = ["shorten", "bullets", "ps", "lower", "json"]
    r.shuffle(ops)
    plain = [u for u in units if not u.startswith(("Q:", "A:")) and ":" not in u.split(" ")[0]]
    for op in ops:
        if op == "shorten" and plain:
            n = r.choice([25, 35, 50, 70])
            out, total = [], 0
            for u in plain:
                if total + wcount(u) > n:
                    break
                out.append(u)
                total += wcount(u)
            if not out or "lowercase" in cons1 and False:
                continue
            resp = " ".join(out)
            if "lowercase" in cons1:
                resp = resp.lower()
            if "uppercase" in cons1:
                resp = resp.upper()
            return f"Now shorten it to at most {n} words.", resp, [{"name": "max_words", "params": {"n": n}}]
        if op == "bullets" and len(plain) >= 3:
            n = r.choice([3, 4, 5])
            if len(plain) < n:
                continue
            resp = "Here are the key points:\n" + "\n".join(f"- {u}" for u in plain[:n])
            if "lowercase" in cons1:
                resp = resp.lower()
            if "uppercase" in cons1:
                resp = resp.upper()
            return f"Now turn it into exactly {n} bullet points.", resp, [{"name": "n_bullets", "params": {"n": n}}]
        if op == "ps" and not ({"json", "quoted", "closer", "lowercase", "uppercase"} & cons1):
            return "Add a postscript at the end that starts with P.S.", text + "\nP.S. " + r.choice(PS_TEXT), [{"name": "postscript", "params": {}}]
        if op == "lower" and "uppercase" not in cons1 and "json" not in cons1:
            resp = text.lower()
            return "Now rewrite it so the whole response is in lowercase letters.", resp, [{"name": "lowercase", "params": {}}]
        if op == "json" and not ({"json", "quoted"} & cons1):
            content = " ".join(plain[:5])
            resp = json.dumps({"title": title_of(row["topic"]), "content": content}, ensure_ascii=False)
            return "Now give it as JSON with the keys \"title\" and \"content\" and nothing outside the JSON object.", resp, [{"name": "json", "params": {}}]
    return None


ALL_TOPICS = list(ifgen.TOPICS) + list(BANK2)
WEIGHTED = list(ifgen.CONSTRAINTS) + list(ifgen.EXTRA) * 2          # the IFEval-specific constraints are sampled twice as often


def _conf(name):
    return (ifgen.CONSTRAINTS.get(name) or ifgen.EXTRA[name])[2]


def build_prompt_v2(r: random.Random) -> dict:
    topic, (form, fkind) = r.choice(ALL_TOPICS), r.choice(ifgen.FORMS)
    base = form.format(t=topic)
    if r.random() < 0.05:                                              # "repeat the request" is exclusive: any other rule would apply to the repeated text too
        text, params = ifgen.EXTRA["repeat_prompt"][0](r)
        return {"prompt": f"{base}\n\n{text}", "constraints": [{"name": "repeat_prompt", "params": {"request": base}}], "form": fkind, "topic": topic}
    chosen, order = [], WEIGHTED[:]
    r.shuffle(order)
    want = r.choice([1, 2, 2, 3])
    for name in order:
        if len(chosen) >= want:
            break
        if name == "repeat_prompt" or name in chosen:
            continue
        if any(name in _conf(c) or c in _conf(name) or "*" in _conf(c) for c in chosen):
            continue
        chosen.append(name)
    specs, texts = [], []
    for name in chosen:
        text, params = (ifgen.CONSTRAINTS.get(name) or ifgen.EXTRA[name])[0](r)
        specs.append({"name": name, "params": params})
        texts.append(text)
    glue = r.choice([" ", "\n\n", " Please follow these rules: ", "\n\nRequirements:\n- "])
    body = texts[0] if len(texts) == 1 else ("\n- ".join(texts) if "- " in glue else " ".join(texts))
    return {"prompt": f"{base}{glue}{body}", "constraints": specs, "form": fkind, "topic": topic, "base": base, "texts": texts}


SYS_INTRO = ["Follow these rules in every reply:", "Always obey these formatting rules when you answer:", "You are a writing assistant. Rules for all your replies:", "Rules that apply to every response you give:"]


def build_system(n: int, seed: int):
    """The rules live in the SYSTEM message and the user only states the task; 30% continue with a second task under the same persistent rules."""
    r = random.Random(seed)
    rows, seen, stats = [], set(), {"tried": 0, "two_turn": 0}
    while len(rows) < n and stats["tried"] < n * 10:
        stats["tried"] += 1
        row = build_prompt_v2(r)
        if "repeat_prompt" in [c["name"] for c in row["constraints"]] or row["prompt"] in seen:
            continue
        text, _ = compose(row, r)
        if not text:
            continue
        rules = "\n".join(f"- {t}" for t in row["texts"])
        msgs = [{"role": "system", "content": f"{r.choice(SYS_INTRO)}\n{rules}"}, {"role": "user", "content": row["base"]}, {"role": "assistant", "content": text}]
        tags = ["verified_constraints", "composed", "system_rules"] + [c["name"] for c in row["constraints"]]
        if r.random() < 0.30:
            for _ in range(8):                                        # a second task, same rules (same parameters), different topic and form
                topic2, (form2, fkind2) = r.choice(ALL_TOPICS), r.choice(ifgen.FORMS)
                row2 = {"constraints": row["constraints"], "form": fkind2, "topic": topic2, "base": form2.format(t=topic2), "texts": row["texts"]}
                text2, _ = compose(row2, r)
                if text2 and text2 != text:
                    msgs += [{"role": "user", "content": r.choice(["Now do the same for this: ", "Next: ", "Another one. ", ""]) + row2["base"]}, {"role": "assistant", "content": text2}]
                    tags.append("two_turn")
                    stats["two_turn"] += 1
                    break
        key = json.dumps(msgs, sort_keys=True)
        if key in seen:
            continue
        seen.add(row["prompt"])
        seen.add(key)
        rows.append({"id": f"ifsys/{len(rows)}", "messages": msgs, "tools": [],
                     "meta": {"family": f"ifsys_{row['form']}", "category": "instruction_following", "difficulty": "medium", "kind": "instruct", "seed": len(rows), "tags": tags,
                              "n_tool_calls": 0, "n_turns": sum(m["role"] == "user" for m in msgs), "n_tools_offered": 0}})
    return rows, stats


def build(n: int, seed: int, forms=None):
    r = random.Random(seed)
    rows, seen, stats = [], set(), {"tried": 0, "composed": 0, "two_turn": 0}
    while len(rows) < n and stats["tried"] < n * 8:
        stats["tried"] += 1
        row = build_prompt_v2(r)
        if row["prompt"] in seen:
            continue
        text, units = compose(row, r)
        if not text:
            continue
        stats["composed"] += 1
        msgs = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": text}]
        tags = ["verified_constraints", "composed"] + [c["name"] for c in row["constraints"]]
        if r.random() < 0.30:
            fu = followup(row, text, units, r)
            if fu:
                u2, a2, c2 = fu
                if verify(a2, c2)[0]:
                    msgs += [{"role": "user", "content": u2}, {"role": "assistant", "content": a2}]
                    tags.append("two_turn")
                    stats["two_turn"] += 1
        key = json.dumps(msgs, sort_keys=True)
        if key in seen:
            continue
        seen.add(row["prompt"])
        seen.add(key)
        rows.append({"id": f"ifs/{len(rows)}", "messages": msgs, "tools": [],
                     "meta": {"family": f"ifs_{row['form']}", "category": "instruction_following", "difficulty": "medium", "kind": "instruct", "seed": len(rows), "tags": tags,
                              "n_tool_calls": 0, "n_turns": sum(m["role"] == "user" for m in msgs), "n_tools_offered": 0}})
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="data/raw/if_structured.jsonl")
    ap.add_argument("--system-rules", action="store_true", help="put the rules in the system prompt (and keep them for a second task)")
    a = ap.parse_args()
    rows, stats = (build_system if a.system_rules else build)(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    from collections import Counter
    c = Counter(t for x in rows for t in x["meta"]["tags"] if t not in ("verified_constraints", "composed", "two_turn"))
    print(f"{len(rows)} examples -> {a.out}; {stats}; constraints: {dict(c.most_common())}")


if __name__ == "__main__":
    main()
