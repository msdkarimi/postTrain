"""Instruction-following PROMPTS with programmatic verifiers (the data side of "context following / alignment" in the KD survey).

    python -m distill.ifgen --n 6000 --out data/raw/if_prompts.jsonl          # prompts + constraint specs, NO answers
    python -m distill.teacher_sample --prompts data/raw/if_prompts.jsonl ...  # a teacher model answers on the GPU box; only passing answers are kept

The prompts are INSTRUCTIONS, not answers: a task (write / explain / compare / plan / rewrite ... about a topic) plus 1-3 verifiable constraints
(word limits, bullet counts, required / forbidden words, no commas, casing, a title in <<double angle brackets>>, JSON output, a closing phrase, ...).
Every constraint has a checker, so a teacher's answer is accepted only if it really satisfies all of them (rejection sampling = the survey's
"data curation" with a verifiable "feedback" signal). Constraints that contradict each other are never combined.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TOPICS = ["a broken heater in a rented flat", "how vaccines train the immune system", "why the sky is blue", "starting a vegetable garden on a balcony",
          "preparing for a job interview", "the benefits of sleeping eight hours", "how a bill becomes law", "saving money on groceries", "learning to play the guitar",
          "the water cycle", "a delayed train and a missed meeting", "choosing between renting and buying a home", "the history of the bicycle", "how to brew good coffee",
          "remote work and team communication", "why bees matter for farming", "planning a three-day trip to Lisbon", "how compound interest works", "training for a first 10k run",
          "writing a clear bug report", "a new recycling programme in a small town", "how the internet routes data", "a lost package that never arrived", "the pros and cons of electric cars",
          "meditation for beginners", "organising a surprise birthday party", "how to review a colleague's code kindly", "the life cycle of a butterfly", "setting up a home network safely",
          "how to prepare a presentation for executives", "a neighbour's noisy dog", "why exercise improves mood", "the basics of budgeting for students", "how tides work",
          "a product launch delayed by two weeks", "learning a language in six months", "the role of libraries today", "choosing a laptop for programming", "how earthquakes happen",
          "a polite refusal of a meeting invitation", "reducing food waste at home", "how to give constructive feedback", "an apology for a missed deadline", "the invention of the printing press",
          "keeping passwords secure", "a customer asking for a refund", "the difference between weather and climate", "building a morning routine", "why maps distort the world"]
FORMS = [("Write a short email about {t}.", "email"), ("Explain {t} to a curious ten-year-old.", "explain"), ("Give me practical advice related to {t}.", "advice"),
         ("Write a blog post intro about {t}.", "blog"), ("Summarise the key ideas behind {t}.", "summary"), ("Draft a message to a colleague about {t}.", "message"),
         ("Write a short story that involves {t}.", "story"), ("List the main points someone should know about {t}.", "list"), ("Compare two different perspectives on {t}.", "compare"),
         ("Write a product-style description related to {t}.", "description"), ("Create a short checklist related to {t}.", "checklist"), ("Write a persuasive paragraph about {t}.", "persuade"),
         ("Describe a common mistake people make when it comes to {t}, and how to avoid it.", "mistake"), ("Write a dialogue between two people discussing {t}.", "dialogue"),
         ("Write a short speech about {t}.", "speech"), ("Write a FAQ with answers about {t}.", "faq")]
KEYWORDS = ["balance", "simple", "careful", "future", "community", "practice", "habit", "clear", "budget", "pattern", "trust", "daily", "gentle", "honest", "growth", "signal"]
BANNED = ["very", "really", "thing", "just", "important", "good", "easy", "always"]
CLOSERS = ["Is there anything else I can help with?", "Let me know if you have any questions.", "Thanks for reading.", "That is all for now."]


def words(s: str) -> list:
    return re.findall(r"[A-Za-z0-9']+", s)


def bullets(s: str) -> int:
    return len(re.findall(r"^\s*[-*] ", s, re.M))


def paragraphs(s: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", s.strip()) if p.strip()])


# name -> (builder(rng) -> (text, params), checker(response, params) -> bool, conflicts)
CONSTRAINTS: dict = {}


def reg(name, conflicts=()):
    def deco(fn):
        build, check = fn()
        CONSTRAINTS[name] = (build, check, set(conflicts))
        return fn
    return deco


@reg("max_words", conflicts={"min_words", "json", "n_paragraphs"})
def _():
    def b(r):
        n = r.choice([40, 60, 80, 100, 150])
        return f"Answer in at most {n} words.", {"n": n}
    return b, lambda s, p: len(words(s)) <= p["n"]


@reg("min_words", conflicts={"max_words", "json", "n_bullets"})
def _():
    def b(r):
        n = r.choice([120, 150, 200, 250])
        return f"Use at least {n} words.", {"n": n}
    return b, lambda s, p: len(words(s)) >= p["n"]


@reg("n_bullets", conflicts={"json", "n_paragraphs", "min_words", "numbered"})
def _():
    def b(r):
        n = r.choice([3, 4, 5, 6])
        return f"Your answer must contain exactly {n} bullet points in markdown format (each on its own line starting with \"- \").", {"n": n}
    return b, lambda s, p: bullets(s) == p["n"]


@reg("numbered", conflicts={"json", "n_bullets", "n_paragraphs", "title"})
def _():
    def b(r):
        n = r.choice([3, 4, 5])
        return f"Present the answer as a numbered list with exactly {n} items (1., 2., ...).", {"n": n}
    return b, lambda s, p: len(re.findall(r"^\s*\d+\. ", s, re.M)) == p["n"]


@reg("n_paragraphs", conflicts={"json", "n_bullets", "numbered", "max_words", "sections"})
def _():
    def b(r):
        n = r.choice([2, 3, 4])
        return f"There should be exactly {n} paragraphs, separated by a blank line.", {"n": n}
    return b, lambda s, p: paragraphs(s) == p["n"]


@reg("keywords")
def _():
    def b(r):
        ks = r.sample(KEYWORDS, 2)
        return f"Include the words \"{ks[0]}\" and \"{ks[1]}\" in your answer.", {"ks": ks}
    return b, lambda s, p: all(re.search(rf"\b{k}\b", s, re.I) for k in p["ks"])


@reg("banned")
def _():
    def b(r):
        k = r.choice(BANNED)
        return f"Do not use the word \"{k}\" anywhere.", {"k": k}
    return b, lambda s, p: not re.search(rf"\b{p['k']}\b", s, re.I)


@reg("no_commas", conflicts={"json"})
def _():
    return (lambda r: ("Do not use any commas in your response.", {})), (lambda s, p: "," not in s)


@reg("lowercase", conflicts={"uppercase", "title", "json", "sections", "closer", "starts_with", "postscript"})
def _():
    return (lambda r: ("Your entire response should be in lowercase letters, with no capital letters at all.", {})), (lambda s, p: s == s.lower())


@reg("uppercase", conflicts={"lowercase", "json", "title", "keywords", "closer", "starts_with"})
def _():
    return (lambda r: ("Write your entire response in capital letters.", {})), (lambda s, p: s == s.upper())


@reg("closer", conflicts={"json", "postscript", "quoted"})
def _():
    def b(r):
        c = r.choice(CLOSERS)
        return f"Finish your response with the exact phrase \"{c}\" and nothing after it.", {"c": c}
    return b, lambda s, p: s.strip().endswith(p["c"])


@reg("title", conflicts={"json", "lowercase", "uppercase", "numbered"})
def _():
    return (lambda r: ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.", {})), \
           (lambda s, p: re.search(r"<<[^\n]+>>", s) is not None)


@reg("postscript", conflicts={"json", "closer", "quoted"})
def _():
    return (lambda r: ("At the end of your response, add a postscript starting with P.S.", {})), (lambda s, p: re.search(r"^\s*P\.S\.", s, re.M) is not None)


@reg("highlights", conflicts={"json", "uppercase"})
def _():
    def b(r):
        n = r.choice([2, 3])
        return f"Highlight at least {n} sections of your answer with markdown, i.e. *highlighted section*.", {"n": n}
    return b, lambda s, p: len(re.findall(r"\*[^\n*]+\*", s)) >= p["n"]


@reg("placeholders", conflicts={"json"})
def _():
    def b(r):
        n = r.choice([2, 3])
        return f"The response must contain at least {n} placeholders in square brackets, such as [address].", {"n": n}
    return b, lambda s, p: len(re.findall(r"\[[^\]\n]+\]", s)) >= p["n"]


@reg("sections", conflicts={"n_paragraphs", "json", "lowercase", "numbered"})
def _():
    def b(r):
        n = r.choice([2, 3, 4])
        return f"Your response must have exactly {n} sections. Mark the beginning of each section with \"SECTION X\", where X is the section number.", {"n": n}
    return b, lambda s, p: len(re.findall(r"^\s*SECTION \d+", s, re.M)) == p["n"]


@reg("json", conflicts={"max_words", "min_words", "n_bullets", "numbered", "n_paragraphs", "no_commas", "lowercase", "uppercase", "closer", "title",
                        "postscript", "highlights", "placeholders", "sections", "quoted", "max_sentences", "starts_with"})
def _():
    def b(r):
        return "Wrap your entire output in JSON format with the keys \"title\" and \"content\" (and nothing outside the JSON object).", {}

    def c(s, p):
        t = s.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        try:
            o = json.loads(t)
        except Exception:
            return False
        return isinstance(o, dict) and {"title", "content"} <= set(o) and all(isinstance(o[k], str) and o[k].strip() for k in ("title", "content"))
    return b, c


@reg("quoted", conflicts={"json", "closer", "postscript", "title", "sections"})
def _():
    return (lambda r: ("Wrap your entire response in double quotation marks.", {})), (lambda s, p: len(s.strip()) > 2 and s.strip()[0] == '"' and s.strip()[-1] == '"')


@reg("max_sentences", conflicts={"json", "min_words", "n_paragraphs", "n_bullets", "numbered", "sections"})
def _():
    def b(r):
        n = r.choice([2, 3, 4, 5])
        return f"Use no more than {n} sentences.", {"n": n}
    return b, lambda s, p: len([x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]) <= p["n"]


@reg("starts_with", conflicts={"json", "quoted", "title", "lowercase", "uppercase", "sections"})
def _():
    def b(r):
        w = r.choice(["Honestly", "First", "Imagine", "Picture this", "Today"])
        return f"Start your response with the word \"{w}\".", {"w": w}
    return b, lambda s, p: s.strip().startswith(p["w"])


def build_prompt(r: random.Random) -> dict:
    topic, (form, fkind) = r.choice(TOPICS), r.choice(FORMS)
    names = list(CONSTRAINTS)
    chosen: list = []
    for name in r.sample(names, len(names)):
        if len(chosen) >= r.choice([1, 2, 2, 3]):
            break
        if any(name in CONSTRAINTS[c][2] or c in CONSTRAINTS[name][2] for c in chosen):
            continue
        chosen.append(name)
    specs, texts = [], []
    for name in chosen:
        text, params = CONSTRAINTS[name][0](r)
        specs.append({"name": name, "params": params})
        texts.append(text)
    base = form.format(t=topic)
    glue = r.choice([" ", "\n\n", " Please follow these rules: ", "\n\nRequirements:\n- "])
    body = (texts[0] if len(texts) == 1 else ("\n- ".join(texts) if "- " in glue else " ".join(texts)))
    prompt = f"{base}{glue}{body}"
    return {"prompt": prompt, "constraints": specs, "form": fkind, "topic": topic}


# ---- extra IFEval-style constraints (kept in a separate registry so build_prompt() above keeps producing the exact same prompts / ids)
EXTRA: dict = {}


def reg_extra(name, conflicts=()):
    def deco(fn):
        build, check = fn()
        EXTRA[name] = (build, check, set(conflicts))
        return fn
    return deco


@reg_extra("two_responses", conflicts={"json", "n_bullets", "numbered", "n_paragraphs", "sections", "max_words", "min_words", "max_sentences", "title", "closer", "postscript", "quoted",
                                         "starts_with", "first_word_paragraph", "repeat_prompt", "highlights", "word_range", "min_sentences"})
def _():
    def b(r):
        return "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******.", {}

    def c(s, p):
        parts = [x.strip() for x in s.split("******")]
        return len(parts) == 2 and all(parts) and parts[0] != parts[1]
    return b, c


@reg_extra("repeat_prompt", conflicts={"*"})
def _():
    return (lambda r: ("First repeat the request above word for word without change, then give your answer. Do not say anything before repeating the request.", {})),            (lambda s, p: s.strip().startswith(p["request"].strip()))


@reg_extra("first_word_paragraph", conflicts={"json", "n_bullets", "numbered", "n_paragraphs", "sections", "title", "max_words", "max_sentences", "quoted", "two_responses", "repeat_prompt", "lowercase",
                                                "uppercase", "closer", "postscript", "highlights", "min_sentences", "word_range", "min_words"})
def _():
    WORDS = ["First", "Next", "Also", "Finally", "Overall", "Together", "Often", "Today", "Remember"]

    def b(r):
        n = r.choice([3, 4, 5])
        i, w = r.randrange(1, n + 1), r.choice(WORDS)
        return f"There should be {n} paragraphs separated by two line breaks. Paragraph {i} must start with the word \"{w}\".", {"n": n, "i": i, "w": w}

    def c(s, p):
        paras = [x for x in re.split(r"\n\s*\n", s.strip()) if x.strip()]
        return len(paras) == p["n"] and paras[p["i"] - 1].lstrip().split(" ")[0].strip('"') == p["w"]
    return b, c


@reg_extra("capital_words", conflicts={"lowercase", "uppercase", "json", "two_responses", "repeat_prompt"})
def _():
    def b(r):
        n = r.choice([3, 4, 5])
        return f"In your response, words with all capital letters should appear at least {n} times.", {"n": n}
    return b, lambda s, p: sum(1 for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 2 and w.isupper()) >= p["n"]


@reg_extra("letter_freq", conflicts={"json", "max_words", "max_sentences", "two_responses", "repeat_prompt", "uppercase"})
def _():
    def b(r):
        ch, n = r.choice(["e", "t", "a", "o"]), r.choice([15, 20, 25, 30])
        return f"In your response, the letter \"{ch}\" should appear at least {n} times.", {"ch": ch, "n": n}
    return b, lambda s, p: s.lower().count(p["ch"]) >= p["n"]


@reg_extra("min_sentences", conflicts={"max_sentences", "max_words", "n_bullets", "numbered", "sections", "n_paragraphs", "json", "two_responses", "repeat_prompt", "first_word_paragraph", "word_range"})
def _():
    def b(r):
        n = r.choice([5, 6, 8, 10])
        return f"Your response should contain at least {n} sentences.", {"n": n}
    return b, lambda s, p: len([x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]) >= p["n"]


@reg_extra("word_range", conflicts={"max_words", "min_words", "json", "n_bullets", "numbered", "sections", "n_paragraphs", "two_responses", "repeat_prompt", "first_word_paragraph", "min_sentences", "letter_freq",
                                      "closer", "postscript"})
def _():
    def b(r):
        lo = r.choice([30, 40, 50, 60])
        hi = lo + r.choice([20, 30, 40])
        return f"Answer with between {lo} and {hi} words.", {"lo": lo, "hi": hi}
    return b, lambda s, p: p["lo"] <= len(words(s)) <= p["hi"]


def check_fn(name: str):
    return (CONSTRAINTS.get(name) or EXTRA[name])[1]


def verify(response: str, constraints: list) -> tuple:
    """-> (all_ok, [names that failed])"""
    failed = [c["name"] for c in constraints if not check_fn(c["name"])(response, c["params"])]
    return (not failed), failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/raw/if_prompts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r, seen, rows = random.Random(a.seed), set(), []
    while len(rows) < a.n:
        row = build_prompt(r)
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        row["id"] = f"if/{len(rows)}"
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    from collections import Counter
    c = Counter(s["name"] for x in rows for s in x["constraints"])
    print(f"{len(rows)} prompts -> {a.out}; constraint usage: {dict(c.most_common())}")


if __name__ == "__main__":
    main()
