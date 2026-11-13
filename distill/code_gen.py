"""Procedural coding tasks for the sandbox agent, accepted only if hidden tests pass.

    python -m distill.code_gen --n 1600 --out data/raw/agent_code_tasks.jsonl

Thirty function templates (strings, lists, numbers, brackets, searching, ciphers, number systems, matrices ...) each carry a reference implementation, a
natural-language description with several phrasings, and an input generator with edge cases. A task asks for the function in a file (random file / module
name), the teacher writes it with a heredoc, tries it on three inputs and reports the real output; then the hidden acceptance tests (six or more inputs
including edge cases, expected values from the reference) are RUN in the final workspace. A task whose code fails them is dropped. Trajectories are
executed for real in the sandbox by distill.teacher_tasks.build, so every observation is genuine output.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from distill import teacher_tasks

T: dict = {}


def tpl(name, fn, params, desc, src, gen, edge, plan=None):
    T[name] = dict(name=name, fn=fn, params=params, desc=desc, src=src, gen=gen, edge=edge, plan=plan)


WORDS = ["river", "stone", "lamp", "garden", "cloud", "bridge", "paper", "violin", "market", "winter", "orange", "signal", "harbor", "pencil", "forest"]


def ints(r, lo=-20, hi=40, n=(3, 8)):
    return [r.randrange(lo, hi) for _ in range(r.randrange(*n))]


def sentence(r, n=(3, 7)):
    return " ".join(r.choice(WORDS) for _ in range(r.randrange(*n)))


tpl("sum_of_squares", "sum_of_squares", "nums", "returns the sum of the squares of the numbers in a list", "def sum_of_squares(nums):\n    return sum(x * x for x in nums)\n",
    lambda r: (ints(r),), [([],), ([0],), ([-3, 3],)], "Squaring each value and adding them with sum() handles negatives and an empty list.")
tpl("count_vowels", "count_vowels", "text", "counts the vowels (a, e, i, o, u, case-insensitive) in a string", "def count_vowels(text):\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n",
    lambda r: (sentence(r).title(),), [("",), ("xyz",), ("AEIOU aeiou",)], "I'll lower-case the text once and count characters that are in 'aeiou'.")
tpl("reverse_words", "reverse_words", "text", "reverses the order of the words in a sentence and joins them with single spaces", "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n",
    lambda r: (sentence(r),), [("",), ("one",), ("  spaced   out  ",)], "split() without arguments also collapses repeated spaces, so reversing and joining gives clean output.")
tpl("is_anagram", "is_anagram", "a, b", "checks whether two strings are anagrams, ignoring case and spaces",
    "def is_anagram(a, b):\n    clean = lambda s: sorted(s.lower().replace(' ', ''))\n    return clean(a) == clean(b)\n",
    lambda r: (lambda w: (w, "".join(r.sample(w, len(w))) if r.random() < 0.6 else r.choice(WORDS)))(r.choice(WORDS)), [("listen", "silent"), ("a", "b"), ("Dormitory", "dirty room")],
    "Two strings are anagrams when their sorted letters match, after lower-casing and removing spaces.")
tpl("second_largest", "second_largest", "nums", "returns the second largest DISTINCT value in a list, or None if there are fewer than two distinct values",
    "def second_largest(nums):\n    distinct = sorted(set(nums), reverse=True)\n    return distinct[1] if len(distinct) > 1 else None\n",
    lambda r: (ints(r, 0, 15),), [([],), ([5, 5],), ([1, 2],)], "Using set() removes duplicates, then I sort descending and take index 1 when it exists.")
tpl("running_total", "running_total", "nums", "returns the running (cumulative) totals of a list of numbers", "from itertools import accumulate\n\n\ndef running_total(nums):\n    return list(accumulate(nums))\n",
    lambda r: (ints(r, 0, 20),), [([],), ([5],), ([1, -1, 1],)], "itertools.accumulate does exactly this.")
tpl("dedupe_keep_order", "dedupe_keep_order", "items", "removes duplicates from a list but keeps the first occurrence of each item in its original order", "def dedupe_keep_order(items):\n    return list(dict.fromkeys(items))\n",
    lambda r: ([r.choice(WORDS[:6]) for _ in range(r.randrange(4, 10))],), [([],), (["a", "a"],), ([3, 1, 3, 2, 1],)], "A dict keeps insertion order, so dict.fromkeys() deduplicates without reordering.")
tpl("top_char", "top_char", "text", "returns the most frequent letter in a string (ignore spaces and case); ties go to the alphabetically first letter, and an empty string gives None",
    "from collections import Counter\n\n\ndef top_char(text):\n    counts = Counter(ch for ch in text.lower() if ch.isalpha())\n    if not counts:\n        return None\n    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]\n",
    lambda r: (sentence(r, (2, 5)),), [("",), ("aabb",), ("Hello World",)], "Count letters with Counter, then sort by count (descending) and letter (ascending) so ties are alphabetical.")
tpl("title_case_words", "title_case_words", "text", "capitalises the first letter of every word and lower-cases the rest", "def title_case_words(text):\n    return ' '.join(w.capitalize() for w in text.split())\n",
    lambda r: (sentence(r).upper() if r.random() < 0.4 else sentence(r),), [("",), ("hELLO wORLD",)], "str.capitalize() already lower-cases the remainder of each word.")
tpl("to_snake_case", "to_snake_case", "name", "converts a camelCase or PascalCase identifier to snake_case",
    "import re\n\n\ndef to_snake_case(name):\n    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()\n",
    lambda r: ("".join(w.capitalize() if i else w for i, w in enumerate(r.sample(WORDS, r.randrange(2, 4)))),), [("word",), ("camelCaseName",), ("PascalCase",)], "An underscore goes before every capital letter that is not at the start, then lower-case everything.")
tpl("chunk_list", "chunk_list", "items, size", "splits a list into consecutive chunks of the given size (the last chunk may be shorter)", "def chunk_list(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]\n",
    lambda r: (ints(r, 0, 30, (4, 11)), r.randrange(2, 5)), [([], 3), ([1, 2, 3], 5), ([1, 2, 3, 4], 2)], "Step through the list in jumps of size and slice.")
tpl("rotate_list", "rotate_list", "items, k", "rotates a list to the right by k positions (k may be larger than the list length; an empty list stays empty)",
    "def rotate_list(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else list(items)\n",
    lambda r: (ints(r, 0, 20, (3, 8)), r.randrange(0, 12)), [([], 3), ([1, 2, 3], 0), ([1, 2, 3], 4)], "Taking k modulo the length handles big k, and slicing from the end moves the tail to the front.")
tpl("digit_sum", "digit_sum", "n", "returns the sum of the decimal digits of an integer (ignore the sign)", "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))\n",
    lambda r: (r.randrange(-99999, 99999),), [(0,), (-123,), (999,)], "Convert abs(n) to a string and add the digits.")
tpl("is_balanced", "is_balanced", "text", "checks whether the brackets (), [] and {} in a string are balanced and correctly nested, ignoring other characters",
    "def is_balanced(text):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n",
    lambda r: (r.choice(["(a[b]{c})", "([)]", "((", "{[()]}", "a)b(", "[]{}()", "(()"]),), [("",), ("(]",), ("{[()]}",)], "A stack works: push openers, and every closer must match the most recent opener.")
tpl("binary_search", "binary_search", "items, target", "returns the index of target in a sorted list using binary search, or -1 if it is not present",
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    lambda r: (lambda xs: (xs, r.choice(xs + [r.randrange(0, 60)])))(sorted(set(r.randrange(0, 50) for _ in range(r.randrange(4, 10))))), [([], 1), ([1], 1), ([1, 3, 5], 4)], "Keep a low and high bound and halve the range until the value is found or the bounds cross.")
tpl("unique_word_count", "unique_word_count", "text", "counts the distinct words in a text, ignoring case and surrounding punctuation",
    "import string\n\n\ndef unique_word_count(text):\n    words = [w.strip(string.punctuation).lower() for w in text.split()]\n    return len({w for w in words if w})\n",
    lambda r: (" ".join(r.choice(WORDS[:7]).title() + r.choice(["", ",", "."]) for _ in range(r.randrange(4, 10))),), [("",), ("Hi, hi. HI!",)], "Strip punctuation and lower-case each token, then count the distinct non-empty ones.")
tpl("longest_common_prefix", "longest_common_prefix", "words", "returns the longest common prefix of a list of strings (an empty string if there is none or the list is empty)",
    "def longest_common_prefix(words):\n    if not words:\n        return ''\n    prefix = words[0]\n    for w in words[1:]:\n        while not w.startswith(prefix):\n            prefix = prefix[:-1]\n    return prefix\n",
    lambda r: (lambda b: ([b + r.choice(WORDS)[:3] for _ in range(r.randrange(2, 5))],))(r.choice(["inter", "pre", "un", "sub"])), [([],), (["a"],), (["flower", "flow", "flight"],), (["dog", "cat"],)], "Start with the first word as the prefix and trim it until every other word starts with it.")
tpl("caesar_shift", "caesar_shift", "text, shift", "applies a Caesar cipher: shift each letter by `shift` positions in the alphabet (wrapping around, keeping case) and leave other characters unchanged",
    "def caesar_shift(text, shift):\n    out = []\n    for ch in text:\n        if ch.isalpha():\n            base = ord('A') if ch.isupper() else ord('a')\n            out.append(chr((ord(ch) - base + shift) % 26 + base))\n        else:\n            out.append(ch)\n    return ''.join(out)\n",
    lambda r: (sentence(r, (2, 4)).title(), r.randrange(-5, 14)), [("", 3), ("xyz", 3), ("Hello, World!", 13)], "Work relative to 'a' or 'A', add the shift modulo 26, and keep punctuation as it is.")
tpl("int_to_roman", "int_to_roman", "n", "converts an integer from 1 to 3999 to a Roman numeral string",
    "def int_to_roman(n):\n    table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n    out = ''\n    for value, symbol in table:\n        while n >= value:\n            out += symbol\n            n -= value\n    return out\n",
    lambda r: (r.randrange(1, 3999),), [(4,), (9,), (1994,), (3999,)], "A greedy pass over the value/symbol table, including subtractive pairs like CM and IV.")
tpl("roman_to_int", "roman_to_int", "s", "converts a Roman numeral string (upper case) to an integer",
    "def roman_to_int(s):\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and vals[ch] < vals[s[i + 1]]:\n            total -= vals[ch]\n        else:\n            total += vals[ch]\n    return total\n",
    lambda r: (r.choice(["III", "IV", "IX", "LVIII", "MCMXCIV", "XL", "MMXXV", "CDXLIV", "XIV"]),), [("I",), ("IV",), ("MCMXCIV",)], "A symbol smaller than the next one is subtracted, otherwise added.")
tpl("transpose", "transpose", "matrix", "transposes a rectangular matrix given as a list of rows (an empty matrix gives an empty list)", "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
    lambda r: (lambda c: ([[r.randrange(0, 10) for _ in range(c)] for _ in range(r.randrange(2, 4))],))(r.randrange(2, 4)), [([],), ([[1, 2, 3]],), ([[1], [2]],)], "zip(*matrix) pairs up the columns.")
tpl("merge_sorted", "merge_sorted", "a, b", "merges two already sorted lists into one sorted list in linear time",
    "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    return out + a[i:] + b[j:]\n",
    lambda r: (sorted(ints(r, 0, 30)), sorted(ints(r, 0, 30))), [([], []), ([1], []), ([1, 3], [2, 4])], "Two pointers walk the lists, always taking the smaller head, then append what is left.")
tpl("fizzbuzz", "fizzbuzz", "n", "returns the list of strings for 1..n where multiples of 3 become 'Fizz', multiples of 5 become 'Buzz' and multiples of both become 'FizzBuzz'",
    "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n    return out\n",
    lambda r: (r.randrange(1, 17),), [(0,), (1,), (15,)], "Test the multiple of 15 first so it is not caught by the 3 or 5 branch.")
tpl("clamp_all", "clamp_all", "nums, lo, hi", "clamps every number of a list into the range [lo, hi]", "def clamp_all(nums, lo, hi):\n    return [max(lo, min(x, hi)) for x in nums]\n",
    lambda r: (ints(r, -10, 30), r.randrange(-5, 3), r.randrange(8, 20)), [([], 0, 5), ([1, 9], 2, 5)], "min() against hi and max() against lo, applied to each element.")
tpl("count_pairs_sum", "count_pairs_sum", "nums, target", "counts the pairs of positions i < j whose values add up to target",
    "def count_pairs_sum(nums, target):\n    count = 0\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                count += 1\n    return count\n",
    lambda r: (ints(r, 0, 10, (4, 9)), r.randrange(5, 15)), [([], 3), ([2, 2, 2], 4), ([1, 2, 3], 10)], "Two loops over positions with j after i, so each pair is counted once.")
tpl("flatten_once", "flatten_once", "lists", "flattens a list of lists by one level, keeping the order", "def flatten_once(lists):\n    return [x for sub in lists for x in sub]\n",
    lambda r: ([ints(r, 0, 9, (0, 4)) for _ in range(r.randrange(2, 5))],), [([],), ([[], []],), ([[1], [2, 3]],)], "A nested comprehension walks each sublist and its items.")
tpl("median_of", "median_of", "nums", "returns the median of a non-empty list of numbers (the mean of the two middle values for an even count)",
    "def median_of(nums):\n    s = sorted(nums)\n    mid = len(s) // 2\n    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2\n",
    lambda r: (ints(r, 0, 50, (1, 9)),), [([5],), ([1, 2],), ([3, 1, 2],)], "Sort a copy, then take the middle element or average the two middle ones.")
tpl("prime_factors", "prime_factors", "n", "returns the prime factors of an integer n >= 2 in ascending order, repeated according to multiplicity",
    "def prime_factors(n):\n    out, d = [], 2\n    while d * d <= n:\n        while n % d == 0:\n            out.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        out.append(n)\n    return out\n",
    lambda r: (r.randrange(2, 400),), [(2,), (12,), (97,), (360,)], "Trial division: divide out each factor completely before moving on, and what remains above 1 is prime.")
tpl("max_subarray_sum", "max_subarray_sum", "nums", "returns the largest sum of any non-empty contiguous slice of a list of integers",
    "def max_subarray_sum(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n",
    lambda r: (ints(r, -9, 12, (3, 9)),), [([5],), ([-3, -1, -2],), ([1, -2, 3, 4],)], "Kadane's algorithm: at each item either extend the current slice or start a new one.")
tpl("move_zeros", "move_zeros", "nums", "returns a new list with all zeros moved to the end while keeping the order of the other numbers", "def move_zeros(nums):\n    return [x for x in nums if x != 0] + [x for x in nums if x == 0]\n",
    lambda r: ([r.choice([0, 0, 1, 2, 3, 4, 5]) for _ in range(r.randrange(4, 10))],), [([],), ([0, 0],), ([0, 1, 0, 2],)], "Keep the non-zeros in order and append the zeros.")

FILES = ["utils", "helpers", "toolbox", "lib", "snippets", "functions", "tools", "core"]
OPEN = ["Create {mod}.py with a function {sig} that {desc}.", "Write a Python function {sig} in {mod}.py that {desc}.", "I need {mod}.py containing {sig}, which {desc}.",
        "Please add {sig} to a new file {mod}.py. It {desc2}."]
CLOSE = ["", " Then test it.", " Try it on a few inputs and show me the results.", " Run a quick test afterwards."]
NOTES = ["Project notes: keep helpers small and tested.\n", "TODO: write the utilities module.\n"]


def run_ref(t: dict, args):
    ns = {}
    exec(t["src"], ns)
    return ns[t["fn"]](*args)


def make_entry(name: str, seed: int):
    t = T[name]
    r = random.Random(f"code:{name}:{seed}")
    mod = r.choice(FILES)
    shown = [t["gen"](r) for _ in range(3)]
    shown[r.randrange(3)] = r.choice(t["edge"])
    hidden = [t["gen"](r) for _ in range(5)] + list(t["edge"])
    sig = f"{t['fn']}({t['params']})"
    desc = t["desc"]
    desc2 = desc[0].lower() + desc[1:] if not desc.startswith("returns") else desc.replace("returns", "returns", 1)
    prompt = r.choice(OPEN).format(mod=mod, sig=sig, desc=desc, desc2=desc2)
    if r.random() < 0.5:
        ex_args = shown[0]
        prompt += f" For example {t['fn']}({', '.join(repr(a) for a in ex_args)}) should give {run_ref(t, ex_args)!r}."
    prompt += r.choice(CLOSE)
    outs = [run_ref(t, a) for a in shown]
    calls = "\n".join(f"print(repr({t['fn']}({', '.join(repr(a) for a in args)})))" for args in shown)
    test_code = f"from {mod} import {t['fn']}\n{calls}"
    files = {"notes.txt": r.choice(NOTES)} if r.random() < 0.4 else {}
    steps = []
    if files:
        steps.append({"plan": r.choice(["Let me see what is already in the workspace.", "First a quick look at the folder."]), "tool": "run_bash", "arg": "ls"})
    steps.append({"plan": t["plan"] or "I'll write the function and then test it.", "tool": "run_bash", "arg": f"cat > {mod}.py <<'EOF'\n{t['src']}EOF"})
    steps.append({"plan": r.choice(["Now a quick test on a few inputs, including an edge case.", "Let me try it on three inputs and look at the output.", "Time to run it on some examples."]),
                  "tool": "run_bash", "arg": "python3 - <<'PY'\n" + test_code + "\nPY"})
    lines = "; ".join(f"{t['fn']}({', '.join(repr(a) for a in args)}) gives {o!r}" for args, o in zip(shown, outs))
    final = f"{mod}.py now defines {sig}. In my test run: {lines}."
    hid = "; ".join(f"assert {t['fn']}({', '.join(repr(a) for a in args)}) == {run_ref(t, args)!r}" for args in hidden)
    return {"id": f"{name}_{seed}", "category": f"code_{name}", "difficulty": "medium", "tags": ["python", "function", "code", "hidden_tests"], "prompt": prompt, "files": files, "steps": steps, "final": final,
            "grounded": [repr(o) for o in outs if len(repr(o)) >= 2], "check": {"files": {f"{mod}.py": {"contains": [f"def {t['fn']}"]}}, "run": [{"cmd": f"python3 -c \"from {mod} import {t['fn']}; {hid}; print('ok')\"", "stdout_contains": ["ok"]}]}}


def _build_chunk(entries):
    rows, drops = teacher_tasks.build(entries, verbose=False)
    return rows, dict(drops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--out", default="data/raw/agent_code_tasks.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    names = list(T)
    entries, seen, k = [], set(), 0
    while len(entries) < a.n and k < a.n * 4:
        e = make_entry(names[k % len(names)], k)
        k += 1
        key = e["prompt"] + json.dumps(e["steps"][-1]["arg"])
        if key in seen:
            continue
        seen.add(key)
        entries.append(e)
    chunks = [entries[i::a.workers * 4] for i in range(a.workers * 4)]
    rows, drops = [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for rr, dd in pool.map(_build_chunk, chunks):
            rows += rr
            for kk, v in dd.items():
                drops[kk] = drops.get(kk, 0) + v
    for r_ in rows:
        r_["id"] = r_["id"].replace("agent/teacher/", "agent/code/")
        r_["meta"]["family"] = r_["meta"]["family"].replace("teacher_", "")
        r_["meta"]["tags"] = [t for t in r_["meta"]["tags"] if t != "teacher_written"] + ["procedural_code"]
        r_["meta"].pop("teacher_written", None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(T)} templates -> {len(entries)} candidate tasks, {len(rows)} verified in {a.out}; rejected {drops}")


if __name__ == "__main__":
    main()
