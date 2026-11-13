"""More Python / bash workspace families (breadth) and multi-turn follow-ups where the workspace persists between turns."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from envs.tasks.base import (PLAN_DO, PLAN_DO_BASH, Followup, Step, Task, family, frame, has_number, inspect_steps, last_error_line, norm_lines, pick,
                             read_text, verify_step, with_recovery)
from envs.tasks.common import CATEGORIES, PRODUCTS, REGIONS, add_distractors, people, rand_date, to_csv, to_json

WORD_POOL = ["budget", "report", "client", "meeting", "deadline", "project", "invoice", "team", "launch", "review", "plan", "risk", "status",
             "update", "goal", "task", "quarter", "revenue", "support", "design", "release", "feature", "bug", "sprint", "demo", "scope"]


# ===================================================================================================
@family("csv_percent_of_total", "data", 1.3)
def csv_percent_of_total(rng):
    cats = rng.sample(CATEGORIES, rng.randint(3, 6))
    rows = [[rng.choice(cats), f"{rng.uniform(10, 900):.2f}"] for _ in range(rng.randint(25, 60))]
    for i, c in enumerate(cats):
        rows[i][0] = c
    tot = defaultdict(float)
    for c, a in rows:
        tot[c] += float(a)
    grand = sum(tot.values())
    ordered = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
    expected = ["category,percent"] + [f"{c},{v / grand * 100:.2f}" for c, v in ordered]
    fname, out = rng.choice(["spend.csv", "expenses.csv", "budget_lines.csv"]), rng.choice(["shares.csv", "percent_of_total.csv"])
    body = pick(rng, [f"From {fname}, work out each category's share of the total amount as a percentage with 2 decimals. Write {out} (columns category,percent), largest share first, and tell me the biggest category.",
                      f"What percentage of the overall spend does each category in {fname} represent? Save it to {out} as category,percent (2 decimals, descending) and name the top one."])
    code = (f"import csv\nfrom collections import defaultdict\ntotals = defaultdict(float)\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n        totals[r['category']] += float(r['amount'])\n"
            "grand = sum(totals.values())\nordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))\n"
            f"with open({out!r}, 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['category', 'percent'])\n    for c, v in ordered:\n        w.writerow([c, f'{{v / grand * 100:.2f}}'])\n"
            f"print(open({out!r}).read())\n")
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    top = ordered[0][0]
    final = f"Done - {out} has {len(ordered)} categories; {top} is the biggest at {ordered[0][1] / grand * 100:.2f}%."

    def check(ws, reply):
        txt = read_text(ws, out)
        return (txt is not None and norm_lines(txt) == expected and top in reply), "shares wrong"

    return Task("csv_percent_of_total", "data", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv(["category", "amount"], rows)}), steps, final, check,
                [top, f"{ordered[0][1] / grand * 100:.2f}"], ["python", "file_output", "percentages"])


# ===================================================================================================
@family("csv_filter_export", "data", 1.3)
def csv_filter_export(rng):
    region, thr = rng.choice(REGIONS), rng.choice([200, 300, 500, 750])
    rows = [[f"ORD-{1000 + i}", rng.choice(REGIONS), rng.choice(PRODUCTS[:6]), f"{rng.uniform(20, 1200):.2f}"] for i in range(rng.randint(25, 55))]
    for i in range(3):
        rows[i][1], rows[i][3] = region, f"{rng.uniform(thr + 20, thr + 600):.2f}"
    sel = [r for r in rows if r[1] == region and float(r[3]) > thr]
    fname, out = rng.choice(["orders.csv", "sales_log.csv"]), rng.choice(["big_orders.csv", "filtered.csv"])
    expected = ["order_id,region,product,amount"] + [",".join(r) for r in sel]
    body = pick(rng, [f"From {fname}, keep only the {region} rows whose amount is greater than {thr} and save them (with the same header) to {out}. How many rows did you keep?",
                      f"Filter {fname}: region must be {region} and amount above {thr}. Write the matching rows to {out}, same columns, and tell me the count."])
    code = (f"import csv\nwith open({fname!r}, newline='') as f:\n    reader = csv.DictReader(f)\n    header = reader.fieldnames\n    keep = [r for r in reader if r['region'] == {region!r} and float(r['amount']) > {thr}]\n"
            f"with open({out!r}, 'w', newline='') as o:\n    w = csv.DictWriter(o, fieldnames=header, lineterminator='\\n')\n    w.writeheader()\n    w.writerows(keep)\nprint(len(keep), 'rows kept')\n")
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code)] + verify_step(rng, f"head -n 3 {out}", 0.4)
    final = f"I kept {len(sel)} {region} rows with amount above {thr} and saved them to {out}."

    def check(ws, reply):
        txt = read_text(ws, out)
        return (txt is not None and norm_lines(txt) == expected and has_number(reply, len(sel), 0.01)), "filtered file wrong"

    return Task("csv_filter_export", "data", "easy", frame(rng, body), add_distractors(rng, {fname: to_csv(["order_id", "region", "product", "amount"], rows)}), steps, final, check,
                [f"{len(sel)} rows kept"], ["python", "file_output", "filtering"])


# ===================================================================================================
@family("text_word_stats", "text", 1.2)
def text_word_stats(rng):
    words = [rng.choice(WORD_POOL) for _ in range(rng.randint(60, 140))]
    sents, i = [], 0
    while i < len(words):
        k = rng.randint(5, 12)
        s = " ".join(words[i:i + k]).capitalize() + rng.choice([".", ".", "!", "?"])
        sents.append(s)
        i += k
    text = " ".join(sents)
    toks = re.findall(r"[a-z']+", text.lower())
    cnt = Counter(toks)
    top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    fname = rng.choice(["essay.txt", "notes.txt", "draft.txt"])
    body = pick(rng, [f"For {fname}: how many words are there in total, how many different words (ignore case and punctuation), and which 3 words appear most often? Break ties alphabetically.",
                      f"Give me word statistics for {fname}: total words, unique words (case-insensitive) and the top 3 most frequent words with their counts (ties: alphabetical)."])
    code = (f"import re\nfrom collections import Counter\ntext = open({fname!r}).read().lower()\nwords = re.findall(r\"[a-z']+\", text)\ncounts = Counter(words)\n"
            "print('total:', len(words))\nprint('unique:', len(counts))\nfor w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:\n    print(f'{w}: {c}')\n")
    steps = [Step(pick(rng, ["Let me look at the file first.", "I'll check the text before counting."]), "run_bash", f"ls; head -c 300 {fname}"), Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"{fname} has {len(toks)} words, {len(cnt)} unique. The top 3: " + ", ".join(f"{w} ({c})" for w, c in top) + "."

    def check(ws, reply):
        return (has_number(reply, len(toks), 0.01) and has_number(reply, len(cnt), 0.01) and all(w in reply and has_number(reply, c, 0.01) for w, c in top)), "stats wrong"

    return Task("text_word_stats", "text", "easy", frame(rng, body), add_distractors(rng, {fname: text + "\n"}), steps, final, check,
                [f"total: {len(toks)}", f"unique: {len(cnt)}"], ["python", "reply_answer", "counting"])


# ===================================================================================================
@family("text_replace_files", "text", 1.2)
def text_replace_files(rng):
    old, new = rng.choice([("Falcon", "Osprey"), ("Atlas", "Meridian"), ("Q3", "Q4"), ("beta", "stable"), ("Acme", "Globex")])
    sentences = ["The {x} rollout is on track.", "Please review the {x} plan before Friday.", "Budget for {x} was approved.", "Nothing to report today.", "Support asked about {x} again.", "The team met to discuss priorities."]
    files, total = {}, 0
    expected_files = {}
    for i in range(rng.randint(2, 4)):
        lines = [rng.choice(sentences).format(x=old) for _ in range(rng.randint(4, 8))]
        text = "\n".join(lines) + "\n"
        files[f"docs/file_{i + 1}.txt"] = text
        expected_files[f"docs/file_{i + 1}.txt"] = text.replace(old, new)
        total += text.count(old)
    body = pick(rng, [f"Replace every occurrence of '{old}' with '{new}' in all the .txt files in docs/ (edit them in place) and tell me how many replacements you made in total.",
                      f"In docs/*.txt, change '{old}' to '{new}' everywhere. Update the files directly and report the total number of replacements."])
    code = (f"import glob\ntotal = 0\nfor path in sorted(glob.glob('docs/*.txt')):\n    text = open(path).read()\n    n = text.count({old!r})\n    open(path, 'w').write(text.replace({old!r}, {new!r}))\n"
            "    print(path, n)\n    total += n\nprint('total replacements:', total)\n")
    steps = [Step(pick(rng, ["Let me see the files in docs/ first.", "I'll list the docs folder."]), "run_bash", "ls docs"), Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"Replaced '{old}' with '{new}' {total} times across {len(files)} files."

    def check(ws, reply):
        if any((ws / p).read_text() != t for p, t in expected_files.items()):
            return False, "files wrong"
        return has_number(reply, total, 0.01), "count wrong"

    return Task("text_replace_files", "text", "easy", frame(rng, body), files, steps, final, check, [f"total replacements: {total}"], ["python", "in_place_edit", "multi_file"])


# ===================================================================================================
@family("log_slowest_requests", "text", 1.3)
def log_slowest_requests(rng):
    paths = ["/api/users", "/api/orders", "/login", "/search", "/api/reports", "/health", "/export", "/api/invoices", "/settings"]
    n = rng.randint(20, 45)
    lat = rng.sample(range(20, 2000), n)
    lines = [f"{rng.choice(['GET', 'GET', 'POST'])} {rng.choice(paths)} {rng.choice([200, 200, 200, 404, 500])} {lat[i]}ms" for i in range(n)]
    k = rng.randint(3, 5)
    top = sorted(zip(lat, lines), reverse=True)[:k]
    avg = round(sum(lat) / n, 1)
    body = pick(rng, [f"In access.log, which {k} requests were slowest? Give the path and latency for each, slowest first, plus the average latency over all requests (1 decimal).",
                      f"Find the {k} slowest requests in access.log (path and latency in ms, descending) and the mean latency across the whole log, rounded to one decimal."])
    code = ("import re\nrows = []\nfor line in open('access.log'):\n    m = re.match(r'(\\w+) (\\S+) (\\d+) (\\d+)ms', line)\n    rows.append((int(m.group(4)), m.group(2)))\n"
            f"rows_sorted = sorted(rows, reverse=True)\nfor ms, path in rows_sorted[:{k}]:\n    print(f'{{path}} {{ms}}ms')\nprint('average:', round(sum(ms for ms, _ in rows) / len(rows), 1))\n")
    steps = inspect_steps(rng, ["access.log"], 5) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"The {k} slowest: " + "; ".join(f"{l.split()[1]} {ms}ms" for ms, l in top) + f". Average latency: {avg} ms."

    def check(ws, reply):
        pos = [reply.find(f"{ms}ms") for ms, _ in top]
        return (min(pos) >= 0 and pos == sorted(pos) and has_number(reply, avg, 0.06)), "answer wrong"

    return Task("log_slowest_requests", "text", "medium", frame(rng, body), add_distractors(rng, {"access.log": "\n".join(lines) + "\n"}), steps, final, check,
                [f"{top[0][0]}ms", f"average: {avg}"], ["python", "reply_answer", "log_parsing"])


# ===================================================================================================
@family("json_merge_dedupe", "files", 1.3)
def json_merge_dedupe(rng):
    names = people(rng, rng.randint(6, 10))
    base = date(2025, rng.randint(1, 6), 1)
    a = [{"id": i + 1, "name": names[i], "updated": (base + timedelta(days=rng.randint(0, 60))).isoformat()} for i in range(len(names))]
    b = []
    for i in rng.sample(range(len(names)), rng.randint(3, 5)):
        b.append({"id": i + 1, "name": names[i] + (" (new)" if rng.random() < 0.7 else ""), "updated": (base + timedelta(days=rng.randint(0, 60))).isoformat()})
    b.append({"id": 50, "name": "Extra Person", "updated": base.isoformat()})
    merged = {r["id"]: r for r in a}
    for r in b:
        if r["id"] not in merged or r["updated"] >= merged[r["id"]]["updated"]:
            merged[r["id"]] = r
    expected = [merged[k] for k in sorted(merged)]
    body = pick(rng, ["Merge people_a.json and people_b.json into merged.json. Records are keyed by id; when an id is in both files keep the one with the later 'updated' date (on a tie prefer people_b). Sort by id.",
                      "Combine the two JSON lists people_a.json and people_b.json into merged.json: one record per id, newest 'updated' wins (tie: people_b), sorted by id."])
    code = ("import json\na = json.load(open('people_a.json'))\nb = json.load(open('people_b.json'))\nmerged = {r['id']: r for r in a}\nfor r in b:\n"
            "    if r['id'] not in merged or r['updated'] >= merged[r['id']]['updated']:\n        merged[r['id']] = r\nresult = [merged[k] for k in sorted(merged)]\n"
            "json.dump(result, open('merged.json', 'w'), indent=2)\nprint(len(a), '+', len(b), '->', len(result), 'records')\n")
    steps = inspect_steps(rng, ["people_a.json", "people_b.json"], 8) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"merged.json now has {len(expected)} records ({len(a)} + {len(b)} inputs, duplicates resolved by the newest update)."

    def check(ws, reply):
        try:
            return json.loads(read_text(ws, "merged.json")) == expected, "merge wrong"
        except Exception:
            return False, "bad json"

    return Task("json_merge_dedupe", "files", "medium", frame(rng, body), add_distractors(rng, {"people_a.json": to_json(a), "people_b.json": to_json(b)}), steps, final, check,
                [f"{len(a)} + {len(b)} -> {len(expected)} records"], ["python", "file_output", "json", "multi_file"])


# ===================================================================================================
@family("csv_overdue_days", "data", 1.4)
def csv_overdue_days(rng):
    today = rand_date(rng, 2025)
    tasks = ["Send quote", "Renew licence", "File expenses", "Update roadmap", "Review contract", "Book venue", "Pay supplier", "Prepare slides", "Audit access", "Archive tickets", "Order laptops", "Draft memo"]
    rows = [[t, (today + timedelta(days=rng.randint(-45, 20))).isoformat()] for t in rng.sample(tasks, rng.randint(7, 11))]
    rows[0][1], rows[1][1] = (today - timedelta(days=40)).isoformat(), (today - timedelta(days=9)).isoformat()
    over = sorted([(t, (today - date.fromisoformat(d)).days) for t, d in rows if d < today.isoformat()], key=lambda x: (-x[1], x[0]))
    expected = ["task,days_overdue"] + [f"{t},{d}" for t, d in over]
    fname, out = rng.choice(["tasks.csv", "todo.csv"]), rng.choice(["overdue.csv", "late_tasks.csv"])
    body = pick(rng, [f"Today is {today.isoformat()}. In {fname}, find the tasks whose due_date is before today and write {out} with columns task,days_overdue (days between the due date and today), most overdue first, ties by task name. Which task is the most overdue?",
                      f"Assuming today is {today.isoformat()}, list the overdue tasks from {fname} in {out} (task,days_overdue; biggest first; ties alphabetical) and name the most overdue one."])
    code = (f"import csv\nfrom datetime import date\ntoday = date.fromisoformat({today.isoformat()!r})\nrows = []\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n"
            "        due = date.fromisoformat(r['due_date'])\n        if due < today:\n            rows.append((r['task'], (today - due).days))\nrows.sort(key=lambda x: (-x[1], x[0]))\n"
            f"with open({out!r}, 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['task', 'days_overdue'])\n    w.writerows(rows)\nprint(len(rows), 'overdue')\nprint(rows[0])\n")
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"{len(over)} tasks are overdue; the most overdue is '{over[0][0]}' ({over[0][1]} days). Details are in {out}."

    def check(ws, reply):
        txt = read_text(ws, out)
        return (txt is not None and norm_lines(txt) == expected and over[0][0] in reply), "overdue list wrong"

    return Task("csv_overdue_days", "data", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv(["task", "due_date"], rows)}), steps, final, check,
                [over[0][0], f"{len(over)} overdue"], ["python", "file_output", "date_logic"])


# ===================================================================================================
@family("ini_to_json", "files", 1.1)
def ini_to_json(rng):
    secs = {"server": {"host": rng.choice(["0.0.0.0", "localhost"]), "port": str(rng.choice([3000, 8080, 9000])), "debug": rng.choice(["true", "false"])},
            "database": {"url": f"postgres://db{rng.randint(1, 5)}.internal/app", "pool_size": str(rng.randint(5, 30))},
            "logging": {"level": rng.choice(["info", "debug", "warning"]), "file": "app.log"}}
    keep = rng.sample(list(secs), rng.randint(2, 3))
    ini = "".join(f"[{s}]\n" + "".join(f"{k} = {v}\n" for k, v in secs[s].items()) + "\n" for s in keep)
    expected = {s: secs[s] for s in keep}
    out = rng.choice(["config.json", "settings.json"])
    body = pick(rng, [f"Convert config.ini to {out}: one object per section with its keys, keeping every value as a string.", f"Turn config.ini into JSON ({out}), grouped by section; all values should stay strings."])
    code = (f"import configparser, json\ncp = configparser.ConfigParser(interpolation=None)\ncp.read('config.ini')\nresult = {{s: dict(cp[s]) for s in cp.sections()}}\n"
            f"json.dump(result, open({out!r}, 'w'), indent=2)\nprint(json.dumps(result, indent=2))\n")
    steps = [Step(pick(rng, ["Let me read the ini file first.", "I'll look at the config before converting it."]), "run_bash", "cat config.ini"), Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"Converted config.ini to {out} with sections: " + ", ".join(keep) + "."

    def check(ws, reply):
        try:
            return json.loads(read_text(ws, out)) == expected, "json wrong"
        except Exception:
            return False, "bad json"

    return Task("ini_to_json", "files", "easy", frame(rng, body), add_distractors(rng, {"config.ini": ini}), steps, final, check, [], ["python", "file_output", "format_conversion"])


# ===================================================================================================
@family("markdown_toc", "text", 1.1)
def markdown_toc(rng):
    heads = rng.sample(["Overview", "Getting Started", "Installation", "Usage: Basics", "Advanced Topics", "FAQ", "Troubleshooting", "Release Notes", "Contributing", "License", "Roadmap (2025)", "Design Goals"], rng.randint(6, 9))
    levels = [1] + [rng.choice([2, 2, 3]) for _ in heads[1:]]
    lines = []
    for h, lv in zip(heads, levels):
        lines += ["#" * lv + " " + h, "", "Some text about this section.", ""]
    anchor = lambda t: re.sub(r"\s", "-", re.sub(r"[^a-z0-9\s-]", "", t.lower()))
    expected = [("  " if lv == 3 else "") + f"- [{h}](#{anchor(h)})" for h, lv in zip(heads, levels) if lv > 1]
    body = pick(rng, ["Generate a table of contents for guide.md and save it to toc.md: skip the level-1 title, list '##' headings as '- [Title](#anchor)' and '###' headings indented by two spaces. Anchors are lowercase, with punctuation removed and spaces turned into hyphens.",
                      "Create toc.md from the headings in guide.md. Level-1 headings are skipped; ## entries are '- [Title](#anchor)', ### entries get two leading spaces. Anchor = lowercase title, punctuation stripped, spaces -> hyphens."])
    code = ("import re\nout = []\nfor line in open('guide.md'):\n    m = re.match(r'(#{2,3}) (.+)', line.rstrip())\n    if not m:\n        continue\n    title = m.group(2)\n"
            "    anchor = re.sub(r'\\s', '-', re.sub(r'[^a-z0-9\\s-]', '', title.lower()))\n    out.append(('  ' if len(m.group(1)) == 3 else '') + f'- [{title}](#{anchor})')\n"
            "open('toc.md', 'w').write('\\n'.join(out) + '\\n')\nprint('\\n'.join(out))\n")
    steps = [Step(pick(rng, ["Let me look at the headings in the guide.", "First I'll list the headings."]), "run_bash", "grep -n '^#' guide.md"), Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"toc.md is written with {len(expected)} entries."

    def check(ws, reply):
        txt = read_text(ws, "toc.md")
        return (txt is not None and [l.rstrip() for l in txt.strip("\n").split("\n")] == [e.rstrip() for e in expected]), "toc wrong"

    return Task("markdown_toc", "text", "medium", frame(rng, body), add_distractors(rng, {"guide.md": "\n".join(lines)}), steps, final, check, [], ["python", "bash", "file_output", "regex"])


# ===================================================================================================
@family("bash_file_counts", "files", 1.2)
def bash_file_counts(rng):
    exts = rng.sample(["txt", "csv", "md", "json", "log"], rng.randint(3, 4))
    files, counts = {}, Counter()
    for d in rng.sample(["alpha", "beta", "gamma", "delta"], 3):
        for _ in range(rng.randint(2, 5)):
            e = rng.choice(exts)
            nm = f"{d}/{rng.choice(['report', 'data', 'notes', 'export'])}_{rng.randint(1, 99)}.{e}"
            if nm not in files:
                files[nm] = "x\n"
                counts[e] += 1
    body = pick(rng, ["How many files of each extension are there in this folder tree (recursively)? List them as 'ext: count'.", "Count the files by extension across all sub-folders and report each as 'ext: count'.",
                      "Give me a per-extension file count for everything under the current folder."])
    cmd = "find . -type f | awk -F. '{c[$NF]++} END {for (e in c) print e \": \" c[e]}' | sort"
    steps = [Step(pick(rng, ["Let me look at the folder structure first.", "I'll list the tree."]), "run_bash", "find . -type f | sort"), Step(pick(rng, PLAN_DO_BASH), "run_bash", cmd)]
    final = "File counts by extension: " + ", ".join(f"{e}: {c}" for e, c in sorted(counts.items())) + "."

    def check(ws, reply):
        return all(f"{e}: {c}" in reply for e, c in counts.items()), "counts wrong"

    return Task("bash_file_counts", "files", "easy", frame(rng, body), files, steps, final, check, [f"{e}: {c}" for e, c in sorted(counts.items())][:2], ["bash", "reply_answer", "filesystem"])


# ===================================================================================================
@family("csv_pivot_month_category", "data", 1.3)
def csv_pivot_month_category(rng):
    year = rng.choice([2024, 2025])
    months = sorted(rng.sample(range(1, 13), rng.randint(3, 5)))
    cats = sorted(rng.sample(["signup", "purchase", "refund", "login", "support"], rng.randint(2, 4)))
    cnt = Counter()
    rows = []
    for i in range(rng.randint(30, 70)):
        m, c = rng.choice(months), rng.choice(cats)
        rows.append([rand_date(rng, year, m).isoformat(), c])
        cnt[(f"{year}-{m:02d}", c)] += 1
    mkeys = sorted({k[0] for k in cnt})
    expected = ["month," + ",".join(cats)] + [f"{m}," + ",".join(str(cnt[(m, c)]) for c in cats) for m in mkeys]
    fname = rng.choice(["events.csv", "activity_log.csv"])
    body = pick(rng, [f"Build a month-by-category table from {fname}: write pivot.csv with header 'month,<categories alphabetically>' and one row per month (YYYY-MM, ascending) holding the number of events (0 if none).",
                      f"Pivot {fname} into pivot.csv: rows are months (YYYY-MM), columns are the categories in alphabetical order, cells are event counts."])
    code = (f"import csv\nfrom collections import Counter\ncnt = Counter()\ncats, months = set(), set()\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n"
            "        m = r['date'][:7]\n        cnt[(m, r['category'])] += 1\n        cats.add(r['category'])\n        months.add(m)\ncats = sorted(cats)\n"
            "with open('pivot.csv', 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['month'] + cats)\n    for m in sorted(months):\n        w.writerow([m] + [cnt[(m, c)] for c in cats])\nprint(open('pivot.csv').read())\n")
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"pivot.csv is written: {len(mkeys)} months x {len(cats)} categories."

    def check(ws, reply):
        txt = read_text(ws, "pivot.csv")
        return (txt is not None and norm_lines(txt) == expected), "pivot wrong"

    return Task("csv_pivot_month_category", "data", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv(["date", "category"], rows)}), steps, final, check, [],
                ["python", "file_output", "pivot"])


# ===================================================================================================
# multi-turn classic tasks: the workspace persists between turns
@family("csv_sum_followup", "multi_turn", 2.0)
def csv_sum_followup(rng):
    r1, r2 = rng.sample(REGIONS, 2)
    month, year = rng.randint(1, 12), rng.choice([2024, 2025])
    rows, tot = [], defaultdict(float)
    for i in range(rng.randint(30, 60)):
        r = r1 if i < 3 else (r2 if i < 6 else rng.choice(REGIONS))
        d = rand_date(rng, year, month if (i < 6 or rng.random() < 0.4) else rng.randint(1, 12))
        amt = round(rng.uniform(20, 1500), 2)
        rows.append([d.isoformat(), r, f"{amt:.2f}"])
        if d.month == month:
            tot[r] += amt
    fname = rng.choice(["sales.csv", "orders.csv"])
    code = lambda reg: (f"import csv\ntotal = 0.0\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n        if r['region'] == {reg!r} and r['date'].startswith('{year}-{month:02d}'):\n"
                        "            total += float(r['amount'])\nprint(f'total: {total:.2f}')\n")
    t1, t2 = f"{tot[r1]:.2f}", f"{tot[r2]:.2f}"
    ml = f"{year}-{month:02d}"
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code(r1))]
    fu = Followup(pick(rng, [f"And what about the {r2} region for the same month?", f"Now the same for {r2}.", f"Thanks. Can you do {r2} as well?"]),
                  [Step(f"Same file and month, just a different region: {r2}.", "run_python", code(r2))],
                  f"For {r2} in {ml} the total is {t2}.", [t2])
    final = f"The total for {r1} in {ml} is {t1}."

    def check(ws, reply):
        return has_number(reply, tot[r2]), "second answer wrong"

    return Task("csv_sum_followup", "multi_turn", "medium", frame(rng, f"What is the total amount for the {r1} region in {ml} (rows of {fname} whose date starts with {ml})?"),
                add_distractors(rng, {fname: to_csv(["date", "region", "amount"], rows)}), steps, final, check, [t1], ["python", "multi_turn", "context_carryover", "reply_answer"], [fu])


@family("csv_group_followup", "multi_turn", 2.0)
def csv_group_followup(rng):
    cats = rng.sample(CATEGORIES, rng.randint(3, 5))
    rows = [[rng.choice(cats), f"{rng.uniform(10, 700):.2f}"] for _ in range(rng.randint(20, 45))]
    for i, c in enumerate(cats):
        rows[i][0] = c
    tot = defaultdict(float)
    for c, a in rows:
        tot[c] += float(a)
    grand = sum(tot.values())
    ordered = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
    exp1 = ["category,total"] + [f"{c},{v:.2f}" for c, v in ordered]
    exp2 = ["category,total,share"] + [f"{c},{v:.2f},{v / grand * 100:.2f}" for c, v in ordered]
    fname = rng.choice(["spend.csv", "costs.csv"])
    c1 = (f"import csv\nfrom collections import defaultdict\ntotals = defaultdict(float)\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n        totals[r['category']] += float(r['amount'])\n"
          "ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))\nwith open('summary.csv', 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['category', 'total'])\n"
          "    for k, v in ordered:\n        w.writerow([k, f'{v:.2f}'])\nprint(open('summary.csv').read())\n")
    c2 = ("import csv\nrows = list(csv.DictReader(open('summary.csv', newline='')))\ngrand = sum(float(r['total']) for r in rows)\n"
          "with open('summary.csv', 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['category', 'total', 'share'])\n"
          "    for r in rows:\n        w.writerow([r['category'], r['total'], f\"{float(r['total']) / grand * 100:.2f}\"])\nprint(open('summary.csv').read())\n")
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", c1)]
    fu = Followup(pick(rng, ["Please also add a 'share' column to summary.csv: each category's percentage of the grand total, 2 decimals.", "Can you extend summary.csv with a third column called share (percent of the total, two decimals)?"]),
                  [Step("I'll reread summary.csv, compute each share from the totals already there, and rewrite the file with the new column.", "run_python", c2)],
                  f"Added the share column to summary.csv; {ordered[0][0]} is the largest at {ordered[0][1] / grand * 100:.2f}%.", [f"{ordered[0][1] / grand * 100:.2f}"])

    def check(ws, reply):
        txt = read_text(ws, "summary.csv")
        return (txt is not None and norm_lines(txt) == exp2), "final summary.csv wrong"

    return Task("csv_group_followup", "multi_turn", "medium",
                frame(rng, f"Total the amount per category in {fname} and save it to summary.csv as category,total (2 decimals), largest first."),
                add_distractors(rng, {fname: to_csv(["category", "amount"], rows)}), steps, f"Saved summary.csv with {len(ordered)} categories; {ordered[0][0]} is the largest.", check,
                [ordered[0][0]], ["python", "multi_turn", "file_output", "state_carryover"], [fu])


@family("notes_edit_followup", "multi_turn", 1.5)
def notes_edit_followup(rng):
    items = rng.sample(["buy coffee", "send invoice", "book room", "call supplier", "review slides", "update roadmap", "file expenses", "order laptops"], 4)
    a, b, c, d = items
    body = pick(rng, [f"Create notes.txt with three lines: '{a}', '{b}' and '{c}' (one per line).", f"Write a file notes.txt that contains these items, one per line: {a}; {b}; {c}."])
    steps = [Step(pick(rng, ["I'll create the file with the three items.", "Let me write notes.txt."]), "run_python",
                  f"open('notes.txt', 'w').write({a!r} + '\\n' + {b!r} + '\\n' + {c!r} + '\\n')\nprint(open('notes.txt').read())\n")]
    fu = Followup(pick(rng, [f"Now put '{d}' at the very top and remove the second item.", f"Update it: add '{d}' as the first line and delete the second line of the original list."]),
                  [Step("I'll read the file, drop the second original item, and put the new one first.", "run_python",
                        f"lines = open('notes.txt').read().splitlines()\nlines = [{d!r}] + [l for i, l in enumerate(lines) if i != 1]\nopen('notes.txt', 'w').write('\\n'.join(lines) + '\\n')\nprint(open('notes.txt').read())\n")],
                  f"notes.txt now reads: {d}, {a}, {c}.", [])
    expected = [d, a, c]

    def check(ws, reply):
        txt = read_text(ws, "notes.txt")
        return (txt is not None and norm_lines(txt) == expected), "final notes wrong"

    return Task("notes_edit_followup", "multi_turn", "easy", frame(rng, body), {}, steps, f"Created notes.txt with {a}, {b} and {c}.", check, [], ["python", "multi_turn", "file_output", "state_carryover"], [fu])
