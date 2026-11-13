"""Text / log / document task families."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from envs.tasks.base import (PLAN_DO, PLAN_DO_BASH, Step, Task, family, frame, has_number, inspect_steps,
                             last_error_line, norm_lines, pick, read_text, verify_step, with_recovery)
from envs.tasks.common import DOMAINS, PROJECTS, add_distractors, people, rand_date, to_json

FILLER = ["We agreed to revisit the budget on Friday.", "The rollout is scheduled for the second week of next month.",
          "Everyone should review the draft before the next sync.", "The vendor asked for a revised timeline.",
          "Customer feedback has been mostly positive so far.", "Let's keep the meeting notes in the shared folder.",
          "Support tickets dropped slightly compared with last quarter.", "The team will demo the prototype on Thursday."]


# ===================================================================================================
@family("extract_emails", "text", 1.5)
def extract_emails(rng):
    n = rng.randint(4, 9)
    names = people(rng, n)
    emails = [f"{nm.lower().replace(' ', '.')}@{rng.choice(DOMAINS)}" for nm in names]
    shown = []
    paras = []
    for i, nm in enumerate(names):
        em = emails[i]
        e_show = em.upper() if rng.random() < 0.2 else (em.capitalize() if rng.random() < 0.2 else em)
        shown.append(e_show)
        tpl = rng.choice([f"Please CC {e_show} on the follow-up.", f"{nm} (reachable at {e_show}) will lead the workstream.",
                          f"Send the contract to {e_show}, or call +39 011 555 {rng.randint(1000, 9999)}.",
                          f"For access requests write to {e_show}."])
        paras.append(tpl + " " + rng.choice(FILLER))
    for _ in range(rng.randint(1, 3)):                       # repeat some addresses
        paras.append(f"Reminder: {rng.choice(shown)} still needs to confirm. " + rng.choice(FILLER))
    paras.append("Obfuscated ones like jane at example dot com should not count.")
    rng.shuffle(paras)
    expected = sorted({e.lower() for e in shown})
    out = rng.choice(["emails.txt", "contacts.txt", "unique_emails.txt"])
    body = pick(rng, [f"Pull every email address out of notes.txt, lowercase them, remove duplicates, sort alphabetically and save one per line to {out}. Tell me how many unique addresses there are.",
                      f"List the unique email addresses mentioned in notes.txt (lowercase, sorted, one per line) in {out} and tell me the count.",
                      f"Extract all email addresses from notes.txt into {out}: lowercase, deduplicated, alphabetical. How many are there?"])
    pat = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    if rng.random() < 0.4:
        good = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                    f"grep -Eo '{pat}' notes.txt | tr 'A-Z' 'a-z' | sort -u > {out}; cat {out}; awk 'END {{print NR \" unique addresses\"}}' {out}")
        steps, tag = [good], "bash"
    else:
        code = (f"import re\ntext = open('notes.txt').read()\nfound = re.findall({pat!r}, text)\nunique = sorted({{e.lower() for e in found}})\n"
                f"open({out!r}, 'w').write('\\n'.join(unique) + '\\n')\nprint('\\n'.join(unique))\nprint(len(unique), 'unique addresses')\n")
        steps, tag = [Step(pick(rng, PLAN_DO), "run_python", code)], "python"
    steps = inspect_steps(rng, ["notes.txt"], 6) + steps
    final = f"I found {len(expected)} unique email addresses and saved them (lowercase, sorted) to {out}."

    def check(ws, reply):
        txt = read_text(ws, out)
        if txt is None or norm_lines(txt) != expected:
            return False, "file mismatch"
        return has_number(reply, len(expected), 0.01), "count missing"

    return Task("extract_emails", "text", "medium", frame(rng, body), add_distractors(rng, {"notes.txt": "\n\n".join(paras) + "\n"}),
                steps, final, check, [f"{len(expected)} unique addresses"], [tag, "file_output", "regex"])


# ===================================================================================================
@family("log_summary", "text", 1.5)
def log_summary(rng):
    comps = ["db", "auth", "api", "cache", "worker", "scheduler"]
    msgs = {"INFO": ["request completed", "user logged in", "job finished", "cache refreshed"],
            "WARN": ["slow response detected", "retrying connection", "disk usage high"],
            "ERROR": ["connection refused", "timeout while calling upstream", "failed to write record", "unhandled exception in handler"],
            "DEBUG": ["entering handler", "payload parsed", "cache miss"]}
    year, month, day = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 27)
    t = rng.randint(6, 9) * 3600
    lines, counts, errs = [], Counter(), []
    for i in range(rng.randint(40, 90)):
        t += rng.randint(5, 400)
        lvl = "ERROR" if i in (7, 19, 33) else rng.choices(["INFO", "WARN", "ERROR", "DEBUG"], [10, 3, 2, 4])[0]
        ts = f"{year}-{month:02d}-{day:02d} {t // 3600 % 24:02d}:{t // 60 % 60:02d}:{t % 60:02d}"
        lines.append(f"{ts} {lvl} {rng.choice(comps)}: {rng.choice(msgs[lvl])}")
        counts[lvl] += 1
        if lvl == "ERROR":
            errs.append(ts)
    mode = rng.choice(["reply", "reply", "json"])
    if mode == "reply":
        body = pick(rng, ["How many ERROR lines are in app.log, and what are the timestamps of the first and last one?",
                          "Count the ERROR entries in app.log and give me the first and last ERROR timestamps.",
                          "In app.log, how many errors are there and when did the first and the last one happen?",
                          "Tell me the number of ERROR-level entries in app.log, plus the time of the earliest and the latest one.",
                          "I need error stats from app.log: the count of ERROR lines and the first and last ERROR timestamps."])
        if rng.random() < 0.45:
            good = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                        "grep -c ' ERROR ' app.log; grep ' ERROR ' app.log | head -n 1 | cut -c1-19; grep ' ERROR ' app.log | tail -n 1 | cut -c1-19")
            tag = "bash"
        else:
            good = Step(pick(rng, PLAN_DO), "run_python",
                        "errors = []\nfor line in open('app.log'):\n    parts = line.split(' ', 3)\n    if parts[2] == 'ERROR':\n        errors.append(parts[0] + ' ' + parts[1])\n"
                        "print('errors:', len(errors))\nprint('first:', errors[0])\nprint('last:', errors[-1])\n")
            tag = "python"
        steps = inspect_steps(rng, ["app.log"], 6) + [good]
        final = f"There are {counts['ERROR']} ERROR lines. The first was at {errs[0]} and the last at {errs[-1]}."
        grounded = [str(counts["ERROR"]), errs[0], errs[-1]]

        def check(ws, reply):
            return (has_number(reply, counts["ERROR"], 0.01) and errs[0] in reply and errs[-1] in reply), "reply facts"
    else:
        out = rng.choice(["log_summary.json", "level_counts.json"])
        body = pick(rng, [f"Count the log lines per level (INFO, WARN, ERROR, DEBUG) in app.log and write the counts as a JSON object to {out}. Only include levels that occur.",
                          f"How is app.log distributed across log levels? Save a JSON object with the number of lines per level (only levels that appear) to {out}.",
                          f"Write {out} as a JSON object that maps each log level found in app.log to its line count."])
        code = ("import json\nfrom collections import Counter\ncounts = Counter()\nfor line in open('app.log'):\n    counts[line.split(' ', 3)[2]] += 1\n"
                f"with open({out!r}, 'w') as f:\n    json.dump(dict(sorted(counts.items())), f, indent=2)\nprint(dict(sorted(counts.items())))\n")
        steps = inspect_steps(rng, ["app.log"], 6) + [Step(pick(rng, PLAN_DO), "run_python", code)] + verify_step(rng, f"cat {out}", 0.5)
        final = f"Saved the per-level counts to {out}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "."
        grounded, tag = [], "python"

        def check(ws, reply):
            txt = read_text(ws, out)
            try:
                return json.loads(txt) == dict(counts), "json mismatch"
            except Exception:
                return False, "bad json"

    return Task("log_summary", "text", "easy", frame(rng, body), add_distractors(rng, {"app.log": "\n".join(lines) + "\n"}), steps, final, check,
                grounded, [tag, "log_parsing", "reply_answer" if mode == "reply" else "file_output"])


# ===================================================================================================
@family("grep_keyword_count", "text", 1.0)
def grep_keyword_count(rng):
    kw = rng.choice(["invoice", "deadline", "budget", "refund", "contract"])
    n_files = rng.randint(3, 5)
    pool = FILLER + ["Please see the attached document.", "Contact the office for details.", "This section is under review."]
    files, count = {}, 0
    for i in range(n_files):
        lines = []
        for _ in range(rng.randint(8, 16)):
            line = rng.choice(pool)
            if rng.random() < 0.25:
                form = rng.choice([kw, kw.upper(), kw.capitalize()])
                line = f"{line} The {form} came up again" + (f" - the {kw} is still open." if rng.random() < 0.3 else ".")
                count += 1
            lines.append(line)
        files[f"docs/report_{i + 1}.txt"] = "\n".join(lines) + "\n"
    files["docs/readme.md"] = f"Notes about the {kw} process.\nThe {kw} template lives elsewhere.\n"     # decoy: not a .txt file
    # ground truth is computed from the generated files themselves (filler sentences may already contain the keyword)
    count = sum(1 for path, text in files.items() if path.endswith(".txt") for line in text.splitlines() if kw in line.lower())
    body = pick(rng, [f"How many lines mention '{kw}' (case-insensitive) across the .txt files in docs/?",
                      f"Count the lines containing the word {kw}, ignoring case, in all .txt files under docs/. Don't count other file types.",
                      f"Across docs/*.txt, how many lines contain '{kw}' in any capitalisation?"])
    if rng.random() < 0.5:
        good, tag = Step(pick(rng, PLAN_DO_BASH), "run_bash", f"cat docs/*.txt | grep -ic '{kw}'"), "bash"
    else:
        good = Step(pick(rng, PLAN_DO), "run_python",
                    f"import glob\ncount = 0\nfor path in sorted(glob.glob('docs/*.txt')):\n    for line in open(path):\n        if {kw!r} in line.lower():\n            count += 1\nprint(count)\n")
        tag = "python"
    steps = [Step(pick(rng, ["Let me see how docs/ is organised.", "I'll list the folder first."]), "run_bash", "ls docs")] + [good]
    final = f"{count} lines in the .txt files under docs/ mention '{kw}' (ignoring case)."

    def check(ws, reply):
        return has_number(reply, count, 0.01), "wrong count"

    return Task("grep_keyword_count", "text", "easy", frame(rng, body), add_distractors(rng, files), steps, final, check, [str(count)],
                [tag, "reply_answer", "distractor_files"])


# ===================================================================================================
@family("todo_extract", "text", 1.2)
def todo_extract(rng):
    owners = rng.sample(["Alice", "Marco", "Priya", "Jonas", "Sofia", "Liam", "Aisha"], rng.randint(2, 4))
    tasks_pool = ["send the revised quote", "book the conference room", "update the onboarding checklist", "review the vendor contract",
                  "prepare the Q3 slides", "fix the broken export link", "schedule the customer call", "archive the old tickets",
                  "draft the release notes", "confirm the budget numbers", "collect feedback from support"]
    rng.shuffle(tasks_pool)
    files, items, k = {}, [], 0
    for i in range(rng.randint(2, 3)):
        lines = [f"# Meeting {i + 1}", "", "Agenda: planning and status updates.", ""]
        for _ in range(rng.randint(2, 4)):
            if k >= len(tasks_pool):
                break
            ow, txt = rng.choice(owners), tasks_pool[k]
            k += 1
            lines.append(f"- TODO({ow}): {txt}")
            items.append((f"docs/meeting_{i + 1}.md", ow, txt))
            if rng.random() < 0.5:
                lines.append(rng.choice(["- Discussed the roadmap.", "- Noted a delay in shipping.", "- TODO: someone should look at pricing"]))
        files[f"docs/meeting_{i + 1}.md"] = "\n".join(lines) + "\n"
    by_owner = defaultdict(list)
    for _, ow, txt in items:
        by_owner[ow].append(txt)
    expected = []
    for ow in sorted(by_owner):
        expected.append(f"## {ow}")
        expected += [f"- [ ] {t}" for t in by_owner[ow]]
    out = rng.choice(["todos.md", "action_items.md"])
    body = pick(rng, [f"Collect every line of the form 'TODO(owner): text' from the markdown files in docs/ and write {out} grouped by owner: a '## Owner' heading per person (alphabetical), then '- [ ] text' lines in the order they appear (files in name order). Ignore TODOs without an owner.",
                      f"Make a checklist {out} from the docs/*.md files: for each owner (sorted) a '## Owner' heading followed by '- [ ] text' items in file order. Only lines like TODO(name): text count.",
                      f"Build an action-item list in {out} from the meeting notes in docs/ (*.md). Every 'TODO(person): task' line becomes '- [ ] task' under a '## person' heading; people sorted alphabetically, tasks in the order they appear across the files (file name order). Skip TODOs that have no person."])
    code = ("import glob, re\nfrom collections import defaultdict\nby_owner = defaultdict(list)\n"
            "for path in sorted(glob.glob('docs/*.md')):\n    for line in open(path):\n"
            "        m = re.match(r'\\s*-?\\s*TODO\\((\\w+)\\):\\s*(.+)', line)\n        if m:\n            by_owner[m.group(1)].append(m.group(2).strip())\n"
            f"lines = []\nfor owner in sorted(by_owner):\n    lines.append('## ' + owner)\n    lines += ['- [ ] ' + t for t in by_owner[owner]]\n"
            f"open({out!r}, 'w').write('\\n'.join(lines) + '\\n')\nprint('\\n'.join(lines))\n")
    steps = [Step(pick(rng, ["Let me look at the docs folder and the notes first.", "First I'll see what the meeting notes look like."]), "run_bash",
                  "ls docs; for f in docs/*.md; do echo \"== $f\"; cat \"$f\"; done")] + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"Done - {out} lists {len(items)} action items for {len(by_owner)} owners."

    def check(ws, reply):
        txt = read_text(ws, out)
        return (txt is not None and norm_lines(txt) == expected), "checklist mismatch"

    return Task("todo_extract", "text", "medium", frame(rng, body), add_distractors(rng, files), steps, final, check, [],
                ["python", "file_output", "regex", "multi_file"])


# ===================================================================================================
@family("json_to_markdown", "text", 1.2)
def json_to_markdown(rng):
    n = rng.randint(4, 8)
    names = rng.sample(PROJECTS, n)
    projs = [{"name": names[i], "status": rng.choice(["active", "paused", "done"]), "budget": rng.randrange(10, 400) * 1000,
              "owner": rng.choice(["Alice", "Marco", "Priya"])} for i in range(n)]
    only = rng.choice([None, "active", "done", "paused"])
    if only and not any(p["status"] == only for p in projs):
        projs[0]["status"] = only
    title = rng.choice(["Project Report", "Budget Overview", "Portfolio Summary", "Status Report"])
    desc = rng.random() < 0.5                                   # highest first or lowest first
    fname = rng.choice(["projects.json", "portfolio.json", "initiatives.json"])
    sel = sorted([p for p in projs if only is None or p["status"] == only], key=lambda p: (-p["budget"] if desc else p["budget"], p["name"]))
    total = sum(p["budget"] for p in sel)
    expected = [f"# {title}", "| Project | Status | Budget |", "|---|---|---|"] + \
               [f"| {p['name']} | {p['status']} | {p['budget']:,} |" for p in sel] + [f"Total budget: {total:,}"]
    flt = f" Include only projects whose status is '{only}'." if only else ""
    order = "highest first" if desc else "lowest first"
    body = pick(rng, [f"Read {fname} and write report.md in exactly this layout: a title line '# {title}', a table with the header '| Project | Status | Budget |' and separator '|---|---|---|', one row per project sorted by budget ({order}), with the budget written with thousands separators, and a last line 'Total budget: N' (also with thousands separators).{flt}",
                      f"Turn {fname} into report.md. Layout: '# {title}' on the first line, then a markdown table (header '| Project | Status | Budget |', separator '|---|---|---|') with projects ordered by budget, {order}; budgets get thousands separators. End with 'Total budget: N' using thousands separators.{flt}"])
    code = (f"import json\nprojects = json.load(open({fname!r}))\n"
            + (f"projects = [p for p in projects if p['status'] == {only!r}]\n" if only else "")
            + f"projects.sort(key=lambda p: ({'-' if desc else ''}p['budget'], p['name']))\n"
            f"lines = ['# {title}', '| Project | Status | Budget |', '|---|---|---|']\n"
            "for p in projects:\n    lines.append(f\"| {p['name']} | {p['status']} | {p['budget']:,} |\")\n"
            "lines.append(f\"Total budget: {sum(p['budget'] for p in projects):,}\")\n"
            "open('report.md', 'w').write('\\n'.join(lines) + '\\n')\nprint('\\n'.join(lines))\n")
    steps = inspect_steps(rng, [fname], 12) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"report.md is written with {len(sel)} project rows; the total budget is {total:,}."

    def check(ws, reply):
        txt = read_text(ws, "report.md")
        return (txt is not None and norm_lines(txt) == expected), "report mismatch"

    return Task("json_to_markdown", "text", "medium", frame(rng, body), add_distractors(rng, {fname: to_json(projs)}), steps, final, check,
                [f"{total:,}"], ["python", "file_output", "formatting"])


