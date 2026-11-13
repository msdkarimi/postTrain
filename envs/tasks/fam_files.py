"""File-system, JSON and abstention task families."""
from __future__ import annotations

import copy
import json
import os

from envs.tasks.base import (PLAN_DO, PLAN_DO_BASH, Step, Task, family, frame, has_number, inspect_steps, last_error_line,
                             norm_lines, pick, read_text, verify_step, with_recovery)
from envs.tasks.common import DISTRACTORS, PRODUCTS, REGIONS, add_distractors, people, rand_date, to_csv, to_json

WORDS = ["budget", "plan", "notes", "summary", "draft", "report", "minutes", "agenda", "forecast", "review", "backlog", "roadmap",
         "invoice", "survey", "memo", "outline"]


def _tree(root) -> set:
    return {os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs}


# ===================================================================================================
@family("organize_by_ext", "files", 1.5)
def organize_by_ext(rng):
    exts = rng.sample(["txt", "csv", "md", "json", "log"], 3)
    a, b, c = exts
    files, names = {}, set()
    for ext in exts:
        for _ in range(rng.randint(2, 4)):
            nm = f"{rng.choice(WORDS)}_{rng.randint(1, 30)}.{ext}"
            if nm not in names:
                names.add(nm)
                files[nm] = f"content of {nm}\n" if ext != "csv" else "a,b\n1,2\n"
    other_pool = [e for e in ["txt", "csv", "md", "json", "log", "pdf", "png"] if e not in exts]
    for _ in range(rng.randint(1, 2)):
        nm = f"{rng.choice(WORDS)}_{rng.randint(31, 60)}.{rng.choice(other_pool)}"
        files[nm] = f"other {nm}\n"
    files = add_distractors(rng, files)
    moves = {n: n.rsplit(".", 1)[1] for n in files if "." in n and not n.startswith(".") and n.rsplit(".", 1)[1] in exts}
    body = pick(rng, [f"Tidy up the workspace: move every .{a} file into a folder called {a}/, every .{b} into {b}/ and every .{c} into {c}/ (create the folders). Leave all other files where they are.",
                      f"Please sort the loose files in the current folder by extension: .{a} files into {a}/, .{b} files into {b}/, .{c} files into {c}/. Don't touch anything else.",
                      f"The workspace is cluttered. Group the .{a}, .{b} and .{c} files into sub-folders named after their extension ({a}, {b}, {c}); everything else stays put.",
                      f"Can you file the .{a}, .{b} and .{c} documents in this folder into {a}/, {b}/ and {c}/ respectively? Create the folders if needed and leave the rest alone."])
    if rng.random() < 0.7:
        good, tag = Step(pick(rng, PLAN_DO_BASH), "run_bash", f"mkdir -p {a} {b} {c} && mv *.{a} {a}/ && mv *.{b} {b}/ && mv *.{c} {c}/ && find . -type f | sort"), "bash"
    else:
        good = Step(pick(rng, PLAN_DO), "run_python",
                    "import os, shutil\nfor name in sorted(os.listdir('.')):\n    if os.path.isfile(name) and not name.startswith('.'):\n        ext = name.rsplit('.', 1)[-1]\n"
                    f"        if ext in {tuple(exts)!r}:\n            os.makedirs(ext, exist_ok=True)\n            shutil.move(name, os.path.join(ext, name))\n"
                    "for root, _, fs in sorted(os.walk('.')):\n    for f in sorted(fs):\n        print(os.path.join(root, f))\n")
        tag = "python"
    steps = [Step(pick(rng, ["Let me see what's in the folder before moving anything.", "First I'll list the files."]), "run_bash", "ls")] + [good]
    final = f"Done - the .{a}, .{b} and .{c} files are now in {a}/, {b}/ and {c}/; the other files were left alone."

    def check(ws, reply):
        tree = _tree(ws)
        for name, ext in moves.items():
            if name in tree or f"{ext}/{name}" not in tree:
                return False, f"{name} not moved correctly"
            if (ws / ext / name).read_text() != files[name]:
                return False, "content changed"
        for name in files:
            if name not in moves and name not in tree:
                return False, f"{name} should have stayed"
        return True, "ok"

    return Task("organize_by_ext", "files", "easy", frame(rng, body), files, steps, final, check, [], [tag, "filesystem", "distractor_files"])


# ===================================================================================================
@family("merge_csv_parts", "files", 1.3)
def merge_csv_parts(rng):
    k = rng.randint(3, 5)
    header = rng.choice([["id", "name", "score"], ["sku", "product", "qty"], ["day", "city", "visits"]])
    files, rows_all, n = {}, [], 0
    for i in range(1, k + 1):
        rows = []
        for _ in range(rng.randint(3, 8)):
            n += 1
            rows.append([f"{n:03d}", rng.choice(PRODUCTS), rng.randint(1, 99)])
        files[f"data/part{i}.csv"] = to_csv(header, rows)
        rows_all += rows
    expected = [",".join(header)] + [",".join(map(str, r)) for r in rows_all]
    out = rng.choice(["all.csv", "merged.csv"])
    body = pick(rng, [f"Merge the part files in data/ (part1.csv ... part{k}.csv, same header) into {out} in the current folder: one header line, then all rows in file order. Tell me how many data rows there are.",
                      f"Combine data/part*.csv into a single {out} with just one header and the rows in order of the file names. How many rows did you end up with?"])
    if rng.random() < 0.5:
        good = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                    f"head -n 1 data/part1.csv > {out} && for f in data/part*.csv; do tail -n +2 \"$f\" >> {out}; done && awk 'END {{print NR-1 \" data rows\"}}' {out}")
        tag = "bash"
    else:
        good = Step(pick(rng, PLAN_DO), "run_python",
                    f"import glob\nheader, rows = None, []\nfor path in sorted(glob.glob('data/part*.csv')):\n    lines = open(path).read().splitlines()\n"
                    f"    header = header or lines[0]\n    rows += lines[1:]\nopen({out!r}, 'w').write('\\n'.join([header] + rows) + '\\n')\nprint(len(rows), 'data rows')\n")
        tag = "python"
    steps = [Step(pick(rng, ["Let me look at the data folder and one of the files.", "First I'll see which part files exist and their format."]), "run_bash",
                  "ls data; head -n 3 data/part1.csv")] + [good] + verify_step(rng, f"head -n 3 {out}", 0.5)
    final = f"{out} now has one header and {len(rows_all)} data rows from {k} files."

    def check(ws, reply):
        txt = read_text(ws, out)
        if txt is None or norm_lines(txt) != expected:
            return False, "merged content mismatch"
        return has_number(reply, len(rows_all), 0.01), "count missing"

    return Task("merge_csv_parts", "files", "medium", frame(rng, body), add_distractors(rng, files), steps, final, check, [f"{len(rows_all)} data rows"],
                [tag, "file_output", "multi_file"])


# ===================================================================================================
@family("rename_extension", "files", 1.0)
def rename_extension(rng):
    old, new, folder, stems, other_ext = rng.choice([
        ("jpeg", "jpg", rng.choice(["photos", "images", "pics"]), ["holiday", "team", "office", "product", "event"], "png"),
        ("htm", "html", rng.choice(["pages", "site", "web"]), ["index", "about", "contact", "pricing", "faq"], "css"),
        ("yml", "yaml", rng.choice(["configs", "settings", "deploy"]), ["app", "db", "cache", "queue", "auth"], "json"),
        ("jpeg", "jpg", rng.choice(["photos", "images", "pics"]), ["logo", "banner", "avatar", "scan", "receipt"], "gif")])
    olds = sorted({f"{rng.choice(stems)}_{rng.randint(1, 40)}.{old}" for _ in range(rng.randint(3, 6))})
    others = sorted({f"{rng.choice(['chart', 'icon', 'misc'])}_{rng.randint(1, 9)}.{other_ext}" for _ in range(rng.randint(1, 3))})
    files = {f"{folder}/{n}": f"data {n}\n" for n in olds + others}
    files[f"{folder}/notes.txt"] = "notes go here\n"
    cut = len(old) + 1
    expected = sorted([n[:-cut] + f".{new}" for n in olds] + others + ["notes.txt"])
    body = pick(rng, [f"Rename every .{old} file in {folder}/ so it ends in .{new} instead (keep the rest of each name). Don't touch the other files.",
                      f"The {folder}/ folder has files ending in .{old}; change those to .{new}. Leave the other files alone, then show me the folder.",
                      f"Please standardise the extensions in {folder}/: all .{old} files should become .{new}. Nothing else should change.",
                      f"In {folder}/, switch the extension of each .{old} file to .{new}, keeping the base names, and list the folder afterwards."])
    if rng.random() < 0.65:
        good, tag = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                         f"for f in {folder}/*.{old}; do mv \"$f\" \"${{f%.{old}}}.{new}\"; done; ls {folder}"), "bash"
    else:
        good = Step(pick(rng, PLAN_DO), "run_python",
                    f"import os\nfor name in sorted(os.listdir({folder!r})):\n    if name.endswith('.{old}'):\n"
                    f"        os.rename(os.path.join({folder!r}, name), os.path.join({folder!r}, name[:-{cut}] + '.{new}'))\n"
                    f"print('\\n'.join(sorted(os.listdir({folder!r}))))\n")
        tag = "python"
    steps = [Step(pick(rng, [f"Let me check which files are in {folder}/ first.", "I'll list the folder to see the current names."]), "run_bash", f"ls {folder}")] + [good]
    final = f"Renamed {len(olds)} .{old} file(s) to .{new}; {folder}/ now contains: " + ", ".join(expected) + "."

    def check(ws, reply):
        got = sorted(os.listdir(ws / folder))
        if got != expected:
            return False, "folder listing mismatch"
        return all((ws / folder / (n[:-cut] + f".{new}")).read_text() == f"data {n}\n" for n in olds), "content changed"

    return Task("rename_extension", "files", "easy", frame(rng, body), add_distractors(rng, files), steps, final, check, [expected[0]], [tag, "filesystem"])


# ===================================================================================================
@family("largest_file", "files", 1.0)
def largest_file(rng):
    names = rng.sample(["app", "db", "auth", "cache", "worker", "audit", "mail", "proxy"], rng.randint(4, 7))
    ks = rng.sample(range(5, 400), len(names))
    files = {f"logs/{n}.log": "log entry\n" * k for n, k in zip(names, ks)}
    sizes = {f"{n}.log": 10 * k for n, k in zip(names, ks)}
    top = max(sizes, key=sizes.get)
    body = pick(rng, ["Which file in logs/ is the largest, and how many bytes is it?", "Find the biggest file in the logs/ folder and tell me its name and size in bytes.",
                      "I'm running low on disk. What's the largest log in logs/ and how big is it (in bytes)?",
                      "Which log under logs/ takes the most space? Give me its name and exact size in bytes.",
                      "Look through logs/ and report the biggest file together with its size in bytes."])
    code = ("import os\nsizes = {n: os.path.getsize(os.path.join('logs', n)) for n in os.listdir('logs')}\n"
            "for name, size in sorted(sizes.items(), key=lambda kv: -kv[1])[:3]:\n    print(f'{name}: {size} bytes')\n")
    steps = [Step(pick(rng, ["Let me list the logs folder first.", "I'll start by listing logs/."]), "run_bash", "ls logs"), Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"The largest file is {top} at {sizes[top]} bytes."

    def check(ws, reply):
        return (top in reply and has_number(reply, sizes[top], 0.01)), "answer mismatch"

    return Task("largest_file", "files", "easy", frame(rng, body), add_distractors(rng, files), steps, final, check, [f"{top}: {sizes[top]} bytes"], ["python", "reply_answer"])


# ===================================================================================================
@family("json_filter_transform", "files", 1.5)
def json_filter_transform(rng):
    n = rng.randint(8, 16)
    nm = people(rng, n)
    users = []
    for i in range(n):
        first, last = nm[i].split(" ", 1)
        users.append({"id": 100 + i, "first": first, "last": last, "email": f"{first.lower()}.{last.lower()}@example.com", "age": rng.randint(18, 70), "active": rng.random() < 0.6})
    min_age = rng.choice([25, 30, 35, 40])
    sel = sorted([u for u in users if u["active"] and u["age"] >= min_age], key=lambda u: f"{u['first']} {u['last']}")
    if not sel:
        users[0].update(active=True, age=min_age + 5)
        sel = [users[0]]
    expected = [{"user_id": u["id"], "full_name": f"{u['first']} {u['last']}", "email": u["email"]} for u in sel]
    out = rng.choice(["out.json", "selected_users.json"])
    body = pick(rng, [f"From users.json keep only active users aged {min_age} or older. Write {out} as a JSON list of objects with keys user_id (the id), full_name (first and last name joined by a space) and email, sorted by full_name.",
                      f"Filter users.json to active users with age >= {min_age} and save {out}: a list sorted by full_name where each item has user_id, full_name and email."])
    code = lambda key: ("import json\nusers = json.load(open('users.json'))\nselected = [u for u in users if u['active'] and u['age'] >= %d]\n"
                        "result = [{'user_id': u['id'], 'full_name': u[%r] + ' ' + u['last'], 'email': u['email']} for u in selected]\n"
                        "result.sort(key=lambda r: r['full_name'])\nwith open(%r, 'w') as f:\n    json.dump(result, f, indent=2)\nprint(len(result), 'users written')\nprint(json.dumps(result[:2], indent=2))\n") % (min_age, key, out)
    good = Step(pick(rng, PLAN_DO), "run_python", code("first"))
    steps, recovered = with_recovery(rng, good, [(code("name"), lambda obs: f"`{last_error_line(obs[-1])}` - the records have 'first' and 'last' fields, not 'name'.")], p=0.3)
    steps = inspect_steps(rng, ["users.json"], 14) + steps
    final = f"Saved {len(expected)} users to {out}, sorted by full_name."

    def check(ws, reply):
        txt = read_text(ws, out)
        try:
            return json.loads(txt) == expected, "json mismatch"
        except Exception:
            return False, "bad json"

    return Task("json_filter_transform", "files", "medium", frame(rng, body), add_distractors(rng, {"users.json": to_json(users)}), steps, final, check,
                [f"{len(expected)} users"], ["python", "file_output", "json"] + (["error_recovery"] if recovered else []))


# ===================================================================================================
@family("config_update", "files", 1.2)
def config_update(rng):
    cfg = {"server": {"host": "0.0.0.0", "port": rng.choice([3000, 5000, 8000]), "debug": rng.choice([True, False])},
           "logging": {"level": rng.choice(["info", "warning"]), "file": "app.log"},
           "features": {"dark_mode": True, "export": rng.choice([True, False])}, "version": f"1.{rng.randint(0, 9)}.{rng.randint(0, 9)}"}
    ops, text, code_lines = [], [], []
    expected = copy.deepcopy(cfg)
    pool = ["port", "level", "feature", "drop_debug", "bump"]
    for op in rng.sample(pool, 3):
        if op == "port":
            p = rng.choice([8080, 9000, 4000])
            expected["server"]["port"] = p
            text.append(f"set server.port to {p}")
            code_lines.append(f"cfg['server']['port'] = {p}")
        elif op == "level":
            lv = rng.choice(["debug", "error"])
            expected["logging"]["level"] = lv
            text.append(f"set logging.level to \"{lv}\"")
            code_lines.append(f"cfg['logging']['level'] = {lv!r}")
        elif op == "feature":
            nm = rng.choice(["beta_search", "new_checkout", "audit_trail"])
            expected["features"][nm] = True
            text.append(f"add features.{nm} = true")
            code_lines.append(f"cfg['features'][{nm!r}] = True")
        elif op == "drop_debug":
            expected["server"].pop("debug")
            text.append("remove server.debug")
            code_lines.append("cfg['server'].pop('debug', None)")
        else:
            major, minor, patch = expected["version"].split(".")
            expected["version"] = f"{major}.{minor}.{int(patch) + 1}"
            text.append("bump the patch number of version by one")
            code_lines.append("major, minor, patch = cfg['version'].split('.')\ncfg['version'] = f'{major}.{minor}.{int(patch) + 1}'")
    body = f"Edit config.json in place: {', '.join(text[:-1])} and {text[-1]}. Keep everything else unchanged."
    code = "import json\ncfg = json.load(open('config.json'))\n" + "\n".join(code_lines) + "\njson.dump(cfg, open('config.json', 'w'), indent=2)\nprint(json.dumps(cfg, indent=2))\n"
    steps = [Step(pick(rng, ["Let me look at the current configuration first.", "I'll read config.json before changing it."]), "run_bash", "cat config.json"),
             Step(pick(rng, PLAN_DO), "run_python", code)]
    final = "config.json is updated: " + "; ".join(text) + "."

    def check(ws, reply):
        txt = read_text(ws, "config.json")
        try:
            return json.loads(txt) == expected, "config mismatch"
        except Exception:
            return False, "bad json"

    return Task("config_update", "files", "easy", frame(rng, body), add_distractors(rng, {"config.json": to_json(cfg)}), steps, final, check, [], ["python", "json", "in_place_edit"])


# ===================================================================================================
@family("missing_information", "abstain", 2.0)
def missing_information(rng):
    """The requested data does not exist. The right behaviour is to look, then say so (never to invent an answer)."""
    variant = rng.choice(["column", "column", "file"])
    if variant == "column":
        hdr = ["date", "region", "product", "amount"]
        rows = [[rand_date(rng, 2024).isoformat(), rng.choice(REGIONS), rng.choice(PRODUCTS), f"{rng.uniform(20, 900):.2f}"] for _ in range(rng.randint(10, 25))]
        files = {"sales.csv": to_csv(hdr, rows)}
        missing = rng.choice(["discount", "tax", "customer_email", "margin"])
        body = pick(rng, [f"What is the total {missing} in sales.csv?", f"Please sum the {missing} column of sales.csv and tell me the result.",
                          f"Give me the average {missing} per sale from sales.csv."])
        steps = [Step(pick(rng, ["I'll look at the file's header to see which columns exist.", "Let me check the columns in sales.csv first."]), "run_bash", "ls; head -n 3 sales.csv"),
                 Step(lambda obs: f"The header only has {', '.join(hdr)} - there is no '{missing}' column. Let me search the file for the word to be sure.", "run_bash",
                      f"grep -ic {missing} sales.csv || true")]
        final = f"I can't answer that: sales.csv has no '{missing}' column. Its columns are {', '.join(hdr)}. If the data is in another file, tell me which one."
        grounded = [",".join(hdr)]                  # the header line as it appears in the tool output
    else:
        files = {"notes.md": "# Notes\n- review the pipeline\n", "budget_2024.csv": "item,cost\nrent,1200\nsoftware,300\n"}
        missing = rng.choice(["forecast_q4.xlsx", "headcount_plan.csv", "sales_2023.csv", "contracts.pdf"])
        stem = missing.split(".")[0].split("_")[0]
        body = pick(rng, [f"Summarise {missing} for me.", f"Open {missing} and tell me what the largest value in it is.", f"What's in {missing}? Give me a quick overview."])
        steps = [Step(pick(rng, ["Let me see which files are in the workspace.", "I'll list the files first."]), "run_bash", "ls"),
                 Step(lambda obs: f"I don't see {missing} in that listing. Let me search for anything with a similar name.", "run_bash",
                      f"ls | grep -i '{stem}' || echo 'no file matching {stem}'")]
        listing = ", ".join(sorted(files))
        final = f"I couldn't find {missing} in the workspace, so I can't summarise it. The files here are: {listing}. Should I look at one of those instead?"
        grounded = [listing.split(", ")[0]]
    initial = set(files)

    def check(ws, reply):
        low = reply.lower()
        if missing.lower() not in low:
            return False, "must name what is missing"
        if not any(w in low for w in ("can't", "cannot", "couldn't", "no '", "not find", "doesn't", "does not", "isn't", "don't")):
            return False, "must say it is unavailable"
        if _tree(ws) != initial:
            return False, "must not create files"
        return True, "ok"

    return Task("missing_information", "abstain", "easy", frame(rng, body), files, steps, final, check, grounded,
                ["bash", "abstention", variant])
