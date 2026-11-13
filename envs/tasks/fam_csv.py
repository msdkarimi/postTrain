"""Spreadsheet-style (CSV) task families: filtered sums, group-bys, top-N, cleaning, joins, statistics, counting."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict

from envs.tasks.base import (PLAN_DO, PLAN_DO_BASH, Step, Task, family, frame, has_number, inspect_steps,
                             last_error_line, norm_lines, pick, read_text, verify_step, with_recovery)
from envs.tasks.common import (CATEGORIES, CITIES, MONTHS, PRODUCTS, REGIONS, add_distractors, delim_name, people,
                               rand_date, to_csv, to_json)


# ===================================================================================================
@family("csv_filtered_sum", "data", 2.0)
def csv_filtered_sum(rng):
    dc, rc, pc, ac = rng.choice([("date", "region", "product", "amount"), ("Date", "Region", "Product", "Amount"),
                                 ("order_date", "region", "item", "total"), ("day", "area", "product", "revenue")])
    delim = rng.choice([",", ",", ",", ";", "\t"])
    iso = rng.random() < 0.7
    messy = rng.random() < 0.35
    fname = rng.choice(["sales.csv", "sales_export.csv", "transactions.csv", "orders_2024.csv", "revenue_log.csv"])
    region, month, year = rng.choice(REGIONS), rng.randint(1, 12), rng.choice([2023, 2024, 2025])
    rows, total, cnt = [], 0.0, 0
    for i in range(rng.randint(24, 55)):
        forced = i < 4
        m = month if (forced or rng.random() < 0.25) else rng.randint(1, 12)
        d = rand_date(rng, year, m)
        r = region if (forced or rng.random() < 0.3) else rng.choice(REGIONS)
        amt = round(rng.uniform(15, 2500), 2)
        if r == region and d.month == month:
            total += amt
            cnt += 1
        rows.append([d.isoformat() if iso else d.strftime("%d/%m/%Y"), r, rng.choice(PRODUCTS),
                     f"${amt:,.2f}" if messy else f"{amt:.2f}"])
    total_s = f"{total:.2f}"
    total = float(total_s)
    mode = rng.choice(["reply", "file"])
    path = rng.choice(["total.txt", "out/total.txt", "result.txt"])
    ml = f"{MONTHS[month - 1]} {year}"
    body = pick(rng, [f"What was the total {ac} for the {region} region in {ml}? The data is in {fname}.",
                      f"Please add up all {region} sales for {ml} from {fname}.",
                      f"From {fname}, give me the sum of the {ac} column for region {region} during {ml}.",
                      f"How much did the {region} region bring in during {ml}? Numbers are in {fname}."])
    if mode == "file":
        body += f" Save just the number (two decimals) to {path}."
    cond = (f"row[{dc!r}].startswith('{year}-{month:02d}')" if iso else f"row[{dc!r}].endswith('/{month:02d}/{year}')")
    amt_expr = f"row[{ac!r}].replace('$', '').replace(',', '')" if messy else f"row[{ac!r}]"

    def code(fn=fname, delimiter=repr(delim), region_key=rc, convert=True):
        amt = amt_expr if convert else f"row[{ac!r}]"
        write = (f"import os\nos.makedirs(os.path.dirname({path!r}) or '.', exist_ok=True)\n"
                 f"with open({path!r}, 'w') as out:\n    out.write(f'{{total:.2f}}\\n')\n") if mode == "file" else ""
        return (f"import csv\ntotal = 0.0\ncount = 0\nwith open({fn!r}, newline='') as f:\n"
                f"    for row in csv.DictReader(f, delimiter={delimiter}):\n"
                f"        if row[{region_key!r}] == {region!r} and {cond}:\n"
                f"            count += 1\n            total += float({amt})\n"
                f"print('rows matched:', count)\nprint('total:', f'{{total:.2f}}')\n" + write)

    use_awk = mode == "reply" and not messy and rng.random() < 0.25
    if use_awk:
        fs = "\\t" if delim == "\t" else delim
        acond = f'substr($1,1,7)=="{year}-{month:02d}"' if iso else f'substr($1,4,7)=="{month:02d}/{year}"'
        good = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                    f"awk -F'{fs}' 'NR>1 && $2==\"{region}\" && {acond} {{s+=$4; n++}} "
                    f"END {{printf \"rows matched: %d\\ntotal: %.2f\\n\", n, s}}' {fname}")
        steps, recovered = [good], False
    else:
        good = Step(pick(rng, PLAN_DO), "run_python", code())
        typo = fname.rsplit(".", 1)[0][:-1] + ".csv"
        bugs = [(code(fn=typo), lambda obs: f"`{last_error_line(obs[-1])}` - I mistyped the file name. The listing shows it is {fname}.")]
        if delim != ",":
            bugs.append((code(delimiter="','"), lambda obs: f"That failed with `{last_error_line(obs[-1])}`. The header I printed shows the "
                                                          f"fields are separated by {delim_name(delim)}, so the reader needs that delimiter."))
        if messy:
            bugs.append((code(convert=False), lambda obs: f"`{last_error_line(obs[-1])}` - the amounts contain a currency symbol and "
                                                         f"thousands separators, so I must strip them before converting."))
        if rc != rc.lower():
            bugs.append((code(region_key=rc.lower()), lambda obs: f"`{last_error_line(obs[-1])}` - column names are case-sensitive and "
                                                                 f"the header spells it '{rc}'."))
        steps, recovered = with_recovery(rng, good, bugs)
    steps = inspect_steps(rng, [fname]) + steps
    if mode == "file":
        steps += verify_step(rng, f"cat {path}", 0.7)
        final = pick(rng, [f"Done - the {region} total for {ml} is {total_s} ({cnt} matching rows), saved to {path}.",
                           f"I saved {total_s} to {path}. That is the {region} total for {ml}, from {cnt} rows."])
    else:
        final = pick(rng, [f"The total {ac} for the {region} region in {ml} is {total_s} (across {cnt} matching rows).",
                           f"{region} region, {ml}: {total_s} in total, based on {cnt} rows."])

    def check(ws, reply):
        if not has_number(reply, total):
            return False, "reply lacks the total"
        if mode == "file":
            txt = read_text(ws, path)
            if txt is None:
                return False, "output file missing"
            try:
                return abs(float(txt.strip()) - total) <= 0.011, "file value"
            except ValueError:
                return False, "file not a number"
        return True, "ok"

    files = add_distractors(rng, {fname: to_csv([dc, rc, pc, ac], rows, delim)})
    tags = ["python" if not use_awk else "bash", "file_output" if mode == "file" else "reply_answer", "schema_variation"]
    if messy:
        tags.append("messy_values")
    if recovered:
        tags.append("error_recovery")
    return Task("csv_filtered_sum", "data", "medium" if messy else "easy", frame(rng, body), files, steps, final, check,
                [total_s, str(cnt)], tags)


# ===================================================================================================
@family("csv_group_totals", "data", 1.5)
def csv_group_totals(rng):
    idc, gc, ac = rng.choice([("invoice_id", "category", "amount"), ("id", "category", "amount"), ("invoice", "dept", "total")])
    cats = rng.sample(CATEGORIES, rng.randint(3, 6))
    messy = rng.random() < 0.3
    fname = rng.choice(["invoices.csv", "expenses.csv", "ledger.csv", "billing_2024.csv"])
    out = rng.choice(["summary.csv", "totals_by_category.csv", "out/summary.csv"])
    totals, rows = defaultdict(float), []
    for i in range(rng.randint(30, 70)):
        c = cats[i % len(cats)] if i < len(cats) else rng.choice(cats)
        amt = round(rng.uniform(20, 900), 2)
        totals[c] += amt
        rows.append([f"INV-{1000 + i}", c, f"${amt:,.2f}" if messy else f"{amt:.2f}"])
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    expected = [f"{gc},total"] + [f"{k},{v:.2f}" for k, v in ordered]
    top_k, top_v = ordered[0][0], f"{ordered[0][1]:.2f}"
    body = pick(rng, [f"Group {fname} by {gc} and write the total {ac} per {gc} to {out}, highest total first. Use the columns {gc} and total.",
                      f"I need {ac} totals per {gc} from {fname}. Write them to {out} (columns: {gc},total), sorted from biggest to smallest.",
                      f"Summarise {fname}: one row per {gc} with the summed {ac}, largest first, saved as {out} with header {gc},total."])
    conv = f"row[{ac!r}].replace('$', '').replace(',', '')" if messy else f"row[{ac!r}]"

    def code(convert=True):
        v = conv if convert else f"row[{ac!r}]"
        return ("import csv, os\nfrom collections import defaultdict\n"
                f"totals = defaultdict(float)\nwith open({fname!r}, newline='') as f:\n    for row in csv.DictReader(f):\n"
                f"        totals[row[{gc!r}]] += float({v})\n"
                "ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))\n"
                f"os.makedirs(os.path.dirname({out!r}) or '.', exist_ok=True)\n"
                f"with open({out!r}, 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n"
                f"    w.writerow([{gc!r}, 'total'])\n    for k, v in ordered:\n        w.writerow([k, f'{{v:.2f}}'])\n"
                f"print(open({out!r}).read())\n")

    good = Step(pick(rng, PLAN_DO), "run_python", code())
    bugs = [(code(convert=False), lambda obs: f"`{last_error_line(obs[-1])}` - the amounts include '$' and thousands separators; I need to strip those first.")] if messy else []
    steps, recovered = with_recovery(rng, good, bugs)
    steps = inspect_steps(rng, [fname]) + steps
    final = pick(rng, [f"Done. {out} has {len(ordered)} rows; {top_k} is the largest at {top_v}.",
                       f"I wrote {out} with one row per {gc}. The top one is {top_k} ({top_v})."])

    def check(ws, reply):
        txt = read_text(ws, out)
        if txt is None:
            return False, "output file missing"
        return norm_lines(txt) == expected, "content mismatch"

    tags = ["python", "file_output", "schema_variation"] + (["messy_values"] if messy else []) + (["error_recovery"] if recovered else [])
    return Task("csv_group_totals", "data", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv([idc, gc, ac], rows)}),
                steps, final, check, [top_k, top_v], tags)


# ===================================================================================================
@family("csv_top_n", "data", 1.5)
def csv_top_n(rng):
    ncol, dcol, mcol = rng.choice([("name", "department", "salary"), ("employee", "team", "salary"), ("name", "department", "score")])
    n_rows, n = rng.randint(12, 30), rng.randint(3, 5)
    names = people(rng, n_rows)
    vals = rng.sample(range(30_000, 150_000, 500), n_rows)
    fname = rng.choice(["employees.csv", "staff.csv", "team_roster.csv", "people.csv"])
    rows = [[names[i], rng.choice(["Sales", "Support", "Engineering", "Finance", "Design"]), vals[i]] for i in range(n_rows)]
    top = sorted(rows, key=lambda r: -r[2])[:n]
    body = pick(rng, [f"Who are the top {n} people by {mcol} in {fname}? List them from highest to lowest with their {mcol}.",
                      f"From {fname}, show me the {n} highest {mcol} values and who they belong to.",
                      f"Find the {n} people with the biggest {mcol} in {fname}, in descending order."])
    use_bash = rng.random() < 0.4
    if use_bash:
        mi = 3
        good = Step(pick(rng, PLAN_DO_BASH), "run_bash", f"tail -n +2 {fname} | sort -t, -k{mi} -nr | head -n {n}")
        steps, recovered = [good], False
    else:
        code = (f"import csv\nwith open({fname!r}, newline='') as f:\n    rows = list(csv.DictReader(f))\n"
                f"rows.sort(key=lambda r: int(r[{mcol!r}]), reverse=True)\n"
                f"for r in rows[:{n}]:\n    print(r[{ncol!r}], r[{mcol!r}], sep=': ')\n")
        bad = code.replace(f"r[{mcol!r}]", f"r[{mcol.capitalize()!r}]")
        good = Step(pick(rng, PLAN_DO), "run_python", code)
        steps, recovered = with_recovery(rng, good, [(bad, lambda obs: f"`{last_error_line(obs[-1])}` - I capitalised the column name by mistake; it is '{mcol}'.")])
    steps = inspect_steps(rng, [fname]) + steps
    lines = [f"{i + 1}. {r[0]} ({r[2]})" for i, r in enumerate(top)]
    final = f"The top {n} by {mcol}:\n" + "\n".join(lines)

    def check(ws, reply):
        pos = [reply.find(r[0]) for r in top]
        if min(pos) < 0 or pos != sorted(pos):
            return False, "names missing or out of order"
        return all(str(r[2]) in reply for r in top), "values missing"

    tags = ["bash" if use_bash else "python", "reply_answer"] + (["error_recovery"] if recovered else [])
    return Task("csv_top_n", "data", "easy", frame(rng, body), add_distractors(rng, {fname: to_csv([ncol, dcol, mcol], rows)}),
                steps, final, check, [r[0] for r in top] + [str(r[2]) for r in top], tags)


# ===================================================================================================
@family("csv_clean_dedupe", "data", 1.5)
def csv_clean_dedupe(rng):
    n = rng.randint(10, 22)
    names, cities = people(rng, n), [rng.choice(CITIES) for _ in range(n)]
    base = [[names[i], f"{names[i].lower().replace(' ', '.')}@{rng.choice(['acme.com', 'globex.net', 'hooli.dev'])}", cities[i]] for i in range(n)]
    raw, clean, seen = [], [], set()
    for row in base:
        variants = [row] + [[f"  {row[0]} ", row[1].upper(), row[2]] for _ in range(rng.choice([0, 0, 1]))]
        for v in variants:
            raw.append(v)
    rng.shuffle(raw)
    dup_total = 0
    for r in raw:
        nm, em, ct = (x.strip() for x in r)
        em = em.lower()
        if em in seen:
            dup_total += 1
            continue
        seen.add(em)
        clean.append([nm, em, ct])
    if dup_total == 0:                       # guarantee at least one duplicate
        r = list(raw[0])
        raw.append([f" {r[0]}", r[1].upper(), r[2]])
        dup_total = 1
    fname, out = rng.choice(["contacts_raw.csv", "customers_export.csv", "leads.csv"]), rng.choice(["contacts_clean.csv", "clean.csv"])
    body = pick(rng, [f"Clean {fname}: trim whitespace in every field, lowercase the email addresses, drop duplicate emails (keep the first one) and save the result to {out} with the same header.",
                      f"{fname} is messy. Strip spaces, lowercase emails, remove repeated emails (first occurrence wins), and write the cleaned file to {out}.",
                      f"Please deduplicate {fname} by email (case-insensitively, keep the first row), trim all fields, and save to {out}."])
    code = ("import csv\nrows_in = []\nwith open(%r, newline='') as f:\n    reader = csv.reader(f)\n    header = next(reader)\n    rows_in = list(reader)\n"
            "seen, rows_out = set(), []\nfor row in rows_in:\n    name, email, city = (c.strip() for c in row)\n    email = email.lower()\n"
            "    if email in seen:\n        continue\n    seen.add(email)\n    rows_out.append([name, email, city])\n"
            "with open(%r, 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(header)\n    w.writerows(rows_out)\n"
            "print(len(rows_in), 'rows in ->', len(rows_out), 'rows out')\n") % (fname, out)
    steps = inspect_steps(rng, [fname], 8) + [Step(pick(rng, PLAN_DO), "run_python", code)] + verify_step(rng, f"head -n 4 {out}", 0.6)
    final = f"Done. {len(raw)} rows became {len(clean)} after removing {dup_total} duplicate email(s); the cleaned data is in {out}."
    expected = ["name,email,city"] + [",".join(r) for r in clean]

    def check(ws, reply):
        txt = read_text(ws, out)
        return (txt is not None and norm_lines(txt) == expected), "cleaned file mismatch"

    return Task("csv_clean_dedupe", "data", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv(["name", "email", "city"], raw)}),
                steps, final, check, [f"{len(raw)} rows in -> {len(clean)} rows out"], ["python", "file_output", "data_cleaning"])


# ===================================================================================================
@family("csv_join_city_totals", "data", 1.5)
def csv_join_city_totals(rng):
    n_c = rng.randint(8, 14)
    names, cids = people(rng, n_c), [f"C{100 + i}" for i in range(n_c)]
    city_of = {cids[i]: rng.choice(CITIES[:6]) for i in range(n_c)}
    customers = [[cids[i], names[i], city_of[cids[i]]] for i in range(n_c)]
    orders, totals = [], defaultdict(float)
    for i in range(rng.randint(30, 60)):
        orphan = i in (3, 11)
        cid = "C999" if orphan else rng.choice(cids)
        amt = round(rng.uniform(10, 800), 2)
        orders.append([f"O{5000 + i}", cid, f"{amt:.2f}"])
        if not orphan:
            totals[city_of[cid]] += amt
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    expected = ["city,total"] + [f"{k},{v:.2f}" for k, v in ordered]
    top = ordered[0][0]
    out = rng.choice(["city_totals.csv", "revenue_by_city.csv"])
    body = pick(rng, [f"Using orders.csv and customers.csv, work out the total order amount per city. Ignore orders from customers that aren't in customers.csv. Write {out} (columns city,total, biggest first) and tell me which city is on top.",
                      f"Join orders.csv with customers.csv on customer_id and sum the amounts by city. Skip orders with unknown customers. Save to {out} as city,total sorted descending, then tell me the best city.",
                      f"I want revenue per city. Match each order in orders.csv to its customer in customers.csv (skip orders whose customer is unknown), add up the amounts per city and store them in {out} (header city,total; biggest total first). Which city leads?",
                      f"Please build {out} from orders.csv and customers.csv: total order value for every city (columns city,total, sorted high to low). Orders from customers missing in customers.csv should be ignored. Then name the top city."])
    code = lambda guard: ("import csv\ncity_of = {}\nwith open('customers.csv', newline='') as f:\n    for r in csv.DictReader(f):\n        city_of[r['customer_id']] = r['city']\n"
                          "totals = {}\nwith open('orders.csv', newline='') as f:\n    for r in csv.DictReader(f):\n"
                          + ("        if r['customer_id'] not in city_of:\n            continue\n" if guard else "")
                          + "        city = city_of[r['customer_id']]\n        totals[city] = totals.get(city, 0.0) + float(r['amount'])\n"
                          "ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))\n"
                          f"with open({out!r}, 'w', newline='') as o:\n    w = csv.writer(o, lineterminator='\\n')\n    w.writerow(['city', 'total'])\n"
                          "    for k, v in ordered:\n        w.writerow([k, f'{v:.2f}'])\nprint(open(%r).read())\n" % out)
    good = Step(pick(rng, PLAN_DO), "run_python", code(True))
    steps, recovered = with_recovery(rng, good, [(code(False), lambda obs: f"`{last_error_line(obs[-1])}` - some orders reference a customer that isn't in customers.csv. I need to skip those.")], p=0.4)
    steps = inspect_steps(rng, ["orders.csv", "customers.csv"]) + steps
    final = f"Done - {out} is written. {top} has the highest total ({ordered[0][1]:.2f})."

    def check(ws, reply):
        txt = read_text(ws, out)
        if txt is None or norm_lines(txt) != expected:
            return False, "file mismatch"
        return top in reply, "reply must name the top city"

    files = add_distractors(rng, {"orders.csv": to_csv(["order_id", "customer_id", "amount"], orders),
                                  "customers.csv": to_csv(["customer_id", "name", "city"], customers)})
    return Task("csv_join_city_totals", "data", "hard", frame(rng, body), files, steps, final, check,
                [top, f"{ordered[0][1]:.2f}"], ["python", "file_output", "multi_file_join"] + (["error_recovery"] if recovered else []))


# ===================================================================================================
@family("csv_stats", "data", 1.0)
def csv_stats(rng):
    col = rng.choice(["reading", "temperature", "latency_ms", "weight_kg"])
    n = rng.randint(20, 50)
    vals, rows = [], []
    for i in range(n):
        if i in (4, 9, 17) or rng.random() < 0.06:
            rows.append([f"S{i % 5 + 1}", rng.choice(["", "N/A"])])
        else:
            v = round(rng.uniform(5, 120), 2)
            vals.append(v)
            rows.append([f"S{i % 5 + 1}", f"{v:.2f}"])
    mean, med, lo, hi = round(statistics.mean(vals), 2), round(statistics.median(vals), 2), min(vals), max(vals)
    fname = rng.choice(["measurements.csv", "sensor_data.csv", "readings.csv"])
    body = pick(rng, [f"Compute the mean, median, minimum and maximum of the {col} column in {fname}, rounded to 2 decimals. Some cells are blank or N/A - ignore those.",
                      f"What are the average, median, min and max {col} in {fname}? Skip missing values and round to two decimals.",
                      f"Give me summary stats (mean, median, min, max) for {col} from {fname}. Ignore empty or N/A entries."])
    code = lambda guard: (f"import csv, statistics\nvals = []\nwith open({fname!r}, newline='') as f:\n    for r in csv.DictReader(f):\n"
                          + ("        if r[%r].strip() in ('', 'N/A'):\n            continue\n" % col if guard else "")
                          + f"        vals.append(float(r[{col!r}]))\nprint('count:', len(vals))\nprint('mean:', round(statistics.mean(vals), 2))\n"
                          "print('median:', round(statistics.median(vals), 2))\nprint('min:', min(vals))\nprint('max:', max(vals))\n")
    good = Step(pick(rng, PLAN_DO), "run_python", code(True))
    steps, recovered = with_recovery(rng, good, [(code(False), lambda obs: f"`{last_error_line(obs[-1])}` - there are blank / N/A cells. I have to skip those before converting to float.")], p=0.4)
    steps = inspect_steps(rng, [fname], 8) + steps
    final = f"Statistics for {col} ({len(vals)} valid values): mean {mean}, median {med}, min {lo}, max {hi}."

    def check(ws, reply):
        return all(has_number(reply, x) for x in (mean, med, lo, hi)), "missing statistic"

    return Task("csv_stats", "data", "easy", frame(rng, body), add_distractors(rng, {fname: to_csv(["sensor_id", col], rows)}), steps, final, check,
                [str(len(vals)), str(mean), str(med)], ["python", "reply_answer", "messy_values"] + (["error_recovery"] if recovered else []))


# ===================================================================================================
@family("csv_count_by_month", "data", 1.0)
def csv_count_by_month(rng):
    year = rng.choice([2023, 2024, 2025])
    months = sorted(rng.sample(range(1, 13), rng.randint(3, 7)))
    with_time = rng.random() < 0.5
    rows, counts = [], Counter()
    for i in range(rng.randint(25, 60)):
        m = months[i % len(months)] if i < len(months) else rng.choice(months)
        d = rand_date(rng, year, m)
        ts = f"{d.isoformat()} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}" if with_time else d.isoformat()
        rows.append([f"E{i + 1}", ts, rng.choice(["signup", "purchase", "refund", "login"])])
        counts[ts[:7]] += 1
    rng.shuffle(rows)
    fname, out = rng.choice(["events.csv", "activity.csv", "log_events.csv"]), rng.choice(["counts.json", "monthly_counts.json"])
    busiest = max(sorted(counts), key=lambda k: counts[k])
    body = pick(rng, [f"Count how many rows in {fname} fall in each month (use the created_at column) and write a JSON object like {{\"YYYY-MM\": count}} to {out}. Also tell me the busiest month.",
                      f"I need monthly event counts from {fname}. Save them to {out} as a JSON object with YYYY-MM keys, and mention which month had the most events."])
    code = ("import csv, json\nfrom collections import Counter\ncounts = Counter()\nwith open(%r, newline='') as f:\n    for r in csv.DictReader(f):\n"
            "        counts[r['created_at'][:7]] += 1\nresult = dict(sorted(counts.items()))\n"
            "with open(%r, 'w') as o:\n    json.dump(result, o, indent=2)\nprint(result)\nprint('busiest:', max(result, key=result.get))\n") % (fname, out)
    steps = inspect_steps(rng, [fname]) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"Saved the monthly counts to {out}. The busiest month was {busiest} with {counts[busiest]} events."

    def check(ws, reply):
        txt = read_text(ws, out)
        try:
            return (json.loads(txt) == dict(counts) and busiest in reply), "json/reply mismatch"
        except Exception:
            return False, "bad json"

    return Task("csv_count_by_month", "data", "easy",
                frame(rng, body), add_distractors(rng, {fname: to_csv(["id", "created_at", "type"], rows)}), steps, final, check,
                [busiest, str(counts[busiest])], ["python", "file_output"])
