"""Calendar, scheduling, email and cross-application task families (calendar + contacts + documents)."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

from envs.tasks.base import (PLAN_DO, PLAN_DO_BASH, Step, Task, family, frame, has_number, inspect_steps,
                             last_error_line, norm_lines, pick, read_text, verify_step, with_recovery)
from envs.tasks.common import (DOMAINS, FIRST, PROJECTS, add_distractors, email_for, fmt_hhmm, people, rand_date,
                               to_csv, to_json)

LO, HI = 9 * 60, 17 * 60                     # working day 09:00 - 17:00 (minutes)
MEETINGS = ["Standup", "Design review", "1:1 sync", "Budget meeting", "Sprint planning", "Customer call", "Retro",
            "Lunch & learn", "Roadmap workshop", "Vendor demo", "Interview", "Team offsite prep", "Security briefing",
            "Hiring panel", "Quarterly review", "Product sync", "Training session", "Incident review"]


def _busy(rng, n):
    out = []
    for _ in range(n):
        s = rng.choice(range(8 * 60, 17 * 60, 15))
        e = min(s + rng.choice([30, 45, 60, 90, 120]), 18 * 60)
        out.append((s, e))
    return out


def earliest_free(busy, dur):
    for start in range(LO, HI - dur + 1):
        if all(not (start < e and s < start + dur) for s, e in busy):
            return start
    return None


def _calendar(rng, attendees, day: date):
    """calendar.json content + the flat busy list on `day`; retries until a (preferably non-trivial) slot exists."""
    dur = rng.choice([30, 45, 60])
    for attempt in range(60):
        cal, busy = {}, []
        for p in attendees:
            evs = []
            for s, e in _busy(rng, rng.randint(2, 5)):
                evs.append({"date": day.isoformat(), "start": fmt_hhmm(s), "end": fmt_hhmm(e)})
                busy.append((s, e))
            for s, e in _busy(rng, rng.randint(1, 3)):                       # other day: must be ignored
                evs.append({"date": (day + timedelta(days=1)).isoformat(), "start": fmt_hhmm(s), "end": fmt_hhmm(e)})
            rng.shuffle(evs)
            cal[p] = evs
        slot = earliest_free(busy, dur)
        if slot is not None and (slot > LO or attempt > 40):
            return cal, dur, slot
    raise RuntimeError("could not build a feasible calendar")


FREE_SLOT_CODE = (
    "import json\ndata = json.load(open('calendar.json'))\npeople = {people}\nday = {day!r}\nduration = {dur}\nbusy = []\n"
    "for person in people:\n    for ev in data[person]:\n        if ev[{dk!r}] == day:\n"
    "            s = int(ev['start'][:2]) * 60 + int(ev['start'][3:])\n            e = int(ev['end'][:2]) * 60 + int(ev['end'][3:])\n"
    "            busy.append((s, e))\nbusy.sort()\nt = 9 * 60\nfor s, e in busy:\n    if s - t >= duration:\n        break\n    t = max(t, e)\n"
    "if t + duration <= 17 * 60:\n    print(f'slot: {{t // 60:02d}}:{{t % 60:02d}}-{{(t + duration) // 60:02d}}:{{(t + duration) % 60:02d}}')\n"
    "else:\n    print('no free slot')\n")


# ===================================================================================================
@family("common_free_slot", "calendar", 2.0)
def common_free_slot(rng):
    day = rand_date(rng, rng.choice([2025, 2026]))
    names = rng.sample(FIRST, rng.randint(2, 4))
    cal, dur, slot = _calendar(rng, names, day)
    s_txt, e_txt = fmt_hhmm(slot), fmt_hhmm(slot + dur)
    who = ", ".join(names[:-1]) + " and " + names[-1]
    mode = rng.choice(["reply", "file"])
    out = rng.choice(["slot.txt", "meeting_time.txt"])
    body = pick(rng, [f"Using calendar.json, find the earliest {dur}-minute slot on {day.isoformat()} between 09:00 and 17:00 when {who} are all free.",
                      f"When can {who} meet for {dur} minutes on {day.isoformat()}? Use calendar.json and pick the earliest common free slot within working hours (09:00-17:00)."])
    if mode == "file":
        body += f" Write just the start time (HH:MM) to {out}."
    else:
        body += " Reply with the start and end time."
    code = lambda dk="date": FREE_SLOT_CODE.format(people=names, day=day.isoformat(), dur=dur, dk=dk) + \
        (f"if t + duration <= 17 * 60:\n    open({out!r}, 'w').write(f'{{t // 60:02d}}:{{t % 60:02d}}\\n')\n" if mode == "file" else "")
    good = Step(["I'll collect everyone's busy blocks for that day, then walk through the day to find the first gap that fits.",
                 "Plan: gather all busy intervals on that date, sort them, and look for the earliest gap of the right length.",
                 "I'll merge the busy times of all attendees for that date and scan for the first free gap."][rng.randrange(3)], "run_python", code())
    steps, recovered = with_recovery(rng, good, [(code("day"), lambda obs: f"`{last_error_line(obs[-1])}` - the events use the key 'date', not 'day'.")])
    steps = inspect_steps(rng, ["calendar.json"], 14) + steps + (verify_step(rng, f"cat {out}", 0.6) if mode == "file" else [])
    final = (f"The earliest slot on {day.isoformat()} when {who} are all free is {s_txt}-{e_txt}" + (f"; I saved {s_txt} to {out}." if mode == "file" else "."))

    def check(ws, reply):
        if s_txt not in reply:
            return False, "reply lacks start time"
        if mode == "file":
            txt = read_text(ws, out)
            return (txt is not None and txt.strip() == s_txt), "file mismatch"
        return e_txt in reply, "reply lacks end time"

    tags = ["python", "interval_reasoning", "file_output" if mode == "file" else "reply_answer"] + (["error_recovery"] if recovered else [])
    return Task("common_free_slot", "calendar", "hard", frame(rng, body), add_distractors(rng, {"calendar.json": to_json(cal)}), steps, final, check,
                [f"{s_txt}-{e_txt}"], tags)


# ===================================================================================================
@family("schedule_conflicts", "calendar", 1.5)
def schedule_conflicts(rng):
    n = rng.randint(7, 12)
    titles = rng.sample(MEETINGS, n)
    day0 = rand_date(rng, 2025)
    events = []
    for i, t in enumerate(titles):
        d = day0 + timedelta(days=rng.randint(0, 1))
        s = rng.choice(range(9 * 60, 16 * 60, 15))
        e = s + rng.choice([30, 45, 60, 90])
        events.append((t, datetime(d.year, d.month, d.day, s // 60, s % 60), datetime(d.year, d.month, d.day, e // 60, e % 60)))
    # force one overlap and one back-to-back (touching) pair
    a, b = events[0], events[1]
    events[1] = (b[0], a[1] + timedelta(minutes=15), a[2] + timedelta(minutes=45))
    c, d_ = events[2], events[3]
    events[3] = (d_[0], c[2], c[2] + timedelta(minutes=30))
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if events[i][1] < events[j][2] and events[j][1] < events[i][2]:
                pairs.add(frozenset((events[i][0], events[j][0])))
    rows = [[t, s.strftime("%Y-%m-%d %H:%M"), e.strftime("%Y-%m-%d %H:%M")] for t, s, e in events]
    rng.shuffle(rows)
    fname, out = rng.choice(["events.csv", "schedule.csv", "calendar_export.csv"]), rng.choice(["conflicts.txt", "overlaps.txt"])
    body = pick(rng, [f"Find all pairs of overlapping events in {fname} (events that merely touch, like one ending exactly when the next starts, are not conflicts). Write one line per pair as 'Title A <-> Title B' to {out} and tell me how many conflicts there are.",
                      f"Check {fname} for scheduling conflicts. A conflict means the time ranges really overlap (back-to-back is fine). Save each pair as 'Title A <-> Title B' in {out} and report the count."])
    code = lambda fmt: ("import csv\nfrom datetime import datetime\nevents = []\nwith open(%r, newline='') as f:\n    for r in csv.DictReader(f):\n"
                        "        events.append((datetime.strptime(r['start'], %r), datetime.strptime(r['end'], %r), r['title']))\n"
                        "events.sort()\npairs = []\nfor i in range(len(events)):\n    for j in range(i + 1, len(events)):\n"
                        "        if events[i][0] < events[j][1] and events[j][0] < events[i][1]:\n            pairs.append(f'{events[i][2]} <-> {events[j][2]}')\n"
                        "open(%r, 'w').write('\\n'.join(pairs) + '\\n')\nprint(len(pairs), 'conflicts')\nprint('\\n'.join(pairs))\n") % (fname, fmt, fmt, out)
    good = Step(["Plan: parse the timestamps, sort by start time, then compare every pair for a real overlap (start < other end and other start < end).",
                 "I'll parse the times and check each pair of events for overlap, treating touching ranges as non-conflicting."][rng.randrange(2)],
                "run_python", code("%Y-%m-%d %H:%M"))
    steps, recovered = with_recovery(rng, good, [(code("%Y-%m-%dT%H:%M"), lambda obs: f"`{last_error_line(obs[-1])}` - the timestamps have a space between date and time, not a 'T'.")], p=0.35)
    steps = inspect_steps(rng, [fname], 6) + steps
    final = f"I found {len(pairs)} conflicts and saved them to {out}."

    def check(ws, reply):
        txt = read_text(ws, out)
        if txt is None:
            return False, "file missing"
        got = set()
        for l in norm_lines(txt):
            parts = [x.strip() for x in l.split("<->")]
            if len(parts) != 2:
                return False, "bad line"
            got.add(frozenset(parts))
        return (got == pairs and has_number(reply, len(pairs), 0.01)), "pairs mismatch"

    return Task("schedule_conflicts", "calendar", "medium", frame(rng, body), add_distractors(rng, {fname: to_csv(["title", "start", "end"], rows)}), steps, final,
                check, [f"{len(pairs)} conflicts"], ["python", "file_output", "interval_reasoning"] + (["error_recovery"] if recovered else []))


# ===================================================================================================
@family("timesheet_hours", "calendar", 1.2)
def timesheet_hours(rng):
    emps, projs = people(rng, rng.randint(3, 5)), rng.sample(PROJECTS, 3)
    d1 = rand_date(rng, 2025)
    d2 = d1 + timedelta(days=rng.randint(5, 13))
    target_e, target_p = rng.choice(emps), rng.choice(projs)
    rows, total = [], 0.0
    for i in range(rng.randint(35, 70)):
        e, p = (target_e, target_p) if i < 5 else (rng.choice(emps), rng.choice(projs))
        d = d1 - timedelta(days=10) + timedelta(days=rng.randint(0, 36))
        h = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 7.5, 8])
        rows.append([d.isoformat(), e, p, h])
        if e == target_e and p == target_p and d1 <= d <= d2:
            total += h
    if total == 0:
        rows.append([d1.isoformat(), target_e, target_p, 4])
        total = 4.0
    rng.shuffle(rows)
    ts = f"{total:g}"
    body = pick(rng, [f"How many hours did {target_e} log on project {target_p} between {d1.isoformat()} and {d2.isoformat()} (both days included)? The data is in timesheet.csv.",
                      f"From timesheet.csv, total the hours for {target_e} on {target_p} from {d1.isoformat()} through {d2.isoformat()} inclusive."])
    code = lambda col: ("import csv\ntotal = 0.0\nwith open('timesheet.csv', newline='') as f:\n    for r in csv.DictReader(f):\n"
                        f"        if r['employee'] == {target_e!r} and r['project'] == {target_p!r} and {d1.isoformat()!r} <= r['date'] <= {d2.isoformat()!r}:\n"
                        f"            total += float(r[{col!r}])\nprint(f'{{total:g}}')\n")
    if rng.random() < 0.35:
        good = Step(pick(rng, PLAN_DO_BASH), "run_bash",
                    f"awk -F, 'NR>1 && $2==\"{target_e}\" && $3==\"{target_p}\" && $1>=\"{d1.isoformat()}\" && $1<=\"{d2.isoformat()}\" {{s+=$4}} END {{print s}}' timesheet.csv")
        steps, recovered, tag = [good], False, "bash"
    else:
        good = Step(pick(rng, PLAN_DO), "run_python", code("hours"))
        steps, recovered = with_recovery(rng, good, [(code("hrs"), lambda obs: f"`{last_error_line(obs[-1])}` - I guessed the column name wrong; the header says 'hours'.")], p=0.3)
        tag = "python"
    steps = inspect_steps(rng, ["timesheet.csv"], 5) + steps
    final = f"{target_e} logged {ts} hours on {target_p} between {d1.isoformat()} and {d2.isoformat()}."

    def check(ws, reply):
        return has_number(reply, total, 0.011), "wrong total"

    return Task("timesheet_hours", "calendar", "easy", frame(rng, body), add_distractors(rng, {"timesheet.csv": to_csv(["date", "employee", "project", "hours"], rows)}),
                steps, final, check, [ts], [tag, "reply_answer"] + (["error_recovery"] if recovered else []))


# ===================================================================================================
@family("invite_draft", "cross_app", 2.0)
def invite_draft(rng):
    day = rand_date(rng, rng.choice([2025, 2026]))
    att = people(rng, rng.randint(2, 3))
    cal, dur, slot = _calendar(rng, att, day)
    others = [p for p in people(rng, 7) if p not in att]
    contacts = [[nm, email_for(nm, rng.choice(DOMAINS))] for nm in att + others]
    rng.shuffle(contacts)
    email_of = {nm: em for nm, em in contacts}
    subject = pick(rng, ["Project kickoff", "Weekly sync", "Planning session", "Design review", "Quarterly check-in"])
    s_txt = fmt_hhmm(slot)
    who = ", ".join(att[:-1]) + " and " + att[-1]
    expected = [f"To: {', '.join(email_of[a] for a in att)}", f"Subject: {subject}", f"When: {day.isoformat()} {s_txt}"]
    body = (f"Set up a {dur}-minute meeting called '{subject}' with {who} on {day.isoformat()}. Find the earliest slot between 09:00 and 17:00 when they are all free "
            f"(calendar.json), look up their email addresses in contacts.csv, and write the invite to out/invite.txt in exactly three lines: "
            f"'To: <emails separated by comma and space, in the order I listed the people>', 'Subject: <the meeting name>', 'When: YYYY-MM-DD HH:MM'.")
    slot_re = re.compile(r"slot: (\d\d:\d\d)")
    step_a = Step("First I need the earliest time when all attendees are free, so I'll compute that from calendar.json.", "run_python",
                  FREE_SLOT_CODE.format(people=att, day=day.isoformat(), dur=dur, dk="date"))
    code_b = lambda obs: ("import csv, os\nslot = %r\nattendees = %r\nemails = {}\nwith open('contacts.csv', newline='') as f:\n    for r in csv.DictReader(f):\n"
                          "        emails[r['name']] = r['email']\nos.makedirs('out', exist_ok=True)\n"
                          "lines = ['To: ' + ', '.join(emails[a] for a in attendees), %r, 'When: ' + %r + ' ' + slot]\n"
                          "open('out/invite.txt', 'w').write('\\n'.join(lines) + '\\n')\nprint(open('out/invite.txt').read())\n") % (
        slot_re.search(obs[-1]).group(1), att, "Subject: " + subject, day.isoformat())
    step_b = Step(lambda obs: f"The earliest common slot is {slot_re.search(obs[-1]).group(1)}. Now I'll look up the attendees' emails in contacts.csv and write the invite.",
                  "run_python", code_b)
    steps = inspect_steps(rng, ["calendar.json", "contacts.csv"], 12) + [step_a, step_b]
    final = f"The invite for '{subject}' is saved in out/invite.txt: {day.isoformat()} at {s_txt} with {who}."

    def check(ws, reply):
        txt = read_text(ws, "out/invite.txt")
        return (txt is not None and norm_lines(txt) == expected), "invite mismatch"

    return Task("invite_draft", "cross_app", "hard", frame(rng, body), add_distractors(rng, {"calendar.json": to_json(cal), "contacts.csv": to_csv(["name", "email"], contacts)}),
                steps, final, check, [s_txt], ["python", "file_output", "multi_file", "multi_step_dependency", "cross_application"])


# ===================================================================================================
@family("inbox_triage", "cross_app", 1.3)
def inbox_triage(rng):
    kw = rng.choice(["urgent", "asap", "critical"])
    hot = {"urgent": ["URGENT: server down", "Urgent - approval needed", "Re: urgent payment", "urgent: customer escalation"],
           "asap": ["ASAP: approve the PO", "Need your review asap", "Re: asap - contract", "asap: lab results"],
           "critical": ["CRITICAL: outage in EU", "Critical bug in checkout", "Re: critical patch window", "critical: security notice"]}[kw]
    senders = [email_for(nm, rng.choice(DOMAINS)) for nm in people(rng, rng.randint(5, 9))]
    subj_ok = ["Meeting notes", "Invoice attached", "Lunch on Friday?", "Q3 report draft", "Welcome aboard", "Newsletter", "Re: contract"]
    fname = rng.choice(["inbox.json", "mailbox.json", "messages.json"])
    msgs = []
    for i in range(rng.randint(15, 30)):
        is_hot = rng.random() < 0.22 or i in (2, 9)
        msgs.append({"from": rng.choice(senders), "subject": rng.choice(hot if is_hot else subj_ok), "unread": rng.random() < 0.45})
    key = f"{kw}_senders"
    expected = {"unread": sum(m["unread"] for m in msgs), key: sorted({m["from"] for m in msgs if kw in m["subject"].lower()})}
    out = rng.choice(["triage.json", "inbox_summary.json", "mail_report.json"])
    body = pick(rng, [f"Look at {fname} and write {out} with two fields: 'unread' (how many messages are unread) and '{key}' (sorted list of unique sender addresses whose subject contains '{kw}' in any capitalisation, read or unread).",
                      f"Triage {fname} for me. Save {out} as a JSON object {{\"unread\": <count of unread messages>, \"{key}\": <sorted unique senders of messages with '{kw}' in the subject, case-insensitive>}}.",
                      f"I need a quick inbox summary from {fname}: write {out} containing the number of unread messages ('unread') and the sorted unique senders whose subject mentions '{kw}', whatever the capitalisation, under the key '{key}'."])
    code = (f"import json\nmsgs = json.load(open({fname!r}))\nresult = {{'unread': sum(1 for m in msgs if m['unread']),\n"
            f"          {key!r}: sorted({{m['from'] for m in msgs if {kw!r} in m['subject'].lower()}})}}\n"
            f"json.dump(result, open({out!r}, 'w'), indent=2)\nprint(json.dumps(result, indent=2))\n")
    steps = inspect_steps(rng, [fname], 12) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"Saved {out}: {expected['unread']} unread messages and {len(expected[key])} senders with '{kw}' subjects."

    def check(ws, reply):
        txt = read_text(ws, out)
        try:
            return json.loads(txt) == expected, "json mismatch"
        except Exception:
            return False, "bad json"

    return Task("inbox_triage", "cross_app", "medium", frame(rng, body), add_distractors(rng, {fname: to_json(msgs)}), steps, final, check,
                [str(expected["unread"])], ["python", "file_output", "case_insensitive"])


# ===================================================================================================
@family("overdue_reminders", "cross_app", 2.0)
def overdue_reminders(rng):
    today = rand_date(rng, 2025)
    clients = rng.sample(["Acme Corp", "Northwind", "Globex", "Initech", "Hooli", "Umbrella Ltd", "Soylent"], rng.randint(3, 5))
    contacts = [[c, f"billing@{c.lower().replace(' ', '')}.com"] for c in clients]
    email_of = dict(contacts)
    inv = []
    for i in range(rng.randint(10, 22)):
        due = today + timedelta(days=rng.randint(-60, 30))
        inv.append([f"INV-{2000 + i}", rng.choice(clients), f"{rng.uniform(100, 5000):.2f}", due.isoformat(), rng.choice(["yes", "no"])])
    inv[0][3], inv[0][4] = (today - timedelta(days=12)).isoformat(), "no"
    inv[1][3], inv[1][4] = (today - timedelta(days=30)).isoformat(), "no"
    over = sorted([r for r in inv if r[4] == "no" and r[3] < today.isoformat()], key=lambda r: (r[3], r[0]))
    expected = [f"{email_of[r[1]]} | {r[0]} | {float(r[2]):.2f} | due {r[3]}" for r in over]
    body = (f"Today is {today.isoformat()}. In invoices.csv, find every invoice that is unpaid (paid = 'no') and past its due date (due_date earlier than today). "
            f"Look up each client's email in contacts.csv and write reminders.txt with one line per overdue invoice in the form "
            f"'<email> | <invoice_id> | <amount with 2 decimals> | due <due_date>', sorted by due date then invoice id. Tell me how many reminders you wrote.")
    code = ("import csv\ntoday = %r\nemails = {}\nwith open('contacts.csv', newline='') as f:\n    for r in csv.DictReader(f):\n        emails[r['client']] = r['email']\n"
            "overdue = []\nwith open('invoices.csv', newline='') as f:\n    for r in csv.DictReader(f):\n"
            "        if r['paid'] == 'no' and r['due_date'] < today:\n            overdue.append(r)\n"
            "overdue.sort(key=lambda r: (r['due_date'], r['invoice_id']))\n"
            "lines = [f\"{emails[r['client']]} | {r['invoice_id']} | {float(r['amount']):.2f} | due {r['due_date']}\" for r in overdue]\n"
            "open('reminders.txt', 'w').write('\\n'.join(lines) + '\\n')\nprint(len(lines), 'reminders')\nprint('\\n'.join(lines))\n") % today.isoformat()
    steps = inspect_steps(rng, ["invoices.csv", "contacts.csv"], 6) + [Step(pick(rng, PLAN_DO), "run_python", code)]
    final = f"I wrote {len(expected)} reminders to reminders.txt (unpaid invoices due before {today.isoformat()})."

    def check(ws, reply):
        txt = read_text(ws, "reminders.txt")
        if txt is None or norm_lines(txt) != expected:
            return False, "reminders mismatch"
        return has_number(reply, len(expected), 0.01), "count missing"

    files = add_distractors(rng, {"invoices.csv": to_csv(["invoice_id", "client", "amount", "due_date", "paid"], inv),
                                  "contacts.csv": to_csv(["client", "email"], contacts)})
    return Task("overdue_reminders", "cross_app", "hard", frame(rng, body), files, steps, final, check, [f"{len(expected)} reminders"],
                ["python", "file_output", "multi_file", "cross_application", "date_logic"])
