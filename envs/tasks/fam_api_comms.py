"""Function-calling families: email, contacts and tasks (draft vs send, clarification, parallel reads, state changes)."""
from __future__ import annotations

import json
import re

from envs.tasks.api_base import (ApiStep, ApiTask, Call, Turn, MEETING_TITLES, email_of, first, free_start, hm, jfind, make_world, names_phrase,
                                 pick_tools, world_with_slot)
from envs.tasks.base import family, frame, pick
from envs.tasks.common import fmt_hhmm
from envs.workspace_api import HI

TASK_TITLES = ["Send the revised quote", "Book the conference room", "Update the onboarding checklist", "Review the vendor contract",
               "Prepare the Q3 slides", "Fix the broken export link", "Schedule the customer call", "Archive the old tickets",
               "Draft the release notes", "Confirm the budget numbers", "Collect feedback from support", "Renew the software licences"]
PROJECTS = ["Apollo", "Borealis", "Cobalt", "Delta", "Everest", "Falcon", "Granite", "Horizon"]
INBOX_TEMPLATES = [("Invoice {n} needs approval", "Hi,\n\nPlease approve invoice {n} for {amt} EUR before Friday.\n\nThanks"),
                   ("Question about the {p} timeline", "Hello,\n\nCould you confirm whether the {p} milestone still ships on schedule?\n\nCheers"),
                   ("Lunch on Thursday?", "Hi!\n\nAre you free for lunch on Thursday? I was thinking noon.\n\nBest"),
                   ("Updated slides for the {p} review", "Hi,\n\nI attached the updated slides for the {p} review. Tell me if anything is missing.\n\nRegards"),
                   ("Access request for the {p} dashboard", "Hello,\n\nCould you grant me access to the {p} dashboard? My manager already approved it.\n\nThanks"),
                   ("Re: contract renewal {n}", "Hi,\n\nThe legal team signed off on contract {n}. We can renew whenever you are ready.\n\nBest regards"),
                   ("Weekly numbers for {p}", "Hi,\n\nThe weekly numbers for {p} are in the shared folder. Revenue is up {amt} percent.\n\nThanks")]


def _inbox(rng, world, n, n_unread, exclude=()):
    senders = [c["email"] for c in world["contacts"] if c["email"] not in exclude]
    msgs = []
    unread_idx = set(rng.sample(range(n), n_unread))
    for i in range(n):
        subj, body = rng.choice(INBOX_TEMPLATES)
        kw = {"n": rng.randint(1000, 9999), "p": rng.choice(PROJECTS), "amt": rng.randint(12, 95)}
        msgs.append({"id": f"m{i + 1}", "from": rng.choice(senders), "subject": subj.format(**kw), "body": body.format(**kw), "unread": i in unread_idx})
    return msgs


def _ids(obs):
    return [m["id"] for m in jfind(obs, "messages")["messages"]]


# ===================================================================================================
@family("api_inbox_digest", "api_comms", 2.0)
def api_inbox_digest(rng):
    w = make_world(rng)
    n_unread = rng.randint(2, 4)
    w["inbox"] = _inbox(rng, w, rng.randint(n_unread + 2, 9), n_unread)
    unread = [m for m in w["inbox"] if m["unread"]]
    body = pick(rng, ["Summarise my unread emails.", "What's new in my inbox? Only the unread ones, please.", "Give me a digest of the emails I haven't read yet."])
    steps = [ApiStep("I'll list the unread messages first.", [Call("email_list_inbox", {"unread_only": True})]),
             ApiStep(lambda obs: f"There are {len(_ids(obs))} unread messages. I'll read them all at once.",
                     [Call("email_read", (lambda i: (lambda obs: {"message_id": _ids(obs)[i]}))(i)) for i in range(len(unread))])]

    def final(obs):
        lines = []
        for o in obs:
            if o.startswith('{"message"'):
                m = json.loads(o)["message"]
                snippet = m["body"].split("\n\n")[1] if "\n\n" in m["body"] else m["body"]
                lines.append(f"- {m['from']} - {m['subject']}: {snippet[:110]}")
        return f"You have {len(lines)} unread emails:\n" + "\n".join(lines)

    def check(api, replies):
        r = replies[-1]
        if not all(m["subject"] in r and m["from"] in r for m in unread):
            return False, "digest incomplete"
        if any(m["unread"] for m in api.inbox if m["id"] in {u["id"] for u in unread}):
            return False, "read messages should be marked read"
        return True, "ok"

    return Task_(rng, "api_inbox_digest", "api_comms", "medium", w, ["email_list_inbox", "email_read"], body, steps, final,
                 [m["subject"] for m in unread][:3], check, ["parallel_calls", "chain", "summarise_tool_output"])


def Task_(rng, name, cat, diff, world, required, prompt, steps, final, grounded, check, tags, extra=(3, 7)):
    return ApiTask(name, cat, diff, world, pick_tools(rng, required, extra), [Turn(frame(rng, prompt), steps, final, grounded)], check, tags)


# ===================================================================================================
@family("api_draft_or_send", "api_comms", 2.5)
def api_draft_or_send(rng):
    w = make_world(rng)
    who = rng.sample([c["name"] for c in w["contacts"]], rng.choice([1, 1, 2]))
    emails = [email_of(w, a) for a in who]
    topic, date, time = rng.choice(PROJECTS), w["day"], fmt_hhmm(rng.choice(range(9 * 60, 17 * 60, 15)))
    kind = rng.choice(["moved", "deadline", "starts"])
    msg = {"moved": (f"the {topic} review has been moved to {date} at {time}", f"The {topic} review has been moved to {date} at {time}."),
           "deadline": (f"they need to send the {topic} numbers by {date} at {time}", f"Please send the {topic} numbers by {date} at {time}."),
           "starts": (f"the {topic} workshop starts at {time} on {date}", f"The {topic} workshop starts at {time} on {date}.")}[kind]
    subject = f"{topic} update" if kind != "deadline" else f"{topic} numbers"
    send = rng.random() < 0.4
    tail = " and send it" if send else ". Don't send it, I want to review it first"
    body = pick(rng, [f"Write an email to {names_phrase(who)} with the subject '{subject}' letting them know {msg[0]}{tail}.",
                      f"Please email {names_phrase(who)} (subject: '{subject}') to say {msg[0]}{'. Send it right away' if send else ', but only save it as a draft'}."])
    greeting = f"Hi {' and '.join(first(a) for a in who)}"
    steps = [ApiStep("I need the recipients' email addresses first.", [Call("contacts_lookup", {"name": a}) for a in who]),
             ApiStep("Now I'll write the draft.", [Call("email_draft", lambda obs: {"to": [json.loads(o)["contact"]["email"] for o in obs if o.startswith('{"contact"')],
                                                                                    "subject": subject, "body": f"{greeting},\n\n{msg[1]}\n\nBest regards"})])]
    if send:
        steps.append(ApiStep("The draft is ready and you asked me to send it, so I'll send it now.", [Call("email_send", lambda obs: {"draft_id": jfind(obs, "draft_id")["draft_id"]})]))
        final = f"Sent your email '{subject}' to {', '.join(emails)}."
    else:
        final = f"I saved a draft '{subject}' to {', '.join(emails)}. It has not been sent."

    def check(api, replies):
        msgs = api.sent if send else list(api.drafts.values())
        if len(msgs) != 1 or sorted(msgs[0]["to"]) != sorted(emails) or msgs[0]["subject"] != subject:
            return False, "message wrong"
        if date not in msgs[0]["body"] or time not in msgs[0]["body"]:
            return False, "body lacks the facts"
        if not send and api.sent:
            return False, "must not send"
        return True, "ok"

    return Task_(rng, "api_draft_or_send", "api_comms", "medium", w, ["contacts_lookup", "email_draft"] + (["email_send"] if send else []), body, steps, final,
                 [], check, ["tool_chain", "send_vs_draft", "instruction_precision"] + (["parallel_calls"] if len(who) > 1 else []), extra=(3, 6) if send else (3, 6))


# ===================================================================================================
@family("api_reply_confirm_meeting", "api_comms", 2.0)
def api_reply_confirm_meeting(rng):
    for _ in range(80):
        w = make_world(rng)
        me, other = rng.sample([c["name"] for c in w["contacts"]], 2)
        after = rng.choice(range(9 * 60, 14 * 60, 30))
        slot = free_start(w["events"], [me, other], w["day"], 30, after, HI)
        if slot is not None:
            break
    day, subj_topic = w["day"], rng.choice(PROJECTS)
    subject = f"Quick sync on {subj_topic}"
    msg = {"id": "m1", "from": email_of(w, other), "subject": subject,
           "body": f"Hi,\n\nCould we meet for 30 minutes on {day}? Any time after {fmt_hhmm(after)} works for me.\n\nBest", "unread": True}
    w["inbox"] = [msg] + [dict(m, id=f"m{i + 2}") for i, m in enumerate(_inbox(rng, w, rng.randint(2, 4), 1, exclude=[msg["from"]]))]
    rng.shuffle(w["inbox"])
    s_txt = fmt_hhmm(slot)
    body = pick(rng, [f"I'm {me}. Reply to {other}'s email about the sync: pick the earliest 30-minute slot on the day they proposed, after the time they suggested, when we are both free. Save it as a draft, don't send.",
                      f"This is {me}. {other} emailed me asking for a quick sync. Draft a reply confirming the earliest time that works for both of us within what they proposed."])
    mid = lambda obs: next(m["id"] for m in jfind(obs, "messages")["messages"] if m["from"] == email_of(w, other) and m["subject"] == subject)
    mbody = lambda obs: jfind(obs, "message")["message"]["body"]
    steps = [ApiStep("I'll check the inbox to find the email.", [Call("email_list_inbox", {})]),
             ApiStep(lambda obs: f"The message from {other} is {mid(obs)}. Let me read what they proposed.", [Call("email_read", lambda obs: {"message_id": mid(obs)})]),
             ApiStep(lambda obs: f"They want 30 minutes on {re.search(r'on (\d{4}-\d{2}-\d{2})', mbody(obs)).group(1)} after {re.search(r'after (\d{2}:\d{2})', mbody(obs)).group(1)}. Let me find a slot that works for both of us.",
                     [Call("calendar_find_free_slot", lambda obs: {"attendees": [me, other], "date": re.search(r"on (\d{4}-\d{2}-\d{2})", mbody(obs)).group(1), "duration_minutes": 30,
                                                                   "earliest": re.search(r"after (\d{2}:\d{2})", mbody(obs)).group(1)})]),
             ApiStep(lambda obs: f"{jfind(obs, 'slot')['slot']['start']} works for both. I'll draft the reply.",
                     [Call("email_draft", lambda obs: {"to": [email_of(w, other)], "subject": f"Re: {subject}",
                                                       "body": f"Hi {first(other)},\n\n{jfind(obs, 'slot')['slot']['start']} on {day} works for me. Does that suit you?\n\nBest regards,\n{first(me)}"})])]
    final = f"I drafted a reply to {other} (not sent) proposing {s_txt} on {day}, the earliest time we are both free."

    def check(api, replies):
        ds = list(api.drafts.values())
        if len(ds) != 1 or ds[0]["to"] != [email_of(w, other)] or ds[0]["subject"] != f"Re: {subject}":
            return False, "draft wrong"
        return (s_txt in ds[0]["body"] and day in ds[0]["body"] and not api.sent and s_txt in replies[-1]), "draft content"

    return Task_(rng, "api_reply_confirm_meeting", "api_comms", "hard", w, ["email_list_inbox", "email_read", "calendar_find_free_slot", "email_draft"], body, steps, final,
                 [s_txt], check, ["tool_chain", "read_then_act", "parse_tool_output", "draft_not_send", "cross_application"], extra=(2, 5))


# ===================================================================================================
@family("api_ambiguous_contact", "api_comms", 2.0)
def api_ambiguous_contact(rng):
    for _ in range(80):
        w = make_world(rng, n_people=(5, 7))
        a = w["contacts"][0]
        fn = first(a["name"])
        last_b = rng.choice(["Berg", "Moreau", "Santos", "Weber", "Novak", "Costa"])
        if any(first(c["name"]) == fn for c in w["contacts"][1:]) or f"{fn} {last_b}" in [c["name"] for c in w["contacts"]]:
            continue
        team_b = next((t for t in ["Sales", "Engineering", "Support", "Finance", "Design", "Marketing"] if t != a["team"]), "Sales")
        b = {"name": f"{fn} {last_b}", "email": f"{fn.lower()}.{last_b.lower()}@{a['email'].split('@')[1]}", "team": team_b}
        w["contacts"].append(b)
        hits = [c for c in w["contacts"] if fn.lower() in c["name"].lower() or fn.lower() in c["email"].lower() or fn.lower() in c["team"].lower()]
        if len(hits) == 2:
            break
    doc = rng.choice(["report", "proposal", "budget draft", "slide deck"])
    chosen = rng.choice([a, b])
    t1 = Turn(frame(rng, f"Draft a short email to {fn} saying the {doc} is ready for review."),
              [ApiStep(f"There may be more than one {fn} in the contacts, so I'll search first.", [Call("contacts_search", {"query": fn})])],
              f"I found two contacts named {fn}: {a['name']} ({a['team']}) and {b['name']} ({b['team']}). Which one should I write to?", [a["name"], b["name"]])
    answer = pick(rng, [f"The one in {chosen['team']}.", f"{chosen['name']}.", f"I mean {first(chosen['name'])} from {chosen['team']}."])
    t2 = Turn(answer, [ApiStep(f"That's {chosen['name']} ({chosen['email']}). I'll write the draft.",
                               [Call("email_draft", {"to": [chosen["email"]], "subject": f"{doc.capitalize()} ready for review",
                                                     "body": f"Hi {fn},\n\nThe {doc} is ready for your review. Let me know if you have any questions.\n\nBest regards"})])],
              f"I saved a draft to {chosen['name']} ({chosen['email']}) saying the {doc} is ready for review. It has not been sent.", [])

    def check(api, replies):
        if "?" not in replies[0] or a["name"] not in replies[0] or b["name"] not in replies[0]:
            return False, "first reply must ask which contact"
        ds = list(api.drafts.values())
        return (len(ds) == 1 and ds[0]["to"] == [chosen["email"]] and not api.sent), "draft wrong"

    tools = pick_tools(rng, ["contacts_search", "email_draft"], extra=(3, 6))
    return ApiTask("api_ambiguous_contact", "api_comms", "hard", w, tools, [t1, t2], check, ["multi_turn", "clarifying_question", "ambiguity", "contacts"])


# ===================================================================================================
@family("api_task_capture", "api_comms", 2.0)
def api_task_capture(rng):
    w = make_world(rng)
    day = w["day"]
    k = rng.randint(2, 5)
    owners = [rng.choice([c["name"] for c in w["contacts"]]) for _ in range(k)]
    titles = rng.sample(TASK_TITLES, k)
    from datetime import date, timedelta
    dues = [(date.fromisoformat(day) + timedelta(days=rng.randint(1, 14))).isoformat() for _ in range(k)]
    lines = [f"# Standup notes ({day})", "- Discussed the release plan.", "- Support tickets are down this week."]
    items = []
    for i in range(k):
        lines.insert(rng.randint(1, len(lines)), f"- ACTION [{owners[i]}]: {titles[i].lower()} by {dues[i]}")
        items.append((titles[i].lower(), owners[i], dues[i]))
    w["files"]["notes/standup.md"] = "\n".join(lines) + "\n"
    w["tasks"] = [{"id": f"task_{i + 1:02d}", "title": t, "owner": None, "due": None, "status": "open"} for i, t in enumerate(rng.sample(TASK_TITLES, rng.randint(0, 2)))]
    body = pick(rng, ["Read notes/standup.md and add every action item (the lines starting with '- ACTION') to my task list with the owner and due date.",
                      "Turn the action items in notes/standup.md into tasks: one task per '- ACTION' line, with its owner and due date."])
    parse = lambda obs: re.findall(r"- ACTION \[([^\]]+)\]: (.+) by (\d{4}-\d{2}-\d{2})", jfind(obs, "content")["content"])
    steps = [ApiStep("I'll read the notes first.", [Call("read_file", {"path": "notes/standup.md"})]),
             ApiStep(lambda obs: f"I found {len(parse(obs))} action items. They're independent, so I'll add them all in one go.",
                     [Call("tasks_add", (lambda i: (lambda obs: {"title": parse(obs)[i][1], "owner": parse(obs)[i][0], "due": parse(obs)[i][2]}))(i)) for i in range(k)])]
    if rng.random() < 0.5:
        steps.append(ApiStep("Let me double-check the task list.", [Call("tasks_list", {"status": "open"})]))
    final = f"I added {k} tasks from the standup notes: " + "; ".join(f"{t} ({o}, due {d})" for t, o, d in items) + "."
    pre = len(w["tasks"])

    def check(api, replies):
        new = api.tasks[pre:]
        return (sorted((t["title"], t["owner"], t["due"]) for t in new) == sorted(items) and len(new) == k), "tasks wrong"

    return Task_(rng, "api_task_capture", "api_comms", "medium", w, ["read_file", "tasks_add"] + (["tasks_list"] if len(steps) == 3 else []), body, steps, final,
                 [items[0][2]], check, ["parallel_calls", "parse_tool_output", "state_change"])


# ===================================================================================================
@family("api_task_review", "api_comms", 1.5)
def api_task_review(rng):
    w = make_world(rng)
    n = rng.randint(5, 8)
    titles = rng.sample(TASK_TITLES, n)
    w["tasks"] = [{"id": f"task_{i + 1:02d}", "title": titles[i], "owner": rng.choice([c["name"] for c in w["contacts"]]), "due": w["day"], "status": "open"} for i in range(n)]
    done = rng.sample(titles, rng.randint(1, 3))
    left = [t for t in titles if t not in done]
    quoted = " and ".join(f"'{t}'" for t in done)
    body = pick(rng, [f"Mark {quoted} as done, then tell me which tasks are still open.", f"I finished {quoted}. Please close them and list what's left on my to-do list."])
    tid = lambda t: (lambda obs: {"task_id": next(x["id"] for x in jfind(obs, "tasks")["tasks"] if x["title"] == t)})
    steps = [ApiStep("I'll look at the open tasks to get their ids.", [Call("tasks_list", {"status": "open"})]),
             ApiStep("I have the ids. The completions are independent, so I'll do them together.", [Call("tasks_complete", tid(t)) for t in done])]
    final = f"Closed {len(done)} task(s). Still open ({len(left)}): " + "; ".join(left) + "." if left else "Closed them - nothing is left open."

    def check(api, replies):
        st = {t["title"]: t["status"] for t in api.tasks}
        if any(st[t] != "done" for t in done) or any(st[t] != "open" for t in left):
            return False, "task states wrong"
        return all(t in replies[-1] for t in left), "reply must list the open tasks"

    return Task_(rng, "api_task_review", "api_comms", "easy", w, ["tasks_list", "tasks_complete"], body, steps, final, left[:2], check,
                 ["parallel_calls", "state_change", "lookup_then_act"])
