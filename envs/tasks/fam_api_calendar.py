"""Function-calling families: calendar & scheduling (typed API tools, parallel calls, API errors, multi-turn)."""
from __future__ import annotations

import json

from envs.tasks.api_base import (ApiStep, ApiTask, Call, MEETING_TITLES, Turn, email_of, free_start, hm, jfind, make_world,
                                 names_phrase, pick_tools, world_with_slot)
from envs.tasks.base import family, frame, pick
from envs.tasks.common import fmt_hhmm

PLAN_FIND = ["I'll start by finding the earliest time when everyone is free.", "First I need a slot that works for all attendees.",
             "Let me look for the earliest common free slot."]
PLAN_LIST = ["Let me check the calendars first.", "I'll pull up the events for that day.", "First I'll look at what's already scheduled."]


def _slot(obs):
    return jfind(obs, "slot")["slot"]


# ===================================================================================================
@family("api_find_and_schedule", "api_calendar", 2.5)
def api_find_and_schedule(rng):
    w, att, dur, slot = world_with_slot(rng)
    day, title = w["day"], rng.choice(MEETING_TITLES)
    s_txt, e_txt, who = fmt_hhmm(slot), fmt_hhmm(slot + dur), names_phrase(att)
    body = pick(rng, [f"Schedule '{title}' on {day} for {dur} minutes with {who}, at the earliest time everyone is free.",
                      f"Please book a {dur}-minute meeting called '{title}' with {who} on {day}. Use the first slot when all of them are available.",
                      f"Can you set up '{title}' ({dur} min) on {day} with {who}? Pick the earliest common free time within working hours.",
                      f"I need {dur} minutes on {day} for {who}. Find the earliest time they're all free and create the event '{title}'."])
    create_args = lambda obs: {"title": title, "date": day, "start": _slot(obs)["start"], "duration_minutes": dur, "attendees": att}
    plan_create = lambda obs: f"The earliest common slot is {_slot(obs)['start']}-{_slot(obs)['end']}. I'll create the event."
    steps = [ApiStep(pick(rng, PLAN_FIND), [Call("calendar_find_free_slot", {"attendees": att, "date": day, "duration_minutes": dur})])]
    recovered = rng.random() < 0.2
    if recovered:
        bad = lambda obs: {**create_args(obs), "start": _slot(obs)["start"].replace(":", ".")}
        steps += [ApiStep(plan_create, [Call("calendar_create_event", bad, expect_error=True)]),
                  ApiStep("The API needs times as HH:MM with a colon. Retrying with the right format.", [Call("calendar_create_event", create_args)])]
    else:
        steps.append(ApiStep(plan_create, [Call("calendar_create_event", create_args)]))
    final = f"Done - '{title}' is booked on {day} from {s_txt} to {e_txt} with {who}."

    def check(api, replies):
        evs = [e for e in api.events.values() if e["title"] == title and e["date"] == day]
        if len(evs) != 1:
            return False, "expected exactly one new event"
        e = evs[0]
        if e["start"] != s_txt or hm(e["end"]) - hm(e["start"]) != dur or sorted(e["attendees"]) != sorted(att):
            return False, "event fields wrong"
        return s_txt in replies[-1], "reply lacks the time"

    tools = pick_tools(rng, ["calendar_find_free_slot", "calendar_create_event"])
    return ApiTask("api_find_and_schedule", "api_calendar", "medium", w, tools, [Turn(frame(rng, body), steps, final, [s_txt, e_txt])], check,
                   ["tool_chain", "tool_selection", "date_time"] + (["api_error_recovery"] if recovered else []))


# ===================================================================================================
@family("api_reschedule", "api_calendar", 2.0)
def api_reschedule(rng):
    w, att, dur, _ = world_with_slot(rng, n_att=(2, 3), want_late=False)
    day, title, owner = w["day"], rng.choice(MEETING_TITLES), att[0]
    s0 = rng.choice(range(10 * 60, 15 * 60, 15))
    ev = {"id": "evt_90", "title": title, "date": day, "start": fmt_hhmm(s0), "end": fmt_hhmm(s0 + dur), "attendees": att}
    w["events"].append(ev)
    others_events = [e for e in w["events"] if e["id"] != "evt_90"]
    cands = [s for s in range(9 * 60, 16 * 60, 15) if s != s0 and free_start(others_events, att, day, dur, s, s + dur) == s]
    explicit = bool(cands) and rng.random() < 0.55
    exp_b = free_start(w["events"], att, day, dur)
    if not explicit and exp_b is None:
        explicit = True
    if explicit and not cands:
        cands = [s0 + 15]
    new = rng.choice(cands) if explicit else exp_b
    new_txt, who = fmt_hhmm(new), names_phrase(att[1:])
    ev_id = lambda obs: next(e["id"] for e in jfind(obs, "events")["events"] if e["title"] == title)
    upd = lambda start: (lambda obs: {"event_id": ev_id(obs), "start": start(obs) if callable(start) else start})
    if explicit:
        body = pick(rng, [f"I'm {owner}. Move '{title}' on {day} to {new_txt}.", f"This is {owner}: please shift my '{title}' meeting on {day} so it starts at {new_txt}."])
        steps = [ApiStep(pick(rng, PLAN_LIST), [Call("calendar_list_events", {"person": owner, "date": day})]),
                 ApiStep(lambda obs: f"'{title}' is event {ev_id(obs)}. I'll move it to {new_txt}.", [Call("calendar_update_event", upd(new_txt))])]
        grounded = [new_txt]
    else:
        body = pick(rng, [f"I'm {owner}. Reschedule '{title}' on {day} to the earliest time when all attendees ({who} and I) are free.",
                          f"I'm {owner}. Find the earliest time on {day} that works for everyone in '{title}' and move the meeting there."])
        steps = [ApiStep("I need the event's id and the earliest common free slot; those two lookups are independent.",
                         [Call("calendar_list_events", {"person": owner, "date": day}),
                          Call("calendar_find_free_slot", {"attendees": att, "date": day, "duration_minutes": dur})]),
                 ApiStep(lambda obs: f"The earliest common slot is {_slot(obs)['start']}. Moving '{title}' there.", [Call("calendar_update_event", upd(lambda o: _slot(o)["start"]))])]
        grounded = [new_txt]
    final = f"I moved '{title}' on {day} to {new_txt}."

    def check(api, replies):
        e = api.events.get("evt_90")
        if not e or e["start"] != new_txt or hm(e["end"]) - hm(e["start"]) != dur or sorted(e["attendees"]) != sorted(att):
            return False, "event not moved correctly"
        return new_txt in replies[-1], "reply lacks the new time"

    tools = pick_tools(rng, ["calendar_list_events", "calendar_update_event"] + ([] if explicit else ["calendar_find_free_slot"]))
    return ApiTask("api_reschedule", "api_calendar", "medium", w, tools, [Turn(frame(rng, body), steps, final, grounded)], check,
                   ["tool_chain", "parallel_calls" if not explicit else "single_lookup", "update_existing"])


# ===================================================================================================
@family("api_agenda_summary", "api_calendar", 1.8)
def api_agenda_summary(rng):
    w = make_world(rng, n_people=(3, 5))
    day = w["day"]
    ppl = rng.sample([c["name"] for c in w["contacts"]], rng.randint(2, 3))
    body = pick(rng, [f"What's on the calendar on {day} for {names_phrase(ppl)}?", f"Give me a quick agenda for {day} covering {names_phrase(ppl)}.",
                      f"Show me {names_phrase(ppl)}'s meetings on {day}, in time order."])
    calls = [Call("calendar_list_events", {"person": p, "date": day}) for p in ppl]

    def final(obs):
        lines = []
        for o in obs:
            d = json.loads(o)
            evs = "; ".join(f"{e['start']}-{e['end']} {e['title']}" for e in d["events"]) or "nothing scheduled"
            lines.append(f"- {d['person']}: {evs}")
        return f"Here is the agenda for {day}:\n" + "\n".join(lines)

    expected = {p: sorted((e["start"], e["title"]) for e in w["events"] if e["date"] == day and p in e["attendees"]) for p in ppl}
    grounded = [f"{s}" for evs in expected.values() for s, _ in evs][:6]

    def check(api, replies):
        r = replies[-1]
        return all(s in r and t in r for evs in expected.values() for s, t in evs), "agenda incomplete"

    steps = [ApiStep("These are independent lookups, so I'll fetch every calendar at once.", calls)]
    return ApiTask("api_agenda_summary", "api_calendar", "easy", w, pick_tools(rng, ["calendar_list_events"]), [Turn(frame(rng, body), steps, final, grounded)],
                   check, ["parallel_calls", "summarise_tool_output"])


# ===================================================================================================
@family("api_conflict_check", "api_calendar", 2.0)
def api_conflict_check(rng):
    w, att, dur, slot = world_with_slot(rng, n_att=(2, 3), want_late=False)
    day, title = w["day"], rng.choice(MEETING_TITLES)
    grid = list(range(9 * 60, 16 * 60, 15))
    free_starts = [s for s in grid if free_start(w["events"], att, day, dur, s, s + dur) == s]
    busy_starts = [s for s in grid if s not in free_starts]
    free_case = (rng.random() < 0.5 and bool(free_starts)) or not busy_starts
    start = rng.choice(free_starts) if free_case else rng.choice(busy_starts)
    s_txt = fmt_hhmm(start)
    who = names_phrase(att)
    body = pick(rng, [f"Is {s_txt} on {day} free for {who}? If everyone is free, book '{title}' for {dur} minutes; if not, tell me who is busy and when we could meet instead.",
                      f"Check whether {who} are all available at {s_txt} on {day} for {dur} minutes. If yes, create '{title}'. If not, say who has a conflict and propose the earliest alternative."])

    def conflicts(obs):
        out = []
        for o in obs:
            d = json.loads(o)
            if "events" in d:
                hit = [e for e in d["events"] if start < hm(e["end"]) and hm(e["start"]) < start + dur]
                if hit:
                    out.append((d["person"], hit))
        return out

    steps = [ApiStep("I'll check each attendee's calendar for that day in parallel.", [Call("calendar_list_events", {"person": p, "date": day}) for p in att])]
    if free_case:
        steps.append(ApiStep(f"Nobody has anything overlapping {s_txt}. I'll book it.",
                             [Call("calendar_create_event", {"title": title, "date": day, "start": s_txt, "duration_minutes": dur, "attendees": att})]))
        final = f"Everyone is free at {s_txt} on {day}, so I booked '{title}' for {dur} minutes with {who}."
        grounded = []
    else:
        alt = free_start(w["events"], att, day, dur)
        alt_txt = fmt_hhmm(alt) if alt is not None else None
        steps.append(ApiStep(lambda obs: "There are conflicts: " + ", ".join(f"{p} ({h[0]['title']} {h[0]['start']}-{h[0]['end']})" for p, h in conflicts(obs)) +
                             ". I'll look for the earliest slot that works for everyone instead.",
                             [Call("calendar_find_free_slot", {"attendees": att, "date": day, "duration_minutes": dur})]))

        def final(obs):
            c = conflicts(obs)
            sl = _slot(obs)
            alt_part = f"The earliest slot that works for everyone is {sl['start']}-{sl['end']}." if sl else "There is no common free slot that day."
            return (f"{s_txt} doesn't work: " + "; ".join(f"{p} is busy with {h[0]['title']} ({h[0]['start']}-{h[0]['end']})" for p, h in c) +
                    f". I didn't book anything. {alt_part} Want me to book that instead?")

        grounded = [alt_txt] if alt_txt else []
    exp_conf = []
    if not free_case:
        exp_conf = [p for p in att if any(start < hm(e["end"]) and hm(e["start"]) < start + dur for e in w["events"] if e["date"] == day and p in e["attendees"])]

    def check(api, replies):
        evs = [e for e in api.events.values() if e["title"] == title]
        if free_case:
            return (len(evs) == 1 and evs[0]["start"] == s_txt and sorted(evs[0]["attendees"]) == sorted(att)), "should have booked"
        if evs:
            return False, "must not book when there is a conflict"
        return all(p in replies[-1] for p in exp_conf), "reply must name the conflicting people"

    tools = pick_tools(rng, ["calendar_list_events", "calendar_create_event", "calendar_find_free_slot"], extra=(2, 5))
    return ApiTask("api_conflict_check", "api_calendar", "hard", w, tools, [Turn(frame(rng, body), steps, final, grounded)], check,
                   ["parallel_calls", "conditional_logic", "conflict" if not free_case else "no_conflict"])


# ===================================================================================================
@family("api_meeting_prep", "api_calendar", 2.5)
def api_meeting_prep(rng):
    w, att, dur, slot = world_with_slot(rng, n_att=(2, 3))
    day, title = w["day"], rng.choice(MEETING_TITLES)
    s_txt, e_txt, who = fmt_hhmm(slot), fmt_hhmm(slot + dur), names_phrase(att)
    emails = [email_of(w, a) for a in att]
    body = (f"Set up '{title}' ({dur} min) on {day} with {who}: book the earliest slot when they are all free, write an invitation email draft to them "
            f"(don't send it), and add a task for me called 'Prepare agenda for {title}' due on {day}.")
    emails_from = lambda obs: [json.loads(o)["contact"]["email"] for o in obs if o.startswith('{"contact"')]
    steps = [ApiStep("I need the attendees' email addresses and a common free slot. These lookups don't depend on each other.",
                     [Call("contacts_lookup", {"name": a}) for a in att] + [Call("calendar_find_free_slot", {"attendees": att, "date": day, "duration_minutes": dur})]),
             ApiStep(lambda obs: f"The earliest common slot is {_slot(obs)['start']}-{_slot(obs)['end']}. I'll book it.",
                     [Call("calendar_create_event", lambda obs: {"title": title, "date": day, "start": _slot(obs)["start"], "duration_minutes": dur, "attendees": att})]),
             ApiStep("The event exists. Now I'll draft the invitation and add the agenda task at the same time.",
                     [Call("email_draft", lambda obs: {"to": emails_from(obs), "subject": f"Invitation: {title}",
                                                       "body": f"Hi,\n\nYou're invited to '{title}' on {day} from {_slot(obs)['start']} to {_slot(obs)['end']}.\n\nBest regards"}),
                      Call("tasks_add", {"title": f"Prepare agenda for {title}", "due": day})])]
    final = (f"All set: '{title}' is booked on {day} {s_txt}-{e_txt} with {who}; an invitation draft to {', '.join(emails)} is saved (not sent), "
             f"and the task 'Prepare agenda for {title}' is due {day}.")

    def check(api, replies):
        evs = [e for e in api.events.values() if e["title"] == title and e["date"] == day]
        if len(evs) != 1 or evs[0]["start"] != s_txt or sorted(evs[0]["attendees"]) != sorted(att):
            return False, "event wrong"
        ds = list(api.drafts.values())
        if len(ds) != 1 or sorted(ds[0]["to"]) != sorted(emails) or title not in ds[0]["subject"] or s_txt not in ds[0]["body"] or day not in ds[0]["body"]:
            return False, "draft wrong"
        if api.sent:
            return False, "must not send"
        t = [x for x in api.tasks if x["title"] == f"Prepare agenda for {title}"]
        return (len(t) == 1 and t[0]["due"] == day), "task wrong"

    tools = pick_tools(rng, ["contacts_lookup", "calendar_find_free_slot", "calendar_create_event", "email_draft", "tasks_add"], extra=(1, 4))
    return ApiTask("api_meeting_prep", "api_calendar", "hard", w, tools, [Turn(frame(rng, body), steps, final, [s_txt, e_txt])], check,
                   ["tool_chain", "parallel_calls", "multi_tool_compound", "draft_not_send", "cross_application"])


# ===================================================================================================
@family("api_followup_adjust", "api_calendar", 2.0)
def api_followup_adjust(rng):
    for _attempt in range(60):                       # regenerate the whole world until the follow-up is feasible too
        w, att, dur, slot = world_with_slot(rng, n_att=(2, 3), busy=(1, 3))
        day, title = w["day"], rng.choice(MEETING_TITLES)
        s1, who = fmt_hhmm(slot), names_phrase(att)
        e1 = {"id": "evt_new", "title": title, "date": day, "start": s1, "end": fmt_hhmm(slot + dur), "attendees": att}
        pool = [c["name"] for c in w["contacts"] if c["name"] not in att]
        found = False
        for _ in range(12):
            p_new, dur2 = rng.choice(pool), rng.choice([d for d in (30, 45, 60) if d != dur])
            att2 = att + [p_new]
            slot2 = free_start(w["events"] + [e1], att2, day, dur2)
            if slot2 is not None:
                found = True
                break
        if found:
            break
    else:
        raise RuntimeError("no feasible follow-up")
    s2 = fmt_hhmm(slot2)
    t1 = Turn(frame(rng, f"Book '{title}' on {day} for {dur} minutes with {who} at the earliest time they're all free."),
              [ApiStep(pick(rng, PLAN_FIND), [Call("calendar_find_free_slot", {"attendees": att, "date": day, "duration_minutes": dur})]),
               ApiStep(lambda obs: f"The earliest common slot is {_slot(obs)['start']}-{_slot(obs)['end']}. Creating the event.",
                       [Call("calendar_create_event", lambda obs: {"title": title, "date": day, "start": _slot(obs)["start"], "duration_minutes": dur, "attendees": att})])],
              f"Booked '{title}' on {day} from {s1} to {fmt_hhmm(slot + dur)} with {who}.", [s1])
    ev_id = lambda obs: jfind(obs, "event")["event"]["id"]
    new_slot = lambda obs: json.loads([o for o in obs if o.startswith('{"slot"')][-1])["slot"]
    t2 = Turn(pick(rng, [f"Actually, please add {p_new} as well and make it {dur2} minutes. Keep it on the same day at the earliest time that works for everyone.",
                         f"Change of plans: {p_new} needs to join too, and we need {dur2} minutes. Same day, earliest time everyone is free."]),
              [ApiStep(f"With {p_new} added and {dur2} minutes, I need a new common slot.", [Call("calendar_find_free_slot", {"attendees": att2, "date": day, "duration_minutes": dur2})]),
               ApiStep(lambda obs: f"The earliest slot for everyone is {new_slot(obs)['start']}-{new_slot(obs)['end']}. I'll update the existing event.",
                       [Call("calendar_update_event", lambda obs: {"event_id": ev_id(obs), "start": new_slot(obs)["start"], "duration_minutes": dur2, "attendees": att2})])],
              f"Updated '{title}': {p_new} is added and it now runs {s2}-{fmt_hhmm(slot2 + dur2)} on {day}.", [s2])

    def check(api, replies):
        evs = [e for e in api.events.values() if e["title"] == title]
        if len(evs) != 1:
            return False, "must update the existing event, not create another"
        e = evs[0]
        return (e["start"] == s2 and hm(e["end"]) - hm(e["start"]) == dur2 and sorted(e["attendees"]) == sorted(att2)), "event not updated correctly"

    tools = pick_tools(rng, ["calendar_find_free_slot", "calendar_create_event", "calendar_update_event"], extra=(2, 5))
    return ApiTask("api_followup_adjust", "api_calendar", "hard", w, tools, [t1, t2], check, ["multi_turn", "tool_chain", "update_existing", "context_carryover"])


# ===================================================================================================
@family("api_today_agenda", "api_calendar", 1.6)
def api_today_agenda(rng):
    w = make_world(rng, n_people=(3, 5))
    w["today"] = w["day"]                                           # "today" is the day that has events
    me = rng.choice([c["name"] for c in w["contacts"]])
    day = w["day"]
    from datetime import date as _date
    weekday = _date.fromisoformat(day).strftime("%A")
    body = pick(rng, [f"I'm {me}. What do I have on my calendar today?", f"This is {me}. Show me today's schedule.", f"{me} here - what's on today?"])
    steps = [ApiStep("I don't know today's date, so I'll ask the date tool first.", [Call("get_current_date", {})]),
             ApiStep(lambda obs: f"Today is {jfind(obs, 'weekday')['weekday']}, {jfind(obs, 'date')['date']}. Now I'll fetch your events for that date.",
                     [Call("calendar_list_events", lambda obs: {"person": me, "date": jfind(obs, "date")["date"]})])]

    def final(obs):
        evs = jfind(obs, "events")["events"]
        listing = "; ".join(f"{e['start']}-{e['end']} {e['title']}" for e in evs) or "nothing scheduled"
        return f"Today is {weekday}, {day}. Your schedule: {listing}."

    expected = sorted((e["start"], e["title"]) for e in w["events"] if e["date"] == day and me in e["attendees"])

    def check(api, replies):
        r = replies[-1]
        return (weekday in r and all(s in r and t in r for s, t in expected)), "agenda wrong"

    return ApiTask("api_today_agenda", "api_calendar", "easy", w, pick_tools(rng, ["get_current_date", "calendar_list_events"]), [Turn(frame(rng, body), steps, final, [day, weekday])], check,
                   ["tool_chain", "current_date", "tool_selection"])


# ===================================================================================================
@family("api_cancel_event", "api_calendar", 1.8)
def api_cancel_event(rng):
    w, att, dur, _ = world_with_slot(rng, n_att=(2, 3), want_late=False)
    day, title, me = w["day"], rng.choice(MEETING_TITLES), att[0]
    s0 = rng.choice(range(10 * 60, 15 * 60, 15))
    w["events"].append({"id": "evt_90", "title": title, "date": day, "start": fmt_hhmm(s0), "end": fmt_hhmm(s0 + dur), "attendees": att})
    notify = rng.random() < 0.5
    others = att[1:]
    ev_id = lambda obs: next(e["id"] for e in jfind(obs, "events")["events"] if e["title"] == title)
    if notify:
        body = pick(rng, [f"I'm {me}. Cancel '{title}' on {day} and draft a short email to the other attendees ({names_phrase(others)}) letting them know. Don't send it.",
                          f"This is {me}: please cancel my '{title}' meeting on {day} and prepare (but don't send) a note to {names_phrase(others)} about the cancellation."])
        steps = [ApiStep("I need the event id and the other attendees' email addresses; these lookups are independent.",
                         [Call("calendar_list_events", {"person": me, "date": day})] + [Call("contacts_lookup", {"name": o}) for o in others]),
                 ApiStep(lambda obs: f"'{title}' is event {ev_id(obs)}. I'll cancel it and write the notification draft.",
                         [Call("calendar_delete_event", lambda obs: {"event_id": ev_id(obs)}),
                          Call("email_draft", lambda obs: {"to": [json.loads(o)["contact"]["email"] for o in obs if o.startswith('{"contact"')], "subject": f"Cancelled: {title}",
                                                           "body": f"Hi,\n\n'{title}' on {day} has been cancelled.\n\nBest regards"})])]
        final = f"I cancelled '{title}' on {day} and saved a draft to {names_phrase(others)} (not sent)."
    else:
        body = pick(rng, [f"I'm {me}. Cancel my '{title}' meeting on {day}.", f"This is {me}. Please delete '{title}' from the calendar on {day}."])
        steps = [ApiStep(pick(rng, PLAN_LIST), [Call("calendar_list_events", {"person": me, "date": day})]),
                 ApiStep(lambda obs: f"'{title}' is event {ev_id(obs)}. I'll delete it.", [Call("calendar_delete_event", lambda obs: {"event_id": ev_id(obs)})])]
        final = f"Done - '{title}' on {day} is cancelled."

    def check(api, replies):
        if "evt_90" in api.events:
            return False, "event not deleted"
        if not notify:
            return not api.drafts, "no draft expected"
        ds = list(api.drafts.values())
        want = sorted(email_of(w, o) for o in others)
        return (len(ds) == 1 and sorted(ds[0]["to"]) == want and title in ds[0]["subject"] and not api.sent), "notification wrong"

    req = ["calendar_list_events", "calendar_delete_event"] + (["contacts_lookup", "email_draft"] if notify else [])
    return ApiTask("api_cancel_event", "api_calendar", "medium", w, pick_tools(rng, req), [Turn(frame(rng, body), steps, final, [])], check,
                   ["tool_chain", "delete_event", "safety_only_what_asked"] + (["parallel_calls", "draft_not_send"] if notify else []))
