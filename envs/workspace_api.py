"""Simulated workspace service used to generate FUNCTION-CALLING data: calendar, contacts, email, tasks, files, utilities.

Each tool is a typed function with a JSON schema (TOOL_DEFS -> tool_schema). A task gets a deterministic world (dict) and a
WorkspaceAPI built from it; every call returns a JSON string, or {"error": ..., "code": ...} for real API errors (not found,
invalid arguments, ...), so error-handling behaviour is learned from real error responses, not invented ones.
The API keeps a log of calls and the final state, which the task checkers inspect.
"""
from __future__ import annotations

import ast
import copy
import csv
import io
import json
import operator
import re
from datetime import date

LO, HI = 9 * 60, 17 * 60

# ------------------------------------------------------------------------------------------------ tool definitions
TOOL_DEFS: dict = {}


def _tool(name, description, params):
    TOOL_DEFS[name] = {"description": description, "params": params}


S, I, N, B, A = "string", "integer", "number", "boolean", "array"

_tool("list_files", "List the files and folders directly inside a workspace folder (folders end with '/').",
      [("path", S, "Folder to list, relative to the workspace root. Use '.' for the root.", False, {})])
_tool("read_file", "Read a text file from the workspace and return its content.",
      [("path", S, "File path relative to the workspace root.", True, {})])
_tool("write_file", "Create or overwrite a text file in the workspace.",
      [("path", S, "File path relative to the workspace root.", True, {}), ("content", S, "Full text content to write.", True, {})])
_tool("search_files", "Search all workspace files for lines that contain every word of the query (case-insensitive).",
      [("query", S, "Words that must all appear in a line.", True, {}), ("path", S, "Folder to search in. Defaults to '.'.", False, {})])
_tool("delete_file", "Permanently delete a file from the workspace.", [("path", S, "File path to delete.", True, {})])
_tool("calendar_list_events", "List one person's calendar events on a given date, sorted by start time.",
      [("person", S, "Full name of the person.", True, {}), ("date", S, "Date as YYYY-MM-DD.", True, {})])
_tool("calendar_find_free_slot", "Find the earliest time slot on a date when all listed people are free.",
      [("attendees", A, "Full names of everyone who must be free.", True, {"items": S}), ("date", S, "Date as YYYY-MM-DD.", True, {}),
       ("duration_minutes", I, "Length of the slot in minutes.", True, {}),
       ("earliest", S, "Earliest allowed start, HH:MM (24h). Defaults to 09:00.", False, {}),
       ("latest", S, "Latest allowed end, HH:MM (24h). Defaults to 17:00.", False, {})])
_tool("calendar_create_event", "Create a calendar event for the given attendees.",
      [("title", S, "Event title.", True, {}), ("date", S, "Date as YYYY-MM-DD.", True, {}), ("start", S, "Start time, HH:MM (24h).", True, {}),
       ("duration_minutes", I, "Length in minutes.", True, {}), ("attendees", A, "Full names of the attendees.", True, {"items": S})])
_tool("calendar_update_event", "Change an existing calendar event. Only the fields you pass are changed.",
      [("event_id", S, "ID of the event.", True, {}), ("date", S, "New date, YYYY-MM-DD.", False, {}), ("start", S, "New start time, HH:MM.", False, {}),
       ("duration_minutes", I, "New length in minutes.", False, {}), ("title", S, "New title.", False, {}),
       ("attendees", A, "New full list of attendees (replaces the old list).", False, {"items": S})])
_tool("calendar_delete_event", "Delete a calendar event.", [("event_id", S, "ID of the event to delete.", True, {})])
_tool("contacts_lookup", "Look up one contact by exact full name and return their email address and team.",
      [("name", S, "Exact full name of the contact.", True, {})])
_tool("contacts_search", "Search contacts by part of a name, email address or team.", [("query", S, "Text to search for.", True, {})])
_tool("email_list_inbox", "List messages in the inbox (id, sender, subject, unread flag).",
      [("unread_only", B, "Only return unread messages. Defaults to false.", False, {}), ("limit", I, "Maximum number of messages. Defaults to 20.", False, {})])
_tool("email_read", "Read one email by id and mark it as read.", [("message_id", S, "ID of the message.", True, {})])
_tool("email_draft", "Create an email draft. It is NOT sent until email_send is called.",
      [("to", A, "Recipient email addresses.", True, {"items": S}), ("subject", S, "Subject line.", True, {}), ("body", S, "Email body text.", True, {})])
_tool("email_send", "Send a previously created draft. This cannot be undone.", [("draft_id", S, "ID of the draft to send.", True, {})])
_tool("tasks_add", "Add a task to the to-do list.",
      [("title", S, "Task title.", True, {}), ("owner", S, "Full name of the person responsible.", False, {}), ("due", S, "Due date as YYYY-MM-DD.", False, {})])
_tool("tasks_list", "List tasks, optionally filtered by status or owner.",
      [("status", S, "Filter by status.", False, {"enum": ["open", "done"]}), ("owner", S, "Filter by owner's full name.", False, {})])
_tool("tasks_complete", "Mark a task as done.", [("task_id", S, "ID of the task.", True, {})])
_tool("get_current_date", "Return today's date and weekday.", [])
_tool("calculator", "Evaluate an arithmetic expression (+, -, *, /, **, %, parentheses) exactly.",
      [("expression", S, "Arithmetic expression, e.g. '(120 - 45) / 120 * 100'.", True, {})])
_tool("table_aggregate", "Aggregate a column of a CSV file: sum, avg, min, max or count, optionally filtered and grouped.",
      [("path", S, "CSV file path.", True, {}), ("column", S, "Column to aggregate (ignored for count).", True, {}),
       ("op", S, "Aggregation to apply.", True, {"enum": ["sum", "avg", "min", "max", "count"]}),
       ("group_by", S, "Column to group by (optional).", False, {}),
       ("where_column", S, "Only rows where this column equals where_value (optional).", False, {}),
       ("where_value", S, "Value for where_column (optional).", False, {})])


def tool_schema(name: str) -> dict:
    d = TOOL_DEFS[name]
    props, req = {}, []
    for pname, ptype, desc, required, extra in d["params"]:
        p = {"type": ptype, "description": desc}
        if ptype == A:
            p["items"] = {"type": extra.get("items", S)}
        if "enum" in extra:
            p["enum"] = extra["enum"]
        props[pname] = p
        if required:
            req.append(pname)
    return {"type": "function", "function": {"name": name, "description": d["description"],
                                              "parameters": {"type": "object", "properties": props, "required": req}}}


# ------------------------------------------------------------------------------------------------ the service
class ApiError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _date(s) -> str:
    try:
        date.fromisoformat(s)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ValueError
        return s
    except (ValueError, TypeError):
        raise ApiError("invalid_argument", f"date must be YYYY-MM-DD, got {s!r}")


def _hm(s) -> int:
    if not isinstance(s, str) or not re.fullmatch(r"\d{2}:\d{2}", s) or int(s[:2]) > 23 or int(s[3:]) > 59:
        raise ApiError("invalid_argument", f"time must be HH:MM in 24-hour format, got {s!r}")
    return int(s[:2]) * 60 + int(s[3:])


def _fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ApiError("invalid_expression", "only numbers and + - * / ** % ( ) are allowed")


class WorkspaceAPI:
    def __init__(self, world: dict):
        w = copy.deepcopy(world)
        self.files: dict = w.get("files", {})
        self.contacts: list = w.get("contacts", [])
        self.events: dict = {e["id"]: e for e in w.get("events", [])}
        self.inbox: list = w.get("inbox", [])
        self.tasks: list = w.get("tasks", [])
        self.today: str = w.get("today", "2025-03-03")
        self.drafts: dict = {}
        self.sent: list = []
        self.log: list = []                       # (tool, args, ok)
        self._n = {"evt": 100, "d": 0, "t": 100}

    # ---- dispatch
    def call(self, name: str, args: dict) -> str:
        try:
            if name not in TOOL_DEFS:
                raise ApiError("unknown_tool", f"no such tool: {name}")
            self._validate(name, args)
            out = getattr(self, name)(**args)
            self.log.append((name, args, True))
            return json.dumps(out, ensure_ascii=False)
        except ApiError as e:
            self.log.append((name, args, False))
            return json.dumps({"error": e.message, "code": e.code}, ensure_ascii=False)
        except Exception as e:                    # a model can send arguments that pass validation but still break a tool (e.g. sum over a text column)
            self.log.append((name, args, False))
            return json.dumps({"error": f"{name} failed: {type(e).__name__}: {str(e)[:200]}", "code": "invalid_argument"}, ensure_ascii=False)

    def _validate(self, name, args):
        if not isinstance(args, dict):
            raise ApiError("invalid_arguments", "arguments must be an object")
        spec = {p[0]: p for p in TOOL_DEFS[name]["params"]}
        for k in args:
            if k not in spec:
                raise ApiError("invalid_arguments", f"unknown argument {k!r}")
        for pname, ptype, _, required, extra in TOOL_DEFS[name]["params"]:
            if required and pname not in args:
                raise ApiError("invalid_arguments", f"missing required argument {pname!r}")
            if pname in args:
                v, ok = args[pname], True
                ok = {S: isinstance(v, str), I: isinstance(v, int) and not isinstance(v, bool), B: isinstance(v, bool),
                      A: isinstance(v, list), N: isinstance(v, (int, float))}[ptype]
                if not ok:
                    raise ApiError("invalid_arguments", f"argument {pname!r} must be of type {ptype}")
                if "enum" in extra and v not in extra["enum"]:
                    raise ApiError("invalid_arguments", f"argument {pname!r} must be one of {extra['enum']}")

    # ---- helpers
    def _person(self, name: str) -> str:
        if not any(c["name"] == name for c in self.contacts):
            raise ApiError("not_found", f"unknown person: {name}")
        return name

    def _busy(self, people, day):
        return [(_hm(e["start"]), _hm(e["end"])) for e in self.events.values() if e["date"] == day and set(e["attendees"]) & set(people)]

    # ---- files
    def _norm(self, p: str) -> str:
        p = p.strip()
        while p.startswith("./"):
            p = p[2:]
        return p.rstrip("/") or "."

    def list_files(self, path: str = "."):
        p = self._norm(path)
        prefix = "" if p == "." else p + "/"
        entries = set()
        for f in self.files:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                entries.add(rest.split("/")[0] + ("/" if "/" in rest else ""))
        if not entries:
            raise ApiError("not_found", f"No such directory: {path}")
        return {"path": path, "entries": sorted(entries)}

    def read_file(self, path: str):
        p = self._norm(path)
        if p not in self.files:
            raise ApiError("not_found", f"No such file: {path}")
        text = self.files[p]
        return {"path": p, "content": text[:4000], "truncated": len(text) > 4000}

    def write_file(self, path: str, content: str):
        self.files[self._norm(path)] = content
        return {"path": self._norm(path), "bytes": len(content.encode())}

    def search_files(self, query: str, path: str = "."):
        words = query.lower().split()
        p = self._norm(path)
        prefix = "" if p == "." else p + "/"
        matches = []
        for f in sorted(self.files):
            if f.startswith(prefix):
                for i, line in enumerate(self.files[f].splitlines(), 1):
                    if words and all(w in line.lower() for w in words):
                        matches.append({"path": f, "line": i, "text": line.strip()[:200]})
        return {"matches": matches[:10], "truncated": len(matches) > 10}

    def delete_file(self, path: str):
        p = self._norm(path)
        if p not in self.files:
            raise ApiError("not_found", f"No such file: {path}")
        del self.files[p]
        return {"deleted": p}

    # ---- calendar
    def calendar_list_events(self, person: str, date: str):
        self._person(person)
        d = _date(date)
        evs = sorted((e for e in self.events.values() if e["date"] == d and person in e["attendees"]), key=lambda e: (e["start"], e["end"], e["title"]))
        return {"person": person, "date": d, "events": [{"id": e["id"], "title": e["title"], "start": e["start"], "end": e["end"]} for e in evs]}

    def calendar_find_free_slot(self, attendees, date, duration_minutes, earliest="09:00", latest="17:00"):
        for a in attendees:
            self._person(a)
        d, lo, hi = _date(date), _hm(earliest), _hm(latest)
        if duration_minutes <= 0:
            raise ApiError("invalid_argument", "duration_minutes must be positive")
        busy = self._busy(attendees, d)
        for start in range(lo, hi - duration_minutes + 1):
            if all(not (start < e and s < start + duration_minutes) for s, e in busy):
                return {"slot": {"date": d, "start": _fmt(start), "end": _fmt(start + duration_minutes)}}
        return {"slot": None, "reason": "no common free slot in that window"}

    def calendar_create_event(self, title, date, start, duration_minutes, attendees):
        for a in attendees:
            self._person(a)
        d, s = _date(date), _hm(start)
        if duration_minutes <= 0 or s + duration_minutes > 24 * 60:
            raise ApiError("invalid_argument", "event must end before midnight")
        busy_people = [a for a in attendees if any(s < be and bs < s + duration_minutes for bs, be in self._busy([a], d))]
        self._n["evt"] += 1
        ev = {"id": f"evt_{self._n['evt']}", "title": title, "date": d, "start": _fmt(s), "end": _fmt(s + duration_minutes), "attendees": list(attendees)}
        self.events[ev["id"]] = ev
        return {"event": ev, "conflicts": busy_people}

    def calendar_update_event(self, event_id, date=None, start=None, duration_minutes=None, title=None, attendees=None):
        if event_id not in self.events:
            raise ApiError("not_found", f"No such event: {event_id}")
        ev = dict(self.events[event_id])
        if date is not None:
            ev["date"] = _date(date)
        if title is not None:
            ev["title"] = title
        if attendees is not None:
            for a in attendees:
                self._person(a)
            ev["attendees"] = list(attendees)
        s = _hm(start) if start is not None else _hm(ev["start"])
        dur = duration_minutes if duration_minutes is not None else _hm(ev["end"]) - _hm(ev["start"])
        if dur <= 0 or s + dur > 24 * 60:
            raise ApiError("invalid_argument", "event must end before midnight")
        ev["start"], ev["end"] = _fmt(s), _fmt(s + dur)
        self.events[event_id] = ev
        return {"event": ev}

    def calendar_delete_event(self, event_id):
        if event_id not in self.events:
            raise ApiError("not_found", f"No such event: {event_id}")
        del self.events[event_id]
        return {"deleted": event_id}

    # ---- contacts
    def contacts_lookup(self, name):
        for c in self.contacts:
            if c["name"].lower() == name.lower():
                return {"contact": c}
        raise ApiError("not_found", f"No contact named {name!r}. Try contacts_search.")

    def contacts_search(self, query):
        q = query.lower()
        return {"results": [c for c in self.contacts if q in c["name"].lower() or q in c["email"].lower() or q in c.get("team", "").lower()]}

    # ---- email
    def email_list_inbox(self, unread_only=False, limit=20):
        msgs = [m for m in self.inbox if m["unread"] or not unread_only]
        return {"messages": [{"id": m["id"], "from": m["from"], "subject": m["subject"], "unread": m["unread"]} for m in msgs[:limit]]}

    def email_read(self, message_id):
        for m in self.inbox:
            if m["id"] == message_id:
                m["unread"] = False
                return {"message": {k: m[k] for k in ("id", "from", "subject", "body")}}
        raise ApiError("not_found", f"No such message: {message_id}")

    def email_draft(self, to, subject, body):
        if not to or not all(isinstance(t, str) and "@" in t for t in to):
            raise ApiError("invalid_argument", "to must be a non-empty list of email addresses")
        self._n["d"] += 1
        did = f"draft_{self._n['d']}"
        self.drafts[did] = {"to": list(to), "subject": subject, "body": body}
        return {"draft_id": did}

    def email_send(self, draft_id):
        if draft_id not in self.drafts:
            raise ApiError("not_found", f"No such draft: {draft_id}")
        self.sent.append(self.drafts.pop(draft_id))
        return {"sent": True, "to": self.sent[-1]["to"]}

    # ---- tasks
    def tasks_add(self, title, owner=None, due=None):
        if owner is not None:
            self._person(owner)
        if due is not None:
            _date(due)
        self._n["t"] += 1
        t = {"id": f"task_{self._n['t']}", "title": title, "owner": owner, "due": due, "status": "open"}
        self.tasks.append(t)
        return {"task": t}

    def tasks_list(self, status=None, owner=None):
        return {"tasks": [t for t in self.tasks if (status is None or t["status"] == status) and (owner is None or t["owner"] == owner)]}

    def tasks_complete(self, task_id):
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = "done"
                return {"task": t}
        raise ApiError("not_found", f"No such task: {task_id}")

    # ---- utilities
    def get_current_date(self):
        d = date.fromisoformat(self.today)
        return {"date": self.today, "weekday": d.strftime("%A")}

    def calculator(self, expression):
        try:
            v = _eval(ast.parse(expression, mode="eval"))
        except ApiError:
            raise
        except ZeroDivisionError:
            raise ApiError("invalid_expression", "division by zero")
        except Exception:
            raise ApiError("invalid_expression", "could not parse the expression")
        return {"result": int(v) if isinstance(v, float) and v.is_integer() else round(v, 6)}

    def table_aggregate(self, path, column, op, group_by=None, where_column=None, where_value=None):
        p = self._norm(path)
        if p not in self.files:
            raise ApiError("not_found", f"No such file: {path}")
        rows = list(csv.DictReader(io.StringIO(self.files[p])))
        cols = list(rows[0].keys()) if rows else []
        for c in (column if op != "count" else None, group_by, where_column):
            if c is not None and c not in cols:
                raise ApiError("invalid_argument", f"unknown column {c!r}; columns are {cols}")
        if where_column is not None:
            rows = [r for r in rows if r[where_column] == where_value]

        def num(r):
            return float(r[column].replace("$", "").replace(",", ""))

        def agg(rs):
            if op == "count":
                return len(rs)
            vals = [num(r) for r in rs]
            if not vals:
                return None
            v = {"sum": sum(vals), "avg": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}[op]
            return round(v, 2)

        if group_by is None:
            return {"result": agg(rows), "rows": len(rows)}
        groups = {}
        for r in rows:
            groups.setdefault(r[group_by], []).append(r)
        return {"groups": {k: agg(v) for k, v in sorted(groups.items())}}
