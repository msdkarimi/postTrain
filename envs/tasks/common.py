"""Shared random data pools and file-rendering helpers for the task generators."""
from __future__ import annotations

import csv
import io
import json
import random
from datetime import date, timedelta

FIRST = ["Alice", "Marco", "Priya", "Jonas", "Sofia", "Liam", "Aisha", "Diego", "Mei", "Oscar", "Nadia", "Ethan", "Yara",
         "Hugo", "Lena", "Omar", "Chloe", "Ravi", "Elena", "Tomas", "Ingrid", "Kofi", "Zara", "Felix", "Amira", "Noah",
         "Ines", "Kenji", "Maya", "Victor", "Hana", "Leo", "Sara", "Pavel", "Irene", "Dmitri", "Bianca", "Arjun", "Nora", "Theo"]
LAST = ["Rossi", "Nguyen", "Patel", "Schmidt", "Garcia", "Okafor", "Tanaka", "Silva", "Kowalski", "Haddad", "Larsen", "Moreau",
        "Kim", "Fischer", "Costa", "Ivanov", "Novak", "Bauer", "Santos", "Lindgren", "Mendez", "Dubois", "Rahman", "Weber",
        "Andersson", "Ferrari", "Popov", "Hughes", "Yilmaz", "Berg"]
COMPANIES = ["Acme Corp", "Northwind", "Globex", "Initech", "Umbrella Ltd", "Hooli", "Stark Industries", "Wayne Enterprises",
             "Soylent", "Vandelay Imports", "Wonka Foods", "Cyberdyne", "Tyrell Labs", "Pied Piper"]
PRODUCTS = ["Laptop", "Monitor", "Keyboard", "Mouse", "Webcam", "Headset", "Docking Station", "Desk Lamp", "Notebook",
            "Standing Desk", "Office Chair", "USB Hub", "Printer", "Router", "Tablet", "Speaker"]
CATEGORIES = ["Hardware", "Software", "Services", "Training", "Support", "Accessories", "Licenses", "Consulting"]
REGIONS = ["North", "South", "East", "West", "Central"]
CITIES = ["Turin", "Zurich", "Lisbon", "Oslo", "Krakow", "Lyon", "Porto", "Graz", "Ghent", "Malmo", "Utrecht", "Bilbao"]
PROJECTS = ["Apollo", "Borealis", "Cobalt", "Delta", "Everest", "Falcon", "Granite", "Horizon", "Ion", "Juniper"]
DOMAINS = ["acme.com", "northwind.io", "globex.net", "initech.org", "hooli.dev", "umbrella.co"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def people(rng: random.Random, n: int) -> list:
    names, seen = [], set()
    while len(names) < n:
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    return names


def email_for(name: str, domain: str) -> str:
    first, last = name.lower().split(" ", 1)
    return f"{first}.{last.replace(' ', '')}@{domain}"


def rand_date(rng: random.Random, year: int, month: int | None = None) -> date:
    if month is None:
        month = rng.randint(1, 12)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    days = (nxt - date(year, month, 1)).days
    return date(year, month, rng.randint(1, days))


def to_csv(header: list, rows: list, delim: str = ",") -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def to_json(obj) -> str:
    return json.dumps(obj, indent=2) + "\n"


def delim_name(d: str) -> str:
    return {",": "commas", ";": "semicolons", "\t": "tabs"}[d]


DISTRACTORS = {
    "README.txt": "Exports are refreshed every Monday.\nContact the data team for access questions.\n",
    "notes.md": "# Notes\n- remember to archive last quarter's files\n- ask finance about the new template\n",
    "old_export_2023.csv": "id,value\n1,10\n2,20\n3,30\n",
    ".hidden_cache": "cache-v2\n",
    "todo.txt": "buy coffee\nreview slides\nbook room for Thursday\n",
}


def add_distractors(rng: random.Random, files: dict) -> dict:
    out = dict(files)
    for name in rng.sample(list(DISTRACTORS), rng.choice([0, 0, 1, 1, 2, 3])):
        if name not in out:
            out[name] = DISTRACTORS[name]
    return out


def fmt_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def daterange_label(d: date) -> str:
    return d.isoformat()
