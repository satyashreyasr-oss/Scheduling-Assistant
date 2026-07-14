"""Mocked-but-functional tools for the Booking Specialist agent.

- resolve_date: turns relative expressions ("tomorrow", "next Friday") into an
  absolute YYYY-MM-DD, anchored to the real current date.
- check_availability: reads/writes an in-memory-style schedule backed by SQLite.
- reserve_slot: books a slot, rejecting collisions so the agent can negotiate.
- send_booking_notification: posts a mock confirmation to a webhook (or
  simulates it locally if no webhook is configured).
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta

import dateparser
import requests
from langchain_core.tools import tool

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_WEEKDAY_PHRASE_RE = re.compile(
    r"^\s*(next|this|coming)?\s*(" + "|".join(_WEEKDAYS) + r")\s*$", re.IGNORECASE
)


def _resolve_weekday_phrase(expression: str, base: datetime):
    """Fallback for phrases like 'next Monday' / 'this Friday' / 'Monday' --
    some dateparser versions fail to parse these reliably (observed with
    'next <weekday>' in this project's environment), so resolve them
    deterministically instead of guessing with an LLM."""
    match = _WEEKDAY_PHRASE_RE.match(expression)
    if not match:
        return None
    modifier = (match.group(1) or "").lower()
    target_weekday = _WEEKDAYS[match.group(2).lower()]
    days_ahead = (target_weekday - base.weekday()) % 7
    if days_ahead == 0 and modifier != "this":
        days_ahead = 7
    return base + timedelta(days=days_ahead)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scheduling.db")

BUSINESS_HOURS = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00", "17:00"]


def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            email TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(date, time)
        )
        """
    )
    conn.commit()
    conn.close()


def _booked_times(date: str):
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT time FROM bookings WHERE date = ?", (date,)).fetchall()
    conn.close()
    return {r[0] for r in rows}


def _free_slots(date: str):
    dt = datetime.strptime(date, "%Y-%m-%d")
    if dt.weekday() >= 5:  # Saturday/Sunday closed
        return []
    booked = _booked_times(date)
    return [t for t in BUSINESS_HOURS if t not in booked]


def _next_available(start_date: str, needed: int = 3):
    """Scan forward from start_date for up to `needed` open date/time slots."""
    suggestions = []
    d = datetime.strptime(start_date, "%Y-%m-%d")
    days_checked = 0
    while len(suggestions) < needed and days_checked < 14:
        ds = d.strftime("%Y-%m-%d")
        for t in _free_slots(ds):
            suggestions.append(f"{ds} {t}")
            if len(suggestions) >= needed:
                break
        d += timedelta(days=1)
        days_checked += 1
    return suggestions


@tool
def resolve_date(expression: str) -> str:
    """Resolve a natural-language date expression (e.g. 'tomorrow', 'next Monday',
    'July 20') into an absolute YYYY-MM-DD date string, anchored to the real
    current date/time. ALWAYS call this before check_availability or reserve_slot
    whenever the user gives a relative date instead of an explicit YYYY-MM-DD."""
    base = datetime.now()
    parsed = _resolve_weekday_phrase(expression, base) or dateparser.parse(
        expression,
        languages=["en"],
        settings={"RELATIVE_BASE": base, "PREFER_DATES_FROM": "future"},
    )
    if not parsed:
        return json.dumps(
            {
                "ok": False,
                "error": f"Could not understand the date '{expression}'. "
                "Ask the user to clarify (e.g. 'next Friday' or '2026-07-20').",
            }
        )
    return json.dumps(
        {"ok": True, "resolved_date": parsed.strftime("%Y-%m-%d"), "today": base.strftime("%Y-%m-%d")}
    )


@tool
def check_availability(date: str) -> str:
    """Check open appointment slots for a given YYYY-MM-DD date. Business hours
    are 09:00-12:00 and 14:00-17:00, Monday-Friday, hourly slots. Returns the
    free slots, or alternative dates/times if the day is fully booked or falls
    on a weekend."""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"ok": False, "error": "date must be in YYYY-MM-DD format. Call resolve_date first."})

    if dt.weekday() >= 5:
        return json.dumps(
            {
                "ok": True,
                "date": date,
                "free_slots": [],
                "note": "closed on weekends",
                "alternatives": _next_available(date),
            }
        )

    free = _free_slots(date)
    result = {"ok": True, "date": date, "free_slots": free}
    if not free:
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        result["note"] = "fully booked"
        result["alternatives"] = _next_available(next_day)
    return json.dumps(result)


@tool
def reserve_slot(date: str, time: str, email: str) -> str:
    """Reserve an appointment slot for the given YYYY-MM-DD date and HH:MM (24h)
    time under the given email. Fails if the slot is already taken, outside
    business hours, or on a weekend -- in that case it returns alternative
    slots so the agent can negotiate with the user instead of failing silently."""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"ok": False, "error": "date must be YYYY-MM-DD. Call resolve_date first."})

    if dt.weekday() >= 5 or time not in BUSINESS_HOURS:
        return json.dumps(
            {
                "ok": False,
                "error": "Outside business hours (09:00-12:00, 14:00-17:00, Mon-Fri).",
                "alternatives": _next_available(date),
            }
        )

    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO bookings (date, time, email, details, created_at) VALUES (?, ?, ?, ?, ?)",
            (date, time, email, "", datetime.now().isoformat()),
        )
        conn.commit()
        return json.dumps({"ok": True, "date": date, "time": time, "email": email})
    except sqlite3.IntegrityError:
        alternatives = _free_slots(date) or _next_available(date)
        return json.dumps(
            {"ok": False, "error": "That slot was just taken.", "alternatives": alternatives}
        )
    finally:
        conn.close()


@tool
def send_booking_notification(email: str, details: str) -> str:
    """Send a mock booking confirmation notification (simulating an email or
    WhatsApp message) to a webhook endpoint. Configure the WEBHOOK_NOTIFY_URL
    env var (e.g. a webhook.site or Pipedream URL) to actually see the
    payload arrive; otherwise the send is simulated locally."""
    webhook_url = os.environ.get("WEBHOOK_NOTIFY_URL")
    payload = {"to": email, "message": details, "sent_at": datetime.now().isoformat()}
    if not webhook_url:
        return json.dumps({"ok": True, "simulated": True, "payload": payload})
    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        return json.dumps({"ok": resp.ok, "status_code": resp.status_code, "payload": payload})
    except requests.RequestException as e:
        return json.dumps({"ok": False, "error": str(e), "payload": payload})
