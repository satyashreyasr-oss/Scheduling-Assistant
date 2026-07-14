# Multi-Agent Scheduling Assistant

A LangGraph-based scheduling assistant with two cooperating agents:

- **Triage Agent** — reads each incoming message and decides whether it's a
  general question (answers directly) or scheduling intent (hands off to the
  Booking Specialist).
- **Booking Specialist** — resolves relative dates ("tomorrow", "next
  Friday") to real `YYYY-MM-DD` values, checks calendar availability, asks
  for missing details (date/time/email), reserves the slot, and sends a mock
  confirmation. If a slot is taken it negotiates alternatives instead of
  failing silently.

A Streamlit chat UI sits on top, and conversation state is persisted to
SQLite so history survives a page refresh or server restart.

## Architecture

```
User message
     │
     ▼
 START ──(active_flow == "booking")──► Booking Specialist
   │                                         │▲
   │ (else)                                  ││
   ▼                                         ││
 Triage Agent                                ││
   │  general question → answer, END          ││
   │  booking intent  ──────────────────────►─┘│
   ▼                                            │
[booking flow continues across turns until]     │
                                                 │
 Booking Specialist ──(tool_calls?)──► tools ────┘
   │  no tool calls                    │
   ▼                                   ▼
  END                            post_tools (checks if
                                  send_booking_notification
                                  succeeded → clears
                                  active_flow, else stays
                                  in the booking flow)
```

Key files:

| File | Purpose |
|---|---|
| `agent_graph.py` | LangGraph `StateGraph`: nodes, routing, system prompts |
| `tools.py` | `resolve_date`, `check_availability`, `reserve_slot`, `send_booking_notification` |
| `persistence.py` | SQLite checkpointer (`langgraph-checkpoint-sqlite`) for thread state |
| `app.py` | Streamlit chat UI |

### Tools (mocked but functional)

- `resolve_date(expression)` — uses `dateparser` anchored to the real current
  date/time (`datetime.now()`) to turn relative expressions into an absolute
  date. The Booking Specialist is instructed to always call this before any
  other tool when the user gives a relative date, satisfying the "resolve
  before executing any tools" requirement.
- `check_availability(date)` — looks up booked slots in a local SQLite table
  (`data/scheduling.db`) against fixed business hours (09:00–12:00,
  14:00–17:00, Mon–Fri) and returns free slots, or alternatives if the day is
  full/weekend.
- `reserve_slot(date, time, email)` — inserts into the same SQLite table with
  a `UNIQUE(date, time)` constraint; a collision returns `ok: false` plus
  alternative slots so the agent can negotiate rather than fail silently.
- `send_booking_notification(email, details)` — POSTs a JSON payload to
  `WEBHOOK_NOTIFY_URL` (point this at a free https://webhook.site or
  Pipedream URL to see it live) or simulates the send locally if unset.

### State persistence

`persistence.py` wires up `SqliteSaver` from `langgraph-checkpoint-sqlite`,
writing checkpoints to `data/checkpoints.db`. The Streamlit UI keeps a
`thread_id` in the page's URL query string (`?thread=...`), so reloading the
page reconnects to the same LangGraph thread and replays its message history
from SQLite instead of starting over.

## Setup

1. **Install dependencies** (Python 3.11+ recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Configure environment variables** — copy `.env.example` to `.env` and
   fill in:

   ```
   GROQ_API_KEY=your-groq-key       # https://console.groq.com/keys (free tier)
   # GROQ_MODEL=llama-3.3-70b-versatile
   # WEBHOOK_NOTIFY_URL=https://webhook.site/your-unique-url
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (typically `http://localhost:8501`).

4. **Optional: sanity-check the tools without an API key**:

   ```bash
   python smoke_test.py
   ```

   Exercises `resolve_date` / `check_availability` / `reserve_slot` /
   `send_booking_notification` directly against a throwaway SQLite DB.

## Trying it out

- General question: *"What are your business hours?"* → Triage Agent
  answers directly, no booking flow entered.
- Booking: *"Book an appointment tomorrow at 10am, my email is a@b.com"* →
  Triage routes to the Booking Specialist, which calls `resolve_date`,
  `check_availability`/`reserve_slot`, and `send_booking_notification`, then
  confirms.
- Conflict negotiation: book the same date/time twice in two different
  browser tabs (different `thread_id`s) — the second attempt gets alternative
  slots offered instead of a silent failure.
- Missing info: *"I'd like to book a call"* → the agent asks for date, time,
  and email before calling any tool.

## Deployment (free tier)

**Deployed on Streamlit Community Cloud** (share.streamlit.io), deploying
directly from this GitHub repo:

1. Sign in to share.streamlit.io with GitHub and authorize access to this
   repo.
2. "Create app" → "Deploy a public app from GitHub" → repository
   `satyashreyasr-oss/Scheduling-Assistant`, branch `master`, main file
   `app.py`.
3. Under "Advanced settings" → Secrets, add (TOML format):
   ```toml
   GROQ_API_KEY = "your-groq-key-here"
   ```
   Streamlit Cloud exposes secrets both via `st.secrets` and as regular
   environment variables, so `os.environ.get(...)` works with no code
   changes.
4. Deploy.

Other free-tier platforms work too, since this is a standard Streamlit app:

- **Hugging Face Spaces (Streamlit SDK)** — create a Space, set SDK to
  `streamlit`, push this repo, and add `GROQ_API_KEY` (and optionally
  `WEBHOOK_NOTIFY_URL`) as Space secrets.
- **Render (free web service)** — new Web Service from this repo, build
  command `pip install -r requirements.txt`, start command
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, and
  add the same env vars under Render's Environment settings.

Note: free-tier disks are typically ephemeral across redeploys/restarts (this
includes Streamlit Community Cloud, which spins the app down after
inactivity), so `data/*.db` (bookings + checkpoints) may reset when the
instance restarts — acceptable for this assignment's mock persistence
requirement, since it does survive ordinary page refreshes and short-lived
restarts within a running instance.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Auth for the Groq LLM API (`langchain-groq`) |
| `GROQ_MODEL` | No | Override model name (default `llama-3.3-70b-versatile`) |
| `WEBHOOK_NOTIFY_URL` | No | Where mock booking-confirmation payloads are POSTed; simulated locally if unset |

No API keys are committed to this repository — see `.env.example`.

## Known limitations

- **Model choice matters.** The default `llama-3.3-70b-versatile` reliably
  calls tools before confirming an action. Smaller/faster Groq models (e.g.
  `llama-3.1-8b-instant`) were observed, during testing, to occasionally
  *skip* the required tool call and fabricate a plausible-sounding
  confirmation message instead. Don't swap `GROQ_MODEL` to a smaller model
  for this app without re-verifying that bookings are actually reaching
  `data/scheduling.db`.
- **Groq free-tier daily quota.** `llama-3.3-70b-versatile` on the free tier
  is capped at 100,000 tokens/day. Heavy interactive testing can exhaust this
  well before midnight, surfacing as a `groq.RateLimitError` (HTTP 429). If
  you hit it, wait for the cooldown window Groq reports in the error, or use
  a different `GROQ_API_KEY`.
- Two independent, code-level safeguards don't depend on the model behaving
  well: (1) `resolve_date` is computed deterministically (`dateparser` +
  a regex fallback for "next/this <weekday>" phrases), never left to the
  LLM's own arithmetic; (2) after any tool call fails, `force_text_reply`
  strips tool-calling ability for the model's next turn, so it structurally
  cannot silently retry a different slot on the user's behalf — it can only
  reply in text and must wait for the user's explicit choice.
