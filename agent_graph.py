"""LangGraph state machine: Triage Agent + Booking Specialist.

Triage Agent classifies each new user message. General questions get answered
directly. Scheduling intent hands control to the Booking Specialist, which
stays in control across turns (via `active_flow`) until a booking is confirmed
or the user drops the topic, so slot-filling ("what's your email?") doesn't
get mis-routed back through triage.
"""

import json
import os
from datetime import datetime
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import check_availability, reserve_slot, resolve_date, send_booking_notification

TOOLS = [resolve_date, check_availability, reserve_slot, send_booking_notification]

TRIAGE_SYSTEM = """You are the Triage Agent for a company's scheduling assistant.

Ground truth about the business (do not contradict this): appointments are
available Monday-Friday only, 09:00-12:00 and 14:00-17:00, one per hourly
slot. There are no weekend appointments.

Read the latest user message in context. Decide:
- If the user is expressing intent to schedule, check, reschedule, or book an
  appointment/meeting/call, respond with EXACTLY the single token
  ROUTE_BOOKING and nothing else.
- Otherwise (general questions, greetings, small talk), answer directly and
  helpfully in plain text using only the ground truth above -- don't invent
  policies you don't actually know. Keep it brief.
"""

BOOKING_SYSTEM_TEMPLATE = """You are the Booking Specialist, handling calendar appointments.

Today's real date is {today} ({weekday}). Business hours: 09:00-12:00 and
14:00-17:00, Monday-Friday, one appointment per hourly slot.

Rules you must follow:
1. Never compute or guess dates yourself. If the user gives a relative date
   ("tomorrow", "next Friday", "in 2 weeks"), ALWAYS call resolve_date first
   to turn it into an absolute YYYY-MM-DD before calling any other tool.
2. Collect date, time, and email before booking. Ask for missing details one
   at a time in plain language.
3. NEVER invent, guess, or use a placeholder value for the email address (for
   example "user@example.com" or "test@example.com") under any circumstance.
   The email must be text the USER literally typed in this conversation. If
   they have not given one yet, your ONLY move is to ask for it and STOP --
   do not call reserve_slot or any other tool until they reply with it.
4. If the user hasn't picked a specific time, call check_availability and
   show them the open slots for that date.
5. Only call reserve_slot once you have a resolved YYYY-MM-DD date, an HH:MM
   (24h) time, and an email address the user actually typed themselves.
6. If check_availability or reserve_slot reports the day/slot unavailable,
   propose the alternatives it returns and ask the user to pick one, then
   stop and wait -- you will not be allowed to call another tool until they
   reply. Never fail silently or give up, and never pick an alternative on
   the user's behalf.
7. After reserve_slot succeeds, call send_booking_notification with a short
   summary of the appointment as `details`, then confirm the booking to the
   user in plain language (date, time, email).
Be concise and friendly.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]
    active_flow: Optional[str]
    force_text_reply: Optional[bool]


def _llm(bind_tools: bool = False):
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    return llm.bind_tools(TOOLS) if bind_tools else llm


def route_entry(state: AgentState) -> str:
    return "booking_specialist" if state.get("active_flow") == "booking" else "triage"


def triage_node(state: AgentState) -> dict:
    llm = _llm()
    msgs = [SystemMessage(content=TRIAGE_SYSTEM)] + state["messages"]
    response = llm.invoke(msgs)
    if "ROUTE_BOOKING" in (response.content or ""):
        return {"route": "booking", "active_flow": "booking"}
    return {"messages": [response], "route": "general", "active_flow": None}


def route_after_triage(state: AgentState) -> str:
    return "booking_specialist" if state.get("route") == "booking" else END


def booking_node(state: AgentState) -> dict:
    # After a tool reports failure (bad date, slot conflict, etc.), tool-calling
    # is disabled for this one turn so the model *cannot* silently retry with
    # its own guess -- it can only reply in text, which forces it to present
    # alternatives and wait for the user's explicit choice on the next turn.
    force_text = state.get("force_text_reply", False)
    llm = _llm(bind_tools=not force_text)
    now = datetime.now()
    system = BOOKING_SYSTEM_TEMPLATE.format(today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A"))
    msgs = [SystemMessage(content=system)] + state["messages"]
    response = llm.invoke(msgs)
    return {"messages": [response], "force_text_reply": False}


def route_after_booking(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def post_tools_node(state: AgentState) -> dict:
    """After tools run: (1) check whether a booking just completed
    (notification sent successfully) so the next turn re-enters triage
    instead of staying locked into the booking flow forever, and (2) if any
    tool in this batch reported failure, force the next booking_specialist
    call to reply in text only -- see booking_node."""
    msgs = state["messages"]
    active = state.get("active_flow", "booking")
    any_failure = False
    i = len(msgs) - 1
    while i >= 0 and isinstance(msgs[i], ToolMessage):
        m = msgs[i]
        try:
            data = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            data = None
        if data is not None and not data.get("ok", True):
            any_failure = True
        if m.name == "send_booking_notification" and data and data.get("ok"):
            active = None
        i -= 1
    return {"active_flow": active, "force_text_reply": any_failure}


def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("booking_specialist", booking_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("post_tools", post_tools_node)

    graph.add_conditional_edges(
        START, route_entry, {"booking_specialist": "booking_specialist", "triage": "triage"}
    )
    graph.add_conditional_edges(
        "triage", route_after_triage, {"booking_specialist": "booking_specialist", END: END}
    )
    graph.add_conditional_edges(
        "booking_specialist", route_after_booking, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "post_tools")
    graph.add_edge("post_tools", "booking_specialist")

    return graph.compile(checkpointer=checkpointer)
