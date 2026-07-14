"""Streamlit chat UI for the Multi-Agent Scheduling Assistant."""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_graph import build_graph
from persistence import get_checkpointer

st.set_page_config(page_title="Scheduling Assistant", page_icon="🗓️")


@st.cache_resource
def load_app():
    return build_graph(get_checkpointer())


graph_app = load_app()

# Keep the thread id in the URL so the conversation survives a page refresh.
if "thread" not in st.query_params:
    st.query_params["thread"] = str(uuid.uuid4())
thread_id = st.query_params["thread"]

st.title("🗓️ Scheduling Assistant")
st.caption(f"Session `{thread_id[:8]}` · Triage Agent + Booking Specialist (LangGraph)")

with st.sidebar:
    st.subheader("About")
    st.write(
        "A Triage Agent routes general questions vs. booking intent to a "
        "Booking Specialist, which resolves relative dates, checks/reserves "
        "calendar slots (SQLite), negotiates alternatives on conflicts, and "
        "sends a mock confirmation via webhook."
    )
    st.markdown("**Try asking:**")
    st.code("What are your business hours?", language=None)
    st.code("Book an appointment tomorrow at 10am, email a@b.com", language=None)

    if st.button("Start new conversation"):
        st.query_params["thread"] = str(uuid.uuid4())
        st.rerun()

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to your .env file or hosting secrets.")

config = {"configurable": {"thread_id": thread_id}}

# Render prior history for this thread from the checkpointer.
snapshot = graph_app.get_state(config)
history = snapshot.values.get("messages", []) if snapshot.values else []

for m in history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        with st.chat_message("assistant"):
            st.write(m.content)
    elif isinstance(m, ToolMessage):
        with st.chat_message("assistant"):
            st.caption(f"🔧 `{m.name}` → {m.content}")

if prompt := st.chat_input("Ask a question or book an appointment..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph_app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
            except Exception as e:  # noqa: BLE001 - surface any LLM/API error to the UI
                st.error(f"Something went wrong: {e}")
                result = None

        if result:
            new_msgs = result["messages"][len(history) + 1 :]  # skip history + the human msg just echoed above
            for m in new_msgs:
                if isinstance(m, ToolMessage):
                    st.caption(f"🔧 `{m.name}` → {m.content}")
                elif isinstance(m, AIMessage) and m.content:
                    st.write(m.content)
