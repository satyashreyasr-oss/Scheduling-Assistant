"""SQLite-backed LangGraph checkpointer so thread history survives restarts
and page refreshes."""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "checkpoints.db"
)


def get_checkpointer() -> SqliteSaver:
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
