import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHECKPOINT_PATH


conn = sqlite3.connect(
    str(CHECKPOINT_PATH),
    check_same_thread=False,
)

memory = SqliteSaver(conn)