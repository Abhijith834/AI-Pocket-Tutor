import os
import re

CHAT_HISTORY_FILE = os.getenv("CHAT_HISTORY_FILE", "chat_history.json")
SESSION_STATE_FILE = os.getenv("SESSION_STATE_FILE", "session_state.json")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chromadb_storage")
MAX_TEXT_LENGTH = 20000
MODEL = "mistral"

