# core/config.py

CHROMA_DB_DIR = "chromadb_storage"
CHAT_HISTORY_FILE = "chat_history.json"
SESSION_STATE_FILE = "session_state.json"
MAX_TEXT_LENGTH = 20000

EPHEMERAL_KEYWORDS = [
    "latest", "recent", "happened to", "happened with",
    "starship", "shenzhou", "jack ma", "breaking", "update",
    "2023", "2024", "2025"
]
