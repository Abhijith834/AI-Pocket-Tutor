import sys
import os
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

chat_session = input("Enter chat session number (or press Enter to create a new session): ").strip()

database_root = os.path.join(current_dir, "database")
if not os.path.exists(database_root):
    os.makedirs(database_root)

if chat_session:
    session_id = chat_session
else:
    existing_sessions = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    max_id = 0
    for s in existing_sessions:
        try:
            num = int(s.split("_")[-1])
            if num > max_id:
                max_id = num
        except:
            continue
    session_id = str(max_id + 1)

session_folder = os.path.join(database_root, f"chat_{session_id}")
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

os.environ["CHAT_HISTORY_FILE"] = os.path.join(session_folder, "chat_history.json")
os.environ["SESSION_STATE_FILE"] = os.path.join(session_folder, "session_state.json")
os.environ["CHROMA_DB_DIR"] = os.path.join(session_folder, "chromadb_storage")
os.environ["SESSION_ID"] = session_id

print(f"Using chat session: {session_id}")
print(f"Session folder: {session_folder}")

from core import chat
if __name__ == "__main__":
    chat.main()
