import sys
import os
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

chat_session = input("Enter chat session number (or press Enter to create a new session): ").strip()
is_new_session = (chat_session == "")

learning_mode_chosen = False
if is_new_session:
    mode_input = input("Which mode do you want for this new session? (enter 'learning' or 'normal'): ").strip().lower()
    if mode_input == "learning":
        learning_mode_chosen = True

database_root = os.path.join(current_dir, "database")
os.makedirs(database_root, exist_ok=True)

if chat_session:
    session_id = chat_session
else:
    # create new session
    existing_sessions = [
        d for d in os.listdir(database_root)
        if os.path.isdir(os.path.join(database_root, d))
    ]
    max_id = 0
    for d in existing_sessions:
        try:
            num = int(d.split("_")[-1])
            if num > max_id:
                max_id = num
        except:
            continue
    session_id = str(max_id + 1)

session_folder = os.path.join(database_root, f"chat_{session_id}")
os.makedirs(session_folder, exist_ok=True)

os.environ["CHAT_HISTORY_FILE"] = os.path.join(session_folder, "chat_history.json")
os.environ["CHROMA_DB_DIR"] = os.path.join(session_folder, "chromadb_storage")
os.environ["SESSION_ID"] = session_id

print(f"Using chat session: {session_id}")
print(f"Session folder: {session_folder}")

from core import chat, db_utils, config
db_utils.load_session_state()

if db_utils.memory_summary:
    print("[Main] Loaded memory summary for this session.")

final_pdf_path = None  # Will store the PDF path if a doc was chosen
if is_new_session and learning_mode_chosen and config.LEARNING_MODE:
    from core.learning_mode import LearningModeAgent
    agent = LearningModeAgent()
    agent.init_learning_mode()
    final_pdf_path = agent.final_pdf_path

print("\n[Main] Switching to normal chat mode.\n")

if __name__ == "__main__":
    # Pass the final PDF path to the normal chat
    chat.main(final_pdf_path=final_pdf_path)
