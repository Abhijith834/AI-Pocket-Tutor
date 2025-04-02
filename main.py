import sys
import os
import logging
import subprocess

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
    # Start the current chat session and capture its return value.
    result = chat.main(final_pdf_path=final_pdf_path)

    # --- NEW CODE (only these lines added/modified) ---
    # If the user typed "new chat (normal)" or "new chat (learning)" in chat,
    # chat.main() will return "NEWCHAT:normal" or "NEWCHAT:learning".
    if isinstance(result, str) and result.startswith("NEWCHAT:"):
        mode = result.split(":")[1].lower().strip()
        print(f"[Main] User requested a NEW chat session in '{mode}' mode.\n")

        # Create a brand-new session folder (replicating our "new session" logic)
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
        new_session_id = str(max_id + 1)
        new_session_folder = os.path.join(database_root, f"chat_{new_session_id}")
        os.makedirs(new_session_folder, exist_ok=True)

        os.environ["CHAT_HISTORY_FILE"] = os.path.join(new_session_folder, "chat_history.json")
        os.environ["CHROMA_DB_DIR"] = os.path.join(new_session_folder, "chromadb_storage")
        os.environ["SESSION_ID"] = new_session_id

        print(f"[Main] Created new session: {new_session_id}")
        print(f"[Main] Session folder: {new_session_folder}")

        # Set learning mode flag for the new session if needed.
        learning_mode_for_new = (mode == "learning") and config.LEARNING_MODE

        db_utils.load_session_state()
        if db_utils.memory_summary:
            print("[Main] Loaded memory summary for this new session.")

        final_pdf_path_new = None
        if learning_mode_for_new:
            from core.learning_mode import LearningModeAgent
            agent = LearningModeAgent()
            agent.init_learning_mode()
            final_pdf_path_new = agent.final_pdf_path

        print("\n[Main] Starting new chat session.\n")
        chat.main(final_pdf_path=final_pdf_path_new)
    # --- END NEW CODE ---

    # After finishing (or if the user just typed exit), restart the entire program.
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    print("\n[Main] Chat session ended. Restarting the program now...\n")
    subprocess.run([python_exe, script_path] + sys.argv[1:], check=False)
    sys.exit(0)
