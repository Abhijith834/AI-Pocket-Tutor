import sys
import os
import logging
import subprocess
import argparse

logging.getLogger("chromadb").setLevel(logging.ERROR)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

# ------------------ 1) Parse command-line arguments ------------------
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--session", type=str, default="", help="Jump directly to this existing session.")
group.add_argument("--newchat", type=str, default="", help="Force creation of a new chat session with mode 'normal' or 'learning'.")
args, unknown = parser.parse_known_args()

if args.session:
    # Forced existing session: skip prompt
    chat_session = args.session
    is_new_session = False
    new_chat_mode = None
elif args.newchat:
    # Forced new session: no session number provided
    chat_session = ""
    is_new_session = True
    new_chat_mode = args.newchat.lower()
else:
    # Normal prompt
    chat_session = input("Enter chat session number (or press Enter to create a new session): ").strip()
    is_new_session = (chat_session == "")
    if is_new_session:
        mode_input = input("Which mode do you want for this new session? (enter 'learning' or 'normal'): ").strip().lower()
        new_chat_mode = mode_input
    else:
        new_chat_mode = None
# -----------------------------------------------------------------------

database_root = os.path.join(current_dir, "database")
os.makedirs(database_root, exist_ok=True)

if chat_session:
    session_id = chat_session
else:
    # Create new session by finding the highest existing ID and adding 1
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

final_pdf_path = None  # Will store the PDF path if a document was chosen
if is_new_session and new_chat_mode == "learning" and config.LEARNING_MODE:
    from core.learning_mode import LearningModeAgent
    agent = LearningModeAgent()
    agent.init_learning_mode()
    final_pdf_path = agent.final_pdf_path

print("\n[Main] Switching to normal chat mode.\n")

if __name__ == "__main__":
    # Start the current chat session and capture its return value.
    result = chat.main(final_pdf_path=final_pdf_path)

    python_exe = sys.executable
    script_path = os.path.abspath(__file__)

    # -------- Seamless new chat creation via "new chat (normal)" or "new chat (learning)" --------
    if isinstance(result, str) and result.startswith("NEWCHAT:"):
        mode = result.split(":")[1].lower().strip()
        print(f"[Main] User requested a NEW chat session in '{mode}' mode.\n")
        # Instead of doing the new session creation inline, relaunch with --newchat <mode> to bypass prompts
        subprocess.run([python_exe, script_path, "--newchat", mode], check=False)
        sys.exit(0)
    # -------- Seamless switching to an existing chat (e.g. "chat (10)") --------
    elif isinstance(result, str) and result.isdigit():
        print(f"\n[Main] Chat session ended. Restarting, going straight to session {result}...\n")
        subprocess.run([python_exe, script_path, "--session", result], check=False)
        sys.exit(0)
    # -------- Normal restart (for exit or other cases) --------
    else:
        print("\n[Main] Chat session ended. Restarting the program now...\n")
        subprocess.run([python_exe, script_path] + sys.argv[1:], check=False)
        sys.exit(0)
