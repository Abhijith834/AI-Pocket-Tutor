import sys
import os
import argparse
import json
import subprocess
import logging

logging.getLogger("chromadb").setLevel(logging.ERROR)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

# Path to the shared session state file
database_root = os.path.join(current_dir, "database")
os.makedirs(database_root, exist_ok=True)
session_state_file = os.path.join(database_root, "session_state.json")

def load_session_state():
    """
    Load the session state from database/session_state.json.
    Returns the session id if available, else an empty string.
    """
    if os.path.exists(session_state_file):
        try:
            with open(session_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            session_id = state.get("session_id", "")
            print(f"[Main] Loaded session state from file: session_id = {session_id}")
            return session_id
        except Exception as e:
            print(f"[Main] Error loading session state file: {e}")
    return ""

def save_session_state_to_file(new_session_id):
    """
    Save the session state (session id) to database/session_state.json.
    """
    try:
        state = {"session_id": new_session_id}
        with open(session_state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(f"[Main] Saved session state: session_id = {new_session_id}")
    except Exception as e:
        print(f"[Main] Error writing session state: {e}")

def initialize_session():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", type=str, default="", help="Jump directly to an existing session.")
    group.add_argument("--newchat", type=str, default="", help="Force creation of a new chat session (normal or learning).")
    args, unknown = parser.parse_known_args()

    # Priority: command-line argument → session state file → user prompt.
    if args.session:
        return args.session, False, None
    if args.newchat:
        return "", True, args.newchat.lower()
    
    stored_session = load_session_state()
    if stored_session:
        return stored_session, False, None

    chat_session = input("Enter chat session number (or press Enter to create a new session): ").strip()
    is_new_session = (chat_session == "")
    new_chat_mode = None
    if is_new_session:
        new_chat_mode = input("Which mode do you want for this new session? (enter 'learning' or 'normal'): ").strip().lower()
    return chat_session, is_new_session, new_chat_mode

def update_session_state(new_session_id):
    import session_config
    session_config.session_id = new_session_id
    session_folder = os.path.join(database_root, f"chat_{new_session_id}")
    os.makedirs(session_folder, exist_ok=True)
    os.environ["CHAT_HISTORY_FILE"] = os.path.join(session_folder, "chat_history.json")
    os.environ["CHROMA_DB_DIR"] = os.path.join(session_folder, "chromadb_storage")
    os.environ["SESSION_ID"] = new_session_id
    print(f"[Main] Now using chat session: {new_session_id}")
    from core import db_utils
    db_utils.load_session_state()
    # Also update the persistent session state file.
    save_session_state_to_file(new_session_id)

if __name__ == "__main__":
    # Initialize session from command-line, session state file, or prompt.
    chat_session, is_new_session, new_chat_mode = initialize_session()
    if chat_session:
        session_id_str = chat_session
    else:
        # Create a new session id by scanning the database folder if no session was provided.
        existing_sessions = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
        max_id = 0
        for d in existing_sessions:
            try:
                num = int(d.split("_")[-1])
                if num > max_id:
                    max_id = num
            except:
                continue
        session_id_str = str(max_id + 1)
    update_session_state(session_id_str)
    print(f"[Main] Using chat session: {session_id_str}")
    print(f"[Main] Session folder: {os.path.join(database_root, 'chat_' + session_id_str)}")

    from core import chat, db_utils, config
    db_utils.load_session_state()

    if db_utils.memory_summary:
        print("[Main] Loaded memory summary for this session.")

    final_pdf_path = None
    if is_new_session and new_chat_mode == "learning" and config.LEARNING_MODE:
        from core.learning_mode import LearningModeAgent
        agent = LearningModeAgent()
        agent.init_learning_mode()
        final_pdf_path = agent.final_pdf_path

    print("\nSwitching to normal chat mode.\n")

    result = chat.main(final_pdf_path=final_pdf_path)

    python_exe = sys.executable
    script_path = os.path.abspath(__file__)

    # Process the result from chat.main
    if isinstance(result, str):
        # Case: result is a session-switch command formatted as "chat (10)"
        if result.startswith("chat (") and result.endswith(")"):
            new_session = result[len("chat ("):-1].strip()
            print(f"[Main] Detected session switch command: {result}")
            update_session_state(new_session)
            subprocess.run([python_exe, script_path, "--session", new_session], check=False)
            sys.exit(0)
        # Case: NEWCHAT command
        elif result.startswith("NEWCHAT:"):
            mode = result.split(":")[1].strip().lower()
            new_session = input("Enter new session id: ").strip()
            update_session_state(new_session)
            subprocess.run([python_exe, script_path, "--newchat", mode], check=False)
            sys.exit(0)
        # Case: plain digit (session id)
        elif result.isdigit():
            update_session_state(result)
            subprocess.run([python_exe, script_path, "--session", result], check=False)
            sys.exit(0)

    print("\nChat session ended. Restarting the program now...\n")
    subprocess.run([python_exe, script_path] + sys.argv[1:], check=False)
    sys.exit(0)
