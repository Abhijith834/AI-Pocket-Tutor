import sys
import os
import argparse
import json
import subprocess
import logging

logging.getLogger("chromadb").setLevel(logging.ERROR)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

database_root      = os.path.join(current_dir, "database")
session_state_file = os.path.join(database_root, "session_state.json")
os.makedirs(database_root, exist_ok=True)

def load_session_state() -> str:
    if os.path.exists(session_state_file):
        try:
            with open(session_state_file, "r", encoding="utf-8") as f:
                return json.load(f).get("session_id", "")
        except:
            pass
    return ""

def save_session_state_to_file(sid: str) -> None:
    try:
        with open(session_state_file, "w", encoding="utf-8") as f:
            json.dump({"session_id": sid}, f, indent=4)
    except:
        pass

def initialize_session():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session",       type=str, default="", help="Use existing session")
    group.add_argument("--newchat",       type=str, default="", help="Create new chat (normal|learning)")

    # ← NEW: flags for learning‐mode ingestion
    parser.add_argument("--learning-file",    type=str, default="", help="Path to PDF for learning mode")
    parser.add_argument("--learning-subject", type=str, default="", help="Search subject for PDF")

    args, _ = parser.parse_known_args()

    # 1) Explicit flags
    if args.session:
        return args, args.session, False, None
    if args.newchat:
        return args, "", True, args.newchat.lower()

    # 2) Stored session
    stored = load_session_state()
    if stored:
        return args, stored, False, None

    # 3) Interactive fallback for session id
    sid = input("Enter chat session (or Enter for new): ").strip()
    is_new = sid == ""
    mode   = None
    if is_new:
        mode = input("Which mode? (normal|learning): ").strip().lower()
    return args, sid, is_new, mode

def update_session_state(sid: str):
    import session_config
    session_config.session_id = sid
    folder = os.path.join(database_root, f"chat_{sid}")
    os.makedirs(folder, exist_ok=True)
    os.environ.update({
        "CHAT_HISTORY_FILE": os.path.join(folder, "chat_history.json"),
        "CHROMA_DB_DIR"   : os.path.join(folder, "chromadb_storage"),
        "SESSION_ID"      : sid
    })
    from core import db_utils
    db_utils.load_session_state()
    save_session_state_to_file(sid)

def get_next_session_id() -> str:
    nums = []
    for d in os.listdir(database_root):
        if d.startswith("chat_"):
            suf = d.split("_",1)[1]
            if suf.isdigit():
                nums.append(int(suf))
    return str(max(nums)+1 if nums else 1)

if __name__ == "__main__":
    args, chat_session, is_new, new_mode = initialize_session()
    session_id = chat_session or get_next_session_id()
    update_session_state(session_id)

    print(f"[Main] Using chat session: {session_id}")

    # ─── Learning Mode: only via CLI flags, no prompts ────────────────
    final_pdf_path = None
    from core import config
    if is_new and new_mode == "learning" and config.LEARNING_MODE:
        from core.learning_mode import LearningModeAgent
        agent = LearningModeAgent()
        fp = args.learning_file    or None
        sb = args.learning_subject or None
        agent.init_learning_mode(file_path=fp, subject=sb)
        final_pdf_path = agent.final_pdf_path

    # ─── Hand off to chat loop ──────────────────────────────────────
    from core import chat
    print("\nSwitching to normal chat mode.\n")
    result = chat.main(final_pdf_path=final_pdf_path)

    # ─── Post‐chat routing (unchanged) ───────────────────────────────
    python_exe = sys.executable
    script     = os.path.abspath(__file__)
    if isinstance(result, str):
        if result.startswith("chat (") and result.endswith(")"):
            sid = result[6:-1].strip()
            subprocess.run([python_exe, script, "--session", sid], check=False)
            sys.exit(0)
        if result.startswith("NEWCHAT:"):
            m = result.split(":",1)[1].strip().lower()
            subprocess.run([python_exe, script, "--newchat", m], check=False)
            sys.exit(0)
        if result.isdigit():
            subprocess.run([python_exe, script, "--session", result], check=False)
            sys.exit(0)

    # default restart
    subprocess.run([python_exe, script, "--session", session_id] + sys.argv[1:], check=False)
    sys.exit(0)
