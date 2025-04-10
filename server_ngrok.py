import os
import uuid
import json
import subprocess
import threading
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core import db_utils, chat, config
from core.learning_mode import LearningModeAgent

import session_config
session_id = session_config.session_id

DATABASE_ROOT = os.path.join("AI-Pocket-Tutor", "database")

# Global list to hold captured output from server.py
server_logs = []

# --- Setup logging ---
import logging
logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)
class ListHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        server_logs.append(log_entry)
list_handler = ListHandler()
list_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
list_handler.setFormatter(formatter)
logger.addHandler(list_handler)
def capture_print(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)
print = capture_print
# --- End logging setup ---

# --- Global structures ---
db_notifications = []
has_new_notification = False  # We'll only notify if something changed

# --- Watchdog Event Handler ---
class DatabaseChangeHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.cooldown = 1  # seconds to ignore repeated triggers
        self.last_event_time = 0.0

    def on_modified(self, event):
        """Trigger only on 'modified' file events, ignoring directories and repeated triggers."""
        global has_new_notification
        if event.is_directory:
            return
        now = time.time()
        if (now - self.last_event_time) < self.cooldown:
            # Skip if this is too soon after a previous event
            return

        self.last_event_time = now
        message = {
            "event_type": "modified",
            "src_path": event.src_path,
            "is_directory": event.is_directory
        }
        db_notifications.append(message)
        has_new_notification = True
        print(f"[Watchdog] {message}")

    def on_created(self, event):
        """Similarly handle 'created' events."""
        global has_new_notification
        if event.is_directory:
            return
        now = time.time()
        if (now - self.last_event_time) < self.cooldown:
            return

        self.last_event_time = now
        message = {
            "event_type": "created",
            "src_path": event.src_path,
            "is_directory": event.is_directory
        }
        db_notifications.append(message)
        has_new_notification = True
        print(f"[Watchdog] {message}")

    def on_deleted(self, event):
        """Similarly handle 'deleted' events."""
        global has_new_notification
        if event.is_directory:
            return
        now = time.time()
        if (now - self.last_event_time) < self.cooldown:
            return

        self.last_event_time = now
        message = {
            "event_type": "deleted",
            "src_path": event.src_path,
            "is_directory": event.is_directory
        }
        db_notifications.append(message)
        has_new_notification = True
        print(f"[Watchdog] {message}")

# --- Create Flask app ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://www.tutorbot.online",
    "https://mint-jackal-publicly.ngrok-free.app"
]}})

@app.route("/api/session", methods=["POST"])
def create_session():
    data = request.get_json() or {}
    sess_id = data.get("session_id", session_id)
    session_folder = os.path.join(DATABASE_ROOT, f"chat_{sess_id}")
    os.makedirs(session_folder, exist_ok=True)
    os.environ["CHAT_HISTORY_FILE"] = os.path.join(session_folder, "chat_history.json")
    os.environ["CHROMA_DB_DIR"] = os.path.join(session_folder, "chromadb_storage")
    os.environ["SESSION_ID"] = sess_id
    db_utils.load_session_state()
    print(f"[Server] Session created/loaded: {sess_id}")
    return jsonify({
        "message": "Session created or loaded successfully",
        "session_id": sess_id
    }), 200

@app.route("/api/session/info", methods=["GET"])
def session_info():
    chat_history_file = os.environ.get("CHAT_HISTORY_FILE", "")
    chroma_db_dir = os.environ.get("CHROMA_DB_DIR", "")
    memory_summary = db_utils.memory_summary
    current_session_id = os.environ.get("SESSION_ID", None)
    return jsonify({
        "session_id": current_session_id,
        "chat_history_file": chat_history_file,
        "chroma_db_dir": chroma_db_dir,
        "memory_summary": memory_summary
    }), 200

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No 'message' provided"}), 400
    db_utils.chat_history.append({"role": "user", "content": user_message})
    db_utils.save_session_state()
    answer = chat.master_answer_flow(user_message)
    db_utils.chat_history.append({"role": "assistant", "content": answer})
    db_utils.save_session_state()
    print("[Server] Chat endpoint processed message.")
    return jsonify({"response": answer}), 200

@app.route("/api/ingest", methods=["POST"])
def ingest_file():
    data = request.get_json() or {}
    file_path = data.get("file_path")
    if not file_path:
        return jsonify({"error": "No 'file_path' provided"}), 400
    try:
        subprocess.run(["python", "input.py", file_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500
    print(f"[Server] File ingestion successful: {file_path}")
    return jsonify({"message": "File ingestion successful", "file_path": file_path}), 200

@app.route("/api/learning-mode", methods=["POST"])
def start_learning_mode():
    agent = LearningModeAgent()
    agent.init_learning_mode()
    final_pdf_path = agent.final_pdf_path
    print(f"[Server] Learning mode completed. Final PDF: {final_pdf_path}")
    return jsonify({
        "message": "Learning mode completed",
        "final_pdf_path": final_pdf_path
    }), 200

@app.route("/api/database/export", methods=["GET"])
def export_database():
    if not os.path.isdir(DATABASE_ROOT):
        return jsonify({"error": "Database folder not found"}), 404
    all_data = {}
    for folder_name in os.listdir(DATABASE_ROOT):
        if not folder_name.startswith("chat_"):
            continue
        sess_id = folder_name.split("_", 1)[1]
        session_folder = os.path.join(DATABASE_ROOT, folder_name)
        chat_history_file = os.path.join(session_folder, "chat_history.json")
        if os.path.isfile(chat_history_file):
            try:
                with open(chat_history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                data = {"error": f"Could not load chat_history.json: {str(e)}"}
            all_data[sess_id] = data
        else:
            all_data[sess_id] = {"notice": "No chat_history.json found"}
    return jsonify(all_data), 200

@app.route("/api/ai-pocket-tutor/database/files", methods=["GET"])
def list_ai_pocket_tutor_database_files():
    try:
        db_path = os.path.join("AI-Pocket-Tutor", "database")
        print(f"[Server] Looking for database folder at: {os.path.abspath(db_path)}")
        if not os.path.isdir(db_path):
            return jsonify({"error": "AI-Pocket-Tutor/database folder not found."}), 404
        session_files = {}
        for folder_name in os.listdir(db_path):
            session_folder = os.path.join(db_path, folder_name)
            if os.path.isdir(session_folder) and folder_name.startswith("chat_"):
                print(f"[Server] Processing session folder: {session_folder}")
                matching_files = []
                for root, dirs, files in os.walk(session_folder):
                    for file in files:
                        if file.lower().endswith((".json", ".pdf")):
                            rel_path = os.path.relpath(os.path.join(root, file), session_folder)
                            matching_files.append(rel_path)
                            print(f"[Server] Found file: {rel_path} in session {folder_name}")
                sess_id = folder_name.split("_", 1)[1]
                session_files[sess_id] = matching_files
        print(f"[Server] Found session files: {session_files}")
        return jsonify(session_files), 200
    except Exception as e:
        return jsonify({"error": f"Error listing files: {str(e)}"}), 500

@app.route("/api/database/file", methods=["GET"])
def get_database_file():
    sess_id = request.args.get("session")
    filepath = request.args.get("filepath")
    if not sess_id or not filepath:
        return jsonify({"error": "Both 'session' and 'filepath' query parameters are required."}), 400
    session_folder = os.path.join(DATABASE_ROOT, f"chat_{sess_id}")
    abs_path = os.path.join(session_folder, filepath)
    if not os.path.abspath(abs_path).startswith(os.path.abspath(session_folder)):
        return jsonify({"error": "Invalid file path."}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "File not found."}), 404
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in [".txt", ".json", ".log"]:
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            return jsonify({"session": sess_id, "filepath": filepath, "content": content}), 200
        except Exception as e:
            return jsonify({"error": f"Could not read file: {str(e)}"}), 500
    else:
        try:
            return send_file(abs_path, as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"Could not send file: {str(e)}"}), 500

@app.route("/api/ai-pocket-tutor/database/folders", methods=["GET"])
def list_ai_pocket_tutor_database_folders():
    try:
        db_path = os.path.join("AI-Pocket-Tutor", "database")
        print(f"[Server] Database folder path: {os.path.abspath(db_path)}")
        if not os.path.isdir(db_path):
            return jsonify({"error": "AI-Pocket-Tutor/database folder not found."}), 404
        items = os.listdir(db_path)
        folders = [item for item in items if os.path.isdir(os.path.join(db_path, item))]
        print(f"[Server] Found folders: {folders}")
        return jsonify({"database_folders": folders}), 200
    except Exception as e:
        return jsonify({"error": f"Could not list folders: {str(e)}"}), 500

@app.route("/api/database/notifications", methods=["GET"])
def get_db_notifications():
    global has_new_notification
    if has_new_notification:
        has_new_notification = False
        return jsonify(db_notifications), 200
    else:
        return jsonify([]), 200

@app.route("/api/database/notifications/clear", methods=["POST"])
def clear_db_notifications():
    global db_notifications
    db_notifications = []
    return jsonify({"message": "Notifications cleared"}), 200

# NEW: Global dictionary to store front-end messages.
received_messages_by_session = {}

@app.route("/api/cli-messages", methods=["GET"])
def get_cli_messages():
    msgs = []
    for key in received_messages_by_session:
        msgs.extend(received_messages_by_session[key])
    return jsonify({"received_messages": msgs}), 200

@app.route("/api/cli-message", methods=["POST", "OPTIONS"])
def enqueue_cli_message():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, ngrok-skip-browser-warning")
        return response, 200
    data = request.get_json(force=True)
    msg = data.get("message")
    timestamp = data.get("timestamp")  # Get the timestamp from the payload
    if not msg:
        return jsonify({"error": "No 'message' provided"}), 400

    # Create an object that includes both the message and timestamp
    message_obj = {
        "message": msg,
        "timestamp": timestamp
    }
    if "global" not in received_messages_by_session:
        received_messages_by_session["global"] = []
    received_messages_by_session["global"].append(message_obj)
    print(f"[Server] Enqueued message: {message_obj}")
    return jsonify({"status": "ok"}), 200

@app.route("/api/root/folders", methods=["GET"])
def list_root_folders():
    try:
        items = os.listdir(".")
        folders = [item for item in items if os.path.isdir(item)]
        return jsonify({"root_folders": folders}), 200
    except Exception as e:
        return jsonify({"error": f"Could not list folders: {str(e)}"}), 500

if __name__ == "__main__":
    observer = Observer()
    event_handler = DatabaseChangeHandler()
    observer.schedule(event_handler, path=DATABASE_ROOT, recursive=True)
    observer_thread = threading.Thread(target=observer.start, daemon=True)
    observer_thread.start()
    try:
        app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
    finally:
        observer.stop()
        observer.join()
