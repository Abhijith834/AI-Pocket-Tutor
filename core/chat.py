import os
import json
import subprocess
import ollama
import re
import sys
import time
import threading
import queue
import requests  # NEW: We'll poll the front-end messages
import core.config as config
import core.db_utils as db_utils
from core.config import CHAT_HISTORY_FILE
from core.web_search import (
    web_search_flow,
    wikipedia_flow,
    duckduckgo_search,
    gather_news_articles,
    extract_publication_date,
    scrape_webpage,
    summarize_article_content
)
import sys_msgs
import shutil

next_chat_session = None

SUMMARIZE_THRESHOLD = 10
INACTIVITY_TIMEOUT = 15 * 60
unsummarized_messages = []
MEMORY_UPDATE_THRESHOLD = 10
messages_since_summary = 0
last_input_time = time.time()
last_processed_timestamp = None

# NEW: Keep track of how many front-end messages we've seen so far
last_front_end_count = 0

def input_with_timeout(prompt, timeout):
    """
    Waits for user input OR new front-end message for 'timeout' seconds.
    1) If console input arrives first, return it.
    2) If a new front-end message arrives first, return it.
    3) If neither arrives within 'timeout', raise TimeoutError.
    """
    # We'll do short polling in a loop, up to 'timeout' seconds.
    start_time = time.time()

    # Create a queue for console input (as before)
    result = queue.Queue()

    def get_input():
        try:
            user_input = input(prompt)
            result.put(user_input)
        except Exception:
            result.put("")

    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()

    # Define both endpoints
    front_end_url_ngrok = "https://mint-jackal-publicly.ngrok-free.app/api/cli-messages"
    front_end_url_local = "http://localhost:5000/api/cli-messages"
    # Header required for ngrok
    ngrok_headers = {"ngrok-skip-browser-warning": "true"}

    global last_processed_timestamp

    # Clear old messages by doing an initial GET and updating last_processed_timestamp.
    try:
        try:
            resp = requests.get(front_end_url_ngrok, headers=ngrok_headers, timeout=2)
            if resp.status_code != 200:
                raise Exception(f"Ngrok response status: {resp.status_code}")
            print("Initial fetch using ngrok URL for cli-messages")
        except Exception as e:
            print("Initial ngrok fetch failed, falling back to localhost:5000:", e)
            resp = requests.get(front_end_url_local, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            messages = data.get("received_messages", [])
            if messages and len(messages) > 0:
                # Set last_processed_timestamp to the latest message's timestamp
                last_processed_timestamp = messages[-1].get("timestamp")
                print("Clearing old messages. Last processed timestamp set to:", last_processed_timestamp)
            else:
                last_processed_timestamp = None
        else:
            last_processed_timestamp = None
    except Exception as e:
        print("Initial polling error:", e)
        last_processed_timestamp = None

    while True:
        # 1) Check if console input is available
        if not result.empty():
            return result.get()

        # 2) Check if we've exceeded the timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise TimeoutError

        # 3) Poll for new front-end messages using fallback
        try:
            try:
                # Attempt to use the ngrok URL first.
                resp = requests.get(front_end_url_ngrok, headers=ngrok_headers, timeout=2)
                if resp.status_code != 200:
                    raise Exception(f"Ngrok response status: {resp.status_code}")
                # print("Using ngrok URL for cli-messages")
            except Exception as e:
                # print("Ngrok fetch failed, falling back to localhost:5000:", e)
                resp = requests.get(front_end_url_local, timeout=2)
            
            if resp.status_code == 200:
                data = resp.json()
                messages = data.get("received_messages", [])
                if messages and len(messages) > 0:
                    # Get the latest message (assumed to be last in the list)
                    latest_msg = messages[-1]
                    # Check if the latest message has a timestamp and if it's new
                    if latest_msg.get("timestamp") and latest_msg.get("timestamp") != last_processed_timestamp:
                        last_processed_timestamp = latest_msg.get("timestamp")
                        return latest_msg
            # If no new message, continue.
        except Exception as e:
            print("Polling error:", e)

        # 4) Short sleep before next check
        time.sleep(0.2)


def build_chunk_text(messages):
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

def do_incremental_summary(text_chunk: str) -> str:
    summary_prompt = (
        "You are an AI summarizer. Summarize the following conversation chunk in a concise form.\n\n"
        f"Conversation Chunk:\n{text_chunk}\n\n"
        "Output only the summary. Do not include any additional commentary."
    )
    try:
        resp = ollama.generate(model=config.MODEL, prompt=summary_prompt)
        summary = resp.get("response", "[No summary generated]")
        summary = db_utils.remove_think_clauses(summary)
        return summary
    except Exception as e:
        return f"(Error in summarization) {e}"

def summarize_if_needed():
    global messages_since_summary
    if messages_since_summary >= SUMMARIZE_THRESHOLD:
        chunk_to_summarize = db_utils.chat_history[-SUMMARIZE_THRESHOLD:]
        text_chunk = build_chunk_text(chunk_to_summarize)
        summary_text = do_incremental_summary(text_chunk)
        db_utils.update_memory_summary([{"role": "assistant", "content": summary_text}])
        db_utils.save_session_state()
        messages_since_summary = 0

def finalize_leftover_messages():
    if unsummarized_messages:
        db_utils.update_memory_summary(unsummarized_messages)
        unsummarized_messages.clear()
    if db_utils.chat_history:
        recent_chunk = db_utils.chat_history[-10:] if len(db_utils.chat_history) >= 10 else db_utils.chat_history[:]
        recent_text_chunk = build_chunk_text(recent_chunk)
        recent_summary = do_incremental_summary(recent_text_chunk)
        db_utils.set_recent_summary(recent_summary)
        db_utils.save_session_state()
    print("[System] Final summaries updated. Exiting now.")

def validate_answer(answer: str, user_query: str) -> bool:
    validation_prompt = (
        f"User Query: {user_query}\n"
        f"Answer: {answer}\n\n"
        "Is this answer satisfactory, accurate, and up-to-date? "
        "Output only 'yes' or 'no'."
    )
    resp = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msgs.answer_validation_msg},
            {"role": "user", "content": validation_prompt}
        ]
    )
    raw_validation = resp["message"]["content"].strip()
    val = db_utils.remove_think_clauses(raw_validation).lower()
    print(f"[validate_answer] validation: '{val}'")
    return val == "yes"

def master_answer_flow(user_input: str) -> str:
    if db_utils.active_collection:
        db_ans = db_utils.rag_ollama_chat(user_input)
        if db_ans and not db_ans.lower().startswith("i couldn't find relevant info"):
            if validate_answer(db_ans, user_input):
                return "[source: document]\n" + db_ans

    if should_search():
        web_ans = web_search_flow(user_input)
        if web_ans.strip():
            return "[source: web]\nI retrieved external information:\n\n" + web_ans

    internal_ans = db_utils.normal_ollama_chat(user_input)
    if validate_answer(internal_ans, user_input):
        return "[source: internal]\n" + internal_ans
    return "[source: internal]\n" + internal_ans

def should_search() -> bool:
    if not db_utils.chat_history:
        return False
    user_message = db_utils.chat_history[-1]
    sys_prompt = sys_msgs.search_or_not_msg
    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msgs.search_or_not_msg},
            user_message
        ]
    )
    raw_content = response["message"]["content"]
    content = db_utils.remove_think_clauses(raw_content).strip().lower()
    print(f"[search_or_not] LLM response: '{content}'")
    return content == "true"

def process_injected_file_command():
    global messages_since_summary

    if not db_utils.chat_history:
        return

    last_msg = db_utils.chat_history[-1]
    if last_msg["role"] == "user" and last_msg["content"].lower().startswith("file"):
        user_input = last_msg["content"]
        start = user_input.find("(")
        end = user_input.find(")")
        if start != -1 and end != -1:
            path = user_input[start+1:end].strip()
            if not os.path.exists(path):
                print(f"File '{path}' does not exist.")
                return
            unsummarized_messages.append({"role": "user", "content": user_input})
            db_utils.save_session_state()
            messages_since_summary += 1
            summarize_if_needed()

            print(f"[Chat] Ingesting file '{path}' immediately...")
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input", "input.py")
            subprocess.run(["python", script_path, path])
            pdf_base = os.path.splitext(os.path.basename(path))[0]
            new_coll_name = db_utils.sanitize_collection_name(pdf_base)
            db_utils.load_collection(new_coll_name)
            if db_utils.active_collection:
                db_utils.auto_summarize_and_suggest()

def main(final_pdf_path=None):
    global messages_since_summary, last_input_time, next_chat_session
    last_input_time = time.time()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                db_utils.chat_history[:] = data.get("chat_history", [])
        except Exception as e:
            print(f"[Chat] Could not load chat history: {e}")
    else:
        db_utils.chat_history[:] = []

    if db_utils.active_collection_name:
        print(f"[Chat] Active collection loaded: {db_utils.active_collection_name}")
    else:
        print("[Chat] No active collection loaded.")

    print("\nType your message to chat. For PDF ingestion, type: file (C:\\path\\to.pdf).")
    print("Type 'exit' to quit.\n")

    if final_pdf_path:
        print("[Chat] Learning mode provided a PDF. Ingesting now...\n")
        user_cmd = f"file ({final_pdf_path})"
        db_utils.chat_history.append({"role": "user", "content": user_cmd})
        unsummarized_messages.append({"role": "user", "content": user_cmd})
        db_utils.save_session_state()
        messages_since_summary += 1
        summarize_if_needed()

        print(f"[Chat] Ingesting file '{final_pdf_path}' automatically...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input", "input.py")
        subprocess.run(["python", script_path, final_pdf_path])
        pdf_base = os.path.splitext(os.path.basename(final_pdf_path))[0]
        new_coll_name = db_utils.sanitize_collection_name(pdf_base)
        db_utils.load_collection(new_coll_name)
        if db_utils.active_collection:
            db_utils.auto_summarize_and_suggest()

    while True:
        remaining = INACTIVITY_TIMEOUT - (time.time() - last_input_time)
        if remaining <= 0:
            print("[System] Inactivity timeout reached. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            return next_chat_session

        try:
            received = input_with_timeout("USER:\n", timeout=remaining)
            if isinstance(received, dict):
                user_input = received.get("message", "")
            else:
                user_input = received.strip()
        except TimeoutError:
            print("[System] Inactivity timeout reached. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            return next_chat_session
        except EOFError:
            print("[System] EOF detected. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            return next_chat_session
        except Exception as e:
            print(f"[System] Error reading input: {e}. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            return next_chat_session

        last_input_time = time.time()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("[System] Exit command received. Finalizing conversation.")
            finalize_leftover_messages()
            return next_chat_session
        
        # ----- TTS Command Handling -----
        # Expected command format: tts (chat_2#0)
        if user_input.lower().startswith("tts (") and user_input.endswith(")"):
            # Extract the command inside the parentheses.
            tts_command = user_input[user_input.find("(") + 1 : user_input.rfind(")")].strip()
            # Expected pattern: chat_<sessionID>#<block_index>
            pattern = re.compile(r"^(chat_\d+)#(\d+)$", re.IGNORECASE)
            match = pattern.match(tts_command)
            if match:
                session_ref = match.group(1)         # e.g., "chat_2"
                block_number = int(match.group(2))     # 0-indexed for assistant messages
                # Build the chat history file path:
                chat_history_file = os.path.join(BASE_DIR, "database", session_ref, "chat_history.json")
                if not os.path.exists(chat_history_file):
                    print(f"Chat history file not found for session {session_ref}")
                    continue
                try:
                    with open(chat_history_file, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                    chat_history = history_data.get("chat_history", [])
                except Exception as e:
                    print(f"Failed to load chat history for {session_ref}: {e}")
                    continue
                # Filter for assistant messages only (indexed from 0)
                assistant_messages = [msg for msg in chat_history if msg.get("role", "").lower() == "assistant"]
                if block_number < 0 or block_number >= len(assistant_messages):
                    print(f"Invalid block number: {block_number} (total assistant messages: {len(assistant_messages)})")
                    continue
                tts_text = assistant_messages[block_number].get("content", "")
                print(f"Extracted TTS text from {session_ref} assistant block {block_number}:\n{tts_text}")
            else:
                # If no pattern match, use the provided text directly.
                tts_text = tts_command
                print(f"Using direct TTS text: {tts_text}")

            # Build the TTS script path.
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "Text_to_Speach.py")
            if not os.path.exists(script_path):
                alt_script_path = script_path + ".py"
                if os.path.exists(alt_script_path):
                    script_path = alt_script_path
                else:
                    print(f"Error: TTS script not found at {script_path} or {alt_script_path}")
                    continue

            print(f"Using TTS script at: {script_path}")
            try:
                subprocess.run(["python", script_path, tts_text], check=True)
                print("TTS conversion completed successfully.")
            except subprocess.CalledProcessError as e:
                print("TTS subprocess failed:", e)
                continue

            # Assume the TTS script writes the audio file to "output.wav" in the tools folder.
            source_audio_path = os.path.join(os.path.dirname(script_path), "output.wav")
            if not os.path.exists(source_audio_path):
                print("Audio file not found after TTS conversion.")
                continue

            # Build the destination folder under the corresponding session's database.
            dest_folder = os.path.join(BASE_DIR, "database", session_ref, "tts")
            os.makedirs(dest_folder, exist_ok=True)
            # Save the audio file with the name "chat_2#0.wav" (using the received command)
            dest_filename = f"{session_ref}#{block_number}.wav"
            dest_audio_path = os.path.join(dest_folder, dest_filename)

            try:
                # Move (rename) the audio file to the destination folder.
                shutil.move(source_audio_path, dest_audio_path)
                print(f"Audio file saved as: {dest_audio_path}")
            except Exception as ex:
                print("Error saving audio file:", ex)
                continue

            # Now send the audio file to the front end.
            try:
                tts_endpoint = f"http://localhost:5000/api/tts/{session_ref}"
                with open(dest_audio_path, "rb") as audio_file:
                    files = {"file": audio_file}
                    # Optionally, you might also pass the desired file name.
                    data = {"filename": dest_filename}
                    response = requests.post(tts_endpoint, files=files, data=data)
                if response.status_code == 200:
                    print("Audio file sent to front end successfully.")
                else:
                    print(f"Failed to send audio file. HTTP status code: {response.status_code}")
            except Exception as ex:
                print("Error sending audio file to front end:", ex)
            # Continue with next input.
            continue
        # ----- End TTS Command Handling -----



        # ----- Start STT Command Handling -----
        if user_input.lower().startswith("stt (") and user_input.endswith(")"):
            # Extract audio file path
            audio_path = user_input[user_input.find("(")+1:user_input.rfind(")")].strip()
            print(f"Detected STT command. Transcribing file: {audio_path}")

            if not os.path.exists(audio_path):
                print(f"Error: File not found: {audio_path}")
                continue

            # Construct the path to the Speach_to_Text.py script
            script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "tools", "Speach_to_Text.py"
            )
            if not os.path.exists(script_path):
                alt_script_path = script_path + ".py"
                if os.path.exists(alt_script_path):
                    script_path = alt_script_path
                else:
                    print(f"Error: STT script not found at {script_path} or {alt_script_path}")
                    continue

            print(f"Using STT script at: {script_path}")

            # Prepare the output path
            import tempfile
            output_file = tempfile.mktemp(suffix="_transcription.txt")

            try:
                subprocess.run(["python", script_path, audio_path, output_file], check=True)
                print("STT transcription completed successfully.")
            except subprocess.CalledProcessError as e:
                print("STT subprocess failed:", e)
                continue

            # Read and display the transcription
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        transcription = f.read().strip()
                    print(f"Transcription result:\n\n{transcription}\n")
                except Exception as e:
                    print("Error reading transcription file:", e)
            else:
                print("Error: Output transcription file not found.")

            continue  # Skip further processing for this input
        # ----- End STT Command Handling -----
        if user_input.lower() in ["new chat(normal)", "new chat (normal)"]:
            print("[System] Creating a new NORMAL chat session now.")
            finalize_leftover_messages()
            return "NEWCHAT:normal"

        if user_input.lower() in ["new chat(learning)", "new chat (learning)"]:
            print("[System] Creating a new LEARNING chat session now.")
            finalize_leftover_messages()
            return "NEWCHAT:learning"

        chat_match = re.match(r'^\s*chat\s*\(\s*(\d+)\s*\)\s*$', user_input.lower())
        if chat_match:
            new_session_id = chat_match.group(1)
            print(f"[System] Jumping to chat session {new_session_id} now.")
            finalize_leftover_messages()
            next_chat_session = new_session_id
            return next_chat_session

        if user_input.lower().startswith("file"):
            start = user_input.find("(")
            end = user_input.find(")")
            if start != -1 and end != -1:
                path = user_input[start+1:end].strip()
                if not os.path.exists(path):
                    print(f"File '{path}' does not exist.")
                    continue
                db_utils.chat_history.append({"role": "user", "content": user_input})
                unsummarized_messages.append({"role": "user", "content": user_input})
                db_utils.save_session_state()
                messages_since_summary += 1
                summarize_if_needed()

                print(f"[Chat] Ingesting file '{path}'...")
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input", "input.py")
                subprocess.run(["python", script_path, path])
                pdf_base = os.path.splitext(os.path.basename(path))[0]
                new_coll_name = db_utils.sanitize_collection_name(pdf_base)
                db_utils.load_collection(new_coll_name)
                if db_utils.active_collection:
                    db_utils.auto_summarize_and_suggest()
                continue
            else:
                print("Please specify file path in parentheses: file (C:\\path\\to\\doc.pdf)")
                continue

        db_utils.chat_history.append({"role": "user", "content": user_input})
        unsummarized_messages.append({"role": "user", "content": user_input})
        db_utils.save_session_state()
        messages_since_summary += 1
        summarize_if_needed()

        answer = master_answer_flow(user_input)
        print(f"Chatbot: {answer}\n")
        db_utils.chat_history.append({"role": "assistant", "content": answer})
        unsummarized_messages.append({"role": "assistant", "content": answer})
        db_utils.save_session_state()
        messages_since_summary += 1
        summarize_if_needed()

        if len(unsummarized_messages) >= MEMORY_UPDATE_THRESHOLD:
            db_utils.update_memory_summary(unsummarized_messages)
            unsummarized_messages.clear()
