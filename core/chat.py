import os
import json
import subprocess
import ollama
import re
import sys
import time
import threading
import queue
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

next_chat_session = None

SUMMARIZE_THRESHOLD = 10
INACTIVITY_TIMEOUT = 5 * 60
unsummarized_messages = []
MEMORY_UPDATE_THRESHOLD = 10
messages_since_summary = 0
last_input_time = time.time()

def input_with_timeout(prompt, timeout):
    """Waits for user input for 'timeout' seconds. Raises TimeoutError if no input is provided."""
    result = queue.Queue()

    def get_input():
        try:
            user_input = input(prompt)
            result.put(user_input)
        except Exception:
            result.put("")
    
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()

    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError
    
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
        # Take the last SUMMARIZE_THRESHOLD messages for summarization
        chunk_to_summarize = db_utils.chat_history[-SUMMARIZE_THRESHOLD:]
        text_chunk = build_chunk_text(chunk_to_summarize)
        summary_text = do_incremental_summary(text_chunk)
        # Update the memory summary with the detailed summary of these messages
        db_utils.update_memory_summary([{"role": "assistant", "content": summary_text}])
        db_utils.save_session_state()
        messages_since_summary = 0

def finalize_leftover_messages():
    if unsummarized_messages:
        db_utils.update_memory_summary(unsummarized_messages)
        unsummarized_messages.clear()
    if db_utils.chat_history:
        # Generate recent summary from the last 10 messages (or fewer if not available)
        recent_chunk = db_utils.chat_history[-10:] if len(db_utils.chat_history) >= 10 else db_utils.chat_history[:]
        recent_text_chunk = build_chunk_text(recent_chunk)
        recent_summary = do_incremental_summary(recent_text_chunk)
        # Update the recent summary in the session state (outside chat_history)
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
        # Optionally validate
        if db_ans and not db_ans.lower().startswith("i couldn't find relevant info"):
            if validate_answer(db_ans, user_input):
                return "[source: document]\n" + db_ans

    # If no active collection or doc-based answer not found:
    # Optionally check if we need external search
    if should_search():
        web_ans = web_search_flow(user_input)
        if web_ans.strip():
            return "[source: web]\nI retrieved external information:\n\n" + web_ans

    # Otherwise fallback to internal
    internal_ans = db_utils.normal_ollama_chat(user_input)
    if validate_answer(internal_ans, user_input):
        return "[source: internal]\n" + internal_ans
    return "[source: internal]\n" + internal_ans

def should_search() -> bool:
    """
    Asks the LLM if external data is needed based on the last user message.
    """
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

#
# --------------------- NEW HELPER FUNCTION ---------------------
#
def process_injected_file_command():
    """
    Checks if the last user message in chat_history is a 'file(...)' command.
    If so, run the ingestion pipeline and set the active collection immediately.
    This ensures that when normal chat starts, the doc is already loaded.
    """
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
    """
    The main chat loop. If final_pdf_path is provided, we wait until after the
    normal chat interface is shown, then ingest that PDF automatically.
    """
    global messages_since_summary, last_input_time, next_chat_session
    last_input_time = time.time()

    # Attempt to load chat history (if not already loaded by main.py)
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                db_utils.chat_history[:] = data.get("chat_history", [])
        except Exception as e:
            print(f"[Chat] Could not load chat history: {e}")
    else:
        db_utils.chat_history[:] = []

    # If there's an active collection, we show it; otherwise, say none.
    if db_utils.active_collection_name:
        print(f"[Chat] Active collection loaded: {db_utils.active_collection_name}")
    else:
        print("[Chat] No active collection loaded.")

    print("\nType your message to chat. For PDF ingestion, type: file (C:\\path\\to.pdf).")
    print("Type 'exit' to quit.\n")

    # ----------- LATE INGEST of final_pdf_path -----------
    if final_pdf_path:
        print("[Chat] Learning mode provided a PDF. Ingesting now...\n")
        # Exactly as if user typed: file (the/path)
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

    # ------------- Now the normal chat loop -------------
    while True:
        remaining = INACTIVITY_TIMEOUT - (time.time() - last_input_time)
        if remaining <= 0:
            print("[System] Inactivity timeout reached. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            return next_chat_session

        try:
            # Use input_with_timeout with the remaining time as timeout
            user_input = input_with_timeout("USER:\n", timeout=remaining).strip()
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

        last_input_time = time.time()  # Update last input time after successful input
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("[System] Exit command received. Finalizing conversation.")
            finalize_leftover_messages()
            return next_chat_session
                # If user typed new chat(...) command in normal chat
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

        # If user typed file(...) command in normal chat
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

        # Normal user query
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

        # Memory update if needed
        if len(unsummarized_messages) >= MEMORY_UPDATE_THRESHOLD:
            db_utils.update_memory_summary(unsummarized_messages)
            unsummarized_messages.clear()
