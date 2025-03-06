# core/chat.py

import os
import json
import subprocess
import ollama
import re
import sys
import time

import core.config as config
import core.db_utils as db_utils
from core.config import CHAT_HISTORY_FILE, SESSION_STATE_FILE
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

################################################
# SETTINGS
################################################

SUMMARIZE_THRESHOLD = 10         # Number of messages (user+assistant) after which to summarize
INACTIVITY_TIMEOUT = 5 * 60      # 5 minutes in seconds


unsummarized_messages = []        # New messages not yet incorporated into long-term memory
MEMORY_UPDATE_THRESHOLD = 10      # Update memory after every 10 messages
messages_since_summary = 0        # Counter for internal summarization of chat_history
last_input_time = time.time()     # Timestamp of last user input

def build_chunk_text(messages):
    """
    Creates a single text block from a list of conversation messages.
    Each message is a dict with keys 'role' and 'content'.
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

def do_incremental_summary(text_chunk: str) -> str:
    """
    Summarizes the provided conversation chunk using the LLM.
    Removes any <think>...</think> blocks from the output.
    """
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
    """
    Checks if enough new messages have been accumulated.
    If messages_since_summary >= SUMMARIZE_THRESHOLD, summarizes the oldest chunk,
    removes them from the active conversation, and appends the summary.
    """
    global messages_since_summary
    if messages_since_summary >= SUMMARIZE_THRESHOLD:
        # Chunk the oldest SUMMARIZE_THRESHOLD messages
        chunk_to_summarize = db_utils.chat_history[:SUMMARIZE_THRESHOLD]
        text_chunk = build_chunk_text(chunk_to_summarize)
        summary_text = do_incremental_summary(text_chunk)
        # Remove the summarized messages and append the summary
        db_utils.chat_history = db_utils.chat_history[SUMMARIZE_THRESHOLD:]
        db_utils.chat_history.append({"role": "assistant", "content": summary_text})
        db_utils.save_session_state()
        messages_since_summary = 0

def finalize_leftover_messages():
    """
    Summarizes any remaining messages in chat_history before exit.
    Also updates the memory summary with unsummarized messages.
    """
    if unsummarized_messages:
        db_utils.update_memory_summary(unsummarized_messages)
        unsummarized_messages.clear()
    if db_utils.chat_history:
        text_chunk = build_chunk_text(db_utils.chat_history)
        summary_text = do_incremental_summary(text_chunk)
        db_utils.chat_history[:] = []
        db_utils.chat_history.append({"role": "assistant", "content": summary_text})
        db_utils.save_session_state()
    print("[System] Final summary complete. Exiting now.")

################################################
# DECISION AGENTS (Same as before)
################################################

def decide_db_or_web(user_input: str) -> str:
    if not db_utils.active_collection:
        return "web"
    qembed = db_utils.embed_query(user_input)
    try:
        results = db_utils.active_collection.query(
            query_embeddings=[qembed],
            n_results=1,
            include=["distances"]
        )
        if not results or not results.get("distances") or not results["distances"][0]:
            return "web"
        distance = results["distances"][0][0]
        print(f"[decide_db_or_web] distance = {distance}")
        if distance < 0.5:
            return "db"
        else:
            return "web"
    except Exception as e:
        print(f"[decide_db_or_web] Error: {e}")
        return "web"

def search_or_not() -> bool:
    """
    Asks the LLM if external data is needed based on the last user message.
    Uses a refined prompt that instructs the agent to err on the side of 'True'
    if there is any doubt.
    """
    user_message = db_utils.chat_history[-1]
    sys_prompt = sys_msgs.search_or_not_msg
    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            user_message
        ]
    )
    raw_content = response["message"]["content"]
    content = db_utils.remove_think_clauses(raw_content).strip().lower()
    print(f"[search_or_not] LLM response: '{content}'")
    return content == "true"

def decide_answer_source(user_input: str) -> str:
    """
    If a PDF (vector DB) is active, always use DB retrieval.
    Otherwise, if search_or_not() returns True, force ephemeral web search.
    Otherwise, use internal (model) answer.
    """
    if db_utils.active_collection:
        print("[decide_answer_source] Active collection detected -> using DB retrieval.")
        return "db"
    if search_or_not():
        return "web"
    return "normal"

def validate_answer(answer: str, user_query: str) -> bool:
    """
    Validates if the provided answer is satisfactory, accurate, and up-to-date.
    """
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
    """
    Returns a single final answer using this priority:
      1) DB (RAG) if file loaded
      2) Internal (normal)
      3) Web search
      4) Final fallback => internal
    The returned answer is prefixed with the source label.
    """
    # Step 1: If PDF (vector DB) is active, try DB retrieval.
    if db_utils.active_collection:
        db_ans = db_utils.rag_ollama_chat(user_input)
        if db_ans and not db_ans.lower().startswith("i couldn't find relevant info"):
            if validate_answer(db_ans, user_input):
                return "[source: document]\n" + db_ans

    # Step 2: If the LLM indicates that external info is needed, force web search.
    if search_or_not():
        web_ans = web_search_flow(user_input)
        if web_ans.strip():
            return "[source: web]\nI retrieved external information:\n\n" + web_ans

    # Step 3: Otherwise, use the internal (normal) answer.
    internal_ans = db_utils.normal_ollama_chat(user_input)
    if validate_answer(internal_ans, user_input):
        return "[source: internal]\n" + internal_ans

    # Fallback: return the internal answer.
    return "[source: internal]\n" + internal_ans


################################################
# MAIN CHAT LOOP
################################################

def main():
    global messages_since_summary, last_input_time
    last_input_time = time.time()

    # Load existing session
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                db_utils.chat_history[:] = json.load(f)
        except Exception as e:
            print(f"[Chat] Could not load chat history: {e}")
    else:
        db_utils.chat_history[:] = []

    if os.path.exists(SESSION_STATE_FILE):
        try:
            with open(SESSION_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                last_coll = data.get("active_collection_name", None)
                if last_coll:
                    db_utils.load_collection(last_coll)
        except Exception as e:
            print(f"[Chat] Could not load session state: {e}")

    if db_utils.active_collection_name:
        print(f"[Chat] Resuming with active collection: {db_utils.active_collection_name}")
    else:
        print("[Chat] No active collection loaded.")

    print("\nType your message to chat. For PDF ingestion, type: file (C:\\path\\to.pdf).\nType 'exit' to quit.\n")

    while True:
        # Check for inactivity timeout
        if time.time() - last_input_time > INACTIVITY_TIMEOUT:
            print("[System] Inactivity timeout reached. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            sys.exit(0)

        try:
            user_input = input("USER:\n").strip()
        except EOFError:
            print("[System] EOF detected. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            sys.exit(0)
        except Exception as e:
            print(f"[System] Error reading input: {e}. Finalizing conversation and exiting.")
            finalize_leftover_messages()
            sys.exit(0)

        last_input_time = time.time()  # Reset inactivity timer

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("[System] Exit command received. Finalizing conversation.")
            finalize_leftover_messages()
            print("Exiting. Goodbye!")
            sys.exit(0)

        # Handle file ingestion command
        if user_input.lower().startswith("file"):
            start = user_input.find("(")
            end = user_input.find(")")
            if start != -1 and end != -1:
                fpath = user_input[start+1:end].strip()
                if not os.path.exists(fpath):
                    print(f"File '{fpath}' does not exist.")
                    continue
                db_utils.chat_history.append({"role": "user", "content": user_input})
                db_utils.save_session_state()
                messages_since_summary += 1
                summarize_if_needed()

                print(f"[Chat] Ingesting file '{fpath}'...")
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input", "input.py")
                subprocess.run(["python", script_path, fpath])
                pdf_base = os.path.splitext(os.path.basename(fpath))[0]
                new_coll_name = db_utils.sanitize_collection_name(pdf_base)
                db_utils.load_collection(new_coll_name)
                if db_utils.active_collection:
                    db_utils.auto_summarize_and_suggest()
                continue
            else:
                print("Please specify file path in parentheses: file (C:\\path\\to\\doc.pdf)")
                continue

        # Process a normal user query
        db_utils.chat_history.append({"role": "user", "content": user_input})
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


if __name__ == "__main__":
    main()
