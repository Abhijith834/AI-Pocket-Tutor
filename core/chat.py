import os
import json
import subprocess
import ollama
import re

# Instead of importing individual globals, import the module as a namespace.
import core.db_utils as db_utils
from core.config import CHAT_HISTORY_FILE, SESSION_STATE_FILE, EPHEMERAL_KEYWORDS
from core.web_search import (wikipedia_flow, duckduckgo_search, gather_news_articles, web_search_flow,
                             extract_publication_date, scrape_webpage, summarize_article_content)
from core.db_utils import rag_ollama_chat, normal_ollama_chat
import sys_msgs

################################################
# DECISION AGENTS (DB vs. WEB, etc.)
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
    # Simple. The LLM decides if external data is needed.
    user_message = db_utils.chat_history[-1]
    sys_prompt = sys_msgs.search_or_not_msg
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": sys_prompt},
            user_message
        ]
    )
    content = response["message"]["content"].strip().lower()
    print(f"[search_or_not] agent response: '{content}'")
    return content == "true"



def decide_answer_source(user_input: str) -> str:
    """
    If a file is loaded, always do 'db'.
    If no file is loaded, check if search_or_not() => 'web' or 'normal'.
    """
    if db_utils.active_collection:
        print("[decide_answer_source] Active collection -> forcing DB retrieval.")
        return "db"
    if search_or_not():
        return "web"
    return "normal"


def validate_answer(answer: str, user_query: str) -> bool:
    # Also purely LLM-based. No ephemeral checks in code.
    validation_prompt = (
        f"User Query: {user_query}\n"
        f"Answer: {answer}\n\n"
        "Is this answer satisfactory, accurate, and up-to-date? "
        "Output only 'yes' or 'no'."
    )
    resp = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": sys_msgs.answer_validation_msg},
            {"role": "user", "content": validation_prompt}
        ]
    )
    validation = resp["message"]["content"].strip().lower()
    print(f"[validate_answer] validation: '{validation}'")
    return validation == "yes"


def master_answer_flow(user_input: str) -> str:
    """
    Returns a single, consolidated answer for the user's query using this priority:
      1) If an active collection exists, try the DB (RAG) answer.
         If the DB answer is validated as satisfactory, return it immediately.
      2) Otherwise, try the internal (normal) answer.
         If validated as satisfactory, return it.
      3) Otherwise, perform a web search.
         If web search yields an answer, return that prefixed with a note.
      4) As a final fallback, return the internal answer.
    """
    # Step 1: Try DB retrieval if an active collection exists.
    if db_utils.active_collection:
        db_ans = rag_ollama_chat(user_input)
        if db_ans and not db_ans.lower().startswith("i couldn't find relevant info"):
            if validate_answer(db_ans, user_input):
                return db_ans

    # Step 2: Try internal (normal) answer.
    internal_ans = normal_ollama_chat(user_input)
    if validate_answer(internal_ans, user_input):
        return internal_ans

    # Step 3: Fallback to web search.
    web_ans = web_search_flow(user_input)
    if web_ans.strip():
        return ("I could not find sufficient internal context, so I retrieved external information.\n\n" 
                + web_ans)

    # Step 4: Final fallback: return the internal answer.
    return internal_ans



################################################
# MAIN CHAT LOOP
################################################

def main():
    # Use db_utils to access global state and session functions.
    # We also import from db_utils at the top of chat.py for references like db_utils.chat_history.

    # Load session state from disk
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

    print("\nType your message to chat. For PDF ingestion, type: file (C:\\path\\to\\pdf).\nType 'exit' to quit.\n")

    while True:
        user_input = input("USER:\n").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # Check file ingestion command
        if user_input.lower().startswith("file"):
            start = user_input.find("(")
            end = user_input.find(")")
            if start != -1 and end != -1:
                fpath = user_input[start+1:end].strip()
                if not os.path.exists(fpath):
                    print(f"File '{fpath}' does not exist.")
                    continue
                db_utils.chat_history.append({"role": "user", "content": user_input})
                db_utils.save_session_state()  # <-- CHANGED
                print(f"[Chat] Ingesting file '{fpath}'...")
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input", "input.py")
                subprocess.run(["python", script_path, fpath])
                pdf_base = os.path.splitext(os.path.basename(fpath))[0]
                new_coll_name = db_utils.sanitize_collection_name(pdf_base)
                db_utils.load_collection(new_coll_name)  # <-- CHANGED
                if db_utils.active_collection:
                    db_utils.auto_summarize_and_suggest()  # <-- Use the function from db_utils
                continue
            else:
                print("Please specify file path in parentheses: file (C:\\path\\to\\doc.pdf)")
                continue

        # Normal user query
        db_utils.chat_history.append({"role": "user", "content": user_input})
        db_utils.save_session_state()  # <-- CHANGED

        answer = master_answer_flow(user_input)
        print(f"Chatbot: {answer}\n")
        db_utils.chat_history.append({"role": "assistant", "content": answer})
        db_utils.save_session_state()  # <-- CHANGED


if __name__ == "__main__":
    main()
