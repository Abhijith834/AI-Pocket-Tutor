# core/db_utils.py

import os
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from core.config import CHROMA_DB_DIR, CHAT_HISTORY_FILE, SESSION_STATE_FILE, MAX_TEXT_LENGTH, MODEL

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
query_model = SentenceTransformer("all-MiniLM-L6-v2")

active_collection = None
active_collection_name = None
chat_history = []

# New globals for long-term memory
memory_summary = ""
memory_included = False

def embed_query(query_text: str):
    return query_model.encode([query_text]).tolist()[0]

def remove_think_clauses(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def sanitize_collection_name(name: str) -> str:
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    if len(name) < 3:
        name = name.ljust(3, "_")
    if len(name) > 63:
        name = name[:63]
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    if not name:
        name = "default"
    return name

def load_session_state():
    global active_collection, active_collection_name, chat_history
    # Load chat history
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                chat_history[:] = json.load(f)
        except Exception as e:
            print(f"[DB] Could not load chat history: {e}")
    else:
        chat_history[:] = []

    if os.path.exists(SESSION_STATE_FILE):
        try:
            with open(SESSION_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                last_collection = data.get("active_collection_name", None)
                if last_collection:
                    load_collection(last_collection)
        except Exception as e:
            print(f"[DB] Could not load session state: {e}")

def save_session_state():
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DB] Could not save chat history: {e}")

    state_data = {"active_collection_name": active_collection_name}
    try:
        with open(SESSION_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DB] Could not save session state: {e}")

def load_collection(collection_name: str):
    global active_collection, active_collection_name
    try:
        c = client.get_collection(name=collection_name)
        active_collection = c
        active_collection_name = collection_name
        print(f"[DB] Active collection loaded: '{active_collection_name}'")
    except Exception as e:
        print(f"[DB] Error loading collection '{collection_name}': {e}")

def auto_summarize_and_suggest():
    if not active_collection:
        return
    try:
        all_data = active_collection.get(include=["documents"])
        docs = []
        for sublist in all_data["documents"]:
            docs.extend(sublist)
        combined_text = " ".join(docs)
        if len(combined_text) > MAX_TEXT_LENGTH:
            combined_text = combined_text[:MAX_TEXT_LENGTH] + "...(truncated)..."

        summarization_prompt = (
            f"You are an AI assistant.\n\n"
            f"Here is the text from a newly ingested PDF (collection: {active_collection_name}):\n\n"
            f"{combined_text}\n\n"
            "1) Summarize this PDF in a few paragraphs.\n"
            "2) Suggest 3 intelligent questions a user might ask about this PDF.\n\nAssistant:"
        )
        import ollama
        resp = ollama.generate(model=MODEL, prompt=summarization_prompt)
        summary_text = resp.get("response", "[No summary generated]")
        summary_text = remove_think_clauses(summary_text)
        print("\n[DB] Auto-Summary + Suggested Questions:\n")
        print(summary_text)
        chat_history.append({"role": "assistant", "content": summary_text})
        save_session_state()
    except Exception as e:
        print(f"[DB] Error summarizing new PDF: {e}")

def normal_ollama_chat(user_input: str) -> str:
    import ollama
    global memory_included, memory_summary
    if memory_summary and not memory_included:
         prompt = (
             f"You are an AI assistant. Here is your long-term memory from previous conversations:\n{memory_summary}\n\n"
             f"User: {user_input}\n\nAssistant:"
         )
         memory_included = True
    else:
         prompt = f"You are an AI assistant.\n\nUser: {user_input}\n\nAssistant:"
    try:
        out = ollama.generate(model=MODEL, prompt=prompt)
        response_text = out.get("response", "No response")
        response_text = remove_think_clauses(response_text)
        return response_text
    except Exception as e:
        return f"(Error in normal Ollama chat) {e}"

def rag_ollama_chat(user_input: str) -> str:
    if not active_collection:
        return normal_ollama_chat(user_input)
    qembed = embed_query(user_input)
    import ollama
    try:
        results = active_collection.query(
            query_embeddings=[qembed],
            n_results=1,
            include=["documents", "distances", "metadatas"]
        )
    except Exception as e:
        return f"(Error querying active collection) {e}"

    if not results or not results.get("documents") or not results["documents"][0]:
        fallback = normal_ollama_chat(user_input)
        return f"I couldn't find relevant info in the vector DB.\n{fallback}"

    relevant_data = results["documents"][0][0]
    prompt = (
        f"You are an AI assistant. Use the following context from your documents:\n\n"
        f"{relevant_data}\n\n"
        f"Now answer the user's question:\nUser: {user_input}\n\nAssistant:"
    )
    try:
        out = ollama.generate(model=MODEL, prompt=prompt)
        response_text = out.get("response", "No response")
        response_text = remove_think_clauses(response_text)
        return response_text
    except Exception as e:
        return f"(Error generating RAG answer) {e}"

def get_session_folder():
    return os.path.dirname(os.getenv("CHAT_HISTORY_FILE", os.getcwd()))

def load_memory_summary():
    global memory_summary
    mem_file = os.path.join(get_session_folder(), "memory_summary.txt")
    if os.path.exists(mem_file):
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                memory_summary = f.read().strip()
        except Exception as e:
            print(f"[DB] Error loading memory summary: {e}")
    return memory_summary

def save_memory_summary():
    mem_file = os.path.join(get_session_folder(), "memory_summary.txt")
    try:
        with open(mem_file, "w", encoding="utf-8") as f:
            f.write(memory_summary)
    except Exception as e:
        print(f"[DB] Could not save memory summary: {e}")

def update_memory_summary(new_messages):
    global memory_summary
    new_text = build_chunk_text(new_messages)
    prompt = (
        "You are an AI assistant maintaining a long-term memory of a conversation. "
        "You already have an existing memory summary (which compresses older parts) and now you have new conversation turns.\n\n"
        f"Existing Memory Summary:\n{memory_summary if memory_summary else '[None]'}\n\n"
        f"New Conversation Turns:\n{new_text}\n\n"
        "Update the memory summary so that older parts are very brief and the new parts are described in detail. "
        "Output only the updated memory summary."
    )
    import ollama
    try:
        resp = ollama.generate(model=MODEL, prompt=prompt)
        updated = resp.get("response", "[No memory update]")
        memory_summary = updated.strip()
        save_memory_summary()
        print("[DB] Memory summary updated.")
    except Exception as e:
        print(f"[DB] Error updating memory summary: {e}")
    return memory_summary

def build_chunk_text(messages):
    """
    Helper function to build text from a list of messages.
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)