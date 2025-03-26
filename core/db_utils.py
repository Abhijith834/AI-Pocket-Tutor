import os
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from core.config import CHROMA_DB_DIR, CHAT_HISTORY_FILE, MAX_TEXT_LENGTH, MODEL

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
query_model = SentenceTransformer("all-MiniLM-L6-v2")

active_collection = None
active_collection_name = None
chat_history = []

# Global long-term memory summary.
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
    """
    Loads the chat history, long-term memory summary, and recent summary from CHAT_HISTORY_FILE.
    Then, if active_collection_name is found, loads that collection from the DB.
    """
    global chat_history, memory_summary, recent_summary, active_collection, active_collection_name
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                chat_history[:] = data.get("chat_history", [])
                memory_summary = data.get("memory_summary", "")
                recent_summary = data.get("recent_summary", "")
                saved_coll = data.get("active_collection_name", None)
                if saved_coll:
                    load_collection(saved_coll)
        except Exception as e:
            print(f"[DB] Could not load session state: {e}")
    else:
        chat_history[:] = []
        memory_summary = ""
        recent_summary = ""


# Add a new global variable for recent summary
recent_summary = ""

def save_session_state():
    """
    Saves the chat history, memory summary, recent summary, and active_collection_name
    into the same JSON file.
    """
    data = {
        "chat_history": chat_history,
        "memory_summary": memory_summary,
        "recent_summary": recent_summary,
        "active_collection_name": active_collection_name
    }
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DB] Could not save session state: {e}")

def set_recent_summary(summary):
    global recent_summary
    recent_summary = summary


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
    """
    If there's an active collection, use it for retrieval-augmented generation.
    Otherwise, fall back to normal chat logic.
    """
    if not active_collection:
        return normal_ollama_chat(user_input)
    import ollama
    qembed = embed_query(user_input)
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

# ----- Long-term Memory Functions -----
def update_memory_summary(new_messages):
    """
    Appends a detailed summary of the last 10 messages to the existing memory summary.
    """
    global memory_summary
    new_text = build_chunk_text(new_messages)
    prompt = (
        "You are an AI assistant maintaining a long-term memory of a conversation. "
        "You already have an existing conversation summary that covers older details. "
        "Now, here are the last 10 conversation turns that need to be kept detailed:\n\n"
        f"Existing Conversation Summary:\n{memory_summary if memory_summary else '[None]'}\n\n"
        f"Last 10 Conversation Turns (detailed):\n{new_text}\n\n"
        "Merge these into an updated conversation summary that retains all key details, "
        "with the older parts compressed and the last 10 turns described in detail. "
        "Output only the updated conversation summary."
    )
    import ollama
    try:
        resp = ollama.generate(model=MODEL, prompt=prompt)
        updated = resp.get("response", "[No memory update]")
        memory_summary = updated.strip()
        save_session_state()
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
