#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# ── NEW ── Ensure we can import core.config when this script lives under tools/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import MODEL  # import the MODEL from your config

def remove_think_clauses(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def load_single_collection(session_id: str):
    tools_dir = os.path.abspath(os.path.dirname(__file__))
    database_root = os.path.join(tools_dir, "..", "database")
    session_folder = os.path.join(database_root, f"chat_{session_id}")
    chroma_db_dir = os.path.join(session_folder, "chromadb_storage")

    if not os.path.isdir(chroma_db_dir):
        sys.exit(f"[MCQ] Error: No chromadb_storage found for session {session_id}.")

    client = chromadb.PersistentClient(path=chroma_db_dir)
    coll_names = client.list_collections()
    if not coll_names:
        sys.exit("[MCQ] Error: No collections found in this session's database.")
    if len(coll_names) > 1:
        sys.exit("[MCQ] Found multiple collections, only one is expected per DB.")
    
    collection_name = coll_names[0]
    collection = client.get_collection(name=collection_name)
    return collection, collection_name, session_folder

# ── NEW ── Support for generating MCQs over *all* collections
def load_all_collections(session_id: str):
    tools_dir = os.path.abspath(os.path.dirname(__file__))
    database_root = os.path.join(tools_dir, "..", "database")
    session_folder = os.path.join(database_root, f"chat_{session_id}")
    chroma_db_dir = os.path.join(session_folder, "chromadb_storage")
    
    if not os.path.isdir(chroma_db_dir):
        sys.exit(f"[MCQ] Error: No chromadb_storage found for session {session_id}.")
    
    client = chromadb.PersistentClient(path=chroma_db_dir)
    coll_names = client.list_collections()
    if not coll_names:
        sys.exit(f"[MCQ] Error: No collections found in session {session_id}.")
    
    return client, session_folder, coll_names

def fetch_combined_text(collection):
    try:
        docs_data = collection.get(include=["documents"])
        docs = []
        for sublist in docs_data["documents"]:
            docs.extend(sublist)
        return " ".join(docs).strip()
    except Exception as e:
        sys.exit(f"[MCQ] Error fetching docs: {e}")

def generate_raw_mcqs(text: str) -> str:
    if len(text) < 50:
        return "Not enough text to generate MCQs."
    if len(text) > 8000:
        text = text[:8000] + "...(truncated)..."

    prompt = (
        "You are an AI specialized in generating multiple-choice questions.\n\n"
        "Here is some text from a document:\n\n"
        f"{text}\n\n"
        "Please create 20 MCQs from this content. For each question:\n"
        "1) Write the question prefixed with a number, e.g. '1. What is...'\n"
        "2) Provide exactly 4 answer options labeled A), B), C), D)\n"
        "3) Indicate the correct answer using 'Answer: X' (where X is A, B, C, or D)\n"
        "4) Provide a short 'Explanation: ...'\n\n"
        "Assistant:"
    )
    try:
        resp = ollama.generate(model=MODEL, prompt=prompt)
        return remove_think_clauses(resp.get("response", "No response"))
    except Exception as e:
        return f"(Error) {e}"

def parse_mcq_output(raw_text: str):
    lines = raw_text.splitlines()
    mcqs = []
    current = {"question": "", "options": [], "answer": "", "explanation": ""}
    in_options = in_answer = in_explanation = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Question start
        m_q = re.match(r'^(?i)(\d+)\.\s*(.*)$', line)
        if m_q:
            if current["question"]:
                mcqs.append(current)
            current = {"question": m_q.group(2).strip(), "options": [], "answer": "", "explanation": ""}
            in_options, in_answer, in_explanation = True, False, False
            continue

        # Option lines
        m_o = re.match(r'^(?i)([ABCD])\)\s+(.*)$', line)
        if m_o and in_options:
            current["options"].append(f"{m_o.group(1).upper()}) {m_o.group(2).strip()}")
            continue

        # Answer line
        m_a = re.match(r'^(?i)answer:\s*(.*)$', line)
        if m_a:
            current["answer"] = m_a.group(1).strip()
            in_options, in_answer, in_explanation = False, True, False
            continue

        # Explanation line
        m_e = re.match(r'^(?i)explanation:\s*(.*)$', line)
        if m_e:
            current["explanation"] = m_e.group(1).strip()
            in_options, in_answer, in_explanation = False, False, True
            continue

        # Continuation
        if in_explanation:
            current["explanation"] += " " + line
        elif in_options:
            current["question"] += " " + line
        elif in_answer:
            current["answer"] += " " + line

    if current["question"]:
        mcqs.append(current)

    final = []
    for mcq in mcqs:
        q = mcq["question"].strip()
        opts = [o.strip() for o in mcq["options"]]
        ans = re.sub(r'^(?i)[ABCD]\)\s*', '', mcq["answer"].strip())
        expl = mcq["explanation"].strip()
        if q and len(opts) == 4 and ans:
            final.append({"question": q, "options": opts, "answer": ans, "explanation": expl})
    if not final:
        final = [{
            "question": "MCQ Generation Failed",
            "options": ["A) N/A","B) N/A","C) N/A","D) N/A"],
            "answer": "N/A",
            "explanation": "Unable to generate valid MCQs from the content."
        }]
    return final

# ── NEW ── Automatically generate a five-word style title if user doesn’t supply one
def generate_title(text: str) -> str:
    sample = text[:500]
    prompt = (
        "You are an AI assistant. Based on the following content, generate a creative, concise, and formal title "
        "for a set of multiple-choice questions (MCQs) in exactly one sentence of about five words. Do not add commentary.\n\n"
        f"Content:\n{sample}\n\nTitle:"
    )
    try:
        resp = ollama.generate(model=MODEL, prompt=prompt)
        title = resp.get("response", "").splitlines()[0].strip().strip('"')
        return title or "Untitled MCQs"
    except:
        return "Untitled MCQs"

def parse_args():
    p = argparse.ArgumentParser(description="Generate MCQs from a session's collection(s).")
    p.add_argument("session_id", help="Session ID (e.g. 7)")
    p.add_argument("--all", action="store_true", help="Include every collection in that session.")
    p.add_argument("--topic", type=str, default=None, help="Narrow the MCQs to a specific topic.")
    p.add_argument("--title", type=str, default="", help="Provide your own title. Otherwise one is auto-generated.")
    return p.parse_args()

def main():
    args = parse_args()
    sid = args.session_id

    # ── NEW ── Combine text from either one collection, all collections, or scoped by topic
    if args.all:
        client, folder, names = load_all_collections(sid)
        combined = ""
        for n in names:
            c = client.get_collection(name=n)
            combined += fetch_combined_text(c) + "\n\n"
        title_src = combined
    else:
        coll, name, folder = load_single_collection(sid)
        combined = fetch_combined_text(coll)
        title_src = combined
        if args.topic:
            combined = f"Focus on {args.topic}. " + combined

    if not combined:
        print("[MCQ] No text found to generate MCQs.")
        sys.exit(0)

    final_title = args.title or generate_title(title_src)

    print(f"[MCQ] Generating MCQs titled '{final_title}'...")
    raw = generate_raw_mcqs(combined)
    mcqs = parse_mcq_output(raw)

    # ── NEW ── Avoid overwriting: if mcqs.json exists, auto-increment the filename
    base = os.path.join(folder, "mcqs.json")
    if os.path.exists(base):
        i = 1
        while os.path.exists(os.path.join(folder, f"mcqs_{i}.json")):
            i += 1
        outfile = os.path.join(folder, f"mcqs_{i}.json")
    else:
        outfile = base

    out = {"title": final_title, "mcqs": mcqs}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[MCQ] Saved to {outfile}\n")

    print(f"\n=== MCQs: {final_title} ===\n")
    for i, m in enumerate(mcqs, 1):
        print(f"Q{i}. {m['question']}")
        for o in m['options']:
            print(o)
        print(f"Answer: {m['answer']}")
        print(f"Explanation: {m['explanation']}\n")

if __name__ == "__main__":
    main()
