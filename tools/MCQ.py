#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

def remove_think_clauses(text: str) -> str:
    """
    Removes <think>...</think> if your model uses hidden chain-of-thought tags.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def load_single_collection(session_id: str):
    """
    Load a single collection from the session. Expects exactly one collection.
    """
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

def load_all_collections(session_id: str):
    """
    Load all collections from the session's chromadb_storage.
    Returns the client, session folder, and the list of collection names.
    """
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

def load_specific_collection(session_id: str, topic: str):
    """
    Load a specific collection (topic) from the session.
    (Not used since there is only one collection.)
    """
    # Since there is only one collection in the database, we always fallback to it.
    return load_single_collection(session_id)

def fetch_combined_text(collection):
    """
    Retrieve all 'documents' from a collection and combine them.
    """
    try:
        docs_data = collection.get(include=["documents"])
        docs = []
        for sublist in docs_data["documents"]:
            docs.extend(sublist)
        return " ".join(docs).strip()
    except Exception as e:
        sys.exit(f"[MCQ] Error fetching docs: {e}")

def generate_raw_mcqs(text: str) -> str:
    """
    Instructs the LLM to produce 20 MCQs from the provided text.
    Each question is expected to have:
      1) "1. Question"
      2) "A) ...", "B) ...", "C) ...", "D) ..."
      3) "Answer: X"
      4) "Explanation: ..."
    Minimal truncation is applied if the text is very long.
    """
    if len(text) < 50:
        return "Not enough text to generate MCQs."

    if len(text) > 8000:
        text = text[:8000] + "...(truncated)..."

    prompt = (
        f"You are an AI specialized in generating multiple-choice questions.\n\n"
        f"Here is some text from a document:\n\n"
        f"{text}\n\n"
        f"Please create 20 MCQs from this content. For each question:\n"
        f"1) Write the question prefixed with a number, e.g. '1. What is...'\n"
        f"2) Provide exactly 4 answer options labeled A), B), C), D)\n"
        f"3) Indicate the correct answer using 'Answer: X' (where X is A, B, C, or D)\n"
        f"4) Provide a short 'Explanation: ...'\n\n"
        f"Assistant:"
    )
    try:
        resp = ollama.generate(model="llama3.1", prompt=prompt)
        raw_text = resp.get("response", "No response")
        return remove_think_clauses(raw_text)
    except Exception as e:
        return f"(Error) {e}"

def parse_mcq_output(raw_text: str):
    """
    Parse the LLM output into a structured list of MCQs.
    Expects lines like:
      1. Some question?
      A) ...
      B) ...
      C) ...
      D) ...
      Answer: A
      Explanation: Because ...
      2. Next question...
    If the parsed output is invalid (e.g. empty questions), a fallback MCQ is returned.
    """
    lines = raw_text.splitlines()
    mcqs = []
    current = {"question": "", "options": [], "answer": "", "explanation": ""}
    in_options = False
    in_answer = False
    in_explanation = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match_q = re.match(r'^(?i)(\d+)\.\s*(.*)$', line)
        if match_q:
            if current["question"] or current["options"] or current["answer"] or current["explanation"]:
                mcqs.append(current)
            current = {"question": match_q.group(2).strip(), "options": [], "answer": "", "explanation": ""}
            in_options = True
            in_answer = False
            in_explanation = False
            continue

        match_opt = re.match(r'^(?i)([ABCD])\)\s+(.*)$', line)
        if match_opt and in_options:
            full_opt = f"{match_opt.group(1).upper()}) {match_opt.group(2).strip()}"
            current["options"].append(full_opt)
            continue

        match_ans = re.match(r'^(?i)answer:\s*(.*)$', line)
        if match_ans:
            current["answer"] = match_ans.group(1).strip()
            in_options = False
            in_answer = True
            in_explanation = False
            continue

        match_expl = re.match(r'^(?i)explanation:\s*(.*)$', line)
        if match_expl:
            current["explanation"] = match_expl.group(1).strip()
            in_options = False
            in_answer = False
            in_explanation = True
            continue

        if in_explanation:
            current["explanation"] += " " + line
        else:
            if in_options:
                current["question"] += " " + line
            elif in_answer:
                current["answer"] += " " + line

    if current["question"] or current["options"] or current["answer"] or current["explanation"]:
        mcqs.append(current)

    final_mcqs = []
    for mcq in mcqs:
        q = mcq["question"].strip()
        opts = [o.strip() for o in mcq["options"]]
        ans = re.sub(r'^(?i)[abcd]\)\s+', '', mcq["answer"].strip())
        ans = re.sub(r'\s+', ' ', ans).strip()
        expl = mcq["explanation"].strip()
        # Only consider valid questions with non-empty question text, exactly 4 options and a non-empty answer.
        if q and len(opts) == 4 and ans:
            final_mcqs.append({
                "question": q,
                "options": opts,
                "answer": ans,
                "explanation": expl
            })
    if not final_mcqs:
        # Fallback in case no valid MCQs were extracted
        fallback_mcq = {
            "question": "MCQ Generation Failed",
            "options": ["A) N/A", "B) N/A", "C) N/A", "D) N/A"],
            "answer": "N/A",
            "explanation": (
                "The system was unable to generate valid MCQs from the provided content. "
                "Please try re-running the command or check the input text."
            )
        }
        final_mcqs.append(fallback_mcq)
    return final_mcqs

def generate_title(text: str) -> str:
    """
    Uses an LLM call to generate a creative, concise, and formal title for the MCQs based on the provided text.
    Only the first 500 characters are used for efficiency.
    The instruction forces exactly one concise sentence of about five words.
    """
    sample_text = text[:500]
    prompt = (
        "You are an AI assistant. Based on the following content, generate a creative, concise, and formal title "
        "for a set of multiple-choice questions (MCQs) that captures the main subject. Answer in exactly one concise sentence of about five words. Do not provide multiple options or any extra commentary.\n\n"
        f"Content:\n{sample_text}\n\n"
        "Title:"
    )
    try:
        resp = ollama.generate(model="llama3.1", prompt=prompt)
        title = resp.get("response", "").strip()
        # Use only the first line of the response and strip extraneous quotes.
        title = title.splitlines()[0] if title else ""
        title = title.strip(' "')
        return title if title else "Untitled MCQs"
    except Exception as e:
        return "Untitled MCQs"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MCQs from a session's collection(s).")
    parser.add_argument("session_id", help="Session ID (e.g. 7)")
    parser.add_argument("--all", action="store_true",
                        help="Generate MCQs for all collections in the session.")
    parser.add_argument("--topic", type=str, default=None,
                        help="Generate MCQs for a specific topic from the content. In this system, content is in one collection, so the topic will be used to focus the prompt.")
    parser.add_argument("--title", type=str, default="",
                        help="Optional title for the MCQ output. If omitted, the AI will generate a title based on the content.")
    return parser.parse_args()

def main():
    args = parse_args()
    session_id = args.session_id

    # Determine which collection(s) to process and combine their text.
    if args.all:
        client, session_folder, coll_names = load_all_collections(session_id)
        combined_text = ""
        for coll in coll_names:
            coll_obj = client.get_collection(name=coll)
            combined_text += fetch_combined_text(coll_obj) + "\n\n"
        base_title = combined_text
    elif args.topic:
        # Since there is only one collection, simply load it.
        collection, coll_name, session_folder = load_single_collection(session_id)
        combined_text = fetch_combined_text(collection)
        # Prepend the desired focus so the LLM is cued to concentrate on that topic.
        combined_text = f"Focus on {args.topic}. " + combined_text
        base_title = combined_text
    else:
        collection, coll_name, session_folder = load_single_collection(session_id)
        combined_text = fetch_combined_text(collection)
        base_title = combined_text

    if not combined_text:
        print("[MCQ] No text found to generate MCQs!")
        sys.exit(0)

    # Auto-generate a title if --title is not provided.
    default_title = args.title if args.title else generate_title(base_title)

    print(f"[MCQ] Generating 20 MCQs for session='{session_id}'...")
    raw_mcqs = generate_raw_mcqs(combined_text)
    parsed = parse_mcq_output(raw_mcqs)
    if not parsed:
        print("\n[MCQ] Could not parse the 20 MCQs. Raw LLM output:\n")
        print(raw_mcqs)
        sys.exit(0)

    # Determine filename: if mcqs.json exists, append _1, _2, etc.
    base_file = os.path.join(session_folder, "mcqs.json")
    if os.path.exists(base_file):
        i = 1
        while True:
            candidate = os.path.join(session_folder, f"mcqs_{i}.json")
            if not os.path.exists(candidate):
                mcq_file = candidate
                break
            i += 1
    else:
        mcq_file = base_file

    # Create output dictionary with title and MCQs.
    output_data = {
        "title": default_title,
        "mcqs": parsed
    }
    
    try:
        with open(mcq_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"[MCQ] Saved 20 MCQs to '{mcq_file}'\n")
    except Exception as e:
        print(f"[MCQ] Error saving to JSON: {e}")

    # Print the header with the generated title.
    print(f"\n=== MCQs on {default_title} ===\n")
    for i, mcq in enumerate(parsed, start=1):
        print(f"Q{i}. {mcq['question']}")
        for opt in mcq['options']:
            print(opt)
        print(f"Answer: {mcq['answer']}")
        print(f"Explanation: {mcq['explanation']}\n")

if __name__ == "__main__":
    main()
