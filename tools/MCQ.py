import os
import sys
import re
import json
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
    1) We locate the chat_<session_id>/chromadb_storage folder.
    2) We expect exactly one collection. If there's none or more than one, we error out.
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

def fetch_combined_text(collection):
    """
    Retrieve all 'documents' from the single collection and combine them.
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

    We do minimal truncation if extremely large.
    """
    if len(text) < 50:
        return "Not enough text to generate MCQs."

    # Truncate if extremely large to keep prompt reasonable
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
    Parse the LLM output into a structured list of 20 MCQs.
    Expects lines like:
      1. Some question?
      A) ...
      B) ...
      C) ...
      D) ...
      Answer: A
      Explanation: Because ...
      2. Next question...
      ...
    """
    lines = raw_text.splitlines()
    mcqs = []
    current = {"question": "", "options": [], "answer": "", "explanation": ""}
    question_num = None

    # Helper states
    in_options = False
    in_answer = False
    in_explanation = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # detect '1. ...' or '2) ...' style question
        match_q = re.match(r'^(\d+)\.\s*(.*)$', line)
        if match_q:
            # If we already have a question started, store it
            if current["question"] or current["options"] or current["answer"] or current["explanation"]:
                mcqs.append(current)
            # start a new question
            qn = match_q.group(1)
            question_text = match_q.group(2).strip()
            question_num = qn
            current = {"question": question_text, "options": [], "answer": "", "explanation": ""}
            in_options = True
            in_answer = False
            in_explanation = False
            continue

        # detect 'A) ...' or 'B) ...'
        match_opt = re.match(r'^([ABCD])\)\s+(.*)$', line, flags=re.IGNORECASE)
        if match_opt and in_options:
            option_letter = match_opt.group(1).upper()
            option_text = match_opt.group(2).strip()
            full_opt = f"{option_letter}) {option_text}"
            current["options"].append(full_opt)
            continue

        # detect 'Answer: X'
        match_ans = re.match(r'^(?i)answer:\s*(.*)$', line)
        if match_ans:
            ans_text = match_ans.group(1).strip()
            # We might see 'A', or 'A) Some text', or something else
            # We'll store raw for user convenience
            current["answer"] = ans_text
            in_options = False
            in_answer = True
            in_explanation = False
            continue

        # detect 'Explanation: ...'
        match_expl = re.match(r'^(?i)explanation:\s*(.*)$', line)
        if match_expl:
            expl_text = match_expl.group(1).strip()
            current["explanation"] = expl_text
            in_options = False
            in_answer = False
            in_explanation = True
            continue

        # If we're in explanation, we might continue reading lines
        if in_explanation:
            current["explanation"] += " " + line
        else:
            # Possibly question lines continuing if the model didn't follow exact format
            # We'll just append to question
            if in_options:
                current["question"] += " " + line
            elif in_answer:
                current["answer"] += " " + line
            # else do nothing

    # Add final question if partial
    if current["question"] or current["options"] or current["answer"] or current["explanation"]:
        mcqs.append(current)

    # Clean up each MCQ
    # Remove leftover 'Answer: ' from answer, etc.
    final_mcqs = []
    for mcq in mcqs:
        q = mcq["question"].strip()
        opts = [o.strip() for o in mcq["options"]]
        ans = mcq["answer"].strip()
        ans = re.sub(r'^(?i)[abcd]\)\s+', '', ans)  # remove 'A) ' if present
        ans = re.sub(r'\s+', ' ', ans).strip()
        expl = mcq["explanation"].strip()

        if q or opts or ans or expl:
            final_mcqs.append({
                "question": q,
                "options": opts,
                "answer": ans,
                "explanation": expl
            })

    return final_mcqs

def main():
    if len(sys.argv) < 2:
        print("Usage: python MCQ.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]

    # 1) Load single collection from that session
    collection, coll_name, session_folder = load_single_collection(session_id)

    # 2) Combine the text
    combined_text = fetch_combined_text(collection)
    if not combined_text:
        print(f"[MCQ] No text found in collection '{coll_name}'!")
        sys.exit(0)

    print(f"[MCQ] Generating 20 MCQs for session='{session_id}', collection='{coll_name}'...")
    # 3) Generate raw text for 20 MCQs
    raw_mcqs = generate_raw_mcqs(combined_text)

    # 4) Parse the output to structure
    parsed = parse_mcq_output(raw_mcqs)
    if not parsed:
        print("\n[MCQ] Could not parse the 20 MCQs. Raw LLM output:\n")
        print(raw_mcqs)
        return

    # 5) Save to JSON
    mcq_file = os.path.join(session_folder, "mcqs.json")
    try:
        with open(mcq_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"[MCQ] Saved 20 MCQs to '{mcq_file}'\n")
    except Exception as e:
        print(f"[MCQ] Error saving to JSON: {e}")

    # 6) Print them to console
    for i, mcq in enumerate(parsed, start=1):
        print(f"Q{i}. {mcq['question']}")
        for opt in mcq['options']:
            print(opt)
        print(f"Answer: {mcq['answer']}")
        print(f"Explanation: {mcq['explanation']}\n")

if __name__ == "__main__":
    main()
