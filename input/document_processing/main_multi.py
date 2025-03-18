import os
import re
import concurrent.futures

from .table_extraction import extract_tables_with_metadata
from .text_extraction import extract_text_without_repetitions
from .pdf_metadata import extract_metadata_and_links, extract_images, extract_audio
import logging

logging.getLogger("chromadb").setLevel(logging.ERROR)

def setup_output_folder(pdf_path, chat_session_folder):
    """
    Instead of 'output/<pdf_name>', store the extracted content
    in the chat session folder so each chat has its own separate output.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(chat_session_folder, f"{pdf_name}_extracted")
    os.makedirs(output_folder, exist_ok=True)
    return pdf_name, output_folder

def extract_text(pdf_path):
    return extract_text_without_repetitions(pdf_path)

def extract_metadata_and_links_with_text(pdf_path):
    metadata, page_data = extract_metadata_and_links(pdf_path)
    return metadata, page_data

def extract_tables(pdf_path):
    return extract_tables_with_metadata(pdf_path)

def update_page_data_with_tables(page_data, tables_with_metadata):
    for table_data in tables_with_metadata:
        page_num = table_data["page"]
        table_id = table_data["table_id"]
        table_entry = {
            "position": table_data["position"],
            "table": table_data["table"],
            "table_id": table_id
        }
        page_key = f"page_{page_num}"
        if page_key in page_data:
            if "tables" in page_data[page_key]:
                page_data[page_key]["tables"].append(table_entry)
            else:
                page_data[page_key]["tables"] = [table_entry]
        else:
            page_data[f"page_{page_num}"] = {"tables": [table_entry]}
    return page_data

def remove_tables_from_text(page_texts, tables_with_metadata):
    replaced_tables = set()
    for table_data in tables_with_metadata:
        table_id = table_data["table_id"]
        page_num = table_data["page"]
        table_content = table_data["table"]
        if table_id in replaced_tables:
            continue
        table_lines = []
        for row in table_content:
            for cell in row:
                if cell:
                    table_lines.extend(cell.split("\n"))
        if table_lines and page_num in page_texts:
            table_pattern = "|".join([re.escape(line.strip()) for line in table_lines if line.strip()])
            table_regex = re.compile(table_pattern, re.IGNORECASE | re.DOTALL)
            replaced_text, num_replacements = table_regex.subn(f"<TABLE|{table_id}>", page_texts[page_num], count=1)
            if num_replacements > 0:
                replaced_tables.add(table_id)
                page_texts[page_num] = replaced_text
                page_texts[page_num] = table_regex.sub("", page_texts[page_num]).strip()
    return page_texts

def remove_excess_newlines(page_texts):
    for page_num, text in page_texts.items():
        text = re.sub(r'(\n\s*)+', '\n', text)
        page_texts[page_num] = text.strip()
    return page_texts

def chunk_text_semantic(text, max_words=200):
    """
    Split text into semantically meaningful chunks:
      1. Split text by double newlines into paragraphs.
      2. Merge paragraphs up to ~max_words.
      3. If a paragraph alone is too big, split by sentence boundaries.
    """
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        word_count = len(para.split())
        if current_count + word_count <= max_words:
            current_chunk.append(para)
            current_count += word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0
            if word_count > max_words:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_count = 0
                for sent in sentences:
                    sent_count = len(sent.split())
                    if temp_count + sent_count <= max_words:
                        temp_chunk.append(sent)
                        temp_count += sent_count
                    else:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [sent]
                        temp_count = sent_count
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                current_chunk.append(para)
                current_count += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def process_and_get_image_description(image_file):
    """
    Calls ollama_images.process_image(image_file) to generate a description
    but does not rely on reading/writing separate .txt. Instead returns the text.
    """

    try:
        from image_processing import ollama_images
        desc = ollama_images.process_image(image_file)
        if desc is None:
            desc = ""
        return desc
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")
        return ""

def deduplicate_images(image_list):
    """
    If the same 'file_path' is repeated, keep only the first occurrence.
    """
    seen = set()
    unique_images = []
    for img in image_list:
        path = img.get("file_path", "")
        if path and path not in seen:
            seen.add(path)
            unique_images.append(img)
    return unique_images

def process_images_and_update_page_data(pdf_path, output_folder, page_data):
    """
    Extract images, deduplicate by file path, then concurrently call
    process_and_get_image_description to fetch a text caption for each.
    """
    from .pdf_metadata import extract_images
    updated_page_data = page_data.copy()
    extract_images(pdf_path, output_folder, updated_page_data)

    # Deduplicate images
    for key, data in updated_page_data.items():
        if "images" in data:
            data["images"] = deduplicate_images(data["images"])

    tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for key, data in updated_page_data.items():
            if "images" in data:
                print("[Step 6] Processing images...")
                for image in data["images"]:
                    image_file = image.get("file_path", "")
                    tasks.append((key, image, executor.submit(process_and_get_image_description, image_file)))

        for key, image, future in tasks:
            description = future.result()
            image["description"] = description

    total_images = sum(len(data.get("images", [])) for data in updated_page_data.values())
    return updated_page_data, total_images

def add_chunks_to_chromadb(collection, pdf_name, page_num, chunks):
    if not chunks:
        return
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        doc_id = f"{pdf_name}_page_{page_num}_chunk_{i}_text"
        documents_to_add.append(chunk)
        metadatas_to_add.append({
            "source_file": pdf_name,
            "page_number": page_num,
            "chunk_index": i,
            "type": "text"
        })
        ids_to_add.append(doc_id)
    if documents_to_add:
        collection.add(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )

def add_image_pointers_with_descriptions(page_texts, page_data):
    """
    Insert <IMAGE|filename|Desc: ...> markers into the text,
    so the chunk later includes the image info.
    """
    for page_key, data in page_data.items():
        try:
            page_num = int(page_key.split("_")[1])
        except:
            continue

        images = data.get("images", [])
        for image in images:
            img_file_name = os.path.basename(image.get("file_path", ""))
            desc_text = (image.get("description") or "").strip()
            if desc_text:
                marker = f"<IMAGE|{img_file_name}|Desc: {desc_text}>"
            else:
                marker = f"<IMAGE|{img_file_name}>"
            if page_num in page_texts:
                page_texts[page_num] += "\n" + marker
    return page_texts

def process_pdf(pdf_path, collection=None):
    """
    1) Extract text, images, audio, and tables from the PDF
    2) Insert chunked text into the specified ChromaDB collection
    3) All extracted content (images, etc.) goes into chat session folder
    """
    # Use the chat session folder from environment variables
    session_folder = os.getenv("CHROMA_DB_DIR", "database/chat_unknown")
    pdf_name, output_folder = setup_output_folder(pdf_path, session_folder)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        print("[Step 1] Extracting text...")
        futures['text'] = executor.submit(extract_text, pdf_path)

        print("[Step 2] Extracting metadata/links...")
        futures['metadata'] = executor.submit(extract_metadata_and_links_with_text, pdf_path)

        metadata, page_data = futures['metadata'].result()
        del futures['metadata']

        print("[Step 3] Extracting & describing images...")
        futures['images'] = executor.submit(process_images_and_update_page_data, pdf_path, output_folder, page_data)

        print("[Step 4] Extracting tables...")
        futures['tables'] = executor.submit(extract_tables, pdf_path)

        print("[Step 5] Extracting audio ...")
        futures['audio'] = executor.submit(extract_audio, pdf_path, output_folder, page_data)

        page_texts = futures['text'].result()
        tables_with_metadata = futures['tables'].result()
        updated_page_data, image_count = futures['images'].result()
        audio_info = futures['audio'].result()

        # Clean up references
        del futures['text'], futures['tables'], futures['images'], futures['audio']

    page_data.update(updated_page_data)
    page_data.update(audio_info)

    # Insert table references
    page_data = update_page_data_with_tables(page_data, tables_with_metadata)
    # Add image markers
    page_texts = add_image_pointers_with_descriptions(page_texts, page_data)
    # Remove table text from the pages
    page_texts = remove_tables_from_text(page_texts, tables_with_metadata)
    # Remove extraneous newlines
    page_texts = remove_excess_newlines(page_texts)

    print(f"[Complete] Finished. Extracted {image_count} images.")

    # Insert chunked text into Chroma if collection provided
    if collection:
        for page_num, text in page_texts.items():
            chunks = chunk_text_semantic(text, max_words=200)
            add_chunks_to_chromadb(collection, pdf_name, page_num, chunks)

    # Return some metrics
    return image_count, 0
