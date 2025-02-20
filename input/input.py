import os
import re
import sys
import subprocess
from pathlib import Path
from document_processing import main_multi, document_to_pdf as documents_to_pdf
from image_processing import ollama_images
from audio_processing.whisper_medium import process_audio
import chromadb
from sentence_transformers import SentenceTransformer
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chromadb_storage")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class ChromaEmbeddingFunction:
    def __call__(self, input):
        """Accepts a list of strings and returns a list of embeddings."""
        return embedding_model.encode(input).tolist()

embedding_function = ChromaEmbeddingFunction()

def sanitize_collection_name(name):
    """
    Sanitize a PDF base name to meet ChromaDB collection name requirements:
      - 3 to 63 characters long,
      - Starts and ends with an alphanumeric character,
      - Contains only alphanumeric characters, underscores, or hyphens,
      - No spaces or consecutive invalid characters.
    """
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

def process_input(input_path):
    """
    Determines whether the user provided a file or a directory, 
    and processes accordingly.
    """
    input_path = Path(input_path)
    print(f"[Debug] Input path: {input_path}")
    if input_path.is_dir():
        for file in input_path.rglob('*'):
            if file.is_file():
                process_file(file)
    elif input_path.is_file():
        process_file(input_path)
    else:
        print(f"Invalid input path: {input_path}")

def process_file(file_path):
    """
    Checks the extension of the file and dispatches the appropriate
    processing function.
    """
    file_extension = file_path.suffix.lower()
    print(f"[Debug] Detected file extension: {file_extension}")
    if file_extension in ['.png', '.jpg', '.jpeg']:
        print(f"Processing image: {file_path}")
        pdf_path = documents_to_pdf.convert_to_pdf(file_path)
        if pdf_path is None:
            print(f"Failed to convert {file_path} to PDF.")
            return
        process_pdf_file(pdf_path)
    elif file_extension == '.pdf':
        print(f"Processing PDF: {file_path}")
        process_pdf_file(file_path)
    elif file_extension in ['.wav', '.mp3', '.flac', '.ogg']:
        print(f"Processing audio: {file_path}")
        process_audio(file_path)
    else:
        print(f"Converting {file_path} to PDF...")
        pdf_path = documents_to_pdf.convert_to_pdf(file_path)
        if pdf_path is not None and pdf_path.exists():
            print(f"Conversion successful. Processing converted PDF: {pdf_path}")
            process_pdf_file(pdf_path)
        else:
            print(f"Conversion failed or unsupported file format: {file_path}")

def process_pdf_file(pdf_path):
    """
    For each PDF, create a dedicated ChromaDB collection using the sanitized PDF name.
    Then call main_multi.process_pdf (which handles text, images, audio, and metadata extraction)
    so that the text is split into meaningful chunks and inserted into the collection.
    """
    pdf_name = Path(pdf_path).stem
    sanitized_name = sanitize_collection_name(pdf_name)
    print(f"[Debug] PDF base name: {pdf_name} -> Sanitized name: {sanitized_name}")
    pdf_collection = client.get_or_create_collection(
        name=sanitized_name,
        embedding_function=embedding_function
    )
    image_count, audio_count = main_multi.process_pdf(pdf_path, pdf_collection)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        user_path = sys.argv[1]
        process_input(user_path)
    else:
        print("[Info] No path argument provided to input.py")
