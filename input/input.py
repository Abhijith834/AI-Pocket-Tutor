import os
import re
import subprocess
from pathlib import Path

# Import our document, image, and audio processing modules.
from document_processing import main_multi, document_to_pdf as documents_to_pdf
from image_processing import ollama_images
from audio_processing.whisper_medium import process_audio

# ============= CHROMADB ADDITIONS =============
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = "chromadb_storage"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Load the embedding model.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a callable class that meets the required embedding function signature.
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
# ============= END CHROMADB ADDITIONS =============

def process_input(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        for file in input_path.rglob('*'):
            if file.is_file():
                process_file(file)
    elif input_path.is_file():
        process_file(input_path)
    else:
        print(f"Invalid input path: {input_path}")

def process_file(file_path):
    file_extension = file_path.suffix.lower()
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
    so that the text (augmented with image descriptions) is split into meaningful chunks
    and inserted into the collection.
    """
    pdf_name = Path(pdf_path).stem
    sanitized_name = sanitize_collection_name(pdf_name)
    pdf_collection = client.get_or_create_collection(
        name=sanitized_name,
        embedding_function=embedding_function
    )
    # Process the PDF and insert its chunks into the dedicated collection.
    image_count, audio_count = main_multi.process_pdf(pdf_path, pdf_collection)

if __name__ == '__main__':
    # Start the Ollama application as a background process (if needed).
    ollama_app_path = r"C:\Users\abhij\AppData\Local\Programs\Ollama\ollama app.exe"
    ollama_process = subprocess.Popen([ollama_app_path])
    
    input_path = 'AI-Pocket-Tutor\\Sentimental Analysis.pdf'  # Replace with your actual input path.
    process_input(input_path)
    
    # After processing, unload the Ollama model.
    subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"], check=False)
