import os
import subprocess
from pathlib import Path
from document_processing import main_multi, document_to_pdf as documents_to_pdf
from image_processing import ollama_images
from audio_processing.whisper_medium import process_audio  # Import the audio processing function from whisper_medium

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
        # Image: Convert to PDF, then process as PDF.
        print(f"Processing image: {file_path}")
        pdf_path = documents_to_pdf.convert_to_pdf(file_path)
        if pdf_path is None:
            print(f"Failed to convert {file_path} to PDF.")
            return
        process_pdf_file(pdf_path)

    elif file_extension == '.pdf':
        # Already a PDF, process directly
        print(f"Processing PDF: {file_path}")
        process_pdf_file(file_path)

    elif file_extension in ['.wav', '.mp3', '.flac', '.ogg']:
        # Audio file, process directly
        print(f"Processing audio: {file_path}")
        process_audio(file_path)
    else:
        # Other document types: attempt conversion with LibreOffice
        print(f"Converting {file_path} to PDF...")
        pdf_path = documents_to_pdf.convert_to_pdf(file_path)
        if pdf_path is not None and pdf_path.exists():
            print(f"Conversion successful. Processing converted PDF: {pdf_path}")
            process_pdf_file(pdf_path)
        else:
            print(f"Conversion failed or unsupported file format: {file_path}")

def process_pdf_file(pdf_path):
    image_count, audio_count = main_multi.process_pdf(pdf_path)
    
    if image_count > 0:
        print(f"{image_count} images found in the PDF. Processing images with Ollama...")
        process_extracted_images(pdf_path)
    
    if audio_count > 0:
        print(f"{audio_count} audio files found in the PDF. Processing audio files...")
        # Additional audio processing if needed

def process_extracted_images(pdf_path):
    images_dir = Path("output") / Path(pdf_path).stem / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg"))
        
        if not image_files:
            print("No images found in the images directory.")
        else:
            for image_file in image_files:
                print(f"Processing image with Ollama: {image_file}")
                ollama_images.process_image(image_file)
    else:
        print(f"Images directory does not exist at expected path: {images_dir}")

if __name__ == '__main__':
    # 1. Start the Ollama application as a background process
    # Adjust the path if Ollama is installed elsewhere
    ollama_app_path = r"C:\Users\abhij\AppData\Local\Programs\Ollama\ollama app.exe"
    ollama_process = subprocess.Popen([ollama_app_path])
    
    input_path = 'AI-Pocket-Tutor/report.docx'  # Replace with your actual input path or a directory
    process_input(input_path)
    
    # 2. Once processing is complete, unload the model from GPU memory
    # by killing the ollama_llama_server.exe process.
    subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"], check=False)
