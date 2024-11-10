import fitz  # PyMuPDF
import pdfplumber
from collections import defaultdict
import os

def extract_text_without_repetitions(pdf_path):
    """
    Extracts text from each page without repetitive headers/footers.
    Falls back to pdfplumber if PyMuPDF fails to retrieve text.
    """
    doc = fitz.open(pdf_path)
    repetitive_texts = defaultdict(int)
    total_pages = doc.page_count
    page_texts = {}

    # First pass: Identify repetitive blocks across pages
    for i in range(total_pages):
        page = doc.load_page(i)
        blocks = page.get_text("blocks")
        for block in blocks:
            block_text = block[4].strip()
            if block_text:
                repetitive_texts[block_text] += 1

    threshold = total_pages * 0.8
    common_texts = {text for text, count in repetitive_texts.items() if count >= threshold}

    # Second pass: Extract and clean text for each page
    for i in range(total_pages):
        page = doc.load_page(i)
        blocks = page.get_text("blocks")
        
        # Try basic extraction if only one block or no repetitive content is detected
        if len(blocks) == 1 or not common_texts:
            extracted_text = page.get_text("text").strip()
            if extracted_text:  # If text is successfully extracted
                page_texts[i + 1] = extracted_text
            else:
                page_texts[i + 1] = fallback_extract_with_pdfplumber(pdf_path, i)  # Use pdfplumber if no text
            continue

        # Apply repetitive text filtering
        cleaned_page_text = []
        for block in blocks:
            block_text = block[4].strip()
            if block_text and block_text not in common_texts:
                cleaned_page_text.append(block_text)

        combined_text = "\n".join(cleaned_page_text).strip()
        page_texts[i + 1] = combined_text if combined_text else fallback_extract_with_pdfplumber(pdf_path, i)

    return page_texts

def fallback_extract_with_pdfplumber(pdf_path, page_num):
    """
    Attempts to extract text from a specific page using pdfplumber as a fallback.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < len(pdf.pages):
            page = pdf.pages[page_num]
            return page.extract_text() or ""
    return ""

def save_text_to_file(output_folder, pdf_name, text):
    """
    Saves the extracted and processed text to a .txt file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"{pdf_name}_cleaned_text.txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return output_file
