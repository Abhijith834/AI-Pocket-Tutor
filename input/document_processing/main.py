import os
import time
from text_extraction import extract_text_without_repetitions, save_text_to_file
from pdf_metadata import extract_metadata_and_links, extract_images, save_metadata_to_json
from table_extraction import extract_tables_with_metadata
import re

def setup_output_folder(pdf_path):
    """
    Sets up the output folder for the given PDF file.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join("output", pdf_name)
    os.makedirs(output_folder, exist_ok=True)
    return pdf_name, output_folder

def extract_text(pdf_path):
    """
    Extracts text from the PDF, removes repetitive elements, and returns
    the text organized by page.
    """
    return extract_text_without_repetitions(pdf_path)

def extract_metadata_and_links_with_text(pdf_path):
    """
    Extracts metadata, links, and surrounding text for each link from the PDF.
    Returns metadata and page data with link information.
    """
    metadata, page_data = extract_metadata_and_links(pdf_path)
    return metadata, page_data

def extract_images_and_update_page_data(pdf_path, output_folder, page_data):
    """
    Extracts images from the PDF and updates the page data with image positions
    and file paths.
    """
    return extract_images(pdf_path, output_folder, page_data)

def extract_tables(pdf_path):
    """
    Extracts tables from the PDF and returns them with metadata (position, content, page number).
    """
    return extract_tables_with_metadata(pdf_path)

def update_page_data_with_tables(page_data, tables_with_metadata):
    """
    Updates the page data with table information.
    """
    for table_data in tables_with_metadata:
        page_num = table_data["page"]
        table_id = table_data["table_id"]

        table_entry = {
            "position": table_data["position"],
            "table": table_data["table"],
            "table_id": table_id
        }

        # Update page data JSON with table metadata
        page_key = f"page_{page_num}"
        if page_key in page_data:
            if "tables" in page_data[page_key]:
                page_data[page_key]["tables"].append(table_entry)
            else:
                page_data[page_key]["tables"] = [table_entry]
        else:
            page_data[f"page_{page_num}"] = {"tables": [table_entry]}

    return page_data
def generate_line_pattern(line):
    """
    Generates a regex pattern that matches a single line from a table cell content.
    """
    # Escape special characters in the line and replace \n with \s* to match any whitespace or line breaks
    line_pattern = re.escape(line).replace(r"\n", r"\s*")
    return rf"{line_pattern}"


def remove_tables_from_text(page_texts, tables_with_metadata):
    """
    Checks each line in the table cells for matches in the extracted text, replacing matched
    table content and providing a detailed summary of matched vs. unmatched lines.
    """
    for table_data in tables_with_metadata:
        page_num = table_data["page"]
        table_id = table_data["table_id"]
        table_content = table_data["table"]

        # Track matched and unmatched lines in cells
        total_lines = 0
        matched_lines = 0
        unmatched_lines = 0

        # Process each row in the table
        for row in table_content:
            for cell in row:
                # Split cell content by lines (account for '\n' in the cell data)
                cell_lines = cell.split("\n")
                for line in cell_lines:
                    if line.strip():  # Skip empty lines
                        total_lines += 1
                        line_pattern = re.compile(generate_line_pattern(line), re.IGNORECASE | re.DOTALL)
                        
                        # Check if the line exists in the page text
                        if page_num in page_texts and line_pattern.search(page_texts[page_num]):
                            matched_lines += 1
                        else:
                            unmatched_lines += 1

        # Calculate match statistics
        matched_percentage = (matched_lines / total_lines * 100) if total_lines > 0 else 0

        # Optional: Replace full table content if overall match is sufficient
        if matched_percentage > 50:  # Only replace if more than 50% of content is matched
            # Construct full table pattern for replacement
            table_patterns = [
                "\s+".join([generate_line_pattern(line) for line in cell.split("\n")])
                for row in table_content for cell in row
            ]
            full_table_pattern = re.compile(r"\s*".join(table_patterns), re.IGNORECASE | re.DOTALL)

            if page_num in page_texts:
                page_texts[page_num], count = full_table_pattern.subn(
                    f"[See table {table_id} in JSON output]", page_texts[page_num]
                )

    return page_texts

def save_final_text_and_metadata(pdf_name, output_folder, page_texts, metadata, page_data):
    """
    Saves the cleaned text for each page as a single text file, and saves the
    metadata, links, images, and tables information as JSON.
    """
    # Save text
    final_text = "\n--- Page ".join([f"{page} ---\n{text}" for page, text in page_texts.items()])
    final_text_path = save_text_to_file(output_folder, pdf_name, final_text)
    print(f"[Info] Cleaned text with unique content saved to: {final_text_path}")

    # Save metadata, links, images, and tables to JSON
    metadata_json_path = save_metadata_to_json(metadata, page_data, output_folder, pdf_name)
    print(f"[Info] Metadata, links, images, and tables saved to: {metadata_json_path}")

def process_pdf(pdf_path):
    """
    Orchestrates the entire process of extracting and saving text, metadata, links, images,
    and tables from the PDF.
    """
    start_time = time.time()
    
    # Step 1: Setup output folder
    print("[Step 1] Setting up output folder...")
    pdf_name, output_folder = setup_output_folder(pdf_path)
    print("[Done] Output folder created.")
    
    # Step 2: Extract text by page
    print("[Step 2] Extracting text from PDF...")
    text_start = time.time()
    page_texts = extract_text(pdf_path)
    print(f"[Done] Text extracted. Time taken: {time.time() - text_start:.2f} seconds.")
    
    # Step 3: Extract metadata and links with associated text
    print("[Step 3] Extracting metadata and links...")
    metadata_start = time.time()
    metadata, page_data = extract_metadata_and_links_with_text(pdf_path)
    print(f"[Done] Metadata and links extracted. Time taken: {time.time() - metadata_start:.2f} seconds.")
    
    # Step 4: Extract images and update page data with image info
    print("[Step 4] Extracting images...")
    images_start = time.time()
    page_data = extract_images_and_update_page_data(pdf_path, output_folder, page_data)
    print(f"[Done] Images extracted. Time taken: {time.time() - images_start:.2f} seconds.")
    
    # Step 5: Extract tables from PDF
    print("[Step 5] Extracting tables...")
    tables_start = time.time()
    tables_with_metadata = extract_tables(pdf_path)
    print(f"[Done] Tables extracted. Time taken: {time.time() - tables_start:.2f} seconds.")
    
    # Step 6: Update page data with tables
    print("[Step 6] Updating page data with tables...")
    page_data = update_page_data_with_tables(page_data, tables_with_metadata)
    print("[Done] Page data updated with tables.")
    
    # Step 7: Remove tables from the extracted text
    print("[Step 7] Removing tables from text...")
    remove_tables_start = time.time()
    page_texts = remove_tables_from_text(page_texts, tables_with_metadata)
    print(f"[Done] Tables removed from text. Time taken: {time.time() - remove_tables_start:.2f} seconds.")
    
    # Step 8: Save the extracted text and metadata to output files
    print("[Step 8] Saving extracted text and metadata to output files...")
    save_start = time.time()
    save_final_text_and_metadata(pdf_name, output_folder, page_texts, metadata, page_data)
    print(f"[Done] Data saved. Time taken: {time.time() - save_start:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"[Complete] PDF processing finished. Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    # Replace with your actual PDF path
    pdf_path = "/home/matrix/Desktop/AI Pocket Tutor/10840.pdf"
    process_pdf(pdf_path)
