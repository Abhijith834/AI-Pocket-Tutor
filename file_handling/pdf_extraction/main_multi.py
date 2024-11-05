import os
import re
import time
import concurrent.futures
from text_extraction import extract_text_without_repetitions, save_text_to_file
from pdf_metadata import extract_metadata_and_links, extract_images, save_metadata_to_json
from table_extraction import extract_tables_with_metadata

def setup_output_folder(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join("output", pdf_name)
    os.makedirs(output_folder, exist_ok=True)
    return pdf_name, output_folder

def extract_text(pdf_path):
    return extract_text_without_repetitions(pdf_path)

def extract_metadata_and_links_with_text(pdf_path):
    metadata, page_data = extract_metadata_and_links(pdf_path)
    return metadata, page_data

def extract_images_and_update_page_data(pdf_path, output_folder, page_data):
    return extract_images(pdf_path, output_folder, page_data)

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

def generate_line_pattern(line):
    line_pattern = re.escape(line).replace(r"\n", r"\s*")
    return rf"{line_pattern}"

def remove_tables_from_text(page_texts, tables_with_metadata):
    for table_data in tables_with_metadata:
        page_num = table_data["page"]
        table_id = table_data["table_id"]
        table_content = table_data["table"]

        total_lines = 0
        matched_lines = 0
        unmatched_lines = 0

        for row in table_content:
            for cell in row:
                cell_lines = cell.split("\n")
                for line in cell_lines:
                    if line.strip():
                        total_lines += 1
                        line_pattern = re.compile(generate_line_pattern(line), re.IGNORECASE | re.DOTALL)
                        
                        if page_num in page_texts and line_pattern.search(page_texts[page_num]):
                            matched_lines += 1
                        else:
                            unmatched_lines += 1

        matched_percentage = (matched_lines / total_lines * 100) if total_lines > 0 else 0

        if matched_percentage > 50:
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
    final_text = "\n--- Page ".join([f"{page} ---\n{text}" for page, text in page_texts.items()])
    final_text_path = save_text_to_file(output_folder, pdf_name, final_text)
    print(f"[Info] Cleaned text with unique content saved to: {final_text_path}")

    metadata_json_path = save_metadata_to_json(metadata, page_data, output_folder, pdf_name)
    print(f"[Info] Metadata, links, images, and tables saved to: {metadata_json_path}")

def process_pdf(pdf_path):
    pdf_name, output_folder = setup_output_folder(pdf_path)

    # Step 1-5: Initial parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}

        print("[Step 2] Extracting text from PDF...")
        futures['text'] = executor.submit(extract_text, pdf_path)

        print("[Step 3] Extracting metadata and links...")
        futures['metadata'] = executor.submit(extract_metadata_and_links_with_text, pdf_path)

        print("[Step 4] Extracting images...")
        futures['images'] = executor.submit(extract_images_and_update_page_data, pdf_path, output_folder, {})

        print("[Step 5] Extracting tables...")
        futures['tables'] = executor.submit(extract_tables, pdf_path)

        # Collect results for Steps 1-5
        page_texts = futures['text'].result()
        metadata, page_data = futures['metadata'].result()
        page_data = futures['images'].result()
        tables_with_metadata = futures['tables'].result()

    # Step 6-7: Parallel processing after Step 5 completion
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}

        print("[Step 6] Updating page data with tables...")
        futures['update_tables'] = executor.submit(update_page_data_with_tables, page_data, tables_with_metadata)

        print("[Step 7] Removing tables from text...")
        futures['remove_tables'] = executor.submit(remove_tables_from_text, page_texts, tables_with_metadata)

        # Collect results for Steps 6 and 7
        page_data = futures['update_tables'].result()
        page_texts = futures['remove_tables'].result()

    # Step 8: Save final results
    print("[Step 8] Saving extracted text and metadata to output files...")
    save_final_text_and_metadata(pdf_name, output_folder, page_texts, metadata, page_data)
    print("[Complete] PDF processing finished.")

if __name__ == "__main__":
    pdf_path = "/home/matrix/Desktop/AI Pocket Tutor/10840.pdf"  # Replace with your actual PDF path
    start_time = time.time()
    print("[Step 1] Setting up output folder...")
    process_pdf(pdf_path)
    print(f"[Done] Total processing time: {time.time() - start_time:.2f} seconds.")
