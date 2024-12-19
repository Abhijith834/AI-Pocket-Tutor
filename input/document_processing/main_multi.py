import os
import re
import time
import concurrent.futures
from .table_extraction import extract_tables_with_metadata
from .text_extraction import extract_text_without_repetitions, save_text_to_file
from .pdf_metadata import extract_metadata_and_links, extract_images, extract_audio, save_metadata_to_json


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
    """
    Extracts images from the PDF, updates the page data with image positions and file paths,
    and returns the updated page data along with the count of images.
    """
    images_info = extract_images(pdf_path, output_folder, page_data)
    image_count = len(images_info)  # Count the extracted images
    return page_data, image_count

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
    """
    Removes tables from text and replaces them with placeholders.
    """
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

        if table_lines:
            table_pattern = "|".join([re.escape(line.strip()) for line in table_lines if line.strip()])
            table_regex = re.compile(table_pattern, re.IGNORECASE | re.DOTALL)

            if page_num in page_texts:
                page_texts[page_num], num_replacements = table_regex.subn(f"<TABLE|{table_id}>", page_texts[page_num], count=1)
                if num_replacements > 0:
                    replaced_tables.add(table_id)

                page_texts[page_num] = table_regex.sub("", page_texts[page_num]).strip()

    return page_texts

def save_final_text_and_metadata(pdf_name, output_folder, page_texts, metadata, page_data):
    final_text = "\n--- Page ".join([f"{page} ---\n{text}" for page, text in page_texts.items()])
    final_text_path = save_text_to_file(output_folder, pdf_name, final_text)
    print(f"[Info] Cleaned text with unique content saved to: {final_text_path}")

    metadata_json_path = save_metadata_to_json(metadata, page_data, output_folder, pdf_name)
    print(f"[Info] Metadata, links, images, and tables saved to: {metadata_json_path}")

def extract_audio_and_update_page_data(pdf_path, output_folder, page_data):
    audio_info = extract_audio(pdf_path, output_folder, page_data)
    audio_count = sum(len(page.get("audios", [])) for page in page_data.values())  # Count extracted audio files
    return page_data, audio_count

def add_image_pointers_to_text(page_texts, page_data):
    """
    Adds placeholders for images in the text at their approximate positions.
    """
    for page_key, data in page_data.items():
        page_num = int(page_key.split("_")[1])
        images = data.get("images", [])

        for image in images:
            img_file_name = os.path.basename(image["file_path"])
            image_pointer = f"<IMAGE|{img_file_name}>"

            if page_num in page_texts:
                # Append the image pointer at the end of the page's text
                page_texts[page_num] += f"\n{image_pointer}"

    return page_texts

def process_pdf(pdf_path):
    pdf_name, output_folder = setup_output_folder(pdf_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}

        print("[Step 2] Extracting text from PDF...")
        futures['text'] = executor.submit(extract_text, pdf_path)

        print("[Step 3] Extracting metadata and links...")
        futures['metadata'] = executor.submit(extract_metadata_and_links_with_text, pdf_path)

        # Wait for metadata first so we have initial page_data
        metadata, page_data = futures['metadata'].result()
        # Remove the metadata future from the dict since it's done
        del futures['metadata']

        # Now that we have `page_data`, we can pass it to images and audio
        print("[Step 4] Extracting images...")
        futures['images'] = executor.submit(extract_images_and_update_page_data, pdf_path, output_folder, page_data)

        print("[Step 5] Extracting tables...")
        futures['tables'] = executor.submit(extract_tables, pdf_path)

        print("[Step 6] Extracting audio files...")
        futures['audio'] = executor.submit(extract_audio_and_update_page_data, pdf_path, output_folder, page_data)

        # Also wait for text
        page_texts = futures['text'].result()
        del futures['text']

        # Collect results for images, audio, tables
        tables_with_metadata = futures['tables'].result()
        del futures['tables']

        page_data_images, image_count = futures['images'].result()
        del futures['images']

        page_data_audio, audio_count = futures['audio'].result()
        del futures['audio']

    # Merge page_data from images and audio back into the main page_data
    # page_data_images and page_data_audio should be merging updates into the main page_data
    page_data.update(page_data_images)
    page_data.update(page_data_audio)

    # Now we have a unified page_data with metadata, images, and audio
    page_data = update_page_data_with_tables(page_data, tables_with_metadata)

    # At this point, page_data should contain 'images' arrays for pages
    page_texts = add_image_pointers_to_text(page_texts, page_data)

    page_texts = remove_tables_from_text(page_texts, tables_with_metadata)

    print("[Step 9] Saving extracted text and metadata to output files...")
    save_final_text_and_metadata(pdf_name, output_folder, page_texts, metadata, page_data)
    print(f"[Complete] PDF processing finished. Extracted {image_count} images and {audio_count} audio files.")

    return image_count, audio_count



if __name__ == "__main__":
    pdf_path = "AI-Pocket-Tutor/Sentimental Analysis.pdf"  # Replace with your actual PDF path
    start_time = time.time()
    print("[Step 1] Setting up output folder...")
    process_pdf(pdf_path)
    print(f"[Done] Total processing time: {time.time() - start_time:.2f} seconds.")
