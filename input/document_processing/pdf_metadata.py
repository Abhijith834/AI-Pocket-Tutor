import fitz  # PyMuPDF
import os
import json
import hashlib

def convert_to_serializable(obj):
    """Helper function to convert unsupported types like Rect and Point to lists."""
    if isinstance(obj, fitz.Rect) or isinstance(obj, fitz.Point):
        return list(obj)
    return obj

def extract_metadata_and_links(pdf_path):
    """
    Extracts metadata and links along with the surrounding text that contains each link.
    """
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    total_pages = doc.page_count
    page_data = {}

    # Extract links from each page
    for i in range(total_pages):
        page = doc.load_page(i)
        links = page.get_links()
        page_links = []
        for link in links:
            link_entry = {
                "position": convert_to_serializable(link.get("from", None)),
                "destination": link.get("uri") or link.get("page")
            }
            link_rect = link.get("from", None)
            if link_rect:
                text = page.get_textbox(link_rect).strip()
                link_entry["text"] = text
            page_links.append(link_entry)

        if page_links:
            page_data[f"page_{i + 1}"] = {"links": page_links}

    return metadata, page_data


def extract_images(pdf_path, output_folder, page_data):
    """
    Extracts images from each page and saves them to the specified output folder.
    Updates page_data with information about image positions and file paths.
    
    Deduplicates images doc-wide by computing an MD5 hash of the image bytes.
    If the same bits appear on multiple pages, it is only extracted once overall.
    """
    import fitz
    import hashlib
    
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # A single set for the entire PDF to track unique image hashes
    seen_hashes = set()

    for i in range(total_pages):
        page_num = i + 1
        page = doc.load_page(i)
        image_list = page.get_images(full=True)

        # We'll store image entries for this page in a local list
        # that we add to page_data at the end of each page
        page_images = []
        # We'll keep a counter for how many unique images we add on this page
        page_image_counter = 0

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Compute MD5 hash for deduplication
            image_hash = hashlib.md5(image_bytes).hexdigest()

            # If we already processed this exact image data, skip
            if image_hash in seen_hashes:
                continue

            # Otherwise, record it as seen
            seen_hashes.add(image_hash)

            # Now we can store it
            image_ext = base_image["ext"]
            page_image_counter += 1
            image_name = f"page_{page_num}_img_{page_image_counter}.{image_ext}"
            image_path = os.path.join(images_folder, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Try to get the bounding box (rect)
            img_rects = page.get_image_rects(xref)
            if img_rects:
                bounding_box = list(img_rects[0])
            else:
                bounding_box = None

            image_entry = {
                "position": bounding_box,
                "file_path": image_path
            }
            page_images.append(image_entry)

        if page_images:
            page_key = f"page_{page_num}"
            if page_key in page_data:
                page_data[page_key]["images"] = page_images
            else:
                page_data[page_key] = {"images": page_images}

    return page_data

def extract_audio(pdf_path, output_folder, page_data):
    """
    Identifies 'Screen' annotations (which may contain multimedia content)
    and logs their positions in page_data.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    for i in range(total_pages):
        page = doc.load_page(i)
        page_audios = []
        for annot in page.annots():
            try:
                if annot.type[0] == fitz.PDF_ANNOT_SCREEN:
                    audio_entry = {
                        "position": convert_to_serializable(annot.rect),
                        "annotation_type": annot.type[1],
                        "description": "Possible multimedia content (audio or video)"
                    }
                    page_audios.append(audio_entry)
            except Exception as e:
                print(f"Error processing annotation on page {i + 1}: {e}")

        if page_audios:
            page_key = f"page_{i + 1}"
            if page_key in page_data:
                page_data[page_key]["audios"] = page_audios
            else:
                page_data[page_key] = {"audios": page_audios}

    print("Identified potential multimedia content on pages with 'Screen' annotations but did not extract due to PyMuPDF limitations.")
    return page_data


def save_metadata_to_json(metadata, page_data, output_folder, pdf_name):
    """
    Combines metadata and page data into a JSON file and saves it.
    """
    data = {
        "metadata": metadata,
        "pages": page_data
    }
    json_path = os.path.join(output_folder, f"{pdf_name}_metadata.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, default=convert_to_serializable)
    return json_path
