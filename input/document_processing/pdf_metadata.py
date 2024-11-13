import fitz  # PyMuPDF
import os
import json

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

        # Process each link
        for link in links:
            link_entry = {
                "position": convert_to_serializable(link.get("from", None)),
                "destination": link.get("uri") or link.get("page")
            }

            # Capture the text associated with the link
            link_rect = link.get("from", None)
            if link_rect:
                text = page.get_textbox(link_rect).strip()  # Extract text in the bounding box of the link
                link_entry["text"] = text

            page_links.append(link_entry)

        # Only add links to page data if there are links on the page
        if page_links:
            page_data[f"page_{i + 1}"] = {"links": page_links}

    return metadata, page_data

def extract_images(pdf_path, output_folder, page_data):
    """
    Extracts images from each page and saves them to the specified output folder.
    Updates page_data with information about image positions and file paths.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)  # Create images folder if it doesn't exist

    for i in range(total_pages):
        page = doc.load_page(i)
        image_list = page.get_images(full=True)
        page_images = []

        # Save each image on the page
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{i + 1}_img_{img_index + 1}.{image_ext}"

            # Save image file
            image_path = os.path.join(images_folder, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Retrieve the bounding box (position) for the image
            img_position = None
            img_rects = page.get_image_rects(xref)
            if img_rects:
                img_position = list(img_rects[0])  # Convert Rect to list for JSON serialization

            # Store positional data and path for each image
            image_entry = {
                "position": img_position,
                "file_path": image_path
            }
            page_images.append(image_entry)

        # Add images only if there are images on the page
        if page_images:
            if f"page_{i + 1}" in page_data:
                page_data[f"page_{i + 1}"]["images"] = page_images
            else:
                page_data[f"page_{i + 1}"] = {"images": page_images}

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

        # Check for annotations that may contain multimedia content
        for annot in page.annots():
            try:
                if annot.type[0] == fitz.PDF_ANNOT_SCREEN:  # Only handle 'Screen' annotations
                    audio_entry = {
                        "position": convert_to_serializable(annot.rect),  # Position of the annotation
                        "annotation_type": annot.type[1],  # Annotation type, e.g., "Screen"
                        "description": "Possible multimedia content (audio or video)"
                    }
                    page_audios.append(audio_entry)
            except Exception as e:
                print(f"Error processing annotation on page {i + 1}: {e}")

        # Add possible audio annotations only if present on the page
        if page_audios:
            if f"page_{i + 1}" in page_data:
                page_data[f"page_{i + 1}"]["audios"] = page_audios
            else:
                page_data[f"page_{i + 1}"] = {"audios": page_audios}

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
