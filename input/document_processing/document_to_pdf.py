import os
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(filename='conversion.log', level=logging.INFO)

def convert_to_pdf(input_path):
    input_path = Path(input_path)
    output_dir = input_path.parent
    output_path = output_dir / f"{input_path.stem}.pdf"
    
    if input_path.suffix.lower() in ['.txt', '.rtf', '.doc', '.docx', '.odt', '.xls', '.xlsx', '.ppt', '.pptx', '.html', '.htm', '.xml', '.md']:
        convert_with_libreoffice(input_path, output_dir)
    elif input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        convert_image_to_pdf(input_path, output_path)
    elif input_path.suffix.lower() == '.pdf':
        return input_path  # Return if already PDF
    else:
        print(f"Unsupported format for {input_path}")
        return None

    return output_path if output_path.exists() else None

def convert_with_libreoffice(input_path, output_dir):
    try:
        cmd = [r"C:\Program Files\LibreOffice\program\soffice.exe",
       '--headless', '--convert-to', 'pdf', '--outdir', str(output_dir), str(input_path)]

        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Conversion failed for {input_path}: {e}")

def convert_image_to_pdf(input_path, output_path):
    from PIL import Image
    try:
        image = Image.open(input_path).convert('RGB')
        image.save(output_path, 'PDF')
    except Exception as e:
        logging.error(f"Image conversion failed: {e}")
