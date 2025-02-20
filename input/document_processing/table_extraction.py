import pdfplumber

def is_valid_table(table):
    """Checks if a table has consistent rows and sufficient columns and rows to likely be a valid table."""
    if len(table) < 2 or any(len(row) < 2 for row in table):
        return False
    
    col_count = len(table[0])
    return all(len(row) == col_count for row in table)

def extract_tables_with_metadata(pdf_path):
    """Extracts valid tables along with their metadata from the PDF and returns them with unique IDs."""
    tables_with_metadata = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            valid_tables = [table for table in tables if is_valid_table(table)]
            
            if valid_tables:
                for table_index, table in enumerate(valid_tables, start=1):
                    bbox = page.find_tables()[table_index - 1].bbox
                    tables_with_metadata.append({
                        "page": page_num,
                        "position": bbox,
                        "table": table,
                        "table_id": f"table_{page_num}_{table_index}"
                    })

    return tables_with_metadata
