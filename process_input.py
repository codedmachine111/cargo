import streamlit as st
import tempfile
import pytesseract

from unstructured.partition.pdf import partition_pdf
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Slower but accurate extraction
def extract_all_data(pdf_path):
    raw_elements = partition_pdf(
        filename=pdf_path,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,

        max_characters=1000,
        new_after_n_chars=1500,
        combine_text_under_n_chars=250,
        strategy="hi_res",
    )  

    text_elements = []
    table_elements = []

    for element in raw_elements:
        if 'unstructured.documents.elements.CompositeElement' in str(type(element)):
            text_elements.append(str(element))
        elif 'unstructured.documents.elements.Table' in str(type(element)):
            table_elements.append(str(element))

    return text_elements, table_elements

# Process input file
def processInput(file):
    '''
        Processes the input file based on file type
    '''
    if file is None:
        st.error("Could not upload files! Try again!")
        return []
    else:
        if file.type == "application/pdf":
            # Extract Text and Tables
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf.seek(0)
                text, tables = extract_all_data(temp_pdf.name)
        else:
            st.error("Please upload only PDF files")
    return text, tables