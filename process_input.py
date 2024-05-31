import streamlit as st
import tempfile
import pytesseract
import uuid
import os
from unstructured.partition.pdf import partition_pdf

# Configure Pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_all_data(pdf_path):
    '''
        Extracts text, tables and images from a PDF file and saves the images to a specified directory.

        Parameters:
        pdf_path (str): Path to the PDF file.

        Returns:
        tuple: Lists of text elements, table elements, and a unique folder ID for the extracted images.
    '''
    text_elements = []
    table_elements = []

    unique_id = str(uuid.uuid4())
    image_dir = os.path.join('figures', unique_id)
    os.makedirs(image_dir, exist_ok=True)

    raw_elements = partition_pdf(
        filename=pdf_path,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=1000,
        new_after_n_chars=1500,
        combine_text_under_n_chars=250,
        strategy="hi_res",
        extract_image_block_output_dir=image_dir
    )

    for element in raw_elements:
        if 'unstructured.documents.elements.CompositeElement' in str(type(element)):
            text_elements.append(str(element))
        elif 'unstructured.documents.elements.Table' in str(type(element)):
            table_elements.append(str(element))

    return text_elements, table_elements, unique_id

def processInput(file):
    '''
        Processes and extracts data from an uploaded PDF file.

        Parameters:
        file: The uploaded file object.

        Returns:
        tuple: Extracted text, tables, and folder ID containing images. 
        Returns an empty list if the file is None or not a PDF.
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
                text, tables, folder_id = extract_all_data(temp_pdf.name)
        else:
            st.error("Please upload only PDF files")
    return text, tables, folder_id