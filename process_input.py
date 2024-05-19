import zipfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import camelot
import tempfile
import re
import fitz
from io import BytesIO

def prune_text(text):
    def replace_cid(match):
        return ''

    # Regular expression to find all (cid:x) patterns
    cid_pattern = re.compile(r'\(cid:(\d+)\)')
    pruned_text = re.sub(cid_pattern, replace_cid, text)
    return pruned_text

def processInput(file):
    '''
        Processes the input file based on file type
    '''
    text_chunks = []
    table_chunks = []
    if file is None:
        st.error("Could not upload files! Try again!")
        return []
    else:
        if file.type == "application/x-zip-compressed":
            # Extract PDF files from zip
            with zipfile.ZipFile(BytesIO(file.read()), "r") as z:
                for pdf_file in z.namelist():
                    with z.open(pdf_file) as pdf:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                            temp_pdf.write(pdf.read())
                            temp_pdf.seek(0)
                            text, tables = extract_text_and_tables(temp_pdf.name)
                            text_chunks += text
                            table_chunks += tables
        elif file.type == "application/pdf":
            # Extract Text and Tables
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf.seek(0)
                text, tables = extract_text_and_tables(temp_pdf.name)
                text_chunks += text
                table_chunks += tables
        else:
            st.error("Please upload a PDF or a zip file.")

    return text_chunks, table_chunks

def extract_text_and_tables(pdf_path):
    '''
        Takes one pdf file path as input and returns chunked text and tables
    '''
    chunks_per_pdf = []
    all_table_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    table_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500, length_function=len)

    doc = fitz.open(pdf_path)

    pdf_chunks = []
    for page in doc:
        text = page.get_text()
        if text:
            # pruned_text = prune_text(text)
            pdf_chunks.extend(text_splitter.split_text(text))
    chunks_per_pdf.append(pdf_chunks)

    # Extract tables
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    for table in tables:
        table.df.replace(to_replace='\(cid\:[0-9]+\)', value='', inplace=True, regex=True)
        table.df.replace(to_replace='\\n', value=' ', inplace=True, regex=True)
        table.df.replace(to_replace='\s+', value=' ', inplace=True, regex=True)
        
        # Remove rows and columns that are completely empty
        table.df.replace("", float("NaN"), inplace=True)
        table.df.dropna(how='all', inplace=True)
        table.df.dropna(axis=1, how='all', inplace=True)
        table_df = table.df
        table_json = table_df.to_json(orient='split')
        all_table_chunks.extend(table_splitter.split_text(table_json))

    return chunks_per_pdf, all_table_chunks
