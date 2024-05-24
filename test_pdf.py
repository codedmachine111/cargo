import zipfile
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile
import re
from io import BytesIO
# import camelot

def prune_text(text):
    def replace_cid(match):
        return ''

    # Regular expression to find all (cid:x) patterns
    cid_pattern = re.compile(r'\(cid:(\d+)\)')
    pruned_text = re.sub(cid_pattern, replace_cid, text)
    return pruned_text

def processInput(file):
    if file is None:
        st.error("Could not upload files! Try again!")
        return [], []
    else:
        text_chunks = []
        # table_chunks = []

        if file.type == "application/x-zip-compressed":
            # Extract PDF files from zip
            with zipfile.ZipFile(BytesIO(file.read()), "r") as z:
                for pdf_file in z.namelist():
                    with z.open(pdf_file) as pdf:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                            temp_pdf.write(pdf.read())
                            temp_pdf.seek(0)
                            text = extract_text_and_tables(temp_pdf.name)
                            text_chunks += text
                            # table_chunks += tables
        elif file.type == "application/pdf":
            # Extract Text and Tables
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf.seek(0)
                text = extract_text_and_tables(temp_pdf.name)
                text_chunks += text
                # table_chunks += tables
        else:
            st.error("Please upload a PDF or a zip file.")
            return [], []

    return text_chunks

def extract_text_and_tables(pdf_path):
    '''
        Takes one pdf file path as input and returns chunked text and tables
    '''
    text_chunks = []
    # all_table_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    
    # Open the PDF file with PyMuPDF
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text = page.get_text("text")
        if text:
            text_chunks.extend(text_splitter.split_text(prune_text(text)))
    
    # Extract tables using camelot
    # tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    # for table in tables:
    #     table_df = table.df
    #     table_text = table_df.to_string(index=False, header=False)
    #     all_table_chunks.extend(text_splitter.split_text(table_text))

    return text_chunks

# Streamlit part
def main():
    st.title("PDF Text and Table Extraction")
    
    uploaded_file = st.file_uploader("Upload PDF or ZIP", type=['pdf', 'zip'])
    
    if uploaded_file:
        text_chunks= processInput(uploaded_file)
        
        st.write("Extracted Text:")
        st.write(f"Number of chunks: {len(text_chunks)}")
        for i, chunk in enumerate(text_chunks):
            st.write(f"Chunk {i + 1}:")
            st.write(chunk)
        
        # st.write("Extracted Tables:")
        # for i, table_chunk in enumerate(table_chunks):
        #     st.write(f"Table Chunk {i + 1}:")
        #     st.write(table_chunk)
    else:
        st.write("Please upload a PDF or ZIP file.")

if __name__ == '__main__':
    main()
