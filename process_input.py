import zipfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import camelot

def processInput(file):
    if file is None:
        print("Could not upload files! Try again!")
        return []
    else:
        chunks = []
        if file.name.endswith(".zip"):
            with zipfile.ZipFile(file, "r") as z:
                z.extractall("./data/")
                for filename in z.namelist():
                    if filename.endswith(".pdf"):
                        with open(f"./data/{filename}", 'rb') as pdf_file:
                            chunks.extend(extract_text_and_tables_from_pdf(pdf_file))
        elif file.name.endswith(".pdf"):
            chunks = extract_text_and_tables_from_pdf(file)
        else:
            print("Please upload a PDF or a zip file.")
            return []

    return chunks

def extract_text_and_tables_from_pdf(file):
    text_chunks = []
    table_chunks = []

    pdf_info = PdfReader(file)
    text = ""
    for page in pdf_info.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(text)
    
    tables = camelot.read_pdf(file.name, pages='all', flavor='stream')
    for table in tables:
        table_chunks.append(table.df.to_string())

    combined_chunks = text_chunks + table_chunks
    return combined_chunks
