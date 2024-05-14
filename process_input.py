import zipfile
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def processInput(file):
    if(file==None):
        print("Could not upload files! Try again!")
        return
    else:
        chunks = []
        if(file.type == "application/x-zip-compressed"):
            # Extract PDF files from zip
            with zipfile.ZipFile(file, "r") as z:
                z.extractall("./data/")
        elif(file.type == "application/pdf"):
            # Read text from the pdf
            pdf_info = PdfReader(file)
            
            # Append all pdf text data into a string
            text = ""
            for page in pdf_info.pages:
                text += page.extract_text()
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)  
        else:
            print("Please upload a PDF or a zip file.")
            return          
    return chunks

