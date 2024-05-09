import os
import pickle
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


# SIDEBAR
with st.sidebar:
    st.title('Abridge')
    st.markdown('''
        ## About
        Summarize all your pdf files at ease in one go!
    ''')
    add_vertical_space(5)
    

def main():
    st.header("Drop your PDF files here")

    # Take PDF files as input
    uploaded_file = st.file_uploader(type=['pdf', 'zip'], label="Upload pdf files")

    if(uploaded_file and uploaded_file.type == "application/x-zip-compressed"):
        # Extract the pdfs from the zip file
        pass
    elif(uploaded_file and uploaded_file.type == "application/pdf"):
        # Read pdf data
        pdf_info = PdfReader(uploaded_file)
        
        # Append all pdf text data into a string
        text = ""
        for page in pdf_info.pages:
            text += page.extract_text()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        # # Convert the text chunks to embeddings
        # try:  
        #     load_dotenv()
        #     os.environ["OPENAI_API_KEY"]
        #     embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        #     vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f'{pdf.name[:-4]}.pkl', "wb") as f:
        #         pickle.dump(vectorStore,f)
            
        # except KeyError: 
        #     print("Please add your OPENAI_API_KEY")
        

        st.write(chunks)
    else:
        st.write('''No pdf uploaded''')

if __name__ == '__main__':
    main()