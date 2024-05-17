import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from process_input import processInput
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import uuid
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-index"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
  pc.create_index(
      name=index_name,
      dimension=768,
      metric="cosine",
      spec=ServerlessSpec(
          cloud='aws', 
          region='us-east-1'
      ) 
  )

# Connect to the index
index = pc.Index(index_name, host=os.getenv("PINECONE_HOST"))

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = []

    # Batch the text chunks to process in groups of 100
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        vectors.extend(batch_vectors)

    # Prepare data for upsert
    pinecone_vectors = [(str(uuid.uuid4()), vector) for vector in vectors]

    max_batch_size = 50  # Adjust this size based on Pinecone's request size limit
    for i in range(0, len(pinecone_vectors), max_batch_size):
        batch = pinecone_vectors[i:i+max_batch_size]
        index.upsert(vectors=batch)
        
    return pinecone_vectors

def main():
    st.set_page_config(page_title="Abridge", page_icon=":brain:")
    st.header("Chat with your all your pdf files with ease! :books:")
    
    input = st.text_input("Ask a question")
    
    # SIDEBAR
    with st.sidebar:
        st.title('Abridge')
        st.markdown('''
            ## About
            Chat with your all your pdf files with ease!
        ''')
        add_vertical_space(3)
        st.divider()
        st.header("Your files")
        
        uploaded_files = st.file_uploader(type=['pdf', 'zip'], label="Click below or drag and drop your files to upload!", accept_multiple_files=True)
        
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []

        if st.button("Process"):
            # Process the input files accordingly
            if uploaded_files:
                new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
                
                if new_files:
                    with st.status("Please wait as we process your input files...", expanded=True) as status:
                        st.write("Extracting data from Documents...")
                        for new_file in new_files:
                            # Extract data
                            text_chunks, table_chunks = processInput(new_file)
                            status.update(label="Successfully Chunked data!", state="running", expanded=True)

                            # Convert to embeddings
                            st.write("Converting text to embeddings...")
                            embeddings = get_vector_store(text_chunks[0])
                            status.update(label="Successfully embedded chunks!", state="running", expanded=True)

                            st.session_state.processed_files.append(new_file.name)
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                else:
                    st.write("All selected files have been processed already.")
            else:
                st.write("No PDFs uploaded.")

if __name__ == '__main__':
    main()
