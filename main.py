import os
import uuid
import asyncio
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

from process_input import processInput
from templates import css, user_template, bot_template

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain

# Load environment variables
load_dotenv()

# Configure Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./secret.json"

# Configure Google AI studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-index"

# Ensure the index exists in pinecone
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

# Connect to the pinecone index
index = pc.Index(index_name, host=os.getenv("PINECONE_HOST"))

def get_vector_store(text_chunks):
    '''
        Converts chunks of data into vector embeddings and saves it to pinecone vector store.
    '''
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = []

    # Batch the text chunks to process in groups of 100
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        vectors.extend(batch_vectors)

    # Prepare data for upsert
    pinecone_vectors = [(str(uuid.uuid4()), vector, {'text': text_chunks[i]}) for i, vector in enumerate(vectors)]

    max_batch_size = 100
    for i in range(0, len(pinecone_vectors), max_batch_size):
        batch = pinecone_vectors[i:i+max_batch_size]
        index.upsert(vectors=batch)

async def get_conversational_chain():
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Your task is to give detailed answers to users questions in a few sentences based on the context provided. If the query is not relevant to the context, you need to respond with: Sorry I could'nt find anything relevant,Try asking again"),
        ("human", "context: {context}\n\nQuestion: {question}")
    ])

    llm = ChatVertexAI(model="chat-bison@002", convert_system_message_to_human=True)
    chain = prompt | llm
    return chain

async def user_input(question):
    if len(question) == 0:
        return "Please enter a valid prompt"
    
    # Convert question to embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_vector = embeddings.embed_query(text=question)

    if isinstance(query_vector, list):
        query_vector = [float(val) for val in query_vector]
    else:
        query_vector = list(query_vector)

    # Query Pinecone
    docs = index.query(vector=query_vector, top_k=2, include_metadata=True).matches

    if not docs:
        return "Sorry, I couldn't find anything relevant in the knowledge base."

    # Collect context from the documents
    context = "\n".join(doc.metadata["text"] for doc in docs)

    # Get the conversational chain
    chain = await get_conversational_chain()
    
    # Prepare the inputs for the chain
    inputs = {
        "context": context,
        "question": question
    }

    # Run the chain and get the response
    response = chain.invoke(inputs)

    output_text = response.content if hasattr(response, 'content') else "Sorry, I couldn't generate a response."
    
    # Update chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": output_text})
    
    return output_text

def display_chat_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        with st.container(height=600, border=True):
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Cargo", page_icon=":ship:")
    st.image('./assets/captain.png', width=200)
    st.header("Ask the Captain about anything! :male-pilot:")
    st.write(css, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.title('Cargo')

        st.markdown('''
            ## About
            Cargo is an RAG designed to allow users to chat with multiple large PDF documents with ease. 
        ''')
        add_vertical_space(3)
        st.divider()
        st.header("Upload your files and start talking with the captain!")

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

                            # Convert to embeddings and store in pinecone
                            st.write("Converting text to embeddings...")
                            get_vector_store(text_chunks[0])
                            status.update(label="Successfully stored embeddings", state="running", expanded=True)

                            st.session_state.processed_files.append(new_file.name)
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                else:
                    st.write("All selected files have been processed already.")
            else:
                st.write("No PDFs uploaded.")

    # Display chat history
    display_chat_history()

    # User input
    input = st.text_input("Ask a question")
    if input is not None and len(input) > 0:
        if len(st.session_state.processed_files) > 0:
            response = asyncio.run(user_input(input))
            # Display updated chat history
            display_chat_history()
        else:
            st.write("Upload files to chat with the Captian!")
    else:
        st.write("Please enter a valid prompt")


if __name__ == '__main__':
    main()
