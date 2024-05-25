import os
import uuid
import time
import asyncio
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

from process_input import processInput
from templates import css, user_template, bot_template
from helpers import *

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.base import ConversationChain

# Load environment variables
load_dotenv()

# Configure Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./secret.json"

# Configure Google AI studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-index"

# Ensure the index exists in pinecone
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the pinecone index
index = pc.Index(INDEX_NAME)

# Define summarization model
MODEL_NAME = "models/gemini-1.5-pro-latest"
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
model = genai.GenerativeModel(model_name=MODEL_NAME, safety_settings=safety_settings)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vector_store(texts, text_sum, tables, tables_sum, image_paths, images_sum):
    '''
        Converts data into vector embeddings and saves it to vector store.
    '''
    vectors = []

    # Embed and store text summaries
    for text, summary in zip(texts, text_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'text_summary', 'content': summary, 'raw_text': text}))

    # Embed and store table summaries
    for table, summary in zip(tables, tables_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'table_summary', 'content': summary, 'raw_table': table}))

    # Embed and store image summariesstore image embeddings using pinecone
    for image_path, summary in zip(image_paths, images_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'image_summary', 'content': summary, 'filepath': image_path}))

    # Upsert to pinecone
    max_batch_size = 100
    for i in range(0, len(vectors), max_batch_size):
        batch = vectors[i:i + max_batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting vectors: {e}")

async def get_conversational_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an automotive assistant at an automobile company. Your task is to give detailed answers to queries in a few sentences based on the CONTEXT provided below. The CONTEXT can include text, table summaries or image summaries. If the QUESTION is not relevant to the CONTEXT, you MUST respond with: SORRY I COULD NOT FIND ANYTHING RELEVANT. If QUESTION is a greeting, reply with HOW MAY I HELP YOU"),
        ("human", "CONTEXT: {context} \n QUESTION: {question}")
    ])
    prompt.input_variables = ["context", "question"]
    llm = ChatVertexAI(model="chat-bison@002", convert_system_message_to_human=True)
    # chain = ConversationChain(llm=llm, prompt=prompt)
    chain = prompt | llm
    return chain

async def user_input(question):
    if len(question) == 0:
        return "Please enter a valid prompt"
    
    # Convert question to embedding
    query_vector = embeddings.embed_query(text=question)

    if isinstance(query_vector, list):
        query_vector = [float(val) for val in query_vector]
    else:
        query_vector = list(query_vector)

    # Query Pinecone
    docs = index.query(vector=query_vector, top_k=4, include_metadata=True).matches

    if not docs:
        return "Sorry, I couldn't find anything relevant in the knowledge base."

    # Collect context from the documents
    text_data = []
    table_data = []
    image_summaries = []
    files_to_display = []
    for doc in docs:
        metadata = doc.metadata
        if 'raw_text' in metadata:
            text_data.append(metadata['raw_text'])
        elif 'raw_table' in metadata:
            table_data.append(metadata['raw_table'])
        elif 'filepath' in metadata:
            files_to_display.append(metadata['filepath'])
            image_summaries.append(metadata['content'])

    context = "Text:\n" + "\n".join(text_data) + "\n\n" + \
              "Tables:\n" + "\n".join(table_data) + "\n\n" + \
              "Image summaries:\n" + "\n".join(image_summaries[:2])

    images_to_display = [load_image_as_base64(filepath) for filepath in files_to_display[:2]]
    images_html = "".join([f'<img src="data:image/png;base64,{img}" id="res_img"/>' for img in images_to_display])

    # Get the conversational chain
    chain = await get_conversational_chain()
    
    # Prepare the inputs for the chain
    inputs = {
        "context": context,
        "question": question,
        # "summary": "\n".join([msg["content"] for msg in st.session_state.chat_history if msg["role"] == "assistant"])
    }

    # Run the chain and get the response
    response = chain.invoke(input=inputs)

    output_text = response.content if hasattr(response, 'content') else "Sorry, I couldn't generate a response."
    
    # Update chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": question, "images": None})
    st.session_state.chat_history.append({"role": "assistant", "content": output_text, "images": images_html})
    
    return output_text, images_html

def main():
    st.set_page_config(page_title="Cargo", page_icon=":ship:")
    st.image('./assets/captain.png', width=200)
    st.header("Ask the Captain about your files! :male-pilot:")
    st.write(css, unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

    # SIDEBAR
    with st.sidebar:
        st.title('Cargo :ship::anchor:')

        st.markdown('''
            Cargo is an RAG designed to allow users to chat with multiple large PDF documents with ease. 
        ''')
        add_vertical_space(3)
        st.divider()
        st.header("Upload your files and start talking with the captain! :open_file_folder:")

        uploaded_files = st.file_uploader(type=['pdf', 'zip'], label="Click below or drag and drop your files to upload!", accept_multiple_files=True)

        if st.button("Process"):
            # Process the input files accordingly
            if uploaded_files:
                new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]

                if new_files:
                    with st.status("Please wait as we process your input files...", expanded=True) as status:
                        st.write("Extracting data from Documents...")
                        start_time = time.time()
                        for new_file in new_files:
                            # Extract data
                            texts, tables, folder_id = processInput(new_file)
                            status.update(label="Successfully extracted data!", state="running", expanded=True)
                            st.write(f'Extracted {len(texts)} text chunks and {len(tables)} tables')
                            st.write(f"Extracted data in : {time.time()-start_time:.2f} s")

                            # Generate summaries of data
                            st.write("Summarizing data...")
                            text_summ, table_summ = generate_summaries(model, texts, tables)
                            image_summ, image_paths = generate_image_summaries(model, folder_id)
                            st.write(f'Extracted {len(image_paths)} images and summarized {len(image_summ)}')

                            status.update(label="Successfully summarized data", state="running", expanded=True)
                            st.write(f"Summarized data in : {time.time()-start_time:.2f} s")

                            # Convert to embeddings and store in pinecone
                            st.write("Converting text to embeddings...")
                            get_vector_store(texts, text_summ, tables, table_summ, image_paths, image_summ)
                            status.update(label="Successfully stored embeddings", state="running", expanded=True)
                            st.write(f"Updated vector store in : {time.time()-start_time:.2f} s")

                            st.session_state.processed_files.append(new_file.name)
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                else:
                    st.write("All selected files have been processed already.")
            else:
                st.write("No PDFs uploaded.")

    # Load initial chat
    for message in st.session_state.chat_history:
        if(message["role"] == "user"):
            with st.chat_message(message["role"], avatar="❓"):
                st.write(user_template.replace("{{MSG}}", message["content"]).replace("{{IMAGES}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"], avatar="⚓"):
                st.write(bot_template.replace("{{MSG}}", message["content"]).replace("{{IMAGES}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)
    
    # User input
    if prompt := st.chat_input("Ask about your files..."):
        if len(st.session_state.processed_files)>=0 :
            # Display user message in chat message container
            with st.chat_message("user", avatar="❓"):
                st.write(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                response, images = asyncio.run(user_input(prompt))
                # Display assistant response in chat message container
                with st.chat_message("assistant", avatar="⚓"):
                    chat_history = st.session_state.chat_history[-1]
                    response_images = chat_history["images"]
                    bot_response = bot_template.replace("{{MSG}}", response).replace("{{IMAGES}}", images)
                    st.write(bot_response, unsafe_allow_html=True)
        else:
            st.write("Please upload some documents to chat with them!")

if __name__ == '__main__':
    main()
