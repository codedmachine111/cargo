import os
import uuid
import time
import asyncio
import streamlit as st
from streamlit_float import float_init
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

from process_input import processInput
from templates import css, user_template, bot_template
from helpers import *
from utils import *

import cloudinary
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from vertexai.preview.language_models import ChatModel, ChatMessage, ChatSession

# Load environment variables
load_dotenv()

# Configure Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./secret.json"

# Configure Google AI studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "main"

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

CLOUD_NAME = os.getenv("CLOUDINARY_NAME")
CLOUD_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(cloud_name=CLOUD_NAME, api_key=CLOUD_API_KEY, api_secret=CLOUD_API_SECRET)

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

system_context = "You are an automotive assistant at an automobile company.\
Your task is to give detailed answers to queries in a few sentences \
based on the CONTEXT provided. The CONTEXT can include text, table summaries\
or image summaries. If the QUESTION is not related to automobiles, you MUST \
respond with: SORRY I COULD NOT FIND ANYTHING RELEVANT. If QUESTION is a greeting,\
reply with HOW MAY I HELP YOU. If the QUESTION is very vague, you MUST ask a follow-up question to get more clarity. \
If you dont know about the car name, you MUST ask for it"

system_image_context = "You are an automotive assistant at an automobile company.\
Your task is to summarize the CONTEXT provided. The CONTEXT can include text, table summaries\
or image summaries."

chat_model = ChatModel.from_pretrained("chat-bison@002")

def get_response(message, context, history):
    chat_history=[]
    for h in history:
        chat_history.append(ChatMessage(h["content"], h["role"]))
    
    chat = ChatSession(model=chat_model, context=context, message_history=chat_history)
    response = chat.send_message(message)
    return response

async def user_input(question="", image=False, image_path="", cloud_url=""):
    summary=""
    if image:
        summary += summarize_image(model, image_path)
        query_vector = embeddings.embed_query(text=summary)
    else:
        if len(question) == 0:
            return "Please enter a valid prompt", ""
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

    query_image_html = f'<img src="{cloud_url}" id="res_img"/>'
    if image:
        images_html = ""
    else:
        images_html = "".join([f'<img src="{img}" id="res_img"/>' for img in files_to_display[:2]])

    if image:
        output_text = get_response(summary, system_image_context+context, st.session_state.chat_history)
    else:
        output_text = get_response(question, system_context+context, st.session_state.chat_history)
    output_text = output_text.text

    # Update chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": question, "images": query_image_html})
    st.session_state.chat_history.append({"role": "assistant", "content": output_text, "images": images_html})

    return output_text, images_html

def main():
    st.set_page_config(page_title="Cargo", page_icon=":ship:", layout="wide")
    st.image('./assets/captain.png', width=200)
    st.header("Ask the Captain about your files! :male-pilot:")
    st.write(css, unsafe_allow_html=True)
    float_init()

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

        uploaded_files = st.file_uploader(type=['pdf'], label="Click below or drag and drop your files to upload!", accept_multiple_files=True)

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

    col1, col2 = st.columns([9, 4])
        
    with col1:
        container = st.container(height=600, border=True)
        # Load initial chat
        for message in st.session_state.chat_history:
            if(message["role"] == "user"):
                with container:
                    with st.chat_message(message["role"], avatar="❓"):
                        st.write(user_template.replace("{{MSG}}", message["content"]).replace("{{IMAGE}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)
            else:
                with container:
                    with st.chat_message(message["role"], avatar="⚓"):
                            st.write(bot_template.replace("{{MSG}}", message["content"]).replace("{{IMAGES}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)

        # User input
        if prompt := st.chat_input("Ask about your files..."):
            if len(st.session_state.processed_files)>=0 :
                # Display user message in chat message container
                with container:
                    with st.chat_message("user", avatar="❓"):
                        st.write(user_template.replace("{{MSG}}", prompt).replace("{{IMAGE}}", ""), unsafe_allow_html=True)

                with st.spinner("Thinking..."):
                    response, images = asyncio.run(user_input(question=prompt))
                    # Display assistant response in chat message container
                    with container:
                        with st.chat_message("assistant", avatar="⚓"):
                            bot_response = bot_template.replace("{{MSG}}", response).replace("{{IMAGES}}", images)
                            st.write(bot_response, unsafe_allow_html=True)
            else:
                st.write("Please upload some documents to chat with them!")

    with col2:
        container1 = st.container()
        with container1:
            image_query = st.file_uploader(type=['png', 'jpg', 'jpeg'], label="Search an image", accept_multiple_files=False)
            if image_query is not None:
                st.image(image_query, width=200)

            if st.button("Search"):
                if image_query is None:
                    st.write("No images selected!")
                else:
                    query_img_path = f"./{image_query.name}"
                    # Save the uploaded file to the local path
                    with open(query_img_path, "wb") as f:
                        f.write(image_query.getbuffer())

                    # Upload image to cloudinary
                    image_query_url = upload_image(query_img_path)
                    image_query_html = f'<img src="{image_query_url}" id="res_img"/>'

                    with container:
                        with st.chat_message("user", avatar="❓"):
                            st.write(user_template.replace("{{MSG}}", "").replace("{{IMAGE}}", image_query_html), unsafe_allow_html=True)

                    with st.spinner("Thinking..."):
                        response, images = asyncio.run(user_input(image=True, image_path=query_img_path, cloud_url=image_query_url))

                        with container:
                            with st.chat_message("assistant", avatar="⚓"):
                                bot_response = bot_template.replace("{{MSG}}", response).replace("{{IMAGES}}", images)
                                st.write(bot_response, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
