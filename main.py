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
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from vertexai.preview.language_models import ChatModel, ChatMessage, ChatSession

# Load environment variables
load_dotenv()

# Configure Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./secret.json"

# Configure Google AI studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Configure pinecone index
INDEX_NAME = "main"
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

# Connect to cloudinary
CLOUD_NAME = os.getenv("CLOUDINARY_NAME")
CLOUD_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
cloudinary.config(cloud_name=CLOUD_NAME, api_key=CLOUD_API_KEY, api_secret=CLOUD_API_SECRET)

# Define summarization model
MODEL_NAME = "models/gemini-1.5-pro-latest"

# Define safety thresholds
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

# Define SUMMARIZATION, EMBEDDINGS and CHAT models
model = genai.GenerativeModel(model_name=MODEL_NAME, safety_settings=safety_settings)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatModel.from_pretrained("chat-bison@002")

# Define prompts
system_context = "You are an automotive assistant at an automobile company.\
Your task is to give detailed answers to queries in a few sentences \
based on the CONTEXT provided. The CONTEXT can include text, table summaries\
or image summaries. If the QUESTION is not related to automobiles or you cannot find an answer in the CONTEXT, you MUST \
respond with: SORRY I COULD NOT FIND ANYTHING RELEVANT. If it is the FIRST QUERY without a car name or If you dont know the name of the car, you MUST ask for it and remember the car name.\
If the QUESTION is very vague or just a keyword, you MUST ask a follow-up question to get more clarity. If the QUERY is asking you to show an image, you need to summarize the available CONTEXT related to the query"

system_image_context = "You are an automotive assistant at an automobile company.\
You are given with a summary of an image, summarized using an LLM. You are also given\
with CONTEXT of similar images in the manual. Your task is to frame a QUERY for retrieving \
relevant documents from the manual related to the image. DO not comment on the image."

system_image_reply="You are an automotive assistant at an automobile company.\
based on the provided CONTEXT, Your task is to give detailed answers to queries in a few sentences \
based on the CONTEXT provided. The CONTEXT can include text, table summaries\
or image summaries.Do not comment on the image."

def add_to_vectorstore(texts, text_sum, tables, tables_sum, image_paths, images_sum, car_name="UNKNOWN"):
    '''
        Processes texts, tables, and images along with their summaries,
        converts these summaries into vector embeddings, and then upserts them
        into Pinecone VectorStore.

        Parameters:
        texts (list): Raw text data.
        text_sum (list): Summaries of the texts.
        tables (list): Raw table data.
        tables_sum (list): Summaries of the tables.
        image_paths (list): Paths to image files.
        images_sum (list): Summaries of the images.

        Returns:
        None
    '''
    vectors = []

    # Embed and store text summaries
    for text, summary in zip(texts, text_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'text_summary', 'content': summary, 'raw_text': text, 'car_name': car_name}))

    # Embed and store table summaries
    for table, summary in zip(tables, tables_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'table_summary', 'content': summary, 'raw_table': table, 'car_name': car_name}))

    # Embed and store image summaries and image embeddings
    for image_path, summary in zip(image_paths, images_sum):
        summary_vector = embeddings.embed_documents([summary])[0]
        vectors.append((str(uuid.uuid4()), summary_vector, {'type': 'image_summary', 'content': summary, 'filepath': image_path, 'car_name': car_name}))

    # Upsert to pinecone in batches
    max_batch_size = 100
    for i in range(0, len(vectors), max_batch_size):
        batch = vectors[i:i + max_batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting vectors: {e}")

def get_response(message, context, history):
    '''
        Generates a response based on the input message, context, and chat history.

        Parameters:
        message (str): The user's message.
        context (str): The context for the conversation.
        history (list): List of previous chat messages with their roles.

        Returns:
        str: The generated response by LLM.
    '''
    chat_history=[]
    for h in history:
        chat_history.append(ChatMessage(h["content"], h["role"]))
    
    chat = ChatSession(model=chat_model, context=context, message_history=chat_history)
    response = chat.send_message(message)
    return response

def handle_image_query(image_path="", cloud_url=""):
    '''
        Processes an image query, generates a summary, retrieves relevant information, and generates a response.

        Parameters:
        image_path (str): Path to the image file.
        cloud_url (str): cloudinary secure URL of the image.

        Returns:
        str: The response generated by LLM.
    '''
    # Summarize the image
    summary = summarize_image(model, image_path)
    st.write("Image summary:")
    st.write(summary)

    # Get a query based on the image
    output_text = get_response(summary, system_image_context, st.session_state.chat_history)
    output_text = output_text.text

    # Embed the query
    query_vector = embeddings.embed_query(text=output_text)

    if isinstance(query_vector, list):
        query_vector = [float(val) for val in query_vector]
    else:
        query_vector = list(query_vector)

    # Find relevant documents
    docs = index.query(vector=query_vector, top_k=4, include_metadata=True).matches
    if not docs:
        return "Sorry, I couldn't find anything relevant in the knowledge base."
    
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

    context = "CONTEXT : \n Text:\n" + "\n".join(text_data) + "\n\n" + \
              "Tables:\n" + "\n".join(table_data) + "\n\n" + \
              "Image summaries:\n" + "\n".join(image_summaries[:2])
    query_image_html = f'<img src="{cloud_url}" id="res_img"/>'

    question = "QUERY: The summary of the input image is : \n" + summary + "\n" + "Based on this, and the CONTEXT, summarize the context."
    response = get_response(question, system_image_reply+context , st.session_state.chat_history)
    response = response.text

    # Update history
    st.session_state.chat_history.append({"role": "user", "content": summary, "images": query_image_html})
    st.session_state.chat_history.append({"role": "assistant", "content": response, "images": ""})

    return response

async def handle_text_query(question):
    '''
        Processes a text query, retrieves relevant information from the vector store, and generates a response.

        Parameters:
        question (str): The user's query.

        Returns:
        str: The response generated by the LLM.
        str: HTML string for displaying related images.
    '''
    if len(question) == 0:
        return "Please enter a valid prompt", ""
    
    # Embed query
    query_vector = embeddings.embed_query(text=question)

    if isinstance(query_vector, list):
        query_vector = [float(val) for val in query_vector]
    else:
        query_vector = list(query_vector)

    # Search relevant documents
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
        car_info = f"\nThe below info is related to car {metadata['car_name']}: \n"
        if 'raw_text' in metadata:
            text_data.append(car_info+metadata['raw_text'])
        elif 'raw_table' in metadata:
            table_data.append(car_info+metadata['raw_table'])
        elif 'filepath' in metadata:
            files_to_display.append(metadata['filepath'])
            image_summaries.append(car_info+metadata['content'])

    context = "CONTEXT : \n TEXT:\n" + "\n".join(text_data) + "\n\n" + \
              "TABLES:\n" + "\n".join(table_data) + "\n\n" + \
              "IMAGE SUMMARIES:\n" + "\n".join(image_summaries[:2])

    images_html = "".join([f'<img src="{img}" id="res_img"/>' for img in files_to_display[:2]])

    output_text = get_response(question, system_context+context, st.session_state.chat_history)
    output_text = output_text.text

    # Update history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": question, "images": ""})
    st.session_state.chat_history.append({"role": "assistant", "content": output_text, "images": images_html})

    return output_text, images_html

def main():
    # Configure streamlit UI
    st.set_page_config(page_title="Cargo", page_icon=":ship:", layout="wide")
    st.image('./assets/captain.png', width=200)
    st.header("Ask the Captain about your files! :male-pilot:")
    st.write(css, unsafe_allow_html=True)
    float_init()

    # Initialize history and processed files
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
            # Process the uploaded files
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
                            car_name = get_car_name(model, texts[:5])
                            st.write("Summarizing data...")
                            text_summ, table_summ = generate_summaries(model, texts, tables)
                            image_summ, image_paths = generate_image_summaries(model, folder_id)
                            st.write(f'Extracted {len(image_paths)} images and summarized {len(image_summ)}')
                            status.update(label="Successfully summarized data", state="running", expanded=True)
                            st.write(f"Summarized data in : {time.time()-start_time:.2f} s")

                            # Convert to embeddings and store in pinecone
                            st.write("Converting text to embeddings...")
                            add_to_vectorstore(texts, text_summ, tables, table_summ, image_paths, image_summ, car_name=car_name)
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
                        st.write(user_template.replace("{{MSG}}", message["content"].capitalize()).replace("{{IMAGE}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)
            else:
                with container:
                    with st.chat_message(message["role"], avatar="⚓"):
                            st.write(bot_template.replace("{{MSG}}", message["content"].capitalize()).replace("{{IMAGES}}", message["images"] if message["images"] is not None else ""), unsafe_allow_html=True)

        # Text input
        if prompt := st.chat_input("Ask about your files..."):
            if len(st.session_state.processed_files)>=0 :
                # Display user message in chat message container
                with container:
                    with st.chat_message("user", avatar="❓"):
                        st.write(user_template.replace("{{MSG}}", prompt.capitalize()).replace("{{IMAGE}}", ""), unsafe_allow_html=True)

                with st.spinner("Thinking..."):
                    response, images = asyncio.run(handle_text_query(question=prompt))
                    # Display assistant response in chat message container
                    with container:
                        with st.chat_message("assistant", avatar="⚓"):
                            bot_response = bot_template.replace("{{MSG}}", response.capitalize()).replace("{{IMAGES}}", images)
                            st.write(bot_response, unsafe_allow_html=True)
            else:
                st.write("Please upload some documents to chat with them!")

    with col2:
        container1 = st.container()
        with container1:
            # Image input
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

                    # Update UI
                    with container:
                        with st.chat_message("user", avatar="❓"):
                            st.write(user_template.replace("{{MSG}}", "").replace("{{IMAGE}}", image_query_html), unsafe_allow_html=True)

                    with st.spinner("Thinking..."):
                        response = handle_image_query(image_path=query_img_path, cloud_url=image_query_url)

                        with container:
                            with st.chat_message("assistant", avatar="⚓"):
                                bot_response = bot_template.replace("{{MSG}}", response.capitalize()).replace("{{IMAGES}}", "")
                                st.write(bot_response, unsafe_allow_html=True)

if __name__ == '__main__':
    main()