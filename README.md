> **Note**
> This project is a part of BGSW (Bosch) GenAI Hackathon.

# Cargo

Cargo is an RAG designed to allow users to chat with multiple large PDF documents with ease. This robust platform enables users to interact seamlessly with their PDF files that include images, text, and tables, through an AI-driven chatbot named "Captain."

## Features

- Upload multiple PDF files.
- Chat Engine integration.
- Analyses texts, tables and images in a PDF document.
- LLM integration for intelligent responses.

## Tools used
<p align="left">
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/492cc671-e6c9-494b-ba2b-296f7c1bad2a" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/f4e6f979-baa6-498c-9d49-b26da8e53cf8" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/1ccc427b-a6bb-45e6-a6b1-ac555b025e94" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/2fb73136-6d50-423f-b4a0-1962b8e6914b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</p>

- **Langchain**: simplifies the integration of language models into applications, facilitating complex natural language processing tasks.
- **Pinecone**:  offers a vector database that allows efficient similarity search and retrieval for high-dimensional data.
- **Vertex AI**: provides robust AI capabilities, enabling the creation and deployment of machine learning models at scale.
- **Streamlit**: allows for the rapid development of interactive web applications with minimal coding effort.

## Installation steps

1. - Fork the [repo](https://github.com/codedmachine111/cargo)
   - Clone the repo to your local machine `git clone https://github.com/codedmachine111/cargo.git`
   - Change current directory `cd cargo`
2. Install latest version of [Python](https://www.python.org/) and install all the dependencies using:

```bash
pip install -r requirements.txt
```
3. Create a .env file in the root directory of the project and add:

```bash
PINECONE_API_KEY = "YOUR-API-KEY"
PINECONE_HOST = "YOUR-API-KEY"
GOOGLE_API_KEY = "YOUR-API-KEY"
```
> **Note**
> You need to get your Google API key from ![here](https://aistudio.google.com/)
> and your Pinecone API key from ![here](https://www.pinecone.io/)

4. Start the app:

```bash
streamlit run main.py
```
