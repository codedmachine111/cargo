> **Note**
> This project is a part of BGSW (Bosch) GenAI Hackathon.
<p align="center">
  <img src="https://github.com/codedmachine111/cargo/assets/88738817/0781bd08-bb16-4c40-ba3d-dc0f0e6110d8" alt="cargo-banner" width="100">
  <img src="https://github.com/codedmachine111/cargo/assets/88738817/004ad416-7e90-4de4-889e-b06a175b00fc" alt="cargo-banner" width="300">
</p>

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
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/350c8f12-ff51-4a76-8ab5-c05daba96c1d" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/bc76a436-7900-4a65-ade3-fb22b70cf08b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/ef687319-cf5c-49d0-9a95-cbd209c0a95b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/2fb73136-6d50-423f-b4a0-1962b8e6914b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</p>

- **Langchain**: simplifies the integration of language models into applications, facilitating complex natural language processing tasks.
- **Pinecone**:  offers a vector database that allows efficient similarity search and retrieval for high-dimensional data.
- **Vertex AI**: Machine learning platform by Google that gives access to Large Language Models.
- **Google AI studio**: platform that gives access to a text embedding model and Gemini.
- **Streamlit**: allows for the rapid development of interactive web applications with minimal coding effort.

## Architecture
<img src="https://github.com/codedmachine111/cargo/assets/88738817/3b1e1be4-cd8d-4763-b4e4-0cd5c2c2fecf" width="900px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## Installation steps

1. - Fork the [repo](https://github.com/codedmachine111/cargo)
   - Clone the repo to your local machine `git clone https://github.com/codedmachine111/cargo.git`
   - Change current directory `cd cargo`
2. Install latest version of [Python](https://www.python.org/) and create a virtual environment:
```bash
python -m venv venv
./venv/Scripts/activate
```

3. Install all dependencies:
```bash
pip install -r requirements.txt
```


4. Create a .env file in the root directory of the project and add:

```bash
PINECONE_API_KEY = "YOUR-API-KEY"
PINECONE_HOST = "YOUR-API-KEY"
GOOGLE_API_KEY = "YOUR-API-KEY"
```
> **Note**
> You need to get your Google API key from ![here](https://aistudio.google.com/)
> and your Pinecone API key from ![here](https://www.pinecone.io/)

5. Google Cloud Platform setup:
   - Login to ![Google Cloud Platform](https://cloud.google.com) and create a new project.
   - Navigate to IAM & Admin > Service Accounts.
   - Click Create Service Account.
   - Grant the necessary permissions to this service account (e.g., Vertex AI User).
   - Create a new key (JSON), rename it to `secret.json` and copy to the root directory.

7. Start the app:

```bash
streamlit run main.py
```
