> **Note**
> This project is a part of BGSW (Bosch) GenAI Hackathon.
<p align="center">

  <img src="https://github.com/codedmachine111/cargo/assets/88738817/3fd48a95-42b1-42c9-bf8e-fcac29bce4c6" alt="cargo-banner" width="120">
  <img src="https://github.com/codedmachine111/cargo/assets/88738817/26f0b566-4950-404a-8757-b711d00fa97c" alt="cargo-banner" width="400">

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
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/e8e4d829-1c2b-4545-a719-093fecb9ac9d" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/2fb73136-6d50-423f-b4a0-1962b8e6914b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

- **Langchain**: simplifies the integration of language models into applications, facilitating complex natural language processing tasks.
- **Pinecone**:  offers a vector database that allows efficient similarity search and retrieval for high-dimensional data.
- **Vertex AI**: Machine learning platform by Google that gives access to Large Language Models.
- **Google AI studio**: platform that gives access to a text embedding model and Gemini.
- **Unstructured**:  Core library for partitioning, cleaning, and chunking documents types for LLM applications.
- **Streamlit**: allows for the rapid development of interactive web applications with minimal coding effort.

## Architecture
<img src="https://github.com/codedmachine111/cargo/assets/88738817/fe8bb9ee-6571-46f7-bf02-4b066abdd6c4" width="900px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## Installation steps

1. - Fork the [repo](https://github.com/codedmachine111/cargo)
   - Clone the repo to your local machine `git clone https://github.com/codedmachine111/cargo.git`
   - Change current directory `cd cargo`
2. Install latest version of [Python](https://www.python.org/) and create a virtual environment:
```bash
python -m venv venv
./venv/Scripts/activate
```

3. Google Cloud Platform setup:
   - Login to [Google Cloud Platform](https://cloud.google.com) and create a new project.
   - Go to the project dashboard.
   - Navigate to IAM & Admin > Service Accounts.
   - Click Create Service Account.
   - Grant the necessary permissions to this service account (e.g., Vertex AI User).
   - Click on your newly created service account.
   - Create a new key (JSON), rename it to `secret.json` and copy to the root directory of project.

4. Create a .env file in the root directory of the project and add:

```bash
PINECONE_API_KEY= "YOUR-API-KEY"
GOOGLE_API_KEY= "YOUR-API-KEY"
PROJECT_ID="YOUR-GOOGLE-CLOUD-PROJECT-ID"
```
> **Note**
> You need to get your Google API key from [here](https://aistudio.google.com/)
> and your Pinecone API key from [here](https://www.pinecone.io/)

5. Install Tesseract OCR on your machine.\
**For Windows:**
- Download tesseract exe from [here](https://github.com/UB-Mannheim/tesseract/wiki).
- Install the `.exe` file in `C:\Program Files\Tesseract-OCR`.
- Add a new path to the system environment variables:
  - On the Windows search bar, search for “Environment Variables.” You will find “Edit the System Variable.”
  - Next, in the “System Properties” window, click on the “Environment Variables” button.
  - Under “System variables,” find the “Path” variable, select it, and click the “Edit” button.
  - Click the “New” button and add the path to the Tesseract installation directory: `C:\Program Files\Tesseract-OCR.`
  - Then, click “OK” to save the changes.

> **Note**
> 
> Follow this [guide](https://builtin.com/articles/python-tesseract) if you are on Mac or Linux.


6. Install all dependencies:
```bash
pip install -r requirements.txt
```

7. Start the app:

```bash
streamlit run main.py
```
