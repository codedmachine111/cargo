> **Note:**
> This project is a part of BGSW (Bosch) GenAI Hackathon.  
> [Pitch Deck](https://www.canva.com/design/DAGG08_sI54/P6Aunmfw38Nw9r0wMHHSaw/view?utm_content=DAGG08_sI54&utm_campaign=designshare&utm_medium=link&utm_source=editor)
<p align="center">

  <img src="https://github.com/codedmachine111/cargo/assets/88738817/3672b024-1f5e-45fa-b179-9aa03e4087dd" alt="cargo-banner" width="700">

</p>

# Cargo

Cargo is an AI-powered advanced multimodal RAG chatbot designed to enable automotive companies to interact effectively with their extensive user manuals. This robust platform enables users to interact seamlessly with their PDF files that includes images, text, and tables.

## Features

- Q&A with multiple PDFs.
- Analyses texts, tables and images in given documents.
- LLM integration for intelligent responses.
- Multimodal search and retrieval.
- Search using images.

## Tools used
<p align="left">
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/492cc671-e6c9-494b-ba2b-296f7c1bad2a" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/350c8f12-ff51-4a76-8ab5-c05daba96c1d" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/bc76a436-7900-4a65-ade3-fb22b70cf08b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/ef687319-cf5c-49d0-9a95-cbd209c0a95b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/e8e4d829-1c2b-4545-a719-093fecb9ac9d" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/cargo/assets/88738817/dcbb61a0-27cf-410b-905a-946782d39cea" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://github.com/codedmachine111/abridge/assets/88738817/2fb73136-6d50-423f-b4a0-1962b8e6914b" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

- **Langchain**: simplifies the integration of language models into applications, facilitating complex natural language processing tasks.
- **Pinecone**:  offers a vector database that allows efficient similarity search and retrieval for high-dimensional data.
- **Vertex AI**: Machine learning platform by Google that gives access to Large Language Models.
- **Google AI studio**: platform that gives access to a text embedding model and Gemini.
- **Unstructured**:  Core library for partitioning, cleaning, and chunking documents types for LLM applications.
- **Cloudinary**: provides a secure and comprehensive API for easily uploading media files 
- **Streamlit**: allows for the rapid development of interactive web applications with minimal coding effort.

## Architecture
<img src="https://github.com/codedmachine111/cargo/assets/88738817/418acd24-d7cb-4e07-bdd2-c739a8559ee7" width="900px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

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
CLOUDINARY_NAME="YOUR-CLOUDINARY-CLOUD-NAME"
CLOUDINARY_API_KEY="YOUR-CLOUDINARY-API-KEY"
CLOUDINARY_API_SECRET="YOUR-CLOUDINARY-API-SECRET"
```
> ### **Note**
> You need to get your Google API key from [here](https://aistudio.google.com/)\
> Pinecone API key from [here](https://www.pinecone.io/)\
> Go to [cloudinary](https://cloudinary.com/), create a new account. Navigate to Media Library -> Settings -> API keys to find your credentials. Your Cloud name will be displayed on top left of console.

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
