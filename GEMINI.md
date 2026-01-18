# GEMINI.md - NotebookLM Clone Project Overview

This document provides a comprehensive overview of the NotebookLM Clone project, intended for developers.

## 1. Project Overview

This project is a Python-based, open-source implementation of Google's NotebookLM. It's a web application that allows users to upload documents (PDFs, text), and then generate a two-person podcast conversation based on the document's content. The application is built with Streamlit and features a user interface inspired by NotebookLM. The core functionality revolves around grounding AI responses in user-provided documents, ensuring that the generated content is verifiable and includes citations.

The application supports:
- Uploading and processing of PDF and text files.
- Generation of AI-powered podcasts from the documents.
- A clean and intuitive web interface.

## 2. Architecture

The application follows a modular architecture, with distinct components for different stages of the data processing and content generation pipeline. The main components are:

1.  **Document Ingestion & Processing**: Users upload documents through the Streamlit web interface. These documents are then processed to extract text content.
2.  **Embedding Generation**: The extracted text is converted into vector embeddings.
3.  **Vector Storage**: The embeddings are stored in a Milvus vector database for efficient semantic search and retrieval.
4.  **Podcast Script Generation**: When a user requests a podcast, relevant text chunks are retrieved from the vector database. This content is then used to generate a podcast script with two speakers.
5.  **Text-to-Speech (TTS)**: The generated script is converted into audio using a TTS model.
6.  **Web Interface**: The entire application is wrapped in a Streamlit web interface, which provides a user-friendly way to interact with the system.

## 3. Key Components

The project is structured into several key Python modules within the `src` directory:

-   **`app.py`**: The main entry point of the application. It's a Streamlit application that handles the user interface, file uploads, and orchestrates the entire workflow.

-   **`src/document_processing/doc_processor.py`**: This module is responsible for processing uploaded documents. It uses libraries like `PyMuPDF` to extract text from PDF files.

-   **`src/embeddings/embedding_generator.py`**: This module generates vector embeddings from the text content extracted by the `doc_processor`.

-   **`src/vector_database/milvus_vector_db.py`**: This module manages the vector database. It uses `Milvus` to store and retrieve document embeddings, which is crucial for finding relevant content for podcast generation.

-   **`src/podcast/script_generator.py`**: This module takes the retrieved text chunks and uses a large language model (via the OpenAI API) to generate a podcast script. The script is designed to be a conversation between two speakers.

-   **`src/podcast/text_to_speech.py`**: This module uses a Text-to-Speech (TTS) model (`Kokoro`) to convert the generated podcast script into an audio file.

## 4. Dependencies

The project relies on a number of open-source libraries. The key dependencies are listed in `pyproject.toml` and `requirements.txt`.

**Core Dependencies:**

-   **`streamlit`**: For building the interactive web application.
-   **`pymupdf`**: For parsing PDF documents.
-   **`pymilvus[milvus-lite]`**: For the vector database.
-   **`fastembed`**: For creating text embeddings.
-p   **`openai`**: For generating the podcast script.
-   **`kokoro`**: For Text-to-Speech functionality.
-   **`python-dotenv`**: For managing environment variables.
-   **`torch`**, **`transformers`**: Required by the TTS and embedding models.

## 5. How to Run the Application

To run the application, follow these steps:

1.  **Install Dependencies**: Make sure you have Python 3.11 installed. It is recommended to use `uv` to install the dependencies as described in the `README.md` file.

    ```bash
    # Install dependencies
    uv sync
    ```

2.  **Set Up Environment Variables**: Create a `.env` file in the root of the project and add your API keys as specified in the `.env.example` file. You will need keys for OpenAI, AssemblyAI, Firecrawl, and Zep.

3.  **Run the Application**:

    ```bash
    streamlit run app.py
    ```

    The application will be available at `http://localhost:8501`.
