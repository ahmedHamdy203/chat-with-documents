# RAG Chat Interface

A sophisticated chatbot interface that employs Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware responses to questions about PDF documents. This application combines modern NLP techniques with an intuitive web interface to make document interaction more natural and efficient.

## Understanding the System

Our application consists of three main architectural components that work together seamlessly:

1. A responsive web-based frontend that provides an intuitive chat interface
2. A robust FastAPI backend that handles document processing and chat functionality
3. An advanced RAG pipeline powered by TinyLlama and FAISS for intelligent question answering

The system processes uploaded PDF documents by breaking them into meaningful chunks, converting these chunks into vector representations, and storing them in a FAISS index. When you ask a question, the system retrieves the most relevant document sections and uses them to generate accurate, contextual responses.

## System Requirements

Before installing the application, ensure your system meets these requirements:

- Python 3.10 or higher (3.10 recommended for optimal compatibility)
- Minimum 4GB of RAM (8GB recommended for better performance)
- At least 2GB of free disk space for model storage
- A modern web browser (Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+)

## Installation Guide

You can install and run this application using either Docker (recommended) or local installation.

### Docker Installation

Docker provides the most straightforward and consistent deployment experience. Our Docker configuration encapsulates all dependencies and services in a single container.

1. First, ensure Docker is installed on your system:
   ```bash
   docker --version
   ```
   If Docker isn't installed, download and install it from the [official Docker website](https://www.docker.com/get-started).

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chat-interface
   ```

3. Build the Docker image:
   ```bash
   docker build -t rag-chat-app .
   ```

4. Run the container:
   ```bash
   docker run -p 8000:8000 -p 8080:8080 -v $(pwd)/backend/uploads:/app/backend/uploads rag-chat-app
   ```

The application will now be available at:
- Frontend interface: http://localhost:8080
- API documentation: http://localhost:8000/docs

### Local Installation

If you prefer a local installation for development or customization:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chat-interface
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:
   ```bash
   python launcher.py
   ```

## Using the Application

Once the application is running, you can interact with it through a simple workflow:

1. Document Upload
   - Navigate to http://localhost:8080 in your web browser
   - Click the "Choose PDF" button and select your document
   - Wait for the processing indicator to show completion

2. Asking Questions
   - Type your question in the chat input
   - The system will analyze your document and provide relevant answers
   - Each answer includes citations to the source material

3. Understanding Responses
   - Responses are formatted in Markdown for readability
   - Source attributions show where the information came from
   - Confidence scores indicate the reliability of the response

## API Documentation

Our REST API provides these core endpoints:

### POST /upload
Handles PDF document upload and processing.
```json
Request: Multipart form data with 'file' field (PDF only)
Response: {
    "session_id": "string",
    "status": "processing"
}
```

### GET /status/{session_id}
Checks document processing status.
```json
Response: {
    "session_id": "string",
    "status": "string",
    "error": "string" (optional)
}
```

### POST /chat
Processes questions and generates responses.
```json
Request: {
    "session_id": "string",
    "question": "string"
}
Response: {
    "answer": "string",
    "sources": [{
        "content": "string",
        "page": "number",
        "score": "number"
    }]
}
```

## Configuration Options

The RAG pipeline can be customized through several parameters in `rag_pipeline.py`:

- `model_name`: Choose the language model (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- `embedding_model`: Select the embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `chunk_size`: Adjust text chunk size (default: 500)
- `chunk_overlap`: Set overlap between chunks (default: 50)
- `retriever_k`: Configure number of retrieved documents (default: 3)

## Troubleshooting Common Issues

1. Memory Errors
   - Reduce `chunk_size` in the RAG pipeline
   - Close unnecessary applications
   - Increase Docker container memory limit

2. Slow Processing
   - Check your system meets the recommended requirements
   - Reduce PDF file size if possible
   - Adjust `retriever_k` for faster responses

3. Connection Issues
   - Verify both servers are running
   - Check if ports 8000 and 8080 are available
   - Ensure firewall settings allow connections
