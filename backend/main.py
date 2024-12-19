from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import aiofiles
from typing import Optional
import uuid
from datetime import datetime

# Import our RAG pipeline
from rag_pipeline import RAGPipeline

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store RAG instances for different sessions
rag_instances = {}

class ChatQuery(BaseModel):
    session_id: str
    question: str

class ProcessingStatus(BaseModel):
    session_id: str
    status: str
    error: Optional[str] = None

# Directory for uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_document(file_path: str, session_id: str):
    """Background task to process the document"""
    try:
        rag = RAGPipeline()
        rag.setup_models()
        rag.load_pdf(file_path)
        rag_instances[session_id] = rag
    except Exception as e:
        rag_instances[session_id] = {"error": str(e)}

@app.post("/upload", response_model=ProcessingStatus)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Endpoint to upload and process a document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save uploaded file
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Start background processing
    background_tasks.add_task(process_document, file_path, session_id)
    
    return ProcessingStatus(
        session_id=session_id,
        status="processing"
    )

@app.post("/chat", response_model=dict)
async def chat(query: ChatQuery):
    """Endpoint to chat with the processed document"""
    if query.session_id not in rag_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    rag_instance = rag_instances[query.session_id]
    if isinstance(rag_instance, dict) and "error" in rag_instance:
        raise HTTPException(status_code=500, detail=rag_instance["error"])
    
    try:
        result = rag_instance.answer_question(query.question)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{session_id}", response_model=ProcessingStatus)
async def get_status(session_id: str):
    """Endpoint to check document processing status"""
    if session_id not in rag_instances:
        return ProcessingStatus(session_id=session_id, status="processing")
    
    rag_instance = rag_instances[session_id]
    if isinstance(rag_instance, dict) and "error" in rag_instance:
        return ProcessingStatus(
            session_id=session_id,
            status="error",
            error=rag_instance["error"]
        )
    
    return ProcessingStatus(session_id=session_id, status="ready")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)