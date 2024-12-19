from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Any
from threading import Thread
import os

class RAGPipeline:
    def __init__(
        self, 
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retriever_k: int = 3
    ):
        """Initialize the RAG pipeline with configurable parameters."""
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.vector_store = None

    def setup_models(self) -> None:
        """Set up all necessary models."""
        print("Setting up models...")
        try:
            # Set up tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
                
            # Set up model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Setup embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("Models setup complete!")
        except Exception as e:
            raise RuntimeError(f"Error setting up models: {str(e)}")

    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a string."""
        try:
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            raise RuntimeError(f"Error formatting documents: {str(e)}")

    def load_pdf(self, pdf_path: str) -> None:
        """Load and process a PDF document."""
        print(f"Loading PDF from {pdf_path}...")
        try:
            # Validate PDF path
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                raise ValueError("No text extracted from PDF")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            print(f"Processed {len(texts)} text chunks from PDF")
            
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")

    def _extract_assistant_response(self, full_response: str) -> str:
        """Extract just the assistant's response from the full response."""
        try:
            # Split on the assistant marker and take everything after it
            if "<|assistant|>" in full_response:
                return full_response.split("<|assistant|>")[-1].strip()
            return full_response.strip()
        except Exception as e:
            raise RuntimeError(f"Error extracting assistant response: {str(e)}")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        if not self.model or not self.tokenizer or not self.vector_store:
            return {"error": "Models not initialized. Please set up the pipeline first!"}
            
        try:
            # Get relevant documents
            sources = self.vector_store.similarity_search(question, k=self.retriever_k)
            context = self._format_docs(sources)
            
            # Create prompt
            template = """<|system|>You are a helpful assistant. Answer the question based on the following context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use markdown formatting when appropriate for:
- Lists and bullet points
- Code blocks using triple backticks
- Emphasis using * or **
- Headers using #
- Tables when presenting structured data

Context: {context}

<|user|>{question}<|assistant|>"""
            
            # Format prompt
            prompt = template.format(context=context, question=question)
            
            # Generate answer
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            # Decode and extract assistant's response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._extract_assistant_response(full_response)
            
            # Prepare source metadata
            sources_with_metadata = [
                {
                    'content': doc.page_content,
                    'page': doc.metadata.get('page', 'Unknown page'),
                    'score': doc.metadata.get('score', 'N/A')
                } for doc in sources
            ]
            
            return {
                'answer': answer,
                'sources': sources_with_metadata
            }
                
        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}"}