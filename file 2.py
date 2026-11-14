import os
import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Production Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production RAG Microservice",
    description="Deployed AI Engineer Portfolio Project: FastAPI + ChromaDB + Gemini",
    version="1.0.0"
)

# Initialize Vector DB (Persistent local storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment variables!")
client = genai.Client(api_key=api_key)

# Request/Response Schemas (Data Validation)
class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Document text to store in vector DB")
    source: str = Field(default="manual_upload", description="Document source metadata")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=2, ge=1, le=5, description="Number of context chunks to retrieve")

@app.get("/health")
def health_check():
    """Kubernetes / Cloud Run health check endpoint"""
    return {"status": "healthy", "vector_db_count": collection.count()}

@app.post("/ingest")
def ingest_document(req: IngestRequest):
    """Chunks, embeds, and stores text into ChromaDB"""
    try:
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[req.text],
            metadatas=[{"source": req.source}],
            ids=[doc_id]
        )
        logger.info(f"Ingested document ID: {doc_id}")
        return {"status": "success", "doc_id": doc_id, "message": "Document embedded and stored."}
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(req: QueryRequest):
    """Retrieves relevant context and generates grounded LLM response"""
    start_time = time.time()
    try:
        # 1. Semantic Search in Vector DB
        results = collection.query(query_texts=[req.question], n_results=req.top_k)
        retrieved_docs = results["documents"][0] if results["documents"] else []
        
        if not retrieved_docs:
            return {"answer": "No relevant context found in the knowledge base.", "latency_ms": 0}

        context = "\n---\n".join(retrieved_docs)
        
        # 2. Grounded Prompt Construction
        prompt = f"""You are a precise enterprise AI assistant. Answer the question strictly using ONLY the provided context. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {req.question}
Answer:"""

        # 3. LLM Generation
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Query processed in {latency_ms}ms")

        return {
            "question": req.question,
            "answer": response.text,
            "sources_used": len(retrieved_docs),
            "latency_ms": latency_ms
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))