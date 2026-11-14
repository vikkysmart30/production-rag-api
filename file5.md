# Production RAG Microservice (FastAPI + ChromaDB + Docker)

A containerized, production-ready Retrieval-Augmented Generation (RAG) REST API designed for low-latency enterprise document Q&A.

## Architecture & Features
- **API Layer:** Asynchronous REST endpoints built with **FastAPI** and strict **Pydantic** schema validation.
- **Vector Store:** Persistent semantic search using **ChromaDB**.
- **LLM Inference:** Grounded response generation powered by **Gemini 2.5 Flash** with strict anti-hallucination prompting.
- **Observability:** Built-in latency tracking (`latency_ms`), structured logging, and `/health` readiness probes.
- **Deployment:** Fully containerized via **Docker** for seamless deployment to AWS ECS, GCP Cloud Run, or Kubernetes.

## Quick Start

### 1. Local Setup
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
uvicorn main:app --reload