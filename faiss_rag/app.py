from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_faiss import initialize, ask_rag


import shutil
import os

UPLOAD_PATH = "uploaded.pdf"

# -------------------------------
# 1. Init FastAPI
# -------------------------------
app = FastAPI(title="RAG API")

# -------------------------------
# 2. Load RAG system (once)
# -------------------------------
print("🚀 Initializing RAG system...")
index, chunks = initialize()
print("✅ RAG system ready!\n")


# -------------------------------
# 3. Request schema
# -------------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------------
# 4. Health check
# -------------------------------
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# -------------------------------
# 5. Main RAG endpoint
# -------------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = ask_rag(request.question, index, chunks)
    return {
        "question": request.question,
        "answer": answer
    }

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Reinitialize RAG with new PDF
    global index, chunks
    index, chunks = initialize()

    return {"message": "PDF uploaded and processed successfully ✅"}