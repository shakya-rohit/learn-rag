from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_faiss import initialize, ask_rag, add_pdf

import shutil
import os

app = FastAPI(title="RAG API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# Load system ONCE
# -------------------------------
print("🚀 Initializing RAG system...")
index, chunks = initialize()
print("✅ RAG ready!\n")


# -------------------------------
# Request schema
# -------------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------------
# Health
# -------------------------------
@app.get("/")
def home():
    return {"message": "RAG API running 🚀"}


# -------------------------------
# Upload (APPEND, NOT RESET)
# -------------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global index, chunks
    index, chunks = add_pdf(file_path, index, chunks)

    return {
        "message": f"{file.filename} added",
        "total_chunks": len(chunks)
    }


# -------------------------------
# Query
# -------------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = ask_rag(request.question, index, chunks)

    return {
        "question": request.question,
        "answer": answer
    }