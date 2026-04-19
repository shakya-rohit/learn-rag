from openai import OpenAI
import numpy as np
from pypdf import PdfReader
import faiss
import pickle
import os

# -------------------------------
# 1. Init OpenAI client
# -------------------------------
client = OpenAI()

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"


# -------------------------------
# 2. Load PDF
# -------------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


# -------------------------------
# 3. Clean text
# -------------------------------
def clean_text(text):
    return " ".join(text.split())


# -------------------------------
# 4. Chunk text
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -------------------------------
# 5. Get embedding
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------------
# 6. Build FAISS index
# -------------------------------
def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))
    return index


# -------------------------------
# 7. Save index + chunks
# -------------------------------
def save_data(index, chunks):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)


# -------------------------------
# 8. Load index + chunks
# -------------------------------
def load_data():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# -------------------------------
# 9. Create or load system
# -------------------------------
def initialize():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("⚡ Loading existing index...")
        return load_data()

    print("📄 Processing PDF...")

    PDF_PATH = "uploaded.pdf" if os.path.exists("uploaded.pdf") else "sample.pdf"
    text = load_pdf(PDF_PATH)
    
    text = clean_text(text)

    chunks = chunk_text(text)
    print(f"📊 Total chunks: {len(chunks)}")

    print("🔄 Creating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]

    print("⚡ Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("💾 Saving index...")
    save_data(index, chunks)

    print("✅ Setup complete!\n")

    return index, chunks


# -------------------------------
# 10. Retrieve top-k
# -------------------------------
def retrieve(query, index, chunks, k=3):
    query_embedding = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0]]


# -------------------------------
# 11. RAG pipeline
# -------------------------------
def ask_rag(query, index, chunks):
    context_docs = retrieve(query, index, chunks)
    context = "\n\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

If the question is general (like greetings), answer normally.

Otherwise, answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=200
    )

    return response.output_text


# -------------------------------
# 12. Main loop
# -------------------------------
if __name__ == "__main__":
    index, chunks = initialize()

    print("🤖 FAISS RAG System Ready!")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Ask: ")

        if query.lower() == "exit":
            break

        answer = ask_rag(query, index, chunks)
        print(f"💡 Answer: {answer}\n")