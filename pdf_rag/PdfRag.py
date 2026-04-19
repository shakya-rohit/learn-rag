from openai import OpenAI
import numpy as np
from pypdf import PdfReader

# -------------------------------
# 1. Init OpenAI client
# -------------------------------
client = OpenAI()

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
        end = start + chunk_size
        chunk = text[start:end]
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
    return response.data[0].embedding


# -------------------------------
# 6. Cosine similarity
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# 7. Build vector store
# -------------------------------
def build_vector_store(chunks):
    print("🔄 Creating embeddings for chunks...")
    embeddings = [get_embedding(chunk) for chunk in chunks]
    print("✅ Embeddings ready!\n")
    return embeddings


# -------------------------------
# 8. Retrieve top-k
# -------------------------------
def find_top_k(query, chunks, embeddings, k=3):
    query_embedding = get_embedding(query)

    scores = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    top_k_indices = np.argsort(scores)[-k:][::-1]

    return [chunks[i] for i in top_k_indices]


# -------------------------------
# 9. RAG pipeline
# -------------------------------
def ask_rag(question, chunks, embeddings):
    context_docs = find_top_k(question, chunks, embeddings, k=3)

    context_text = "\n\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

If the question is general (like greetings), answer normally.

Otherwise, answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context_text}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=200
    )

    return response.output_text


# -------------------------------
# 10. Main
# -------------------------------
if __name__ == "__main__":
    print("📄 Loading PDF...")
    raw_text = load_pdf("sample.pdf")

    print("🧹 Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("✂️ Chunking text...")
    chunks = chunk_text(cleaned_text)

    print(f"📊 Total chunks: {len(chunks)}\n")

    embeddings = build_vector_store(chunks)

    print("🤖 PDF RAG System Ready!")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Ask: ")

        if query.lower() == "exit":
            break

        answer = ask_rag(query, chunks, embeddings)
        print(f"💡 Answer: {answer}\n")