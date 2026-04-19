from openai import OpenAI
import numpy as np

# -------------------------------
# 1. Init OpenAI client
# -------------------------------
client = OpenAI()

# -------------------------------
# 2. Your knowledge base
# -------------------------------
documents = [
    "RAG stands for Retrieval Augmented Generation.",
    "It combines retrieval with language models.",
    "Embeddings convert text into vectors.",
    "Vector databases help find similar content.",
    "Chunking is the process of splitting large documents into smaller parts.",
    "Cosine similarity is used to measure similarity between vectors."
]

# -------------------------------
# 3. Get embedding
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# -------------------------------
# 4. Precompute document embeddings
# -------------------------------
print("🔄 Creating embeddings for documents...")
doc_embeddings = [get_embedding(doc) for doc in documents]
print("✅ Embeddings ready!\n")


# -------------------------------
# 5. Cosine similarity
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# 6. Retrieve top-k relevant docs
# -------------------------------
def find_top_k(query, k=2):
    query_embedding = get_embedding(query)

    scores = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    top_k_indices = np.argsort(scores)[-k:][::-1]

    return [documents[i] for i in top_k_indices]


# -------------------------------
# 7. RAG pipeline
# -------------------------------
def ask_rag(question):
    context_docs = find_top_k(question, k=2)

    context_text = "\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context_text}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=150
    )

    return response.output_text


# -------------------------------
# 8. Run interactive loop
# -------------------------------
if __name__ == "__main__":
    print("🤖 In-Memory RAG System Ready!")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Ask: ")

        if query.lower() == "exit":
            break

        answer = ask_rag(query)
        print(f"💡 Answer: {answer}\n")