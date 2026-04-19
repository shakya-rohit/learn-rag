from openai import OpenAI
import pickle
import os
import numpy as np

client = OpenAI()

KG_FILE = "kg.pkl"

# -------------------------------
# Load / Save KG
# -------------------------------
def load_kg():
    if os.path.exists(KG_FILE):
        with open(KG_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_kg(kg):
    with open(KG_FILE, "wb") as f:
        pickle.dump(kg, f)


# -------------------------------
# Get embedding
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------------
# Extract triplets (same as before)
# -------------------------------
def extract_triplets(text):
    prompt = f"""
Extract relationships as triplets:
(subject, relation, object)

Return ONLY in format:
("A","relation","B")

Text:
{text[:1000]}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=200
    )

    raw = response.output_text

    triplets = []

    for line in raw.split("\n"):
        if "(" in line and ")" in line:
            parts = line.strip("()").split(",")
            if len(parts) == 3:
                triplets.append(tuple([p.strip().strip('"') for p in parts]))

    return triplets


# -------------------------------
# Update KG (NOW WITH EMBEDDINGS)
# -------------------------------
def update_kg(chunks):
    kg = load_kg()

    for chunk in chunks:
        triplets = extract_triplets(chunk["text"])

        for s, r, o in triplets:
            text_repr = f"{s} {r} {o}"

            entry = {
                "subject": s,
                "relation": r,
                "object": o,
                "text": text_repr,
                "embedding": get_embedding(text_repr).tolist()
            }

            kg.append(entry)

    # remove duplicates (basic)
    unique = {}
    for item in kg:
        key = item["text"]
        unique[key] = item

    kg = list(unique.values())

    save_kg(kg)


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# EMBEDDING-BASED KG QUERY
# -------------------------------
def query_kg(question, top_k=5):
    kg = load_kg()

    if not kg:
        return []

    query_emb = get_embedding(question)

    scored = []

    for item in kg:
        emb = np.array(item["embedding"])   # ✅ convert back

        score = cosine_similarity(query_emb, emb)

        if item["subject"].lower() in question.lower():
            score += 0.2

        if item["object"].lower() in question.lower():
            score += 0.2

        scored.append((score, item))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [x[1] for x in scored[:top_k]]