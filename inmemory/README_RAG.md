# 🧠 In-Memory RAG System (Python + OpenAI)

This project demonstrates a **basic Retrieval-Augmented Generation
(RAG)** system built from scratch using Python and OpenAI APIs.

------------------------------------------------------------------------

## 🚀 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique where: 1. Retrieve
relevant data 2. Pass it to LLM 3. Generate grounded answers

------------------------------------------------------------------------

## 🏗️ Architecture

User Query → Embedding → Similarity Search → Context + Query → LLM →
Answer

------------------------------------------------------------------------

## 📦 Features

-   In-memory storage\
-   OpenAI Embeddings\
-   Cosine similarity\
-   Top-K retrieval\
-   CLI interface

------------------------------------------------------------------------

## ⚙️ Setup

### Install

    pip install openai numpy scikit-learn

### Set API Key

    setx OPENAI_API_KEY "your_key"

### Run

    python rag_in_memory.py

------------------------------------------------------------------------

## 🧪 Example

    Ask: what is rag
    Answer: RAG stands for Retrieval Augmented Generation.

------------------------------------------------------------------------

## ⚠️ Limitations

-   No PDF support\
-   No chunking\
-   Not scalable

------------------------------------------------------------------------

## 🔮 Future Work

-   Add PDF support\
-   Add FAISS\
-   Add UI

------------------------------------------------------------------------

## 🎯 Use Cases

-   Chatbots\
-   Document Q&A\
-   Learning systems

------------------------------------------------------------------------
