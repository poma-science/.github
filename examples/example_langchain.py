#!/usr/bin/env python3
"""
POMA Example - LangChain Integration

This example demonstrates how to integrate POMA with LangChain for document processing and retrieval.

Requirements:
  pip install poma-integrations
  pip install langchain-openai chromadb sentence-transformers

See also:
  - example.py - Standalone implementation with simple keyword-based retrieval
  - example_llamaindex.py - Integration with LlamaIndex
"""

import os
import sys
from dotenv import load_dotenv
import sqlite3

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load API keys from environment or .env file
def load_api_keys():
    """Load API keys from .env file or environment variables."""
    # Try to load from the current directory first
    load_dotenv()
    
    required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Set these as environment variables or create a .env file.")
        print("Example .env file:")
        for key in missing_keys:
            print(f"{key}=your_{key.lower()}_here")
        return False
    return True

# Load API keys
api_keys_loaded = load_api_keys()

# Import POMA integration classes for LangChain
from poma_integrations.langchain_poma import Doc2PomaLoader, PomaChunksetSplitter, PomaCheatsheetRetriever


# Configuration for document conversion and chunking using LiteLLM-compatible providers and models.
cfg = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}

# 1️⃣ ingest
# Convert the document to a .poma archive using the Doc2PomaLoader.
loader = Doc2PomaLoader(cfg)

# Process the example PDF file in the same directory
pdf_path = os.path.join(os.path.dirname(__file__), "example.pdf")
print(f"Processing PDF file: {pdf_path}")
docs_md = loader.load(pdf_path)

# Split the document into structured chunksets and raw text chunks using the configured chunking model.
chunksets, raw_chunks = PomaChunksetSplitter(cfg).split_documents(docs_md)


# Store the chunks in a SQLite database (or another key-value store / db) for later retrieval.
con = sqlite3.connect("chunks.db")

def chunk_store(con, chunksets, raw_chunks):
    """
    Store the chunksets in a SQLite database.
    """
    con.executescript(
        "CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, idx INT, depth INT, "
        "content TEXT, PRIMARY KEY(doc_id,idx));"
    )
    doc_id = chunksets[0].metadata["doc_id"]
    con.executemany(
        "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
        [(doc_id, c["chunk_index"], c["depth"], c["content"]) for c in raw_chunks],
    )
    con.commit()

# Store the chunksets and raw chunks in the SQLite database.
chunk_store(con, chunksets, raw_chunks)

# Fetch all chunks for a given document ID from the SQLite database.
def chunk_fetcher(doc_id, con=con):
    """
    Return *all* chunks belonging to the document.
    """
    rows = con.execute(
        "SELECT idx, depth, content FROM chunks WHERE doc_id=? ORDER BY idx",
        (doc_id,),
    ).fetchall()
    return [
        {"chunk_index": r[0], "depth": r[1], "content": r[2]}
        for r in rows
    ]

# Check if API keys are loaded before creating the vector store
if not api_keys_loaded:
    print("\nSkipping vector store creation and query due to missing API keys.")
    print("Please set the required API keys and try again.")
    sys.exit(1)

# Create a vector store from the Poma chunksets using HuggingFace embeddings.
try:
    print("\nCreating vector store...")
    # You can use any other vector DB and embedding provider.
    vec = Chroma.from_documents(chunksets, HuggingFaceEmbeddings())
    print("Vector store created successfully!")
except Exception as e:
    print(f"\nError creating vector store: {e}")
    print("This could be due to issues with the embedding model or Chroma DB.")
    sys.exit(1)


# 2️⃣ Retrieve: Initialize the retriever and QA chain
try:
    print("\nInitializing retriever...")
    retriever = PomaCheatsheetRetriever(vec, chunk_fetcher)
    print("Retriever initialized successfully!")
    
    # Create a RetrievalQA chain with the retriever and your preferred language model.
    print("\nCreating RetrievalQA chain...")
    llm = ChatOpenAI(model="gpt-4o")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    print("RetrievalQA chain created successfully!")
    
    # Query the chain to get an answer based on information retrieved from the cheatsheet.
    print("\nQuerying the chain...")
    query = "How much is a vanity plate with 4 letters? List all fees."
    print(f"Query: {query}")
    response = chain(query)
    print("\nResponse:")
    print(response)
except Exception as e:
    print(f"\nError during retrieval or query: {e}")
    print("This could be due to invalid API keys, network issues, or problems with the LLM.")
    print("Please check your API keys and try again.")
    sys.exit(1)
