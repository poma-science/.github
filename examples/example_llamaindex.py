"""
POMA Example - LlamaIndex Integration

This example demonstrates how to integrate POMA with LlamaIndex for document processing and retrieval.

Requirements:
  pip install poma-integrations
  pip install llama-index-llms-openai

See also:
  - example.py - Standalone implementation with simple keyword-based retrieval
  - example_langchain.py - Integration with LangChain
"""

import os
import sys
from dotenv import load_dotenv
import sqlite3
from poma_integrations.llamaindex_poma import (
    Doc2PomaReader,
    PomaChunksetNodeParser,
    PomaCheatsheetPostProcessor,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI


def load_api_keys():
    """Load API keys from .env file or environment variables."""
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


api_keys_loaded = load_api_keys()


# Use the following configuration to set up the reader and chunking,
# or use any other provider/model in the LiteLLM format.
cfg = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}


# 1️⃣ ingest

# Chunk the document and convert it to Poma nodes.
pdf_path = os.path.join(os.path.dirname(__file__), "example.pdf")
print(f"Processing PDF file: {pdf_path}")
reader = Doc2PomaReader(cfg)
nodes_md = reader.load_data(pdf_path)
nodes, raw_chunks = PomaChunksetNodeParser(cfg).get_nodes_from_documents(nodes_md)


def chunk_store(con, nodes, raw_chunks):
    """
    Store the chunksets in a SQLite database.
    """
    # Create a table for the chunks if it does not exist.
    con.executescript(
        "CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, idx INT, depth INT, "
        "content TEXT, PRIMARY KEY(doc_id,idx));"
    )
    doc_id = nodes[0].metadata["doc_id"]
    con.executemany(
        "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
        [(doc_id, c["chunk_index"], c["depth"], c["content"]) for c in raw_chunks],
    )
    con.commit()


# Store the result in a SQLite database (or any prefered key-value DB) for later retrieval.
con = sqlite3.connect("chunks.db")
chunk_store(con, nodes, raw_chunks)


# Create a vector index from the Poma nodes, or any other index type / vector database.
# Check if API keys are loaded before creating the vector index
if not api_keys_loaded:
    print("\nSkipping vector index creation and query due to missing API keys.")
    print("Please set the required API keys and try again.")
    sys.exit(1)
try:
    print("\nCreating vector index...")
    index = VectorStoreIndex(nodes)
    print("Vector index created successfully!")
except Exception as exception:
    print(f"\nError creating vector index: {exception}")
    print("This could be due to invalid API keys or network issues.")
    print("Please check your API keys and try again.")
    sys.exit(1)


# 2️⃣ retrieve


def chunk_fetch(doc_id, con=con):
    """
    Return *all* chunks belonging to the document.
    """
    rows = con.execute(
        "SELECT idx, depth, content FROM chunks WHERE doc_id=? ORDER BY idx",
        (doc_id,),
    ).fetchall()
    return [{"chunk_index": r[0], "depth": r[1], "content": r[2]} for r in rows]


# Fetch all chunks for a given document ID
# and query the index to get a cheatsheet as result.
try:
    print("\nQuerying the vextor index...")
    postprocessor = PomaCheatsheetPostProcessor(chunk_fetcher=chunk_fetch)
    query_engine = index.as_query_engine(
        node_postprocessors=[postprocessor],
        llm=OpenAI(model="gpt-4.1-nano"),
        chat_mode=ChatMode.REACT,
    )
    query = "How much is a vanity plate with 4 letters? List all fees."
    print(f"Query: {query}")
    response = query_engine.query(query)
    print("\nResponse:")
    print(response)
except Exception as exception:
    print(f"\nError during query: {exception}")
    print(
        "This could be due to invalid API keys, network issues, or problems with the LLM."
    )
    print("Please check your API keys and try again.")
    sys.exit(1)
