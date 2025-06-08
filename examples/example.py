#!/usr/bin/env python3
"""
POMA Example - Universal Document Processor

This example demonstrates how to process HTML and PDF files with POMA using a simple
standalone implementation with keyword-based retrieval.

Usage:
  python example.py ingest <file>      # Process a document (HTML or PDF)
  python example.py retrieve <query>   # Search for information in processed documents

Requirements:
  pip install doc2poma poma-chunker python-dotenv

See also:
  - example_langchain.py - Integration with LangChain
  - example_llamaindex.py - Integration with LlamaIndex
"""

import json, sys, re, os
from pathlib import Path
from dotenv import load_dotenv
import doc2poma, poma_chunker


STORE = Path("store")
CONFIG = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}


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


def ingest(src: str) -> None:
    """Convert a document to POMA format and extract chunks/chunksets."""
    if not load_api_keys():
        sys.exit("Missing required API keys. Please set them and try again.")

    # Create store directory if it doesn't exist
    STORE.mkdir(exist_ok=True)

    src_path = Path(src)
    if not src_path.exists():
        sys.exit(f"File not found: {src}")

    file_type = src_path.suffix.lower()
    if file_type not in [".html", ".pdf"]:
        sys.exit(
            f"Unsupported file type: {file_type}. Only .html and .pdf are supported."
        )

    print(f"📥 Converting {file_type[1:].upper()} document to POMA archive...")

    try:
        archive_path, conversion_costs = doc2poma.convert(
            str(src_path), config=CONFIG, base_url=None
        )
        print(
            f"✅ Generated POMA archive: {archive_path} – for USD {conversion_costs:.5f}"
        )

        print("🪄 Extracting chunks and chunksets...")
        result = poma_chunker.process(archive_path, CONFIG)
        chunks, chunksets = result["chunks"], result["chunksets"]
        chunking_costs = result.get("costs", 0.0)
        print(
            f"✅ Processed {len(chunks)} chunks and {len(chunksets)} chunksets – for USD {chunking_costs:.5f}"
        )

        doc_id = src_path
        with open(STORE / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "chunksets": chunksets}, f)
        print(f"✅ Saved to {STORE / f'{doc_id}.json'}")

        print("\n📊 Document Structure Overview:")
        print(f"  • Document: {src_path.name}")
        print(f"  • Chunks: {len(chunks)}")
        print(f"  • Chunksets: {len(chunksets)}")
        print("  • Ready for retrieval with 'python example.py retrieve <query>'")

    except Exception as exception:
        sys.exit(f"Error processing document: {exception}")


def _tok(text: str) -> list[str]:
    """Simple tokenization for keyword matching."""
    return re.findall(r"[\w']+", text.lower())


def retrieve(query: str, top_k: int = 2) -> None:
    """Find relevant chunksets and generate a query-specific cheatsheet."""
    if not STORE.exists():
        sys.exit(
            "No store/ directory found. Run 'python example.py ingest <file>' first."
        )

    print(f"🔍 Searching for: '{query}'")

    # Load all document data
    all_data = []
    for doc in STORE.glob("*.json"):
        with open(doc) as file:
            all_data.append((doc.stem, json.load(file)))

    if not all_data:
        sys.exit(
            "No documents found in store. Run 'python example.py ingest <file>' first."
        )

    # Simple keyword matching (replace with vector search in production)
    query_tokens = set(_tok(query))
    scored_chunksets = []

    # Score each chunkset based on keyword overlap
    for doc_id, data in all_data:
        chunks, chunksets = data["chunks"], data["chunksets"]
        chunk_by_id = {c["chunk_index"]: c for c in chunks}

        for cs in chunksets:
            # Get text from all chunks in this chunkset
            text = " ".join(
                chunk_by_id[cid]["content"]
                for cid in cs["chunks"]
                if cid in chunk_by_id
            )
            # Score based on keyword overlap
            score = len(query_tokens & set(_tok(text)))
            if score > 0:
                scored_chunksets.append((score, doc_id, cs, chunks))

    if not scored_chunksets:
        print("No relevant information found.")
        return

    # Sort results by score
    scored_chunksets.sort(key=lambda x: x[0], reverse=True)

    unique_docs = set(doc_id for _, doc_id, _, _ in scored_chunksets)
    print(
        f"✅ Found {len(scored_chunksets)} relevant chunksets across {len(unique_docs)} documents"
    )

    # Group results by document
    results_by_doc = {}
    doc_chunks_by_id = {doc_id: data["chunks"] for doc_id, data in all_data}

    # Process all documents with relevant results
    for score, doc_id, cs, _ in scored_chunksets:
        if doc_id not in results_by_doc:
            results_by_doc[doc_id] = []
        results_by_doc[doc_id].append((score, cs))

    # Generate a cheatsheet for each document with hits
    for doc_id, doc_results in results_by_doc.items():
        # Get all chunk IDs from this document's results
        doc_chunk_ids = []
        for _, cs in doc_results:
            doc_chunk_ids.extend(cs["chunks"])

        # Get the chunks data for this document
        doc_chunks = doc_chunks_by_id[doc_id]

        # Get relevant chunks and generate cheatsheet
        relevant_chunks = poma_chunker.get_relevant_chunks(doc_chunk_ids, doc_chunks)
        cheatsheet = poma_chunker.generate_cheatsheet(relevant_chunks)

        print(f"\n📚 Cheatsheet for '{query}' from document '{doc_id}'\n{'-'*80}")
        print(cheatsheet)
        print("-" * 80)

    print("These cheatsheets preserve each document's hierarchical structure,")
    print("making them ideal context for LLM prompts.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python example.py [ingest|retrieve] [file|query]")

    if sys.argv[1] == "ingest":
        if len(sys.argv) < 3:
            sys.exit("Usage: python example.py ingest <file>")
        src = sys.argv[2]
        ingest(src)
    elif sys.argv[1] == "retrieve":
        if len(sys.argv) < 3:
            sys.exit("Usage: python example.py retrieve <query>")
        retrieve(" ".join(sys.argv[2:]))
    else:
        sys.exit("Unknown command. Use 'ingest' or 'retrieve'.")
