#!/usr/bin/env python3
"""
POMA × LlamaIndex quick-start

Requirements:
  pip install poma-integrations
  pip install llama-index-llms-openai

See also:
  - example.py - Standalone implementation with simple keyword-based retrieval
  - example_langChain.py - Integration with LangChain
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy
from db_mini_example import (
    connect,
    save_chunks_and_chunksets,
    fetch_chunks,
    fetch_chunkset,
)

#################
# Configuration #
#################

HERE = Path(__file__).parent
INPUT_PATH = HERE / "example.html"  # set to .pdf if you like

POMA_CONFIG = dict(
    conversion_provider="gemini",
    conversion_model="gemini-2.0-flash",
    chunking_provider="openai",
    chunking_model="gpt-4.1-mini",
)
LLM_MODEL = "gpt-4o"

#############
# Env Check #
#############

load_dotenv()
for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
    if not os.getenv(key):
        raise SystemExit(f"Set {key} as env-var or in a .env file")

#################
# Heavy Imports #
#################

from poma_integrations.llamaindex_poma import (
    Doc2PomaReader,
    PomaChunksetNodeParser,
    PomaCheatsheetPostProcessor,
)
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine


#############
# Ingestion #
#############

connect()

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

try:
    print("Starting ingestion...")
    reader = Doc2PomaReader(POMA_CONFIG)
    doc = reader.load_data(str(INPUT_PATH))[0]
    parser = PomaChunksetNodeParser(POMA_CONFIG)
    doc_id, ingest_nodes, raw_chunks, chunksets = parser.get_nodes_from_documents([doc])

    print("\nSaving results from documents splitter...")
    save_chunks_and_chunksets(doc_id, raw_chunks, chunksets)

    print("\nCreating vector store...")
    index = VectorStoreIndex(
        ingest_nodes,
        embed_model=OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
    )
    print("Vector store created successfully!")
except Exception as exception:
    print(f"\nError during ingestion: {exception}")
    print(
        "This could be due to invalid API keys, network issues, or problems with the embedding model."
    )
    print("Please check your API keys and try again.")
    sys.exit(1)


#############
# Retrieval #
#############


class CheatsheetQueryEngine(CustomQueryEngine):
    """Custom query engine that uses cheatsheet generation"""

    retriever: BaseRetriever
    cheatsheet_processor: PomaCheatsheetPostProcessor
    llm: OpenAI
    prompt_template: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        cheat_nodes = self.cheatsheet_processor.postprocess_nodes(nodes)
        if not cheat_nodes:
            context = "No relevant information found."
        else:
            context = "\n\n".join([node.node.text for node in cheat_nodes])
        prompt = self.prompt_template.format(context=context, question=query_str)
        response = self.llm.complete(prompt)
        return response


try:
    print("\nStarting retrieval...")

    # Vector sanity check
    vecs = index.vector_store._data.embedding_dict
    num_nonzero_vectors = sum(1 for v in vecs.values() if numpy.linalg.norm(v) > 0.01)
    print(f"[vector-check] {num_nonzero_vectors}/{len(vecs)} vectors non-zero")

    retriever = index.as_retriever(similarity_top_k=3)
    cheatsheet_processor = PomaCheatsheetPostProcessor(
        chunk_fetcher=lambda doc_id, ids=None: fetch_chunks(doc_id),
        chunkset_fetcher=lambda doc_id, chunkset_index: fetch_chunkset(
            doc_id, chunkset_index
        ),
    )
    llm = OpenAI(model=LLM_MODEL)
    prompt_template = PromptTemplate(
        "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}"
    )
    query_engine = CheatsheetQueryEngine(
        retriever=retriever,
        cheatsheet_processor=cheatsheet_processor,
        llm=llm,
        prompt_template=prompt_template,
    )
    query = "Please list all costs (one-off and annual) of an Andorran vanity plate with a combination of 4 letters and 1 number (like ALEX1)."
    print(f"Query: {query}")
    response = query_engine.query(query)
    print("\nResponse:")
    print(response)

except Exception as exception:
    print(f"\nError during retrieval: {exception}")
    print(
        "This could be due to invalid API keys, network issues, or problems with the embedding model."
    )
    print("Please check your API keys and try again.")
    sys.exit(1)
