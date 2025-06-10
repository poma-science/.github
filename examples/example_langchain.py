#!/usr/bin/env python3
"""
POMA × LangChain quick-start

Requirements:
  pip install poma-integrations
  pip install faiss-cpu langchain langchain-openai langchain-community

See also:
  - example.py - Standalone implementation with simple keyword-based retrieval
  - example_llamaindex.py - Integration with LlamaIndex
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from dotenv import load_dotenv
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

from poma_integrations.langchain_poma import (
    Doc2PomaLoader,
    PomaChunksetSplitter,
    PomaCheatsheetRetriever,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


#############
# Ingestion #
#############

connect()

try:
    print("Starting ingestion...")
    loader = Doc2PomaLoader(POMA_CONFIG)
    doc = loader.load(str(INPUT_PATH))
    splitter = PomaChunksetSplitter(POMA_CONFIG)
    doc_id, chunkset_content_docs, raw_chunks, chunksets = splitter.split_documents(doc)

    print("\nSaving results from documents splitter...")
    save_chunks_and_chunksets(doc_id, raw_chunks, chunksets)

    print("\nCreating vector store...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    vector_store = FAISS.from_documents(
        documents=chunkset_content_docs, embedding=embeddings
    )
    vector_store.save_local("faiss_index")
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

try:
    print("\nStarting retrieval...")
    retriever = PomaCheatsheetRetriever(
        vector_store, fetch_chunks, fetch_chunkset, top_k=3
    )
    llm = ChatOpenAI(model=LLM_MODEL)
    prompt = PromptTemplate.from_template(
        "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}"
    )
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    query = "Please list all costs (one-off and annual) of an Andorran vanity plate with a combination of 4 letters and 1 number (like ALEX1)."
    print(f"Query: {query}")
    response = qa_chain.invoke(query)
    print("\nResponse:")
    print(response)
except Exception as exception:
    print(f"\nError during retrieval: {exception}")
    print(
        "This could be due to invalid API keys, network issues, or problems with the embedding model."
    )
    print("Please check your API keys and try again.")
    sys.exit(1)
