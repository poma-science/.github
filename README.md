# 📚 POMA: Preserving Optimal Markdown Architecture

Most RAG tools split documents linearly and destroy structure, causing hallucinations, context loss, and token waste. POMA solves this by preserving the document's tree, enabling context-preserving retrieval.

**POMA** is a toolkit for turning unstructured and structured documents—like PDFs, HTML, or images—into structure-preserving "chunksets" for Retrieval-Augmented Generation (RAG) with Large Language Models.

## Key Concepts

### Chunksets: Structure-Preserving Document Paths

A **chunkset** is a sequence of sentences that preserves the complete hierarchical context from document root to specific details. Unlike traditional linear chunks, chunksets maintain the document's tree structure, ensuring that:

- Headings are never separated from their content
- Lists remain intact with all items
- Hierarchical relationships between sections are preserved
- Context is never lost during retrieval

Chunksets are the fundamental unit of retrieval in POMA, allowing for more accurate and contextually rich information retrieval. [See detailed examples below](#what-is-a-chunkset).

### Cheatsheets: Optimized LLM Context

A **cheatsheet** is a compact, deduplicated representation of the retrieved information, optimized for LLM consumption. When you retrieve relevant chunksets, POMA:

1. Collects all chunks from the relevant chunksets
2. Deduplicates overlapping content
3. Preserves structural relationships
4. Formats the information hierarchically

The resulting cheatsheet provides the LLM with precisely the context it needs to answer queries accurately, without wasting tokens on redundant information. [See a cheatsheet example below](#real-world-performance-example).

[Learn more about POMA's approach to document structure](#how-poma-works-re-generating-document-structure) | [View example implementations](#example-implementations)

## Quick-Start Guide

### Installation

Requires **Python 3.10+**. Install the core packages:

```bash
pip install poma-senter doc2poma
pip install https://github.com/poma-science/poma-chunker/releases/latest/download/poma_chunker-latest.whl
```

For integrations, add `pip install poma-integrations` and framework-specific packages as needed.

> 🔑 **Note:** You'll need API keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`) for the models you plan to use.

[See full installation details](#installation) including recommended models and additional requirements.

### Example Implementations

We provide three example implementations to help you get started with POMA:

1. **[example.py](./examples/example.py)** - A standalone implementation showing the basic POMA workflow with simple keyword-based retrieval
2. **[example_langchain.py](./examples/example_langchain.py)** - Integration with LangChain, demonstrating how to use POMA with LangChain's RetrievalQA
3. **[example_llamaindex.py](./examples/example_llamaindex.py)** - Integration with LlamaIndex, showing how to use POMA with LlamaIndex's query engine

All examples follow the same two-phase process (ingest → retrieve) but demonstrate different integration options for your RAG pipeline.

### Basic Workflow Example

POMA offers a simple two-step process for document processing and retrieval:

1. **Ingest** a document to create structured chunks/chunksets:
   ```bash
   python example.py ingest ./docs/example.html
   ```
   (use example.pdf also in this repo if you want to try PDF)

2. **Retrieve** with a query to find relevant information (here just a simple keyword search as an example):
   ```bash
   python example.py retrieve "quota"
   ```

Let's see how the basic workflow works in practice with a minimal example:

```python
#!/usr/bin/env python3
import json, sys, re, os
from pathlib import Path
from dotenv import load_dotenv
import doc2poma, poma_chunker

# Simple configuration and setup
load_dotenv()  # Load API keys from .env file (otherwise, set them as environment variables)
STORE = Path("store")
CONFIG = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}
```

#### Step 1: Document Ingestion

```python
def ingest(src: str) -> None:
    """Convert a document to POMA format and extract chunks/chunksets."""
    STORE.mkdir(exist_ok=True)
    
    # Convert to POMA archive
    archive_path, costs = doc2poma.convert(src, config=CONFIG, base_url=None)
    
    # Process the archive into chunks and chunksets
    result = poma_chunker.process(archive_path, CONFIG)
    chunks, chunksets = result["chunks"], result["chunksets"]
    
    # Save to store
    doc_id = Path(src).stem
    with open(STORE / f"{doc_id}.json", "w") as file:
        json.dump({"chunks": chunks, "chunksets": chunksets}, file)
```

#### Step 2: Information Retrieval

The retrieval process finds relevant chunksets based on a query and generates a cheatsheet for each document with matches:

```python
def _tokenize(text: str) -> list[str]:
    """Simple tokenization for keyword matching"""
    return re.findall(r"[\w']+", text.lower())

def retrieve(query: str, top_k: int = 2) -> None:
    """Find relevant chunksets and generate document-specific cheatsheets."""
    if not STORE.exists():
        sys.exit("No store/ found — run ingest first.")
    
    # Load all document data
    all_data = []
    for path in STORE.glob("*.json"):
        with open(path) as file:
            all_data.append((path.stem, json.load(file)))
    
    # Simple keyword matching (replace with vector search in production)
    query_tokens = set(_tokenize(query))
    scored_chunksets = []
    
    # Score each chunkset based on keyword overlap
    for doc_id, data in all_data:
        chunks, chunksets = data["chunks"], data["chunksets"]
        chunk_by_id = {c["chunk_index"]: c for c in chunks}
        
        for cs in chunksets:
            # Get text from all chunks in this chunkset
            text = " ".join(chunk_by_id[cid]["content"] for cid in cs["chunks"] if cid in chunk_by_id)
            # Score based on keyword overlap
            score = len(query_tokens & set(_tokenize(text)))
            if score > 0:
                scored_chunksets.append((score, doc_id, cs, chunks))
    
    # Sort by score and take top_k results
    scored_chunksets.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_chunksets[:top_k]

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
```

#### Running the Pipeline

The main function ties everything together:

```python
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python example.py [ingest|retrieve] [file|query]")
    
    if sys.argv[1] == "ingest":
        if len(sys.argv) < 3:
            sys.exit("Usage: python example.py ingest <file>")
        ingest(sys.argv[2])
    elif sys.argv[1] == "retrieve":
        if len(sys.argv) < 3:
            sys.exit("Usage: python example.py retrieve <query>")
        retrieve(" ".join(sys.argv[2:]))
```

Swap the simple keyword search with your vector/full-text DB and you have a minimal RAG loop!

Check out [example_langchain.py](./examples/example_langchain.py) and [example_llamaindex.py](./examples/example_llamaindex.py) for more advanced examples using LangChain and LlamaIndex for a full RAG pipeline.

## 🔗 Quick Links
- [Why POMA? (Problem Overview)](#why-poma-problem-overview)
- [How POMA Works](#how-poma-works-re-generating-document-structure)
- [Example Implementations](#example-implementations)
- [Detailed Workflow Example](#the-poma-processing-pipeline-detailed)
- [Real-World Performance Example](#real-world-performance-example)
- [FAQ](#faq)



---

## Why POMA? (Problem Overview) 

Retrieval-augmented generation (RAG) lets LLMs answer questions using external documents. But if you feed LLMs *linear, structureless* chunks, you get: 

* **Orphaned headings** (a title with no details)   
* **Fragmented lists** (missing key info)   
* **Ambiguous articles** (context lost)   
* **Bloated prompts** (wasted tokens)   
* **Hallucinated or incomplete answers** 

*Linear chunking* splits docs by tokens or lines—ignoring real-world structure. Tools like LlamaIndex default to this, but it fails for anything hierarchical: laws, manuals, policies, contracts, technical docs.   
**POMA preserves the true tree of your documents—so every answer comes with context, not confusion.** 

### **Failure Cases of Linear Chunking (with Real-World Impact)** 

To illustrate the shortcomings of traditional linear chunking, consider these common scenarios with real-world documents:   
#### Common Failure #1: Isolated Headings (leads to incomplete information) 

**Chunk A:** *(retrieved)*
```
Article 26. Personalized License Plate Fees
```

**Chunk B:** *(missing during retrieval)*
```
The fees vary by character count and composition.
```

*Real-world impact:* A retrieval system might return only one of these chunks, leaving the user with either a heading without details or details without a heading. This can lead to incomplete answers, confusion about what fees apply, and potentially costly misunderstandings in legal or financial contexts.  
#### Common Failure #2: Fragmented Lists (causes partial information retrieval) 

**Chunk A:** *(retrieved)*
```
Personalized license plate fees by combination:
a) 2 letters and 3 digits: 300 euros
b) 3 letters and 2 digits: 500 euros
```

**Chunk B:** *(missing during retrieval)*
```
c) 4 letters and 1 digit: 1,000 euros
d) 5 letters: 3,000 euros
e) Less than 5 characters: 6,000 euros
```

*Real-world impact:* A model retrieving only Chunk A would miss the higher fees for premium configurations, potentially causing serious misunderstandings about pricing. In legal or regulatory contexts, this partial information retrieval can lead to compliance failures, incorrect advice, or financial errors.  
#### Common Failure #3: Chapter-Article Disconnection (creates ambiguity and misattribution) 

**Chunk A:** *(retrieved)*
```
Chapter 5. Reservation Fee for Personalized License Plates
```

**Chunk B:** *(missing during retrieval)*
```
Article 21. Tax Quota
The tax quota for the reservation fee is a fixed amount of 40.74 euros.
```

*Real-world impact:* Without seeing that Article 21 belongs to Chapter 5, a model might incorrectly associate it with a previously mentioned chapter. This structural break creates ambiguity about which regulatory section governs the fee, potentially leading to incorrect legal interpretations or compliance issues. 

#### Current Workarounds Are Insufficient 

To compensate for these failures, many RAG systems attempt to include neighboring chunks. However, this approach: 

* Bloats prompts with potentially irrelevant information   
* Consumes valuable token context   
* Still frequently misses important structural boundaries   
* Relies on heuristic guesswork rather than document structure   
* Risks introducing more hallucinations through automatic summarization / relation detection 

---

## How POMA Works: Re-Generating Document Structure

In the age of retrieval-augmented generation (RAG), the ability to transform real-world documents into **structured, LLM-friendly formats** is a game-changer. Rather than *extracting* structure from messy documents using brittle heuristics, POMA **re-generates** it by combining powerful vision-language models and deterministic post-processing.
The best results are consistently achieved with the most capable respective models. While we provide flexible model configuration, we generally recommend models like `gpt-4.1-mini` for their balance of performance and reliability, ensuring placeholders aren't dropped. For Google users, `Gemini 2.0 Flash` is highly reliable in this regard but not ideal in terms of later retrieval efficiency.
This unique approach ensures that whether your input is a PDF, HTML, or an image, it is processed into a clean, semantically structured markdown with consistent metadata.
POMA runs as a two-stage pipeline:

1. **Conversion:** Converts documents (PDF, HTML, images, etc.) to a normalized markdown format with one sentence per line. Handles OCR, table/image extraction, etc. 
2. **Structural Chunking:** Analyzes the markdown, assigns each sentence a depth in the hierarchy, and groups them into **chunksets**: paths from document root to leaf content. 

Result:   
You get chunks (sentences + structure) and chunksets (full context paths) ready for use in any RAG pipeline.   
You can use POMA output as the input for LlamaIndex, LangChain, Haystack, or custom retrieval engines—*in place of their default flat chunkers*. 

<a id="what-is-a-chunkset"></a>
### **What is a Chunkset?** 

A **chunkset** is a sequence of sentences that preserves the complete hierarchical context from document root to specific details, enabling accurate context retrieval.   
Instead of slicing documents blindly, POMA: 

1. Parses the **full heading hierarchy** (title → chapter → section → clause)   
2. Assigns each **sentence** a depth within this hierarchy   
3. Creates **chunksets**: complete root-to-leaf paths that maintain structural integrity 

<details>   
<summary>📂 <strong>See Chunkset Examples</strong> (click to expand)</summary> 

**Chunkset Example 1 (Article 26 with fee schedule):**

```
Chunk IDs: [0, 1, 132, 133, 194, 195, 196, 197, 198, 199, 200, 201]

Path: "Law 24/2014" → Approval Note → Chapter 6 → Article 26 → Tax Quota → Combinations → a) through e)

Content:
  Law 24/2014, of October 30, on personalized vehicle license plates
  Given that the General Council in its session of October 30, 2014 has approved the following:
  [...]
  Chapter Four. Infractions and sanctions
  [...]
  Article 26
  Tax Quota
  Combinations
  a) 2 letters and 3 digits: 300 euros
  b) 3 letters and 2 digits: 500 euros
  c) 4 letters and 1 digit: 1,000 euros
  d) 5 letters: 3,000 euros
  e) Less than 5 characters: 6,000 euros
```

**Chunkset Example 2 (Article 21 with fixed fee):**

```
Chunk IDs: [0, 1, 132, 133, 172, 173, 174]

Path: "Law 24/2014" → Approval Note → Chapter 5 → Article 21 → Tax Quota → Fixed fee of 40.74 euros

Content:
  Law 24/2014, of October 30, on personalized vehicle license plates
  Given that the General Council in its session of October 30, 2014 has approved the following:
  [...]
  Chapter Five. Reservation Fee for Personalized License Plates
  [...]
  Article 21
  Tax Quota
  The tax quota for the reservation fee of a personalized license plate is a fixed amount of 40.74 euros.
```

**Chunkset Example 3 (Article 30 with annual tax):**

```
Chunk IDs: [0, 1, 132, 133, 217, 218, 219]

Path: "Law 24/2014" → Approval Note → Chapter 7 → Article 30 → Annual Tax → Fixed fee of 200 euros

Content:
  Law 24/2014, of October 30, on personalized vehicle license plates
  Given that the General Council in its session of October 30, 2014 has approved the following:
  [...]
  Chapter Seven. Annual Tax for Possession of a Personalized License Plate
  [...]
  Article 30
  Tax Quota
  The tax quota corresponding to the annual tax for possession of a personalized license plate is a fixed annual amount of 200 euros.
```

</details> 

---

## Features

* 🔛️ **Structure-preserving chunking** (headings, lists, articles, etc.)
* 📄 **Multi-format input:** PDF, HTML, images (with OCR), Office (via conversion to PDF), ePub (via conversion to HTML)
* ✂️ **One-sentence-per-line segmentation** for precise retrieval
* 🗃️ **Unified .poma archive** output (Markdown + assets + metadata in ZIP container)     
* ⚡ **80%+ token savings** in prompt context for structured docs   
* 🔗 **Plug-in to any RAG pipeline**   
* 🔓 **Open-source for document conversion and segmentation**   
* 🟦 **Binary wheel for structural chunker:** freely downloadable for non-commercial and evaluation use (commercial use currently free, [registration](https://poma.science#register) encouraged, subject to future licensing terms)   
* 🛠️ **Model/config via [litellm](https://github.com/BerriAI/litellm) providers** (ensure API keys are set as environment variables!)   
* 💪 **Fine-tuning friendly:** POMA's output is optimized for models fine-tuned on our structured data, which can enable almost any model to be effectively used for conversion and chunking. We encourage sharing datasets for even better results. Contact us at **fine-tuning@poma.science** to learn more about fine-tuning possibilities!

---

## Example Implementations

POMA provides three example implementations to demonstrate different ways of integrating POMA into your RAG pipeline:

### 1. Standalone Example ([example.py](./examples/example.py))

This is a complete, self-contained implementation that demonstrates the core POMA workflow:

- **Ingest Phase**: Converts documents to POMA format, processes them into chunks and chunksets, and stores the results locally
- **Retrieve Phase**: Uses simple keyword matching to find relevant chunksets and generate a cheatsheet
- **Key Features**:
  - Command-line interface with `ingest` and `retrieve` commands
  - Local JSON storage for processed documents
  - Detailed output showing document structure
  - No external dependencies beyond POMA core libraries

### 2. LangChain Integration ([example_langchain.py](./examples/example_langchain.py))

This example shows how to integrate POMA with LangChain's retrieval and QA components:

- **Integration Classes**: Uses `Doc2PomaLoader`, `PomaChunksetSplitter`, and `PomaCheatsheetRetriever` from `poma_integrations.langchain_poma`
- **Key Features**:
  - Stores chunks in SQLite for persistence
  - Uses Chroma vector store with HuggingFace embeddings
  - Integrates with LangChain's `RetrievalQA` for question answering
  - Shows how to create a custom chunk fetcher function

### 3. LlamaIndex Integration ([example_llamaindex.py](./examples/example_llamaindex.py))

This example demonstrates how to use POMA with LlamaIndex's document processing and query engine:

- **Integration Classes**: Uses `Doc2PomaReader`, `PomaChunksetNodeParser`, and `PomaCheatsheetPostProcessor` from `poma_integrations.llamaindex_poma`
- **Key Features**:
  - Similar SQLite storage pattern as the LangChain example
  - Uses LlamaIndex's `VectorStoreIndex` for retrieval
  - Implements a post-processor to generate cheatsheets from retrieved nodes
  - Shows integration with LlamaIndex's query engine and REACT chat mode

### Common Patterns Across Examples

All three examples follow the same core workflow:

1. **Document Conversion**: Using either `doc2poma.convert()` directly or through integration classes
2. **Chunking**: Processing the POMA archive into structured chunks and chunksets
3. **Storage**: Persisting the chunks and chunksets for later retrieval
4. **Retrieval**: Finding relevant chunksets based on a query
5. **Cheatsheet Generation**: Creating a hierarchically structured context for the LLM

The main differences are in how they integrate with different RAG frameworks and the specific retrieval mechanisms they use.

---

## Module Matrix

| Module | What it does | PyPI / wheel | License |
|--------|-------------|-------------|----------|
| **doc2poma** | Convert any input file into .poma | [doc2poma](https://pypi.org/project/doc2poma/) | [MPL-2.0](https://github.com/poma-science/doc2poma/LICENSE) |
| **poma-senter** | Clean & split into one sentence per line | [poma-senter](https://pypi.org/project/poma-senter/) | [MPL-2.0](https://github.com/poma-science/poma-senter/LICENSE) |
| **poma-chunker** | Build depth-aware chunks & chunksets | [poma-chunker](https://pypi.org/project/poma-chunker/) | [Proprietary](https://github.com/poma-science/poma-chunker/LICENSE.txt) (free for non-commercial & evaluation) |
| **poma-integrations** | Drop-in classes for LangChain / LlamaIndex | [poma-integrations](https://pypi.org/project/poma-integrations/) | [MPL-2.0](https://github.com/poma-science/poma-integrations/LICENSE) |
| **meta-README** | How all pieces fit | – | - |

---

<a id="installation"></a>
**Installation**

Requires **Python 3.10+**.

```
pip install poma-senter doc2poma
pip install https://github.com/poma-science/poma-chunker/releases/latest/download/poma_chunker-latest.whl
```

For the integration examples, you will also need:

```
pip install poma-integrations
```

And depending on which integration you are using:

```
# For LangChain example
pip install langchain-openai chromadb sentence-transformers

# For LlamaIndex example
pip install llama-index-llms-openai
```

* poma-senter: standalone text cleaner & segmenter (one sentence per line) before conversion, see [https://github.com/poma-science/poma-senter](https://github.com/poma-science/poma-senter) 
* doc2poma: converts PDFs, HTML, images, and Markdown into a standardized .poma archive. 
* poma-chunker: a high-performance, compiled chunking & context-extraction engine for .poma archives. 
* poma-integrations: ready-to-use integration classes for popular frameworks like LangChain and LlamaIndex.
* doc2poma, poma-senter, and poma-integrations are open source. 
* poma-chunker is distributed under a **proprietary license**. Non-commercial and evaluation use are free. Commercial use is currently free but [registration is encouraged](https://poma.science#register) and subject to future licensing terms. Patents pending. 

*For OCR and model-based conversion/chunking, you will need credentials for your preferred vision-language/completion API. Model/provider names use the [litellm](https://github.com/BerriAI/litellm) interface—[see supported models/providers here](https://github.com/BerriAI/litellm#providers). Ensure your API keys are set as environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`). While the best results are achieved with the most intelligent models (e.g., `gpt-4.1-mini` for reliability in not dropping placeholders), `gpt-4.1-nano` is quite good and `gpt-4o-mini` offers a solid balance. For Google users, `Gemini 2.0 Flash` is very safe for content retention, though its structuring might be "flatter." We are continuously evaluating other models like Claude Haiku and DeepSeek.*

Also note: For SVG processing, you need to have at least one of the following tools installed:
- Inkscape
- CairoSVG (requires libcairo)
- Wand (requires ImageMagick)
- svglib + reportlab

---

## The POMA Processing Pipeline (Detailed) 

### POMA Processing Flow Diagram

```
+----------------+         +----------------+
| doc2poma       |         | poma-chunker   |
+--------+-------+         +--------+-------+
         |                          |
         v                          v
PDF/HTML → .poma -----------------> (chunks[], chunksets[])
                                    |         |
                                    v         v
  Vector/Keyword  <---- Index chunksets in your DB, also store chunks
  search/retrieve ----> Retrieve relevant chunksets (context trees)
                                    |
                                    v
                        Get all chunk IDs referenced
                        in the retrieved chunksets
                                    |
                                    v
                        Get chunks with content for IDs
                                    |
                                    v
                        generate_cheatsheet(chunks)
                                    |
                                    v
                        Use cheatsheet in LLM prompt
```

### **Input Formats and Conversion** 

We support diverse document types through the doc2poma pipeline, ensuring that virtually anything can be converted to a consistent, LLM-friendly markdown format with assets extracted and catalogued. This includes: 

* **PDFs** (via page screenshots + OCR)   
* **Images** (JPG, PNG, TIFF, etc.)   
* **HTML** (raw structural DOM)   
* **Office documents** (DOCX, PPTX, etc. via conversion to PDF)   
* **E-books** (EPUB, MOBI, etc. via conversion to HTML) 

*(Note: Even complex media like video transcripts can be processed by treating transcripts as text and keyframes as inline images.)* 

#### **📄 PDF (via screenshot + OCR)** 

Instead of parsing messy PDF text directly, doc2poma renders each page to an image and sends that image to a vision-language model (e.g., Mistral Vision, GPT-4o Vision). This enables the model to **re-generate** clean, semantically structured markdown.   
The output per page includes: 

* Paragraphs in GFM markdown.   
* Headings using ATX-style #, ##, etc.   
* Tables returned as **inline HTML** for accurate span representation.   
* Fenced code blocks for preformatted text.   
* Verbose image placeholders for diagrams or figures. 

Each page's result is appended to content.md, separated by page break placeholders, and its screenshot saved to assets/. 


#### **🌐 HTML Processing** 

HTML documents are parsed **directly as structured data**, without rendering. This process: 

1. Parses the **DOM structure**, stripping layout noise and irrelevant elements.   
2. Converts content to structured markdown with proper heading hierarchy.   
3. Extracts and replaces **images** with descriptive placeholders ([🖼️IMAGE X PLACEHOLDER around here]).   
4. Extracts and replaces **tables** with descriptive placeholders ([📋TABLE X PLACEHOLDER around here]).   
5. Marks page breaks with descriptive placeholders ([📄PAGE X begins here]). 

This approach preserves the semantic structure of the original document while removing presentation-specific elements that don't contribute to meaning.   
<details>   
<summary>🌐 <strong>Example: HTML Conversion Flow</strong> (click to expand)</summary> 

**HTML**

```
<div class="content">   
  <h1>Product Features</h1>   
  <p>Our product offers several key features:</p>   
  <ul>   
    <li>Feature A: This is a description of feature A.</li>   
    <li>Feature B: This is a description of feature B.</li>   
  </ul>   
  <table>   
    <tr><td>Metric</td><td>Value</td></tr>   
    <tr><td>Users</td><td>10,000</td></tr>   
  </table>   
  <img src="chart.png" alt="Sales Chart">   
</div>
```

**Markdown**

```
# Product Features   
Our product offers several key features:   
- Feature A: This is a description of feature A.   
- Feature B: This is a description of Feature B.   
[📋TABLE PLACEHOLDER 1 around here]   
[🖼️IMAGE PLACEHOLDER 1 around here]
```

</details> 

### **🗃️ Output Format: The .poma Archive** 

Every processed document becomes a .poma archive (a ZIP-based format, similar to .epub), with structure and assets tightly coupled. The filename is the MD5 hash of the source file, ensuring unique identification. **Metadata for the archive is stored directly within the ZIP container's metadata storage, eliminating the need for a separate manifest.json file.** 

```
5f4dcc3b5aa765d61d8327deb882cf99.poma  # MD5 hash of source file
├── content.md         # Sentence-aligned markdown
└── assets/
    ├── page_00001.png       # Screenshot of page 1 (in case of PDF conversion)   
    ├── image_00001.png      # Extracted image (in case of HTML conversion)
└── tables/ 
    └── table_00001.html     # Extracted table (HTML from inline tables) 
```

This standardized structure ensures that all document types are processed consistently, regardless of their original format. 

---

**✂️ Sentence Segmentation with poma-senter** 

A cornerstone of the POMA format is having **one sentence per line** in content.md. This structure is ideal for indexing, chunking, and token-efficient LLM input.   
To achieve this, we use [poma-senter](https://github.com/poma-science/poma-senter)—a **deterministic, fast** sentence segmentation tool based purely on mathematical heuristics. It: 

* Splits paragraphs into atomic sentences.   
* Re-concatenates stray sentence fragments or broken lines.   
* Ensures that *no NLP model is needed*, avoiding unnecessary latency or complexity. 

<details>   
<summary>⚡ <strong>Example: Input vs. Output with poma-senter</strong> (click to expand)</summary> 

**Input paragraph:**
```
This is a sentence. This is another. This is broken   
across two lines, but should be one sentence.
```

**Output:**
```
This is a sentence.   
This is another.   
This is broken across two lines, but should be one sentence.
```

</details>   
This prepares the document for structural chunking and downstream processing. 

---

**🔧 Structural Chunking with poma-chunker *(Patent Pending)*** 

The core of the POMA format is the generation of **structurally-aware chunks** and **lossless chunksets** from the sentence-aligned markdown. 

### The Two-Phase Ingest Process 

POMA's chunking happens in two distinct phases, each serving a specific purpose:

**Phase 1: Creating Chunks with Depth**   

**Input:** .poma archive with sentence-aligned markdown   

**Process:** 

* Analyze each sentence to determine its hierarchical depth.   
* Identify structural relationships between sentences.   
* Process table HTML from /assets to maintain tabular context. 

**Output:** chunks[] array with each sentence assigned a depth value.   

<details>   
<summary>📦 <strong>POMA Chunk Examples</strong> (click to expand)</summary> 

```
Chunk 0, depth 0:

Law 24/2014, of October 30, on personalized vehicle license plates 


Chunk 1, depth 1:

Given that the General Council in its session of October 30, 2014 has approved the following: 


Chunk 2, depth 2:

Explanatory statement 


Chunk 3, depth 3:

License plates on vehicles serve the function of identifying and individualizing each vehicle 
from others, with a representation that is usually a combination of alphanumeric characters 

[...] 

Chunk 34, depth 2:

Chapter One 


Chunk 35, depth 3:

General provisions 


Chunk 36, depth 4:

Article 1 


Chunk 37, depth 5:

Purpose 


Chunk 38, depth 6:

This Law aims to define the general criteria and conditions for requesting and obtaining a 
personalized license plate, as well as regulating the corresponding tax framework 

Chunk 39, depth 4:

Article 2 


Chunk 40, depth 5:

Applicants 


Chunk 41, depth 6:

The following may request the reservation and attribution of a personalized license plate: 


Chunk 42, depth 7:

a) Natural persons legally residing in Andorra   [...]
```

</details>

**Phase 2: Building Chunksets**   

**Input:** chunks[] array with depth information   

**Process:** 

* Group sentences into meaningful semantic units.   
* Create complete root-to-leaf paths through the document.   
* Preserve parent-child relationships.   
* Maintain full hierarchical context. 

**Output:** chunksets[] array containing complete contextual paths.   
<details>   
<summary>🪜 <strong>POMA Chunkset Example</strong> (root-to-leaf path) (click to expand)</summary> 

```
Chunkset no. 47, chunk IDs [0, 1, 34, 35, 36, 37, 38]: 

Law 24/2014, of October 30, on personalized vehicle license plates   
Given that the General Council in its session of October 30, 2014 has approved the following:   
[...]   
Chapter One   
General provisions   
Article 1   
Purpose   
This Law aims to define the general criteria and conditions for requesting and obtaining a personalized license plate, as well as regulating the corresponding tax framework
```

</details>   
This two-phase approach ensures that document structure is fully preserved, enabling much more accurate retrieval than traditional chunking methods. This spatial mapping is the key innovation that enables high-quality retrieval and context assembly. 

---

**Detailed Workflow Example** 

POMA provides a complete end-to-end workflow for document processing, chunking, and retrieval: 

* **Convert** a document to .poma   
* **Segment** into one sentence per line (poma-senter)   
* **Chunk** into structure-aware units (poma-chunker)   
* **Index** chunksets in your favorite search or vector DB   
* **Retrieve** relevant chunksets for user queries   
* **Assemble** a "cheatsheet" (compact, deduplicated context for your LLM prompt) 

### **Complete End-to-End Workflow** 

The example implementation can be found in the examples/flow.py file, which demonstrates the following steps: 

1. **Document Conversion**: Using doc2poma.convert() to transform files into the standardized .poma archive format.   
2. **Chunking Process**: Using poma_chunker.process() to generate both chunks and chunksets from the .poma archive. These would then be saved/indexed/embedded in the user's (vector and/or fulltext) database.   
3. **Chunkset Selection**: In a real-world scenario, you would retrieve relevant chunksets based on a user query using vector or fulltext search. The example demonstrates this by selecting the broadest and narrowest chunksets.   
4. **Relevant Chunk Extraction**: Using poma_chunker.get_relevant_chunks() to extract all chunks referenced by the selected chunksets.   
5. **Cheatsheet Generation**: Using poma_chunker.generate_cheatsheet() to create a concise, context-preserving text from the relevant chunks. 

This workflow can be customized with different language models for various processing steps. The examples folder includes configuration options for conversion and chunking models.

#### Example: Chunk ID Enrichment and Cheatsheet Assembly 

At retrieval time, you **do not fetch isolated chunks**—you work with **chunksets**. 

1. Use **your own embedding/RAG stack** to retrieve relevant chunksets.   
2. Pass the list of their chunk IDs into:   
   ```python
   poma_chunker.get_relevant_chunks(chunk_ids, all_available_chunks)
   ```

3. This function:   
   * **Deduplicates** overlapping chunk IDs.   
   * **Adds parents, children, and adjacent chunks** as needed to ensure **structural continuity** of the chunksets.   
   * Returns a complete set of Chunk objects (containing content and metadata) for final context assembly. 

<details style="margin: 1em; padding: 1em; border: 1px solid #ddd; border-radius: 5px;">   
<summary style="font-weight: bold; cursor: pointer;">🧠 Why Chunkset Enrichment Matters</summary>   
This approach: 

* Guarantees that context contains not only the answer but the framing, legal basis, and full hierarchical context.   
* Avoids cutoff mid-section or ambiguous “floating sentences”.   
* Preserves spatial awareness—e.g., what chapter, article, or provision a sentence comes from.   
* Reduces hallucination risk by providing complete information.   
* Improves explainability in RAG traces.   
* Avoids false positives or ambiguous interpretations.   
* Supports accurate, traceable responses. 

</details>   
Input to get_relevant_chunks() (a list of chunk IDs from retrieved chunksets): 

```python
# Vector search returns these chunksets (already complete root-to-leaf paths)   
retrieved_chunksets = [   
    # Chunkset 1: Article 26 fee combinations (complete path)   
    [0, 1, 132, 133, 194, 195, 196, 197, 198, 199, 200, 201],   
    # Chunkset 2: Article 30 annual tax (complete path)   
    [0, 1, 207, 208, 217, 218, 219],   
    # Chunkset 3: Article 21 reservation fee (complete path)   
    [0, 1, 162, 163, 172, 173, 174]   
] 

# Flattened to a single list of chunk IDs (this is what you'd pass)   
chunk_ids = [0, 1, 132, 133, 194, 195, 196, 197, 198, 199, 200, 201,   
             0, 1, 207, 208, 217, 218, 219,   
             0, 1, 162, 163, 172, 173, 174]
```

Output from get_relevant_chunks() (a list of Chunk objects, not just IDs, deduplicated and enriched): 

```python
# Deduplicated and enriched set of chunk IDs (represented here as their IDs for brevity)   
[   
    # Common document root   
    0,   # Law 24/2014 title   
    1,   # Approval note
    # Path to Article 26 (Chapter Four)   
    132, # Chapter Four   
    133, # Infractions and sanctions   
    194, # Article 26   
    195, # Tax Quota   
    196, # Combinations   
    197, # a) 2 letters and 3 digits: 300 euros   
    198, # b) 3 letters and 2 digits: 500 euros   
    199, # c) 4 letters and 1 digit: 1,000 euros   
    200, # d) 5 letters: 3,000 euros   
    201, # e) Less than 5 characters: 6,000 euros

    # Path to Article 21 (Chapter Five)   
    162, # Chapter Five   
    163, # Reservation Fee for Personalized License Plates   
    172, # Article 21   
    173, # Tax Quota   
    174, # Fixed fee of 40.74 euros 

    # Path to Article 30 (Chapter Seven)   
    207, # Chapter Seven   
    208, # Annual Tax for Possession of a Personalized License Plate   
    217, # Article 30   
    218, # Tax Quota   
    219  # Fixed annual amount of 200 euros   
]
```

---

**📄 Cheatsheet Assembly (Final LLM Input)** 

The final stage is assembling a **POMA Cheatsheet** from the retrieved and enriched chunks: 

```python
cheatsheet = poma_chunker.generate_cheatsheet(all_relevant_chunks) # all_relevant_chunks is the output from get_relevant_chunks
```

This builds one compact, deduplicated, sentence-aligned context for use in an LLM prompt. It: 

* Creates a single coherent context block.   
* Preserves hierarchical relationships.   
* Organizes content in a logical, structured way.   
* Inserts LLM-friendly ellipses ([...]) to indicate original content that is not included.   
* Deduplicates redundant content. 

The result is a **deduplicated, sentence-aligned** string that provides complete context while minimizing token usage.   
<details>   
<summary>📝 <strong>Cheatsheet Example</strong> (click to expand)</summary> 

```
Atès que el Consell General en la seva sessió del dia 30 d’octubre del 2014 ha aprovat la següent:   
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]   
Capítol quart. Infraccions i sancions   
[...]   
Article 26   
Quota tributària   
Combinacions   
a) 2 lletres i 3 xifres: 300 euros   
b) 3 lletres i 2 xifres: 500 euros   
c) 4 lletres i 1 xifra: 1.000 euros   
d) 5 lletres: 3.000 euros   
e) Menys de 5 caràcters: 6.000 euros   
[...]   
Capítol cinquè. Taxa de reserva de placa de matrícula personalitzada   
[...]   
Article 21   
Quota tributària   
La quota tributària de la taxa de reserva de placa de matrícula personalitzada és un import fix de 40,74 euros.   
[...]   
Capítol setè. Taxa de tinença anual de placa de matrícula personalitzada   
[...]   
Article 30   
Quota tributària   
La quota tributària corresponent a la taxa de tinença anual de placa de matrícula personalitzada és un import fix anual de 200 euros.
```

</details> 

---

**Integration with RAG Tools** 

**POMA is designed to be the chunker inside your RAG pipeline.** 

* Use POMA output with LlamaIndex, LangChain, Haystack, Weaviate, Pinecone, etc.   
* Replace their built-in linear chunkers with POMA’s structure-preserving chunksets for more accurate retrieval.   
* Works with both vector search and keyword/fulltext search backends. 

---

**⚖️ Real-World Performance Example** 

POMA significantly outperforms traditional chunking approaches in both token efficiency and retrieval accuracy. While we don't have a dedicated benchmark repository yet, the following real-world comparison highlights the substantial difference.   
<details>   
<summary>🔍 <strong>Example retrieval comparison:</strong> LlamaIndex vs. POMA (click to expand)</summary> 

### 1. LlamaIndex default/auto chunker: 

```
--------------------------------------------------------------------------------------------------------------   
Chunk 1 (956 tokens)   
--------------------------------------------------------------------------------------------------------------   
Exposició de motius   
Les plaques de matrícula en els vehicles tenen la funció d'identificar i individualitzar cada vehicle respecte als altres, amb una representació que acostuma a ser una combinació de caràcters alfanumèrics. Però alhora, la multiplicitat de formats, colors, símbols i anagrames ha fet que les plaques de matrícula siguin també una identificació del país emissor i una forma d'exportar la marca del país quan els vehicles circulen a l'estranger. La placa de matrícula és, doncs, un símbol propi dels estats sobirans.   
A Andorra, les matrícules han evolucionat d'acord amb la imatge que des de l'Administració s'ha volgut donar a aquest identificatiu, i també per adaptar-se a l'evolució del parc automobilístic del país, que té una ràtio que s'apropa a un vehicle per habitant.   
[...]

--------------------------------------------------------------------------------------------------------------   
Chunk 2 (277 tokens)   
--------------------------------------------------------------------------------------------------------------   
Article 7   
e) No s'accepten combinacions que dificultin diferenciar entre les lletres i les xifres o que prestin confusió entre elles.   
f) No s'accepten combinacions que continguin la menció "AND" i altres pròpies de l'Estat o dels poders públics andorrans.   
g) La matrícula ha de tenir un mínim de 2 caràcters i un màxim de 5 caràcters.   
h) La matrícula de 5 caràcters ha de tenir un mínim de 2 lletres.   
[...]

--------------------------------------------------------------------------------------------------------------   
Chunk 3 (124 tokens)   
--------------------------------------------------------------------------------------------------------------   
Article 26   
Quota tributària   
La quota tributària d'aquesta taxa es determina segons els criteris següents:   
Combinacions    Quota   
a) 2 lletres i 3 xifres 300 euros   
b) 3 lletres i 2 xifres 500 euros   
c) 4 lletres i 1 xifra  1.000 euros   
d) 5 lletres    3.000 euros   
e) Menys de 5 caràcters 6.000 euros 

--------------------------------------------------------------------------------------------------------------   
Chunk 4 (40 tokens)   
--------------------------------------------------------------------------------------------------------------   
Quota tributària   
La quota tributària de la taxa de reserva de placa de matrícula personalitzada és un import fix de 40,74 euros. 

--------------------------------------------------------------------------------------------------------------   
Chunk 5 (95 tokens)   
--------------------------------------------------------------------------------------------------------------   
Article 2   
Sol·licitants   
Poden sol·licitar la reserva i l'atribució d'una placa de matrícula personalitzada:   
a) Les persones físiques legalment residents a Andorra.   
b) Les societats de dret andorrà.   
c) Les entitats sense ànim de lucre legalment establertes a Andorra i inscrites als registres corresponents. 

--------------------------------------------------------------------------------------------------------------   
Chunk 6 (50 tokens)   
--------------------------------------------------------------------------------------------------------------   
Article 30   
Quota tributària   
La quota tributària corresponent a la taxa de tinença anual de placa de matrícula personalitzada és un import fix anual de 200 euros. 

--------------------------------------------------------------------------------------------------------------   
Total context: 956+277+124+40+95+50 = 1542 tokens   
--------------------------------------------------------------------------------------------------------------
```

### 2. POMA: 

(important note: the retrieval unit for POMA is not chunks but chunksets—   
and these are not used directly but compiled into one final cheatsheet per source) 

```
--------------------------------------------------------------------------------------------------------------   
Chunkset 1 (185 tokens) – NOT directly used as context   
--------------------------------------------------------------------------------------------------------------   
Atès que el Consell General en la seva sessió del dia 30 d’octubre del 2014 ha aprovat la següent:   
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]   
Capítol quart. Infraccions i sancions   
[...]   
Article 26   
Quota tributària   
Combinacions   
a) 2 lletres i 3 xifres: 300 euros   
b) 3 lletres i 2 xifres: 500 euros   
c) 4 lletres i 1 xifra: 1.000 euros   
d) 5 lletres: 3.000 euros   
e) Menys de 5 caràcters: 6.000 euros 

--------------------------------------------------------------------------------------------------------------   
Chunkset 2 (144 tokens) – NOT directly used as context   
--------------------------------------------------------------------------------------------------------------   
Atès que el Consell General en la seva sessió del dia 30 d’octubre del 2014 ha aprovat la següent:   
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]   
Capítol setè. Taxa de tinença anual de placa de matrícula personalitzada   
[...]   
Article 30   
Quota tributària   
La quota tributària corresponent a la taxa de tinença anual de placa de matrícula personalitzada és un import fix anual de 200 euros. 

--------------------------------------------------------------------------------------------------------------   
Chunkset 3 (135 tokens) – NOT directly used as context   
--------------------------------------------------------------------------------------------------------------   
Atès que el Consell General en la seva sessió del dia 30 d’octubre del 2014 ha aprovat la següent:   
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]   
Capítol cinquè. Taxa de reserva de placa de matrícula personalitzada   
[...]   
Article 21   
Quota tributària   
La quota tributària de la taxa de reserva de placa de matrícula personalitzada és un import fix de 40,74 euros.

--------------------------------------------------------------------------------------------------------------
Final POMA Cheatsheet (185 tokens total)
--------------------------------------------------------------------------------------------------------------
Atès que el Consell General en la seva sessió del dia 30 d’octubre del 2014 ha aprovat la següent:   
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]
Article 26
Quota tributària
Combinacions
a) 2 lletres i 3 xifres: 300 euros
b) 3 lletres i 2 xifres: 500 euros
c) 4 lletres i 1 xifra: 1.000 euros
d) 5 lletres: 3.000 euros
e) Menys de 5 caràcters: 6.000 euros

La quota tributària de la taxa de reserva de placa de matrícula personalitzada és un import fix de 40,74 euros.
llei 24/2014, del 30 d’octubre, de plaques de matrícula personalitzada de vehicles   
[...]   
Capítol quart. Infraccions i sancions   
[...]   
Article 26   
Quota tributària   
Combinacions   
a) 2 lletres i 3 xifres: 300 euros   
b) 3 lletres i 2 xifres: 500 euros   
c) 4 lletres i 1 xifra: 1.000 euros   
d) 5 lletres: 3.000 euros   
e) Menys de 5 caràcters: 6.000 euros   
[...]   
Capítol cinquè. Taxa de reserva de placa de matrícula personalitzada   
[...]   
Article 21   
Quota tributària   
La quota tributària de la taxa de reserva de placa de matrícula personalitzada és un import fix de 40,74 euros.   
[...]   
Capítol setè. Taxa de tinença anual de placa de matrícula personalitzada   
[...]   
Article 30   
Quota tributària   
La quota tributària corresponent a la taxa de tinença anual de placa de matrícula personalitzada és un import fix anual de 200 euros. 

**Token savings: 1542 vs 185 = 88% reduction in tokens** 
```

</details>   
This efficiency allows for significant energy savings (and therefore cost savings) and/or more context to be included within token limits. 

---

**Licensing & Legal Status** 

### **🔐 Core IP and Patent Protection** 

The core of poma-chunker is **not open source** and protected by: 

* A **pending US patent** (US 19/170,097, filed April 2025)   
* A **granted German utility model** (DE 20 2025 101 876.4, granted April 2025) 

The compiled Cython modules inside poma-chunker implement core logic (e.g., chunkset root-to-leaf resolution, structural propagation, tree embedding) and are **not to be reverse engineered or reimplemented**.   
⚠️ Any reverse-engineering or publication of derivative works is a violation of applicable IP law and will be prosecuted. 

### **🧪 Free Usage Conditions** 

We currently offer poma-chunker **for free under the following condition**: 

* ✅ **Allowed**: Non-commercial use (educational, personal, tinkering)   
* ✅ **Allowed**: Commercial or production use (currently free, [registration encouraged](https://poma.science#register))   
* ❌ **Restricted**: Reverse engineering or unauthorized redistribution 

We aim to keep the non-commercial tier free indefinitely. If we ever introduce fees for commercial use, there will be a **3–6 month advance notice**. 

**📣 Join the Ecosystem** 

We’re building this openly, with a long-term focus on usable tooling, research alignment, and future integrations. 

* 🔧 Open-source components: [doc2poma](https://github.com/poma-science/doc2poma), [poma-senter](https://github.com/poma-science/poma-senter)   
* 🤝 Contributions welcome: issues, discussions, and PRs   
* 🧑‍🏫 Support: coming soon via GitHub Discussions and docs portal 

--- 

## Release History and Roadmap 

* **v0.1.0** (May 2025): Initial public release 
* **v0.1.1** (June 2025): Support any document length 

--- 

**Citation** 

If you use POMA in academic work, please cite as: 

@software{poma_2025,   
  author = {TIGON S.L.U.},   
  title = {POMA: Preserving Optimal Markdown Architecture},   
  year = {2025},   
  url = {https://github.com/poma-science/poma}   
} 

--- 

**Acknowledgments** 

POMA builds on open source and academic work in OCR, NLP, and document analysis.   
See each repo for detailed third-party notices.
