# Architecture Deep Dive: Askly Enterprise RAG Pipeline

## 1. System Overview

Askly is an enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed for local, high-privacy, and high-performance document Q&A. Unlike cloud-dependent SaaS wrappers, Askly processes sensitive PDFs, generates embeddings, and executes LLM generation entirely on local or self-hosted GPU infrastructure. 

The architecture is built around a highly modular, decoupled pipeline. This allows data scientists to swap out embedding models, chunking strategies, or generative LLMs without rewriting the core orchestration logic.

## 2. Core Architectural Components

### 2.1. The Extraction Module (`processors/extraction.py`)
Responsible for ingesting unstructured data.
- **PDF Parsing:** Utilizes `PyMuPDF` (`fitz`) for extremely fast text extraction from binary PDFs, preserving basic paragraph structures and handling multi-column layouts better than standard tools.
- **Text Cleansing:** Implements SpaCy pipelines (or regular expressions) to remove headers, footers, page numbers, and repetitive boilerplate that degrades embedding quality.
- **Chunking Engine:** Uses a sliding window algorithm (defined by `chunk_size` and `overlap`) to slice long documents into semantically meaningful blocks. This prevents context loss at the boundaries of chunks.

### 2.2. The Embedding Module (`processors/embedding.py`)
Converts raw text chunks into dense mathematical vectors.
- **Model Layer:** Leverages Hugging Face's `SentenceTransformers` (defaulting to highly efficient models like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`).
- **Hardware Acceleration:** Automatically detects CUDA devices. It processes chunks in batches to maximize GPU memory bandwidth, utilizing `tqdm` to provide real-time CLI feedback.

### 2.3. The Vector Store (`models/vector_store.pkl` / FAISS)
The persistence layer for the embedded vectors.
- **Storage:** For simplicity and portability, the basic implementation serializes the vectors alongside their metadata (source file, chunk index, raw text) into a Pickle file. For enterprise scale, this module interfaces seamlessly with FAISS or ChromaDB.
- **Retrieval Math:** Uses Cosine Similarity (or Inner Product) to calculate the distance between the user's query vector and the document vectors, returning the top-K nearest neighbors.

### 2.4. The Generator / LLM Layer (`rag_pipeline.py`)
The "Brain" that reads the context and writes the answer.
- **Integration:** Utilizes Hugging Face `transformers` library, optimized with `accelerate` and `bitsandbytes` (for 8-bit or 4-bit quantization), allowing massive models like Llama 2 7B or Mistral 7B to run on consumer-grade GPUs (e.g., RTX 3090/4090).
- **Prompt Engineering:** Constructs a strict prompt template that binds the LLM: *"Context: {retrieved_text} \n\n Question: {user_query} \n\n Answer based ONLY on the context:"*

## 3. Data Flow Diagram

1. **Ingestion:** User places PDFs in the `data/` directory.
2. **Chunking:** `main.py` initializes `PDFExtractor`. PDF -> Raw Text -> Array of overlapping text chunks.
3. **Vectorization:** `DocumentEmbedder` passes chunks through the SentenceTransformer model in batches.
4. **Indexing:** Vectors and metadata are saved to disk (`models/vector_store.pkl`).
5. **Querying:** User runs `run_rag.py --query "..."`. The query is embedded using the *same* SentenceTransformer model.
6. **Search:** The system computes similarity scores across the vector store and retrieves the top 3-5 chunks.
7. **Generation:** The retrieved chunks are concatenated and passed alongside the query to the generative LLM.
8. **Output:** The LLM streams the final human-readable answer to the console.

## 4. Design Decisions & Trade-offs

- **Local Processing vs. Cloud APIs:** Askly is explicitly designed to avoid sending sensitive corporate documents to OpenAI/Anthropic. The trade-off is higher local hardware requirements (CUDA GPUs), but the benefit is zero recurring API costs and absolute data privacy.
- **Pickle vs. Dedicated Vector DB:** The default architecture uses Pickle for immediate, zero-config startup. However, the `RAGPipeline` class is abstracted so that migrating to LanceDB, Milvus, or Qdrant requires changing only the `search()` method implementation.
- **Overlapping Chunks:** A strict 50-token overlap was chosen mathematically to ensure that sentences bisected by the chunk limit do not lose their semantic meaning, which is a common failure point in naive RAG systems.
