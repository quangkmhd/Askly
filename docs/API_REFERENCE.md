# API & Class Reference: Askly

This document provides a comprehensive reference for the core Python classes, methods, and CLI execution scripts that make up the Askly RAG Pipeline.

## 1. CLI Commands

### `main.py`
The primary script for offline data ingestion and embedding generation.

**Usage:**
```bash
python main.py [--data_dir PATH] [--output_store PATH] [--chunk_size INT]
```

**Arguments:**
- `--data_dir`: (Default: `./data`) Directory containing the source PDF files.
- `--output_store`: (Default: `./models/vector_store.pkl`) Destination path for the generated vector database.
- `--chunk_size`: (Default: `500`) Number of characters/tokens per chunk.
- `--overlap`: (Default: `50`) Overlap between consecutive chunks.

### `run_rag.py`
The execution script for querying the generated vector store and generating answers.

**Usage:**
```bash
python run_rag.py --query "Your question here" [--top_k INT]
```

**Arguments:**
- `--query` (Required): The natural language question to ask the pipeline.
- `--top_k`: (Default: `3`) The number of context chunks to retrieve and feed to the LLM.
- `--store_path`: (Default: `./models/vector_store.pkl`) Path to the pre-computed vector store.

## 2. Core Python Classes

### `processors.extraction.PDFExtractor`
Handles the ingestion of raw documents.

#### `__init__(self, chunk_size: int = 500, overlap: int = 50)`
Initializes the extractor with specific sliding window parameters.

#### `process_directory(self, dir_path: str) -> List[Dict]`
Recursively scans a directory for `.pdf` files, extracts text, and applies the chunking strategy.
- **Returns:** A list of dictionaries: `[{"source": "file.pdf", "chunk_id": 0, "text": "Extracted text..."}]`

#### `extract_text_from_pdf(self, file_path: str) -> str`
Core PyMuPDF implementation to strip text from a single binary PDF file.

---

### `processors.embedding.DocumentEmbedder`
Manages the SentenceTransformers logic and GPU acceleration.

#### `__init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None)`
- `model_name`: Hugging Face model identifier.
- `device`: "cuda" or "cpu". If None, auto-detects.

#### `embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]`
Takes the output of `PDFExtractor`, computes dense vectors for the `text` field, and appends a `vector` key to each dictionary.
- Includes a `tqdm` progress bar for CLI visibility.

#### `save_to_disk(self, path: str)` / `load_from_disk(self, path: str)`
Serializes and deserializes the state of the vector embeddings to standard storage.

---

### `rag_pipeline.RAGPipeline`
The orchestrator that marries retrieval and generation.

#### `__init__(self, vector_store_path: str, llm_model_id: str = "meta-llama/Llama-2-7b-chat-hf")`
Initializes the pipeline, loading the vector store into RAM and spinning up the Hugging Face generative model.

#### `search(self, query: str, top_k: int = 3) -> List[Dict]`
1. Embeds the `query` using the `DocumentEmbedder` model.
2. Computes Cosine Similarity against all vectors in the loaded store.
3. Returns the top `k` most similar chunks along with their similarity scores.

#### `generate_answer(self, query: str, temperature: float = 0.7, max_tokens: int = 256) -> str`
1. Calls `self.search(query)`.
2. Constructs the prompt template.
3. Invokes the Hugging Face `transformers.pipeline('text-generation')`.
4. Returns the sanitized string output from the LLM.
