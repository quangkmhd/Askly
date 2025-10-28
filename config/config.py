"""
Configuration settings for the RAG pipeline
"""
import os
from pickle import TRUE
from dotenv import load_dotenv
from pathlib import Path

# Check if torch is available for CUDA detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADED_PDFS_DIR = DATA_DIR / "uploaded_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADED_PDFS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# NOTE: PDFs are now loaded from data/uploaded_pdfs/ directory (incremental processing)
# Legacy PDF_FILENAME and PDF_PATH removed - see rag_pipeline.py for dynamic PDF handling

# Text processing settings (used by processors/document_chunker.py)
# Optimized for procedural/instructional documents (30-50 pages)
# Each chunk should contain complete procedures/sections for better context
NUM_SENTENCE_CHUNK_SIZE = 25  # Number of sentences per chunk (~800-1000 tokens)
CHUNK_OVERLAP = 5  # Number of sentences to overlap between chunks (~20% overlap)
MIN_TOKEN_LENGTH = 0  # Keep ALL chunks, no filtering (set to 0 to disable)

# Embedding settings (used by models/embedding_manager.py)
# NOTE: Embedding model URL and device are configured in embedding_manager.py
# - Model: TF-Hub Universal Sentence Encoder Multilingual v3 (supports Vietnamese)
# - Device: Auto-detected (GPU if available, else CPU)
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding creation

# Load .env for remote inference settings (explicit path)
load_dotenv(dotenv_path=BASE_DIR / '.env', override=True)
_get = lambda k: (os.getenv(k) or '').strip() or None
API_KEY = _get("API_KEY")              # Gemini API key for remote inference
REMOTE_MODEL_NAME = _get("MODEL")      # Remote model name (e.g., gemini-1.5-flash)
# NOTE: BASE_URL removed - llm_manager.py uses hardcoded Gemini API endpoint

# LLM Settings (used by models/llm_manager.py)
# WARNING: USE_REMOTE=False requires local model at LLM_MODEL_PATH
# If you don't have a local model, set USE_REMOTE=True to use Gemini API
USE_REMOTE = TRUE  # True = Gemini API, False = Local model
LLM_MODEL_PATH = MODELS_DIR / "model"  # Path to local model (e.g., Qwen, Vi-Qwen2-RAG)
LLM_DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
USE_QUANTIZATION = True  # Enable 4-bit quantization to save VRAM

# Generation settings (used by rag_pipeline.py and llm_manager.py)
DEFAULT_TEMPERATURE = 0.1  # VERY LOW: prevent hallucination, stick to facts
DEFAULT_MAX_NEW_TOKENS = 300  # MEDIUM: allow detailed answers but prevent rambling (was 100)
DEFAULT_N_RESOURCES_TO_RETURN = 10  # Number of relevant chunks to retrieve

# File paths for embeddings persistence (used by embedding_manager.py and rag_pipeline.py)
EMBEDDINGS_CSV_PATH = OUTPUTS_DIR / "text_chunks_and_embeddings_df.csv"

# GPU memory thresholds for model selection (used by utils.py)
# NOTE: These thresholds are also used in utils.recommend_model_config()
# If you change these, the changes will be reflected automatically
GPU_MEMORY_THRESHOLDS = {
    "low": 5.1,
    "medium": 8.1,
    "high": 19.0
}
