"""
Configuration settings for the RAG pipeline
"""
import os
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

# File paths for persistence
EMBEDDINGS_INDEX_FILE = EMBEDDINGS_DIR / "embeddings_index.json"  # Store file paths and their embeddings mapping
EMBEDDINGS_DATA_FILE = EMBEDDINGS_DIR / "embeddings_data.npz"     # Store the actual embeddings
TEXT_CHUNKS_FILE = EMBEDDINGS_DIR / "text_chunks.json"            # Store text chunks
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"                # Store chat history

# PDF settings
PDF_FILENAME = "human-nutrition-text.pdf"
PDF_PATH = DATA_DIR / PDF_FILENAME  # PDF should be in the data directory

# Text processing settings
# Optimized for procedural/instructional documents (30-50 pages)
# Each chunk should contain complete procedures/sections for better context
NUM_SENTENCE_CHUNK_SIZE = 25  # Number of sentences per chunk (~800-1000 tokens)
CHUNK_OVERLAP = 5  # Number of sentences to overlap between chunks (~20% overlap)
MIN_TOKEN_LENGTH = 0  # Keep ALL chunks, no filtering (set to 0 to disable)

# Embedding settings - Using TensorFlow Universal Sentence Encoder
EMBEDDING_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
# Alternative multilingual model: "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
EMBEDDING_DEVICE = "cuda"  # Use "cuda" if RTX GPU available, otherwise "cpu"
EMBEDDING_BATCH_SIZE = 32

# Load .env for remote inference settings (explicit path)
load_dotenv(dotenv_path=BASE_DIR / '.env', override=True)
_get = lambda k: (os.getenv(k) or '').strip() or None
API_KEY = _get("API_KEY")
BASE_URL = _get("BASE_URL")
REMOTE_MODEL_NAME = _get("MODEL")

# LLM Settings - Chỉ cần paste model vào models/model/ là chạy
USE_REMOTE = False  # True = Gemini API, False = Local model
LLM_MODEL_PATH = MODELS_DIR / "model"  # Paste model vào đây
LLM_DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
USE_QUANTIZATION = True  # Tiết kiệm VRAM


# Generation settings (OPTIMIZED: Accuracy + Conciseness)
DEFAULT_TEMPERATURE = 0.1  # VERY LOW: prevent hallucination, stick to facts
DEFAULT_MAX_NEW_TOKENS = 100  # SHORT: force concise, prevent rambling
DEFAULT_N_RESOURCES_TO_RETURN = 10  # Retrieve more chunks to find relevant info

# File paths
EMBEDDINGS_CSV_PATH = OUTPUTS_DIR / "text_chunks_and_embeddings_df.csv"

# GPU memory thresholds for model selection
GPU_MEMORY_THRESHOLDS = {
    "low": 5.1,
    "medium": 8.1,
    "high": 19.0
}
