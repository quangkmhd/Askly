"""
Configuration settings for the RAG pipeline
"""
import os
from dotenv import load_dotenv
from pathlib import Path

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

# PDF settings
PDF_FILENAME = "human-nutrition-text.pdf"
PDF_PATH = DATA_DIR / PDF_FILENAME  # PDF should be in the data directory

# Text processing settings
NUM_SENTENCE_CHUNK_SIZE = 200
MIN_TOKEN_LENGTH = 50

# Embedding settings
#EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DEVICE = "cuda" #thay bằng cuda nếu có RTX
EMBEDDING_BATCH_SIZE = 32

# Load .env for remote inference settings (explicit path)
load_dotenv(dotenv_path=BASE_DIR / '.env', override=True)
_get = lambda k: (os.getenv(k) or '').strip() or None
API_KEY = _get("API_KEY")
BASE_URL = _get("BASE_URL")
REMOTE_MODEL_NAME = _get("MODEL")
USE_REMOTE = REMOTE_MODEL_NAME is not None

# ----- Local LLM settings (fallback, not used when REMOTE_MODEL_NAME is set) -----
LLM_MODEL_ID = "google/gemma-7b-it"
LLM_DEVICE = "cpu"
LLM_TORCH_DTYPE = "float16"
USE_QUANTIZATION = False
USE_FLASH_ATTENTION = True
# LLM_MODEL_ID = "google/gemma-7b-it"
# LLM_DEVICE = "cpu"
# LLM_TORCH_DTYPE = "float16"
# USE_QUANTIZATION = False
# USE_FLASH_ATTENTION = True


# Generation settings
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 10000
DEFAULT_N_RESOURCES_TO_RETURN = 10

# File paths
EMBEDDINGS_CSV_PATH = OUTPUTS_DIR / "text_chunks_and_embeddings_df.csv"

# GPU memory thresholds for model selection
GPU_MEMORY_THRESHOLDS = {
    "low": 5.1,
    "medium": 8.1,
    "high": 19.0
}
