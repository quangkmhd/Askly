"""
Embedding management module for the RAG pipeline (TensorFlow-only, TF Hub USE)
"""
import os
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm
import tensorflow_hub as hub
from pathlib import Path

from config.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_CSV_PATH,
    MODELS_DIR,
)

# Set TF-Hub cache directory to persist models across reboots
# Default /tmp gets cleared on restart
TFHUB_CACHE_DIR = MODELS_DIR / "tfhub_cache"
TFHUB_CACHE_DIR.mkdir(exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = str(TFHUB_CACHE_DIR)

# --------- Cấu hình model TF Hub ----------
# Bạn có thể đổi sang bản đa ngôn ngữ nếu cần:
#   "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
TF_HUB_URL = os.getenv(
    "EMBEDDING_TF_HUB_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)

def _pick_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return "/GPU:0"
        except Exception:
            pass
    return "/CPU:0"

def _l2_normalize(x: tf.Tensor, axis: int = 1, eps: float = 1e-12) -> tf.Tensor:
    return tf.math.l2_normalize(x, axis=axis, epsilon=eps)


class EmbeddingManager:
    """
    TF-only EmbeddingManager dùng Universal Sentence Encoder (TF Hub).
    - Có hàm .encode(...) signature giống SentenceTransformers
      để tương thích với RetrievalSystem hiện tại.
    - Trả về tf.Tensor cho pipeline; save/load bằng CSV (JSON list) như cũ.
    """

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE, l2_normalize: bool = True):
        self.model_url = TF_HUB_URL
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize

        self.device = _pick_device()      # "/GPU:0" hoặc "/CPU:0"
        self.model = None                 # TF Hub module
        self.embeddings: Optional[tf.Tensor] = None
        self.text_chunks: Optional[List[Dict[str, Any]]] = None

    # ---------- API tương thích SentenceTransformers ----------
    def encode(
        self,
        texts: List[str] | str,
        batch_size: Optional[int] = None,
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
    ):
        """Giống sbert.encode: trả numpy (mặc định) hoặc tf.Tensor."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model is None:
            self.load_model()

        if batch_size is None:
            batch_size = self.batch_size

        with tf.device(self.device):
            vecs = []
            it = range(0, len(texts), batch_size)
            it = tqdm(it, desc="[TF] Embedding") if show_progress_bar else it
            for i in it:
                batch = texts[i : i + batch_size]
                emb = self.model(batch)  # [B, D] tf.Tensor
                vecs.append(emb)
            out = tf.concat(vecs, axis=0) if len(vecs) > 1 else vecs[0]

            if self.l2_normalize:
                out = _l2_normalize(out, axis=1)

        if convert_to_tensor:
            return out  # tf.Tensor
        return out.numpy()  # numpy array

    # ---------- Lifecycle ----------
    def load_model(self):
        print(f"[INFO] Loading TF-Hub embedding model: {self.model_url}")
        with tf.device(self.device):
            self.model = hub.load(self.model_url)
        print(f"[INFO] Embedding model loaded on device: {self.device}")

    def create_embeddings(self, text_chunks: List[Dict[str, Any]], batch_size: int = EMBEDDING_BATCH_SIZE) -> tf.Tensor:
        """Tạo embeddings cho các đoạn text (trả tf.Tensor)."""
        if self.model is None:
            self.load_model()

        texts = [c["sentence_chunk"] for c in text_chunks]
        t0 = time.perf_counter()
        emb = self.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        dt = time.perf_counter() - t0
        print(f"[INFO] Embedding creation completed in {dt:.2f}s")

        self.embeddings = emb  # tf.Tensor [N, D]
        self.text_chunks = text_chunks
        return emb

    # ---------- Save / Load ----------
    def save_embeddings_fast(self, base_path: Optional[str] = None) -> str:
        """Lưu embeddings dạng numpy binary (NHANH HƠN 10x so với CSV)."""
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks to save")

        if base_path is None:
            base_path = str(EMBEDDINGS_CSV_PATH).replace('.csv', '')
        
        # Convert to numpy if needed
        if hasattr(self.embeddings, 'numpy'):
            emb_np = self.embeddings.numpy()
        else:
            emb_np = self.embeddings
        
        # Save embeddings as numpy binary (FAST!)
        emb_file = f"{base_path}.npy"
        np.save(emb_file, emb_np)
        print(f"[INFO] ⚡ Saved embeddings to {emb_file} (numpy binary)")
        
        # Save text chunks as JSON (small file)
        chunks_file = f"{base_path}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(self.text_chunks)} chunks to {chunks_file}")
        
        return emb_file
    
    def save_embeddings(self, file_path: Optional[str] = None) -> str:
        """Lưu CSV: cột 'embedding' là JSON list (giống định dạng cũ - CHẬM)."""
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks to save")

        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)

        print(f"[INFO] Saving embeddings to {file_path}")
        # Handle both tf.Tensor and numpy array
        if hasattr(self.embeddings, 'numpy'):
            emb_np = self.embeddings.numpy()
        else:
            emb_np = self.embeddings

        rows = []
        for i, chunk in enumerate(self.text_chunks):
            row = dict(chunk)
            row["embedding"] = json.dumps(emb_np[i].tolist(), ensure_ascii=False)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved {len(rows)} embeddings to {file_path}")
        
        # Also save fast version
        self.save_embeddings_fast(file_path.replace('.csv', ''))
        return file_path

    def load_embeddings_fast(self, base_path: Optional[str] = None) -> Tuple[tf.Tensor, List[Dict[str, Any]]]:
        """Đọc embeddings từ numpy binary (NHANH HƠN 10x so với CSV)."""
        if base_path is None:
            base_path = str(EMBEDDINGS_CSV_PATH).replace('.csv', '')
        
        emb_file = f"{base_path}.npy"
        chunks_file = f"{base_path}_chunks.json"
        
        # Check if fast files exist
        if not (Path(emb_file).exists() and Path(chunks_file).exists()):
            print(f"[WARN] Fast embeddings not found, falling back to CSV...")
            return self.load_embeddings()
        
        print(f"[INFO] ⚡ Loading embeddings from {emb_file} (numpy binary - FAST!)")
        
        # Load embeddings (FAST!)
        emb_np = np.load(emb_file)
        
        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
        
        # Flatten metadata to top level for easier access
        for chunk in text_chunks:
            if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                for key, value in chunk['metadata'].items():
                    if key not in chunk:  # Don't overwrite existing keys
                        chunk[key] = value
        
        print(f"[INFO] Loaded {len(text_chunks)} chunks and embeddings")
        
        # Convert to tensor
        with tf.device(self.device):
            emb_tensor = tf.constant(emb_np, dtype=tf.float32)
            if self.l2_normalize:
                emb_tensor = _l2_normalize(emb_tensor, axis=1)
        
        self.embeddings = emb_tensor
        self.text_chunks = text_chunks
        return emb_tensor, text_chunks
    
    def load_embeddings(self, file_path: Optional[str] = None) -> Tuple[tf.Tensor, List[Dict[str, Any]]]:
        """Đọc CSV: parse JSON list → tf.Tensor; giữ nguyên text_chunks (CHẬM - dùng load_embeddings_fast thay thế)."""
        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)
        
        # Try fast loading first
        base_path = file_path.replace('.csv', '')
        if Path(f"{base_path}.npy").exists():
            print(f"[INFO] Fast embeddings found, using numpy binary...")
            return self.load_embeddings_fast(base_path)

        print(f"[INFO] Loading embeddings from {file_path} (CSV - SLOW)")

        if self.model is None:
            self.load_model()

        df = pd.read_csv(file_path)
        if "embedding" not in df.columns:
            raise ValueError("Invalid embeddings file: missing 'embedding' column")

        emb_list = df["embedding"].apply(lambda s: json.loads(s) if isinstance(s, str) else s).tolist()
        emb_np = np.array(emb_list, dtype=np.float32)

        with tf.device(self.device):
            emb = tf.convert_to_tensor(emb_np, dtype=tf.float32)
            if self.l2_normalize:
                emb = _l2_normalize(emb, axis=1)

        text_chunks = df.drop(columns=["embedding"]).to_dict(orient="records")

        self.embeddings = emb
        self.text_chunks = text_chunks
        print(f"[INFO] Loaded {len(text_chunks)} embeddings on device: {self.device}")
        return emb, text_chunks

    # ---------- Tiện ích ----------
    def get_embedding_for_text(self, text: str) -> tf.Tensor:
        """Encode 1 câu (trả tf.Tensor [D])."""
        vec = self.encode(text, convert_to_tensor=True)  # [1, D]
        return tf.squeeze(vec, axis=0)

    def get_embedding_stats(self) -> Dict[str, Any]:
        if self.embeddings is None:
            return {}
        return {
            "num_embeddings": int(self.embeddings.shape[0]),
            "embedding_dim": int(self.embeddings.shape[1]),
            "device": self.device,
            "dtype": self.embeddings.dtype.name,
            "normalized": self.l2_normalize,
            "model_url": self.model_url,
        }

    def test_embedding_similarity(self, text1: str, text2: str) -> float:
        v1 = self.get_embedding_for_text(text1)
        v2 = self.get_embedding_for_text(text2)
        v1 = tf.nn.l2_normalize(v1, axis=0)
        v2 = tf.nn.l2_normalize(v2, axis=0)
        sim = tf.reduce_sum(v1 * v2)
        return float(sim.numpy())

    def batch_encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> tf.Tensor:
        return self.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
