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

from config.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_CSV_PATH,
)

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
    def save_embeddings(self, file_path: Optional[str] = None) -> str:
        """Lưu CSV: cột 'embedding' là JSON list (giống định dạng cũ)."""
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks to save")

        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)

        print(f"[INFO] Saving embeddings to {file_path}")
        emb_np = self.embeddings.numpy()

        rows = []
        for i, chunk in enumerate(self.text_chunks):
            row = dict(chunk)
            row["embedding"] = json.dumps(emb_np[i].tolist(), ensure_ascii=False)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved {len(rows)} embeddings to {file_path}")
        return file_path

    def load_embeddings(self, file_path: Optional[str] = None) -> Tuple[tf.Tensor, List[Dict[str, Any]]]:
        """Đọc CSV: parse JSON list → tf.Tensor; giữ nguyên text_chunks."""
        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)

        print(f"[INFO] Loading embeddings from {file_path}")

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
