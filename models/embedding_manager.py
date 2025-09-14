"""
Embedding management module for the RAG pipeline
"""
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE, EMBEDDINGS_CSV_PATH
from utils.utils import format_time


class EmbeddingManager:
    """Handles embedding creation, storage, and loading"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embeddings = None
        self.text_chunks = None
        
    def load_model(self):
        """Load the embedding model"""
        print(f"[INFO] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device
        )
        print(f"[INFO] Model loaded on device: {self.device}")
    
    def create_embeddings(self, text_chunks: List[Dict[str, Any]], batch_size: int = EMBEDDING_BATCH_SIZE) -> torch.Tensor:
        """Create embeddings for text chunks"""
        if self.model is None:
            self.load_model()
        
        print(f"[INFO] Creating embeddings for {len(text_chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk["sentence_chunk"] for chunk in text_chunks]
        
        # Create embeddings in batches
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            print(f"[INFO] Embedding creation completed in {format_time(elapsed_time)}")
        
        # Store embeddings and text chunks
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        
        return embeddings
    
    def save_embeddings(self, file_path: Optional[str] = None) -> str:
        """Save embeddings and text chunks to CSV file"""
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks to save")
        
        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)
        
        print(f"[INFO] Saving embeddings to {file_path}")
        
        # Create DataFrame with text chunks and embeddings
        data = []
        for i, chunk in enumerate(self.text_chunks):
            chunk_data = chunk.copy()
            chunk_data["embedding"] = self.embeddings[i].cpu().numpy()
            data.append(chunk_data)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        print(f"[INFO] Saved {len(data)} embeddings to {file_path}")
        return file_path
    
    def load_embeddings(self, file_path: Optional[str] = None) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Load embeddings and text chunks from CSV file"""
        if file_path is None:
            file_path = str(EMBEDDINGS_CSV_PATH)
        
        print(f"[INFO] Loading embeddings from {file_path}")
        
        # Load the embedding model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Load DataFrame
        df = pd.read_csv(file_path)
        
        # Convert embedding column back to numpy arrays
        df["embedding"] = df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x
        )
        
        # Convert to torch tensor
        embeddings = torch.tensor(
            np.stack(df["embedding"].tolist(), axis=0),
            dtype=torch.float32
        ).to(self.device)
        
        # Convert to list of dictionaries
        text_chunks = df.drop("embedding", axis=1).to_dict(orient="records")
        
        # Store loaded data
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        
        print(f"[INFO] Loaded {len(text_chunks)} embeddings")
        return embeddings, text_chunks
    
    def get_embedding_for_text(self, text: str) -> torch.Tensor:
        """Get embedding for a single text"""
        if self.model is None:
            self.load_model()
        
        return self.model.encode(text, convert_to_tensor=True)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings"""
        if self.embeddings is None:
            return {}
        
        return {
            "num_embeddings": len(self.embeddings),
            "embedding_dim": self.embeddings.shape[1],
            "device": str(self.embeddings.device),
            "dtype": str(self.embeddings.dtype)
        }
    
    def test_embedding_similarity(self, text1: str, text2: str) -> float:
        """Test similarity between two texts"""
        if self.model is None:
            self.load_model()
        
        emb1 = self.get_embedding_for_text(text1)
        emb2 = self.get_embedding_for_text(text2)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()
    
    def batch_encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> torch.Tensor:
        """Encode a batch of texts"""
        if self.model is None:
            self.load_model()
        
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
