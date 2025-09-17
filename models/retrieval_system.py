"""
Retrieval system module for the RAG pipeline (TensorFlow version)
"""
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
from time import perf_counter as timer

from config.config import DEFAULT_N_RESOURCES_TO_RETURN
from utils.utils import print_wrapped, format_time


class RetrievalSystem:
    """Handles semantic search and retrieval of relevant documents (TensorFlow version)"""

    def __init__(self, embedding_model, embeddings: tf.Tensor,
                 text_chunks: List[Dict[str, Any]]):
        self.embedding_model = embedding_model
        self.embeddings = embeddings  # tf.Tensor [N, D]
        self.text_chunks = text_chunks
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    
    def update_embeddings(self, embeddings: tf.Tensor, text_chunks: List[Dict[str, Any]]):
        """Update the system with new embeddings and text chunks"""
        self.embeddings = embeddings
        self.text_chunks = text_chunks

    def _cosine_similarity(self, query_embedding: tf.Tensor) -> tf.Tensor:
        """Compute cosine similarity between query and stored embeddings"""
        with tf.device(self.device):
            # Normalize embeddings
            query_norm = tf.nn.l2_normalize(query_embedding, axis=-1)  # [1,D]
            emb_norm = tf.nn.l2_normalize(self.embeddings, axis=1)    # [N,D]
            
            # Compute similarity
            sims = tf.matmul(emb_norm, query_norm, transpose_b=True)  # [N,1]
            return tf.squeeze(sims, axis=-1)  # [N]

    def retrieve_relevant_resources(self, query: str,
                                    n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN,
                                    print_time: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve relevant resources for a query.
        Returns numpy arrays: (scores, indices)
        """
        # Embed query
        with tf.device(self.device):
            # Get query embedding and ensure shape is [D]
            if hasattr(self.embedding_model, "encode"):
                query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)  # [D] or [1,D]
                if len(query_embedding.shape) == 2:
                    query_embedding = query_embedding[0]  # Take first if batch
            else:
                # Direct TF Hub module call returns [1,D]
                query_embedding = self.embedding_model([query])[0]  # Take first
            
            # Convert to tensor and ensure shape
            query_embedding = tf.convert_to_tensor(query_embedding, dtype=tf.float32)
            if len(query_embedding.shape) == 1:
                query_embedding = tf.expand_dims(query_embedding, 0)  # Make [1,D]

            start_time = timer()
            sims = self._cosine_similarity(query_embedding)  # tf.Tensor [N]
            scores, indices = tf.math.top_k(sims, k=n_resources)
            end_time = timer()

        if print_time:
            elapsed = end_time - start_time
            print(f"[INFO] Time taken to get scores on {len(self.embeddings)} embeddings: {format_time(elapsed)}")

        return scores.numpy(), indices.numpy()

    def print_top_results(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN,
                          wrap_length: int = 80) -> None:
        """Print top results for a query"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources)
        print(f"Query: '{query}'\n")
        print("Results:")
        for score, idx in zip(scores, indices):
            print(f"Score: {score:.4f}")
            print("Text:")
            print_wrapped(self.text_chunks[int(idx)]["sentence_chunk"], wrap_length)
            print(f"Page number: {self.text_chunks[int(idx)]['page_number']}")
            print("\n")

    def get_context_items(self, query: str,
                          n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[Dict[str, Any]]:
        """Get context items for a query"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        context_items = []
        for score, idx in zip(scores, indices):
            item = self.text_chunks[int(idx)].copy()
            item["score"] = float(score)
            context_items.append(item)
        return context_items

    def search_by_similarity(self, query: str, threshold: float = 0.5,
                             n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[Dict[str, Any]]:
        """Search for documents above a similarity threshold"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        results = []
        for score, idx in zip(scores, indices):
            if score >= threshold:
                item = self.text_chunks[int(idx)].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def get_most_relevant_page(self, query: str) -> int:
        """Get the page number of the most relevant result"""
        _, indices = self.retrieve_relevant_resources(query, n_resources=1, print_time=False)
        return self.text_chunks[int(indices[0])]["page_number"]

    def batch_search(self, queries: List[str],
                     n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[List[Dict[str, Any]]]:
        """Perform batch search for multiple queries"""
        results = []
        for q in queries:
            results.append(self.get_context_items(q, n_resources))
        return results

    def get_search_statistics(self, query: str,
                              n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> Dict[str, Any]:
        """Get statistics about search results"""
        scores, _ = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        return {
            "query": query,
            "num_results": len(scores),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "avg_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }

    def rerank_results(self, query: str, initial_results: List[Dict[str, Any]],
                       rerank_model: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Placeholder for reranking"""
        return initial_results if rerank_model is None else initial_results

    def filter_by_page_range(self, results: List[Dict[str, Any]],
                             start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """Filter results by page range"""
        return [r for r in results if start_page <= r["page_number"] <= end_page]

    def filter_by_score_threshold(self, results: List[Dict[str, Any]],
                                  threshold: float) -> List[Dict[str, Any]]:
        """Filter results by minimum score threshold"""
        return [r for r in results if r["score"] >= threshold]

    def get_diverse_results(self, query: str,
                            n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN,
                            diversity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get diverse results by ensuring different pages"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources * 2, print_time=False)
        diverse_results = []
        used_pages = set()
        for score, idx in zip(scores, indices):
            page_num = self.text_chunks[int(idx)]["page_number"]
            if page_num not in used_pages or score > diversity_threshold:
                item = self.text_chunks[int(idx)].copy()
                item["score"] = float(score)
                diverse_results.append(item)
                used_pages.add(page_num)
                if len(diverse_results) >= n_resources:
                    break
        return diverse_results
