"""
Retrieval system module for the RAG pipeline
"""
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from time import perf_counter as timer
from sentence_transformers import util, SentenceTransformer

from config.config import DEFAULT_N_RESOURCES_TO_RETURN
from utils.utils import print_wrapped, format_time


class RetrievalSystem:
    """Handles semantic search and retrieval of relevant documents"""
    
    def __init__(self, embedding_model: SentenceTransformer, embeddings: torch.Tensor, 
                 text_chunks: List[Dict[str, Any]]):
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        self.device = embeddings.device
    
    def retrieve_relevant_resources(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN, 
                                  print_time: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant resources for a query
        Returns (scores, indices) of top-k results
        """
        # Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        
        # Get similarity scores
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        end_time = timer()
        
        if print_time:
            elapsed_time = end_time - start_time
            print(f"[INFO] Time taken to get scores on {len(self.embeddings)} embeddings: {format_time(elapsed_time)}")
        
        # Get top-k results
        scores, indices = torch.topk(dot_scores, k=n_resources)
        
        return scores, indices
    
    def print_top_results(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN, 
                         wrap_length: int = 80) -> None:
        """Print top results for a query"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources)
        
        print(f"Query: '{query}'\n")
        print("Results:")
        
        for score, idx in zip(scores, indices):
            print(f"Score: {score:.4f}")
            print("Text:")
            print_wrapped(self.text_chunks[idx]["sentence_chunk"], wrap_length)
            print(f"Page number: {self.text_chunks[idx]['page_number']}")
            print("\n")
    
    def get_context_items(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[Dict[str, Any]]:
        """Get context items (text chunks) for a query"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        
        context_items = []
        for score, idx in zip(scores, indices):
            item = self.text_chunks[idx].copy()
            item["score"] = score.cpu().item()
            context_items.append(item)
        
        return context_items
    
    def search_by_similarity(self, query: str, threshold: float = 0.5, 
                           n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[Dict[str, Any]]:
        """Search for documents above a similarity threshold"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        
        results = []
        for score, idx in zip(scores, indices):
            if score.item() >= threshold:
                item = self.text_chunks[idx].copy()
                item["score"] = score.cpu().item()
                results.append(item)
        
        return results
    
    def get_most_relevant_page(self, query: str) -> int:
        """Get the page number of the most relevant result"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources=1, print_time=False)
        return self.text_chunks[indices[0].item()]["page_number"]
    
    def batch_search(self, queries: List[str], n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[List[Dict[str, Any]]]:
        """Perform batch search for multiple queries"""
        results = []
        
        for query in queries:
            context_items = self.get_context_items(query, n_resources)
            results.append(context_items)
        
        return results
    
    def get_search_statistics(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> Dict[str, Any]:
        """Get statistics about search results"""
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        
        return {
            "query": query,
            "num_results": len(scores),
            "max_score": scores[0].item(),
            "min_score": scores[-1].item(),
            "avg_score": scores.mean().item(),
            "score_std": scores.std().item()
        }
    
    def rerank_results(self, query: str, initial_results: List[Dict[str, Any]], 
                      rerank_model: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Rerank search results using a reranking model
        This is a placeholder for future reranking implementation
        """
        if rerank_model is None:
            # For now, just return the original results
            return initial_results
        
        # TODO: Implement reranking with a dedicated reranking model
        # Example: https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1
        return initial_results
    
    def filter_by_page_range(self, results: List[Dict[str, Any]], 
                           start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """Filter results by page range"""
        return [
            result for result in results 
            if start_page <= result["page_number"] <= end_page
        ]
    
    def filter_by_score_threshold(self, results: List[Dict[str, Any]], 
                                threshold: float) -> List[Dict[str, Any]]:
        """Filter results by minimum score threshold"""
        return [
            result for result in results 
            if result["score"] >= threshold
        ]
    
    def get_diverse_results(self, query: str, n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN,
                          diversity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get diverse results by ensuring they don't come from the same page
        This is a simple diversity implementation
        """
        scores, indices = self.retrieve_relevant_resources(query, n_resources * 2, print_time=False)
        
        diverse_results = []
        used_pages = set()
        
        for score, idx in zip(scores, indices):
            page_num = self.text_chunks[idx]["page_number"]
            
            # Add result if we haven't used this page or if score is very high
            if page_num not in used_pages or score.item() > diversity_threshold:
                item = self.text_chunks[idx].copy()
                item["score"] = score.cpu().item()
                diverse_results.append(item)
                used_pages.add(page_num)
                
                if len(diverse_results) >= n_resources:
                    break
        
        return diverse_results
