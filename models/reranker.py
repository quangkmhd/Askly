"""
Reranking module using cross-encoder for better retrieval accuracy
"""
from typing import List, Dict, Any
import numpy as np


class Reranker:
    """Rerank search results using cross-encoder or simple heuristics"""
    
    def __init__(self, use_cross_encoder: bool = False):
        """
        Initialize reranker
        
        Args:
            use_cross_encoder: Whether to use cross-encoder model (requires sentence-transformers)
        """
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                # Use multilingual cross-encoder
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("[INFO] Cross-encoder loaded for reranking")
            except ImportError:
                print("[WARNING] sentence-transformers not installed, using heuristic reranking")
                self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results
        
        Args:
            query: User query
            results: List of search results with 'sentence_chunk' and 'score'
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        if self.use_cross_encoder and self.cross_encoder:
            return self._rerank_with_cross_encoder(query, results, top_k)
        else:
            return self._rerank_heuristic(query, results, top_k)
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder model"""
        # Prepare pairs for cross-encoder
        pairs = [[query, result['sentence_chunk']] for result in results]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to results
        for result, ce_score in zip(results, ce_scores):
            result['ce_score'] = float(ce_score)
            # Combine with original score (weighted average)
            result['final_score'] = 0.7 * ce_score + 0.3 * result.get('score', 0)
        
        # Sort by final score
        reranked = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:top_k]
    
    def _rerank_heuristic(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using simple heuristics:
        - Keyword matching
        - Header relevance
        - Length preference
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            text_lower = result['sentence_chunk'].lower()
            
            # 1. Keyword overlap score
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            keyword_score = overlap / max(len(query_words), 1)
            
            # 2. Exact phrase match bonus
            phrase_bonus = 1.5 if query_lower in text_lower else 1.0
            
            # 3. Header relevance (if chunk has headers)
            header_bonus = 1.0
            headers = result.get('headers', [])
            if headers:
                # Check if any header contains query keywords
                header_text = ' '.join(headers).lower()
                if any(word in header_text for word in query_words):
                    header_bonus = 1.3
            
            # 4. Length preference (prefer medium-length chunks)
            chunk_len = len(result['sentence_chunk'])
            if 500 < chunk_len < 3000:
                length_bonus = 1.1
            else:
                length_bonus = 1.0
            
            # Combine scores
            original_score = result.get('score', 0)
            heuristic_score = keyword_score * phrase_bonus * header_bonus * length_bonus
            
            # Weighted combination
            result['final_score'] = 0.6 * original_score + 0.4 * heuristic_score
        
        # Sort by final score
        reranked = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:top_k]


if __name__ == "__main__":
    # Test heuristic reranking
    reranker = Reranker(use_cross_encoder=False)
    
    test_results = [
        {
            'sentence_chunk': 'Học phí năm 1: 31.600.000đ/kỳ',
            'score': 0.35,
            'headers': ['C. Học phí', '1. Nhóm ngành CNTT']
        },
        {
            'sentence_chunk': 'Thời gian học: 4 năm',
            'score': 0.40,
            'headers': ['A. Thông tin chung']
        },
        {
            'sentence_chunk': 'C. Học phí\n1. Nhóm ngành Công nghệ thông tin\n- Học kỳ 1-3: 31.600.000đ/kỳ\n- Học kỳ 4-6: 33.600.000đ/kỳ',
            'score': 0.38,
            'headers': ['C. Học phí']
        }
    ]
    
    query = "học phí là bao nhiêu"
    
    reranked = reranker.rerank(query, test_results, top_k=3)
    
    print(f"Query: {query}\n")
    print("Reranked results:")
    for i, result in enumerate(reranked, 1):
        print(f"\n{i}. Score: {result['final_score']:.4f}")
        print(f"   Headers: {result.get('headers', [])}")
        print(f"   Text: {result['sentence_chunk'][:100]}...")
