"""
Retrieval system module for the RAG pipeline (TensorFlow version)
"""
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
from time import perf_counter as timer

from config.config import DEFAULT_N_RESOURCES_TO_RETURN
from utils.utils import print_wrapped, format_time
from models.reranker import Reranker


class RetrievalSystem:
    """Handles semantic search and retrieval of relevant documents (TensorFlow version)"""

    def __init__(self, embedding_model, embeddings: tf.Tensor,
                 text_chunks: List[Dict[str, Any]], use_reranking: bool = True):
        self.embedding_model = embedding_model
        self.embeddings = embeddings  # tf.Tensor [N, D]
        self.text_chunks = text_chunks
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        self.use_reranking = use_reranking
        self.reranker = Reranker(use_cross_encoder=False) if use_reranking else None
    
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
        # Check if we have embeddings
        if self.embeddings is None or len(self.text_chunks) == 0:
            print("[WARNING] No embeddings available for search")
            return np.array([]), np.array([])
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
            
            # Ensure k doesn't exceed the number of available embeddings
            available_embeddings = tf.shape(sims)[0]
            k = tf.minimum(n_resources, available_embeddings)
            
            scores, indices = tf.math.top_k(sims, k=k)
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
        """Get context items for a query with enhanced search"""
        # First, try keyword-based search for specific topics
        keyword_results = self._keyword_search(query, n_resources)
        if keyword_results:
            print(f"[INFO] Using keyword search results ({len(keyword_results)} found)")
            return keyword_results
        
        # Fallback to semantic search
        scores, indices = self.retrieve_relevant_resources(query, n_resources, print_time=False)
        
        # Handle empty results
        if len(scores) == 0:
            print(f"[WARNING] No search results found for query: {query}")
            return []
        
        # If scores are too low, try expanded search with related terms
        # DISABLED: Query expansion causes irrelevant results (e.g., English HR docs for Vietnamese queries)
        # Only expand if similarity is EXTREMELY low (< 0.2) - lowered from 0.25 to be more conservative
        if len(scores) > 0 and scores[0] < 0.2:
            expanded_query = self._expand_query(query)
            if expanded_query != query:
                print(f"[INFO] Very low similarity ({scores[0]:.3f}), expanding query...")
                exp_scores, exp_indices = self.retrieve_relevant_resources(expanded_query, n_resources, print_time=False)
                # Only combine if we got results from expanded search
                if len(exp_scores) > 0:
                    # Combine results, prioritizing original query
                    all_scores = np.concatenate([scores, exp_scores * 0.8])  # Slightly lower weight for expanded
                    all_indices = np.concatenate([indices, exp_indices])
                    
                    # Remove duplicates efficiently - keep highest score for each index
                    unique_items = {}
                    for score, idx in zip(all_scores, all_indices):
                        idx_int = int(idx)
                        if idx_int not in unique_items or score > unique_items[idx_int]:
                            unique_items[idx_int] = score
                    
                    # Sort by score and take top n_resources
                    sorted_items = sorted(unique_items.items(), key=lambda x: x[1], reverse=True)[:n_resources]
                    scores = np.array([item[1] for item in sorted_items])
                    indices = np.array([item[0] for item in sorted_items])
        
        context_items = []
        for score, idx in zip(scores, indices):
            item = self.text_chunks[int(idx)].copy()
            item["score"] = float(score)
            context_items.append(item)
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker and context_items:
            print(f"[INFO] Reranking {len(context_items)} results...")
            context_items = self.reranker.rerank(query, context_items, top_k=n_resources)
        
        return context_items
    
    def _keyword_search(self, query: str, n_resources: int) -> List[Dict[str, Any]]:
        """Keyword-based search - STRICT matching for better accuracy"""
        query_lower = query.lower()
        
        # Define keyword patterns - STRICT and PRECISE (16 intents)
        keyword_patterns = {
            # Original 6 intents
            "học phí": ["học phí", "31.600.000", "33.600.000", "35.800.000", "chi phí đào tạo", "tiền học"],
            "điểm": ["điểm số", "gpa", "xếp loại", "điểm tối thiểu", "học lực"],
            "tuyển sinh": ["tuyển sinh", "xét tuyển", "top50", "school rank", "nhập học", "điều kiện"],
            "lịch": ["lịch học", "thời gian học", "học kỳ", "thời khóa biểu", "kỳ học"],
            "tốt nghiệp": ["tốt nghiệp", "điều kiện tốt nghiệp", "bằng cấp", "văn bằng"],
            "ngành": ["ngành học", "chuyên ngành", "chương trình", "khóa học"],
            
            # New 10 intents
            "nghiên cứu": ["nghiên cứu", "nckh", "đề tài", "khoa học", "hội nghị", "công bố"],
            "nội quy thi": ["nội quy", "thi cử", "ký thi", "phòng thi", "gian lận", "kiểm tra"],
            "ký túc xá": ["ký túc xá", "ktx", "chỗ ở", "nội trú", "hòa lạc", "phòng ở"],
            "thực tập": ["ojt", "thực tập", "đồ án", "capstone", "dự án tốt nghiệp"],
            "thủ tục": ["thủ tục", "hồ sơ", "giấy tờ", "giấy xác nhận", "đăng ký"],
            "quy tắc": ["quy tắc", "ứng xử", "đạo đức", "hành vi", "kỷ luật"],
            "khen thưởng": ["khen thưởng", "học bổng", "giải thưởng", "sinh viên giỏi"],
            "thạc sĩ": ["thạc sĩ", "sau đại học", "cao học", "master", "luận văn"],
            "liên hệ": ["liên hệ", "số điện thoại", "email", "địa chỉ", "hotline"],
            "công nghệ": ["công nghệ", "hệ thống", "wifi", "phòng lab", "phần mềm"],
        }
        
        # Find matched topic
        matched_keywords = []
        matched_topic = None
        for topic, keywords in keyword_patterns.items():
            if topic in query_lower:
                matched_keywords = keywords
                matched_topic = topic
                break
        
        if not matched_keywords:
            return []
        
        # Search chunks - STRICT MATCHING
        results = []
        for idx, chunk in enumerate(self.text_chunks):
            text_lower = chunk.get('sentence_chunk', '').lower()
            
            # Count exact matches
            match_count = 0
            for kw in matched_keywords:
                if kw in text_lower:
                    match_count += 1
            
            # STRICT: Require at least 2 keyword matches for tuition_fee
            min_matches = 2 if matched_topic == "học phí" else 1
            
            if match_count >= min_matches:
                item = chunk.copy()
                item['score'] = 0.95 + (match_count * 0.01)
                results.append((item, match_count))
        
        # Sort by match count
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Only return if we have good quality matches
        if results and results[0][1] >= min_matches:
            return [item for item, _ in results[:n_resources]]
        
        return []
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better search - expands ALL matching keywords"""
        query_lower = query.lower()
        
        # Vietnamese expansion rules - VIETNAMESE ONLY to prevent English doc retrieval
        expansions = {
            "học phí": "chi phí đào tạo tiền học lệ phí",
            "trường": "đại học cơ sở giáo dục trường học",
            "ngành": "ngành học chuyên ngành",
            "kỹ thuật phần mềm": "công nghệ thông tin IT lập trình",
            "điểm": "điểm số đánh giá",
            "tuyển sinh": "đăng ký nhập học",
            "thời gian": "khoảng thời gian lịch trình",
            "chương trình": "khóa học môn học",
            "giảng viên": "thầy cô giáo viên",
            "sinh viên": "học sinh người học",
            "thi": "kiểm tra đánh giá kỳ thi",
            "học": "giáo dục đào tạo",
            "việc làm": "công việc nghề nghiệp",
            "kỹ năng": "năng lực khả năng",
            "công nghệ": "kỹ thuật",
            "kinh doanh": "thương mại doanh nghiệp",
            "quản lý": "điều hành quản trị",
            "đăng ký": "ghi danh",
            "học kỳ": "kỳ học",
            "bằng cấp": "văn bằng chứng chỉ",
            "thực tập": "thực hành",
        }
        
        # Expand ALL matching keywords (not just the first one)
        expansion_terms = []
        for key, expansion in expansions.items():
            if key in query_lower:
                expansion_terms.append(expansion)
        
        # Combine original query with all expansions
        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}"
            return expanded
        
        return query

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
        if len(indices) == 0:
            print("[WARNING] No results found for query")
            return -1
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
        
        # Handle empty results
        if len(scores) == 0:
            return {
                "query": query,
                "num_results": 0,
                "max_score": 0.0,
                "min_score": 0.0,
                "avg_score": 0.0,
                "score_std": 0.0,
            }
        
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
        """Rerank results (currently returns original results - can be extended with cross-encoder)"""
        # TODO: Implement cross-encoder reranking for better relevance
        # Example: Use sentence-transformers cross-encoder models
        return initial_results

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
