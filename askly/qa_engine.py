"""
Q&A Engine for processing questions and finding relevant answers in documents.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np

# Optional imports for advanced features
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class QAEngine:
    """Question-Answer engine for document-based Q&A."""
    
    def __init__(self):
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None
        self.document_chunks = []
        self.chunk_vectors = None
        self.is_fitted = False
    
    def add_document(self, document: Dict[str, Any], chunk_size: int = 500):
        """
        Add a document to the knowledge base by splitting into chunks.
        
        Args:
            document: Document dictionary from DocumentProcessor
            chunk_size: Size of text chunks in characters
        """
        content = document['content']
        filename = document['filename']
        
        # Split document into chunks
        chunks = self._split_into_chunks(content, chunk_size)
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                self.document_chunks.append({
                    'text': chunk,
                    'source': filename,
                    'chunk_id': i,
                    'metadata': {
                        'file_type': document.get('file_type', 'unknown'),
                        'word_count': len(chunk.split())
                    }
                })
    
    def build_index(self):
        """Build the search index for fast retrieval."""
        if not self.document_chunks:
            raise ValueError("No documents added. Please add documents first.")
        
        if not HAS_SKLEARN:
            # Simple fallback without scikit-learn
            self.is_fitted = True
            return
        
        # Extract text from all chunks
        texts = [chunk['text'] for chunk in self.document_chunks]
        
        # Create TF-IDF vectors
        self.chunk_vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question based on the indexed documents.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to consider
            
        Returns:
            Dictionary containing answer and relevant information
        """
        if not self.is_fitted:
            raise ValueError("Index not built. Please call build_index() first.")
        
        if not HAS_SKLEARN:
            # Simple fallback without machine learning
            return self._simple_search(question, top_k)
        
        # Vectorize the question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.chunk_vectors)[0]
        
        # Get top-k most relevant chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                chunk = self.document_chunks[idx].copy()
                chunk['similarity'] = float(similarities[idx])
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            return {
                'answer': "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                'confidence': 0.0,
                'sources': [],
                'relevant_chunks': []
            }
        
        # Generate answer based on most relevant chunk
        best_chunk = relevant_chunks[0]
        answer = self._generate_answer(question, best_chunk['text'])
        
        return {
            'answer': answer,
            'confidence': best_chunk['similarity'],
            'sources': list(set([chunk['source'] for chunk in relevant_chunks])),
            'relevant_chunks': relevant_chunks
        }
    
    def extract_information(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Extract specific information based on a query.
        
        Args:
            query: Information extraction query
            top_k: Number of relevant results to return
            
        Returns:
            List of relevant information chunks
        """
        if not self.is_fitted:
            raise ValueError("Index not built. Please call build_index() first.")
        
        if not HAS_SKLEARN:
            # Simple fallback without machine learning
            return self._simple_search(query, top_k)['relevant_chunks']
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        
        # Get top-k most relevant chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for information extraction
                chunk = self.document_chunks[idx].copy()
                chunk['relevance_score'] = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _simple_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Simple text-based search without machine learning.
        Used as fallback when scikit-learn is not available.
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        scored_chunks = []
        for chunk in self.document_chunks:
            text_words = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
            
            # Simple word overlap scoring
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(query_words.union(text_words))
                chunk_copy = chunk.copy()
                chunk_copy['similarity'] = score
                chunk_copy['relevance_score'] = score
                scored_chunks.append(chunk_copy)
        
        # Sort by score and take top-k
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        relevant_chunks = scored_chunks[:top_k]
        
        if not relevant_chunks:
            return {
                'answer': "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                'confidence': 0.0,
                'sources': [],
                'relevant_chunks': []
            }
        
        # Generate answer based on most relevant chunk
        best_chunk = relevant_chunks[0]
        answer = self._generate_answer(query, best_chunk['text'])
        
        return {
            'answer': answer,
            'confidence': best_chunk['similarity'],
            'sources': list(set([chunk['source'] for chunk in relevant_chunks])),
            'relevant_chunks': relevant_chunks
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer based on question and context.
        This is a simple implementation - in production, you might use a more sophisticated model.
        """
        # Simple answer generation based on context
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "Không tìm thấy thông tin phù hợp."
        
        # Find the most relevant sentence (simple keyword matching)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence.strip()
        else:
            # Return first meaningful sentence
            return sentences[0] if sentences else "Không tìm thấy thông tin phù hợp."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if not self.document_chunks:
            return {"total_chunks": 0, "total_documents": 0}
        
        sources = set([chunk['source'] for chunk in self.document_chunks])
        total_words = sum([chunk['metadata']['word_count'] for chunk in self.document_chunks])
        
        return {
            "total_chunks": len(self.document_chunks),
            "total_documents": len(sources),
            "total_words": total_words,
            "sources": list(sources)
        }