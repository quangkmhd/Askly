"""
Text processing module for the RAG pipeline
"""
import re
from typing import List, Dict, Any
from tqdm.auto import tqdm

from config.config import NUM_SENTENCE_CHUNK_SIZE, MIN_TOKEN_LENGTH
from utils.utils import split_list, clean_text


class TextProcessor:
    """Handles text processing, sentence splitting, and chunking"""
    
    def __init__(self, chunk_size: int = NUM_SENTENCE_CHUNK_SIZE, min_token_length: int = MIN_TOKEN_LENGTH):
        self.chunk_size = chunk_size
        self.min_token_length = min_token_length
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting without spaCy"""
        # Split on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_into_sentences(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split pages into sentences using simple regex"""
        print("[INFO] Splitting pages into sentences...")
        
        for item in tqdm(pages_and_texts, desc="Processing sentences"):
            item["sentences"] = self._simple_sentence_split(item["text"])
            item["page_sentence_count_spacy"] = len(item["sentences"])
        
        return pages_and_texts
    
    def create_sentence_chunks(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create sentence chunks from pages"""
        print(f"[INFO] Creating sentence chunks with size {self.chunk_size}...")
        
        for item in tqdm(pages_and_texts, desc="Creating chunks"):
            item["sentence_chunks"] = split_list(
                input_list=item["sentences"],
                slice_size=self.chunk_size
            )
            item["num_chunks"] = len(item["sentence_chunks"])
        
        return pages_and_texts
    
    def create_text_chunks(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert sentence chunks into individual text chunks"""
        print("[INFO] Creating individual text chunks...")
        
        pages_and_chunks = []
        
        for item in tqdm(pages_and_texts, desc="Processing chunks"):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]
                
                # Join sentences into paragraph-like structure
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = clean_text(joined_sentence_chunk)
                
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                
                # Get stats on chunks
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 chars
                
                pages_and_chunks.append(chunk_dict)
        
        print(f"[INFO] Created {len(pages_and_chunks)} text chunks")
        return pages_and_chunks
    
    def filter_short_chunks(self, pages_and_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out chunks that are too short"""
        print(f"[INFO] Filtering chunks with token count > {self.min_token_length}...")
        
        filtered_chunks = [
            chunk for chunk in pages_and_chunks 
            if chunk["chunk_token_count"] > self.min_token_length
        ]
        
        print(f"[INFO] Filtered from {len(pages_and_chunks)} to {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    def process_text(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Complete text processing pipeline"""
        # Split into sentences
        pages_and_texts = self.split_into_sentences(pages_and_texts)
        
        # Create sentence chunks
        pages_and_texts = self.create_sentence_chunks(pages_and_texts)
        
        # Create individual text chunks
        pages_and_chunks = self.create_text_chunks(pages_and_texts)
        
        # Filter short chunks
        filtered_chunks = self.filter_short_chunks(pages_and_chunks)
        
        return filtered_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        token_counts = [chunk["chunk_token_count"] for chunk in chunks]
        char_counts = [chunk["chunk_char_count"] for chunk in chunks]
        word_counts = [chunk["chunk_word_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "avg_words": sum(word_counts) / len(word_counts)
        }
    
    def print_chunk_examples(self, chunks: List[Dict[str, Any]], n_examples: int = 5):
        """Print examples of short chunks"""
        short_chunks = [
            chunk for chunk in chunks 
            if chunk["chunk_token_count"] <= self.min_token_length
        ]
        
        if short_chunks:
            print(f"\n[INFO] Examples of short chunks (≤{self.min_token_length} tokens):")
            for i, chunk in enumerate(short_chunks[:n_examples]):
                print(f"Chunk {i+1} - Token count: {chunk['chunk_token_count']:.1f}")
                print(f"Text: {chunk['sentence_chunk'][:100]}...")
                print()
