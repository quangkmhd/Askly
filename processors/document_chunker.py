"""
Document Chunking Module - Combines PDF processing with intelligent text chunking
Integrates footer removal, sentence splitting, and semantic chunking with metadata
"""
import re
import json
import fitz  # PyMuPDF
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
from pathlib import Path
import pytesseract
from PIL import Image
import io

from config.config import NUM_SENTENCE_CHUNK_SIZE, MIN_TOKEN_LENGTH, CHUNK_OVERLAP
from utils.utils import clean_text
from processors.semantic_chunker import SemanticChunker


class DocumentChunker:
    """
    Handles complete document processing pipeline:
    - PDF text extraction with footer removal
    - Sentence splitting
    - Intelligent chunking with overlap
    - Metadata preservation
    """
    
    def __init__(
        self, 
        chunk_size: int = NUM_SENTENCE_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_token_length: int = MIN_TOKEN_LENGTH,
        footer_threshold: int = 70,
        use_ocr: bool = True,
        ocr_lang: str = 'vie+eng',
        use_semantic_chunking: bool = False,  # NEW: Enable semantic chunking (set False for backward compatibility)
        semantic_max_tokens: int = 2000       # NEW: Max tokens for semantic chunks
    ):
        """
        Initialize DocumentChunker
        
        Args:
            chunk_size: Number of sentences per chunk
            chunk_overlap: Number of sentences to overlap between chunks
            min_token_length: Minimum tokens to keep a chunk
            footer_threshold: Pixels from bottom to consider as footer
            use_ocr: Whether to use OCR for scanned PDFs
            ocr_lang: Tesseract language (vie+eng for Vietnamese + English)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_token_length = min_token_length
        self.footer_threshold = footer_threshold
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.use_semantic_chunking = use_semantic_chunking
        self.semantic_max_tokens = semantic_max_tokens
        
        # Initialize semantic chunker if enabled
        if self.use_semantic_chunking:
            self.semantic_chunker = SemanticChunker(
                max_tokens=semantic_max_tokens,
                min_tokens=min_token_length
            )
    
    # ==================== PDF EXTRACTION WITH FOOTER REMOVAL ====================
    
    def _extract_text_with_ocr(self, page: fitz.Page) -> str:
        """
        Extract text from page using OCR (for scanned PDFs)
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            return text.strip()
        except Exception as e:
            print(f"[WARNING] OCR failed: {e}")
            return ""
    
    def _is_scanned_pdf(self, page: fitz.Page, threshold: int = 50) -> bool:
        """
        Check if page is scanned (has little/no text)
        
        Args:
            page: PyMuPDF page object
            threshold: Minimum characters to consider as text-based
            
        Returns:
            True if page appears to be scanned
        """
        text = page.get_text().strip()
        return len(text) < threshold
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str, 
        metadata: Optional[Dict[str, Any]] = None,
        remove_footer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with optional footer removal
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to attach to each page
            remove_footer: Whether to remove footer regions
            
        Returns:
            List of page dictionaries with text and metadata
        """
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_height = page.rect.height
            
            # Check if page needs OCR
            if self.use_ocr and self._is_scanned_pdf(page):
                print(f"[INFO] Page {page_num + 1} appears to be scanned, using OCR...")
                page_text = self._extract_text_with_ocr(page)
            else:
                # Extract text normally
                if remove_footer:
                    # Extract text blocks with coordinates
                    lines = []
                    for block in page.get_text("blocks"):
                        x0, y0, x1, y1, text, *_ = block
                        # Skip blocks in footer region
                        if y1 > page_height - self.footer_threshold:
                            continue
                        lines.append(text.strip())
                    page_text = "\n".join([line for line in lines if line])
                else:
                    # Extract all text
                    page_text = page.get_text()
            
            # Clean and format text
            page_text = page_text.replace("\n", " ").strip()
            
            page_data = {
                "page_number": page_num,
                "text": page_text,
                "page_char_count": len(page_text),
                "page_word_count": len(page_text.split()),
                "page_token_count": len(page_text) / 4  # Approximate
            }
            
            # Add metadata if provided
            if metadata:
                page_data["metadata"] = metadata.copy()
            
            pages_and_texts.append(page_data)
        
        doc.close()
        return pages_and_texts
    
    # ==================== SENTENCE SPLITTING ====================
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        Handles periods, exclamation marks, and question marks
        """
        sentences = []
        parts = re.split(r'([.!?]+(?:\s+|$))', text)
        
        # Combine parts to preserve delimiters
        i = 0
        while i < len(parts) - 1:
            if parts[i].strip():
                sentence = (parts[i] + parts[i+1]).strip()
                # Handle special cases for lists and acronyms
                if re.match(r'^[a-zA-Z]\.$', sentence) or re.match(r'^\d+\.$', sentence):
                    if i + 2 < len(parts):
                        parts[i+2] = sentence + ' ' + parts[i+2]
                else:
                    sentences.append(sentence)
            i += 2
        
        # Add the last part if it exists
        if i < len(parts) and parts[i].strip():
            sentences.append(parts[i].strip())
        
        # Clean up and remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_into_sentences(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split pages into sentences"""
        print("[INFO] Splitting pages into sentences...")
        
        for item in tqdm(pages_and_texts, desc="Processing sentences"):
            item["sentences"] = self._simple_sentence_split(item["text"])
            item["sentence_count"] = len(item["sentences"])
        
        return pages_and_texts
    
    # ==================== INTELLIGENT CHUNKING ====================
    
    def create_sentence_chunks(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create sentence chunks with overlap for better context
        Uses sliding window approach
        """
        print(f"[INFO] Creating chunks: {self.chunk_size} sentences, overlap {self.chunk_overlap}...")
        
        for item in tqdm(pages_and_texts, desc="Creating chunks"):
            sentences = item["sentences"]
            chunks = []
            
            # Create chunks with overlap using sliding window
            i = 0
            while i < len(sentences):
                # Take chunk_size sentences
                chunk = sentences[i:i + self.chunk_size]
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
                
                # Move forward by (chunk_size - overlap)
                step = max(1, self.chunk_size - self.chunk_overlap)
                i += step
            
            item["sentence_chunks"] = chunks
            item["num_chunks"] = len(chunks)
        
        return pages_and_texts
    
    # ==================== CHUNK FINALIZATION ====================
    
    def create_text_chunks(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert sentence chunks into final text chunks with statistics
        Preserves metadata from source pages
        """
        print("[INFO] Creating final text chunks...")
        
        pages_and_chunks = []
        chunk_id = 1
        
        for item in tqdm(pages_and_texts, desc="Finalizing chunks"):
            for chunk_idx, sentence_chunk in enumerate(item["sentence_chunks"]):
                # Join sentences
                joined_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
                joined_chunk = clean_text(joined_chunk)
                
                chunk_dict = {
                    "chunk_id": chunk_id,
                    "page_number": item["page_number"],
                    "chunk_index": chunk_idx,
                    "sentence_chunk": joined_chunk,
                    "chunk_char_count": len(joined_chunk),
                    "chunk_word_count": len(joined_chunk.split()),
                    "chunk_token_count": len(joined_chunk) / 4  # Approximate
                }
                
                # Preserve metadata if exists
                if "metadata" in item:
                    chunk_dict["metadata"] = item["metadata"].copy()
                
                # Preserve source file info if exists
                if "source_file" in item:
                    chunk_dict["source_file"] = item["source_file"]
                if "source_path" in item:
                    chunk_dict["source_path"] = item["source_path"]
                
                pages_and_chunks.append(chunk_dict)
                chunk_id += 1
        
        print(f"[INFO] Created {len(pages_and_chunks)} text chunks")
        return pages_and_chunks
    
    def filter_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out chunks that are too short"""
        print(f"[INFO] Filtering chunks with token count > {self.min_token_length}...")
        
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk["chunk_token_count"] > self.min_token_length
        ]
        
        print(f"[INFO] Filtered from {len(chunks)} to {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    # ==================== COMPLETE PIPELINE ====================
    
    def process_pdf_to_chunks(
        self, 
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        remove_footer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: PDF → Text → Chunks (semantic or sentence-based)
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to attach
            remove_footer: Whether to remove footer regions
            
        Returns:
            List of processed chunks with metadata
        """
        # Extract text from PDF
        pages_and_texts = self.extract_text_from_pdf(pdf_path, metadata, remove_footer)
        
        # Use semantic chunking if enabled
        if self.use_semantic_chunking:
            print("[INFO] Using semantic chunking (2000 tokens, header preservation)")
            chunks = self.semantic_chunker.chunk_document(pages_and_texts, metadata)
        else:
            # Original sentence-based chunking
            print("[INFO] Using sentence-based chunking")
            # Split into sentences
            pages_and_texts = self.split_into_sentences(pages_and_texts)
            
            # Create sentence chunks
            pages_and_texts = self.create_sentence_chunks(pages_and_texts)
            
            # Create final text chunks
            chunks = self.create_text_chunks(pages_and_texts)
            
            # Filter short chunks
            chunks = self.filter_short_chunks(chunks)
        
        return chunks
    
    def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        remove_footer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            metadata_list: Optional list of metadata dicts (one per PDF)
            remove_footer: Whether to remove footer regions
            
        Returns:
            Combined list of chunks from all PDFs
        """
        all_chunks = []
        
        if metadata_list is None:
            metadata_list = [None] * len(pdf_paths)
        
        for pdf_path, metadata in zip(pdf_paths, metadata_list):
            print(f"\n[INFO] Processing {Path(pdf_path).name}...")
            
            # Add source file to metadata
            if metadata is None:
                metadata = {}
            metadata["source_file"] = Path(pdf_path).name
            
            chunks = self.process_pdf_to_chunks(pdf_path, metadata, remove_footer)
            all_chunks.extend(chunks)
        
        print(f"\n[INFO] Total chunks from all PDFs: {len(all_chunks)}")
        return all_chunks
    
    # ==================== UTILITIES ====================
    
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
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(chunks)} chunks to {output_path}")
    
    def print_chunk_examples(self, chunks: List[Dict[str, Any]], n_examples: int = 3):
        """Print example chunks"""
        print(f"\n[INFO] Example chunks (showing first {n_examples}):")
        for i, chunk in enumerate(chunks[:n_examples]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Tokens: {chunk['chunk_token_count']:.1f}")
            print(f"Source: {chunk.get('source_file', 'N/A')}")
            print(f"Text: {chunk['sentence_chunk'][:200]}...")
