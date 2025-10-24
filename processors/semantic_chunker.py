"""
Semantic chunking with header preservation for better context retention
Chunks by paragraphs/sections instead of sentences, preserves headers
"""
import re
from typing import List, Dict, Any, Optional
import tiktoken


class SemanticChunker:
    """Semantic chunking that preserves document structure and headers"""
    
    def __init__(
        self,
        max_tokens: int = 2000,
        min_tokens: int = 100,
        encoding_name: str = "cl100k_base"  # GPT-4/text-embedding-3 tokenizer
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def detect_headers(self, text: str) -> List[Dict[str, Any]]:
        """Detect section headers (A., B., C., 1., 2., etc.)"""
        headers = []
        
        # Patterns for headers
        patterns = [
            r'^([A-Z]\.\s+[^\n]+)',  # A. Header, B. Header
            r'^(\d+\.\s+[^\n]+)',     # 1. Header, 2. Header
            r'^([IVX]+\.\s+[^\n]+)',  # I. Header, II. Header
            r'^(Điều\s+\d+[^\n]+)',   # Điều 1, Điều 2
            r'^(Chương\s+[IVX\d]+[^\n]+)',  # Chương I, Chương 1
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    headers.append({
                        'text': match.group(1).strip(),
                        'line_num': i,
                        'level': self._get_header_level(match.group(1))
                    })
                    break
        
        return headers
    
    def _get_header_level(self, header: str) -> int:
        """Determine header hierarchy level"""
        if header.startswith('Chương'):
            return 1
        elif header.startswith('Điều'):
            return 2
        elif re.match(r'^[A-Z]\.', header):
            return 3
        elif re.match(r'^\d+\.', header):
            return 4
        else:
            return 5
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n+', text)
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def chunk_with_headers(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text semantically while preserving headers
        Each chunk includes relevant headers for context
        """
        # Detect headers
        headers = self.detect_headers(text)
        
        # Split into paragraphs
        paragraphs = self.split_by_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_headers = []  # Stack of active headers
        
        lines = text.split('\n')
        header_texts = {h['line_num']: h for h in headers}
        
        paragraph_idx = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # Check if paragraph is a header
            is_header = any(h['text'] in para for h in headers)
            
            if is_header:
                # Update header stack
                for h in headers:
                    if h['text'] in para:
                        # Remove headers of same or lower level
                        current_headers = [
                            ch for ch in current_headers 
                            if ch['level'] < h['level']
                        ]
                        current_headers.append(h)
                        break
                # Skip adding header to chunk - it will be prepended later
                continue
            
            # Check if adding this paragraph exceeds max_tokens
            header_text = '\n'.join([h['text'] for h in current_headers])
            header_tokens = self.count_tokens(header_text)
            total_tokens = current_tokens + para_tokens + header_tokens
            
            if total_tokens > self.max_tokens and current_chunk:
                # Save current chunk with headers
                chunk_text = header_text + '\n\n' + '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'tokens': self.count_tokens(chunk_text),
                    'headers': [h['text'] for h in current_headers],
                    'metadata': metadata or {}
                })
                
                # Start new chunk
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens
            
            paragraph_idx += 1
        
        # Add last chunk
        if current_chunk:
            header_text = '\n'.join([h['text'] for h in current_headers])
            chunk_text = header_text + '\n\n' + '\n\n'.join(current_chunk)
            
            # Only add if meets minimum token requirement
            chunk_tokens = self.count_tokens(chunk_text)
            if chunk_tokens >= self.min_tokens:
                chunks.append({
                    'text': chunk_text,
                    'tokens': chunk_tokens,
                    'headers': [h['text'] for h in current_headers],
                    'metadata': metadata or {}
                })
        
        return chunks
    
    def chunk_document(
        self,
        pages_and_texts: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk entire document (multiple pages) with semantic chunking
        
        Args:
            pages_and_texts: List of dicts with 'page_number' and 'page_text'
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunks with text, tokens, headers, page_number, metadata
        """
        all_chunks = []
        chunk_id = 1
        
        for page_info in pages_and_texts:
            page_num = page_info.get('page_number', 0)
            page_text = page_info.get('page_text', '')
            
            if not page_text.strip():
                continue
            
            # Get metadata from page_info (priority) or function param
            page_metadata = page_info.get('metadata', {}) or {}
            combined_metadata = {**page_metadata}  # Start with page metadata
            if metadata:  # Override with function param if provided
                combined_metadata.update(metadata)
            
            # Also get source_file from page_info root if exists
            if 'source_file' in page_info and 'source_file' not in combined_metadata:
                combined_metadata['source_file'] = page_info['source_file']
            
            # Chunk this page
            page_chunks = self.chunk_with_headers(page_text, combined_metadata)
            
            # Add page number and chunk ID
            for chunk in page_chunks:
                chunk['chunk_id'] = chunk_id
                chunk['page_number'] = page_num
                chunk['sentence_chunk'] = chunk['text']  # For compatibility
                chunk['chunk_char_count'] = len(chunk['text'])
                chunk['chunk_word_count'] = len(chunk['text'].split())
                chunk['chunk_token_count'] = chunk['tokens']
                
                # Flatten metadata to chunk root level for easier access
                if combined_metadata:
                    for key, value in combined_metadata.items():
                        if key not in chunk:  # Don't override existing keys
                            chunk[key] = value
                
                all_chunks.append(chunk)
                chunk_id += 1
        
        return all_chunks


if __name__ == "__main__":
    # Test
    chunker = SemanticChunker(max_tokens=500)
    
    test_text = """
A. Tuyển sinh và nhập học

Các thủ tục đăng ký:
- Nhập thông tin cơ bản như họ tên, email, sdt
- Nộp hồ sơ trực tuyến

B. Yêu cầu điểm

Với School Rank: Top50 (tương đương 21 điểm)

C. Học phí

1. Nhóm ngành Công nghệ thông tin
- Học kỳ 1-3: 31.600.000đ/kỳ
- Học kỳ 4-6: 33.600.000đ/kỳ
- Học kỳ 7-9: 35.800.000đ/kỳ

2. Nhóm ngành Ngôn ngữ
- Học kỳ 1-3: 28.000.000đ/kỳ
"""
    
    chunks = chunker.chunk_with_headers(test_text)
    
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Tokens: {chunk['tokens']}")
        print(f"  Headers: {chunk['headers']}")
        print(f"  Text: {chunk['text'][:200]}...")
        print()
