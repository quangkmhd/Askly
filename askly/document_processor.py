"""
Document processor for handling various document formats and extracting text content.
"""

import os
import re
from typing import List, Dict, Any
# Optional imports - will be imported when needed
try:
    import PyPDF2
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    import docx
    HAS_DOCX_SUPPORT = True
except ImportError:
    HAS_DOCX_SUPPORT = False
from pathlib import Path


class DocumentProcessor:
    """Handles document loading and text extraction from various formats."""
    
    def __init__(self):
        self.supported_formats = ['.txt']
        if HAS_PDF_SUPPORT:
            self.supported_formats.append('.pdf')
        if HAS_DOCX_SUPPORT:
            self.supported_formats.append('.docx')
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load and extract text from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata and extracted text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if file_ext == '.txt':
            return self._load_txt(file_path)
        elif file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.docx':
            return self._load_docx(file_path)
    
    def _load_txt(self, file_path: str) -> Dict[str, Any]:
        """Load text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {
            'filename': os.path.basename(file_path),
            'file_type': 'txt',
            'content': content,
            'word_count': len(content.split()),
            'char_count': len(content)
        }
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Load text from a .pdf file."""
        if not HAS_PDF_SUPPORT:
            raise ValueError("PDF support not available. Please install PyPDF2: pip install PyPDF2")
        
        content = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
        
        return {
            'filename': os.path.basename(file_path),
            'file_type': 'pdf',
            'content': content,
            'word_count': len(content.split()),
            'char_count': len(content)
        }
    
    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """Load text from a .docx file."""
        if not HAS_DOCX_SUPPORT:
            raise ValueError("DOCX support not available. Please install python-docx: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {str(e)}")
        
        return {
            'filename': os.path.basename(file_path),
            'file_type': 'docx',
            'content': content,
            'word_count': len(content.split()),
            'char_count': len(content)
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for better information extraction.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\;\:]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better processing.
        
        Args:
            text: Text content
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text using simple frequency analysis.
        
        Args:
            text: Text content
            top_k: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                     'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]