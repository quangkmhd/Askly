"""
PDF processing module for the RAG pipeline
"""
import os
import requests
import fitz  # PyMuPDF
from tqdm.auto import tqdm
from typing import List, Dict, Any
from pathlib import Path

from config.config import PDF_URL, PDF_PATH
from utils.utils import clean_text


class PDFProcessor:
    """Handles PDF downloading and text extraction"""
    
    def __init__(self, pdf_url: str = PDF_URL, pdf_path: Path = PDF_PATH):
        self.pdf_url = pdf_url
        self.pdf_path = pdf_path or PDF_PATH
    
    def download_pdf(self) -> bool:
        """Download PDF from URL if it doesn't exist locally"""
        if self.pdf_path.exists():
            print(f"File {self.pdf_path} already exists.")
            return True
        
        print("[INFO] File doesn't exist, downloading...")
        
        try:
            response = requests.get(self.pdf_url)
            if response.status_code == 200:
                with open(self.pdf_path, "wb") as file:
                    file.write(response.content)
                print(f"[INFO] File downloaded and saved as {self.pdf_path}")
                return True
            else:
                print(f"[ERROR] Failed to download file. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Error downloading PDF: {e}")
            return False
    
    def text_formatter(self, text: str) -> str:
        """Performs minor formatting on text"""
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text
    
    def open_and_read_pdf(self) -> List[Dict[str, Any]]:
        """Extract text from PDF and return structured data"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        doc = fitz.open(str(self.pdf_path))
        pages_and_texts = []
        
        for page_number, page in tqdm(enumerate(doc), desc="Processing PDF pages"):
            text = page.get_text()
            text = self.text_formatter(text)
            
            pages_and_texts.append({
                "page_number": page_number - 41,  # Adjust for PDF page offset
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 characters
                "text": text
            })
        
        doc.close()
        return pages_and_texts
    
    def process_pdf(self) -> List[Dict[str, Any]]:
        """Complete PDF processing pipeline"""
        # Download PDF if needed
        if not self.download_pdf():
            raise RuntimeError("Failed to download PDF")
        
        # Extract text from PDF
        pages_and_texts = self.open_and_read_pdf()
        
        print(f"[INFO] Successfully processed {len(pages_and_texts)} pages")
        return pages_and_texts
    
    def get_page_image(self, page_number: int, dpi: int = 300) -> Any:
        """Get image of a specific page for visualization"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        doc = fitz.open(str(self.pdf_path))
        # Adjust page number for PDF offset
        actual_page = page_number + 41
        page = doc.load_page(actual_page)
        
        # Get the image of the page
        img = page.get_pixmap(dpi=dpi)
        doc.close()
        
        return img
