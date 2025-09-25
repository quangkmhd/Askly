#!/usr/bin/env python3
"""
Script to process the latest uploaded PDF and create embeddings
"""
import os
import glob
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

def find_latest_pdf():
    """Find the latest uploaded PDF"""
    pdf_dir = Path("data/uploaded_pdfs")
    if not pdf_dir.exists():
        print("No uploaded PDFs directory found")
        return None
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found")
        return None
    
    # Sort by modification time, get latest
    latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
    print(f"Found latest PDF: {latest_pdf}")
    return latest_pdf

def main():
    """Process the latest PDF"""
    print("🔄 Processing latest uploaded PDF...")
    
    # Find latest PDF
    pdf_path = find_latest_pdf()
    if not pdf_path:
        print("❌ No PDF found to process")
        return False
    
    try:
        # Initialize RAG pipeline with the PDF path
        print("🚀 Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(pdf_path=pdf_path)
        
        # Setup pipeline (this will process the PDF)
        print(f"📄 Processing PDF: {pdf_path.name}")
        success = rag_pipeline.setup_pipeline(load_existing_embeddings=False)
        
        if success:
            print("✅ PDF processed successfully!")
            print(f"📊 Created {len(rag_pipeline.text_chunks)} text chunks")
            print(f"🔍 Generated embeddings: {rag_pipeline.embeddings.shape if rag_pipeline.embeddings is not None else 'None'}")
            return True
        else:
            print("❌ Failed to process PDF")
            return False
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
