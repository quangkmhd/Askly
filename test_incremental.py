#!/usr/bin/env python3
"""
Test script for incremental PDF processing
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

def test_incremental_update():
    """Test incremental PDF processing"""
    print("🧪 Testing incremental PDF processing...")
    
    try:
        # Initialize RAG pipeline
        print("🚀 Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        
        # Setup pipeline with incremental processing
        print("📄 Setting up pipeline with incremental processing...")
        success = rag_pipeline.setup_pipeline(load_existing_embeddings=True)
        
        if success:
            print("✅ Pipeline setup successful!")
            print(f"📊 Total text chunks: {len(rag_pipeline.text_chunks)}")
            print(f"🔍 Total embeddings: {rag_pipeline.embeddings.shape if rag_pipeline.embeddings is not None else 'None'}")
            
            # Show what PDFs have been processed
            processed_pdfs_file = Path("data/processed_pdfs.json")
            if processed_pdfs_file.exists():
                import json
                with open(processed_pdfs_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                    processed_files = processed_data.get('processed_files', [])
                    print(f"📋 Processed PDFs: {len(processed_files)}")
                    for pdf_file in processed_files:
                        print(f"  - {pdf_file}")
            
            # Test a query
            print("\n🔍 Testing query...")
            test_questions = [
                "thông tin liên hệ của trường",
                "dinh dưỡng là gì",
                "email của trường FPT"
            ]
            
            for question in test_questions:
                print(f"\n❓ Question: {question}")
                try:
                    answer = rag_pipeline.ask(question)
                    print(f"💬 Answer: {answer[:200]}...")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            return True
        else:
            print("❌ Pipeline setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_incremental_update()
    sys.exit(0 if success else 1)
