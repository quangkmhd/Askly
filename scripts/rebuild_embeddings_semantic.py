#!/usr/bin/env python3
"""
Rebuild embeddings with semantic chunking
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from processors.document_chunker import DocumentChunker
from models.embedding_manager import EmbeddingManager
from config.config import DATA_DIR, OUTPUTS_DIR

def main():
    print("="*80)
    print("REBUILDING EMBEDDINGS WITH SEMANTIC CHUNKING")
    print("="*80)
    
    # Initialize chunker with semantic chunking enabled
    chunker = DocumentChunker(
        use_semantic_chunking=True,      # Enable semantic chunking
        semantic_max_tokens=2000,         # 2000 tokens per chunk
        min_token_length=100,             # Min 100 tokens
        use_ocr=True,                     # Keep OCR enabled
        ocr_lang='vie+eng'                # Vietnamese + English
    )
    
    # Get all PDFs
    pdf_dir = DATA_DIR / "uploaded_pdfs"
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_paths:
        print(f"❌ No PDFs found in {pdf_dir}")
        return
    
    print(f"\n📁 Found {len(pdf_paths)} PDF files")
    print(f"📊 Chunking strategy: Semantic (2000 tokens, header preservation)")
    print()
    
    # Process PDFs with semantic chunking
    chunks = chunker.process_multiple_pdfs(
        pdf_paths=[str(p) for p in pdf_paths],
        remove_footer=False  # Keep all content
    )
    
    print(f"\n✅ Created {len(chunks)} semantic chunks")
    
    # Show statistics
    stats = chunker.get_chunk_statistics(chunks)
    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Avg tokens: {stats['avg_tokens']:.1f}")
    print(f"   Min tokens: {stats['min_tokens']}")
    print(f"   Max tokens: {stats['max_tokens']}")
    print(f"   Avg words: {stats['avg_words']:.1f}")
    
    # Save chunks
    chunks_json_path = OUTPUTS_DIR / "text_chunks_semantic.json"
    chunker.save_chunks_to_json(chunks, str(chunks_json_path))
    print(f"\n💾 Saved chunks to {chunks_json_path}")
    
    # Create embeddings
    print(f"\n🔄 Creating embeddings...")
    embedding_manager = EmbeddingManager()
    embedding_manager.text_chunks = chunks
    
    # Generate embeddings
    embeddings = embedding_manager.create_embeddings(chunks)
    print(f"✅ Created {len(embeddings)} embeddings")
    
    # Save embeddings
    embedding_manager.save_embeddings()
    print(f"💾 Saved embeddings to {OUTPUTS_DIR}")
    
    print("\n" + "="*80)
    print("✅ REBUILD COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Restart backend: bash start_all.sh")
    print("2. Test with queries like 'học phí là bao nhiêu?'")
    print("3. Check if retrieval finds correct chunks")

if __name__ == "__main__":
    main()
