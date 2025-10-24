#!/usr/bin/env python3
"""
Main entry point for Askly RAG system
Choose to build embeddings or run interactive Q&A
"""
import sys
import os
from pathlib import Path

from config.config import EMBEDDINGS_CSV_PATH, EMBEDDINGS_DIR

def check_embeddings_exist():
    """Check if embeddings have been built"""
    # Check for both numpy binary (.npy) and CSV formats
    embeddings_base = str(EMBEDDINGS_CSV_PATH).replace('.csv', '')
    npy_file = Path(f"{embeddings_base}.npy")
    chunks_file = Path(f"{embeddings_base}_chunks.json")
    csv_file = EMBEDDINGS_CSV_PATH
    
    return npy_file.exists() or chunks_file.exists() or csv_file.exists()

def build_embeddings():
    """Build embeddings from PDFs"""
    print("\n" + "=" * 80)
    print("BUILDING EMBEDDINGS FROM PDFs")
    print("=" * 80)
    
    # Use rebuild_clean_database module with semantic chunking and OCR
    from scripts.rebuild_clean_database import main as build_main
    build_main()

def run_interactive():
    """Run interactive Q&A"""
    print("\n" + "=" * 80)
    print("ASKLY RAG SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    
    from rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    print("\n[1/2] Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Setup pipeline
    print("\n[2/2] Loading embeddings and LLM model...")
    success = pipeline.setup_pipeline(load_existing_embeddings=True)
    
    if not success:
        print("\n[ERROR] Failed to setup pipeline!")
        print("Try running: python run.py --build")
        return
    
    print("\n" + "=" * 80)
    print("Pipeline ready! Ask your questions (type 'quit' to exit)")
    print("=" * 80)
    
    # Initialize chat history for context
    chat_history = []
    
    # Interactive loop
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            print("\n[Searching and generating answer...]")
            answer = pipeline.ask(
                query=query,
                chat_history=chat_history,  # Pass conversation context
                temperature=0.7,
                max_new_tokens=500,
                n_resources=10
            )
            
            print("\n" + "-" * 80)
            print("Answer:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            # Update chat history (keep last 6 messages = 3 turns)
            chat_history.append({'role': 'user', 'content': query})
            chat_history.append({'role': 'assistant', 'content': answer})
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]  # Keep only last 3 turns
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--build', '-b']:
            build_embeddings()
            return
        elif sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python run.py           # Run interactive Q&A")
            print("  python run.py --build   # Build embeddings from PDFs")
            print("  python run.py --help    # Show this help")
            return
    
    # Check if embeddings exist
    if not check_embeddings_exist():
        print("\n[WARNING] No embeddings found!")
        print("You need to build embeddings first.")
        print("\nRun: python run.py --build")
        
        response = input("\nBuild embeddings now? (y/n): ").strip().lower()
        if response == 'y':
            build_embeddings()
            print("\n[SUCCESS] Embeddings built! Starting interactive mode...")
            run_interactive()
        return
    
    # Run interactive mode
    run_interactive()

if __name__ == "__main__":
    main()
