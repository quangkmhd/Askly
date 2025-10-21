#!/usr/bin/env python3
"""
Main entry point for Askly RAG system
Choose to build embeddings or run interactive Q&A
"""
import sys
import os
from pathlib import Path

def check_embeddings_exist():
    """Check if embeddings have been built"""
    embeddings_file = Path("data/embeddings/text_chunks.json")
    csv_file = Path("outputs/text_chunks_and_embeddings_df.csv")
    return embeddings_file.exists() or csv_file.exists()

def build_embeddings():
    """Build embeddings from PDFs"""
    print("\n" + "=" * 80)
    print("BUILDING EMBEDDINGS FROM PDFs")
    print("=" * 80)
    
    # Use rebuild_embeddings_semantic module with semantic chunking
    from rebuild_embeddings_semantic import main as build_main
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
                temperature=0.7,
                max_new_tokens=500,
                n_resources=10
            )
            
            print("\n" + "-" * 80)
            print("Answer:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
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
