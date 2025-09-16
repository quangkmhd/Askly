 #!/usr/bin/env python3
"""
Main entry point for the RAG pipeline
"""
import argparse
import sys
from pathlib import Path

from rag_pipeline import RAGPipeline
from config.config import PDF_URL, PDF_PATH


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Pipeline - Local Retrieval Augmented Generation")
    
    parser.add_argument(
        "--mode", 
        choices=["interactive", "demo", "single"], 
        default="interactive",
        help="Run mode: interactive, demo, or single question"
    )
    
    parser.add_argument(
        "--question", 
        type=str,
        help="Single question to ask (for single mode)"
    )
    
    parser.add_argument(
        "--pdf-url", 
        type=str,
        default=PDF_URL,
        help="URL to download PDF from"
    )
    
    parser.add_argument(
        "--pdf-path", 
        type=Path,
        default=PDF_PATH,
        help="Path to PDF file"
    )
    
    parser.add_argument(
        "--load-existing", 
        action="store_true",
        help="Load existing embeddings if available"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(pdf_url=args.pdf_url, pdf_path=args.pdf_path)
    
    # Setup pipeline
    if not pipeline.setup_pipeline(load_existing_embeddings=args.load_existing):
        print("Failed to setup pipeline. Exiting.")
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "interactive":
        pipeline.interactive_mode()
    
    elif args.mode == "demo":
        pipeline.demo_questions()
    
    elif args.mode == "single":
        if not args.question:
            print("Error: --question is required for single mode")
            sys.exit(1)
        
        print(f"Question: {args.question}")
        print("-" * 50)
        
        try:
            print(f"🤖 Bot: ", end='', flush=True)
            answer = pipeline.ask(
                query=args.question,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                stream=True
            )
            print()  # New line after streaming completes
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#test