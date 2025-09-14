#!/usr/bin/env python3
"""
Simple script to run the RAG pipeline
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_pipeline import RAGPipeline


def main():
    """Simple main function for quick testing"""
    print("Starting RAG Pipeline...")
    
    # Initialize and setup pipeline
    pipeline = RAGPipeline()
    
    if not pipeline.setup_pipeline():
        print("Failed to setup pipeline")
        return
    
    print("Pipeline ready! Starting interactive mode...")
    pipeline.interactive_mode()


if __name__ == "__main__":
    main()
