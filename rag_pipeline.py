"""
Main RAG pipeline module
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from processors.pdf_processor import PDFProcessor
from processors.text_processor import TextProcessor
from models.embedding_manager import EmbeddingManager
from models.retrieval_system import RetrievalSystem
from models.llm_manager import LLMManager
from config.config import EMBEDDINGS_CSV_PATH, DEFAULT_N_RESOURCES_TO_RETURN, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS
from utils.utils import print_wrapped


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""
    
    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_processor = PDFProcessor(pdf_path)
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.retrieval_system = None
        self.llm_manager = LLMManager()
        
        # Pipeline state
        self.is_initialized = False
        self.text_chunks = None
        self.embeddings = None
    
    def setup_pipeline(self, load_existing_embeddings: bool = True) -> bool:
        """Setup the complete RAG pipeline with incremental PDF processing"""
        try:
            print("[INFO] Setting up RAG pipeline...")
            
            # Always try to load existing embeddings first
            if load_existing_embeddings and EMBEDDINGS_CSV_PATH.exists():
                print("[INFO] Loading existing embeddings...")
                self.embeddings, self.text_chunks = self.embedding_manager.load_embeddings()
            else:
                print("[INFO] No existing embeddings found")
                self.embeddings = None
                self.text_chunks = []
            
            # Check for unprocessed PDFs to add incrementally
            unprocessed_pdfs = self._get_unprocessed_pdfs()
            
            if unprocessed_pdfs:
                print(f"[INFO] Found {len(unprocessed_pdfs)} unprocessed PDF(s)")
                for pdf_path in unprocessed_pdfs:
                    print(f"[INFO] Processing new PDF: {pdf_path.name}")
                    self._process_and_append_pdf(pdf_path)
            elif self.embeddings is None:
                # No existing embeddings and no PDFs to process
                print("[WARNING] No embeddings or PDFs found to process")
                return False
            else:
                print("[INFO] All PDFs already processed, using existing embeddings")
                self.embeddings = self.embedding_manager.create_embeddings(self.text_chunks)
                
                # Save embeddings
                self.embedding_manager.save_embeddings()
            
            # Initialize retrieval system
            # Pass the EmbeddingManager instance (has .encode) rather than the raw TF-Hub module
            self.retrieval_system = RetrievalSystem(
                embedding_model=self.embedding_manager,
                embeddings=self.embeddings,
                text_chunks=self.text_chunks
            )
            
            # Load LLM
            self.llm_manager.load_model()
            
            self.is_initialized = True
            print("[INFO] RAG pipeline setup complete!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to setup pipeline: {e}")
            return False
    
    def ask(self, query: str, temperature: float = DEFAULT_TEMPERATURE,
           max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
           n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN,
           return_context: bool = False, stream: bool = False,
           chat_history: List[Dict[str, str]] = None) -> str:
        """
        Ask a question to the RAG system
        Returns the answer or (answer, context_items) if return_context=True
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
        
        # Retrieve relevant context
        context_items = self.retrieval_system.get_context_items(query, n_resources)
        
        # Generate response
        answer = self.llm_manager.generate_rag_response(
            query=query,
            context_items=context_items,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=stream,
            chat_history=chat_history
        )
        
        if return_context:
            return answer, context_items
        return answer
    
    def search(self, query: str, n_results: int = DEFAULT_N_RESOURCES_TO_RETURN,
              print_results: bool = True) -> List[Dict[str, Any]]:
        """Search for relevant documents without generating an answer"""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
        
        results = self.retrieval_system.get_context_items(query, n_results)
        
        if print_results:
            self.retrieval_system.print_top_results(query, n_results)
        
        return results
    
    def batch_ask(self, queries: List[str], temperature: float = DEFAULT_TEMPERATURE,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 n_resources: int = DEFAULT_N_RESOURCES_TO_RETURN) -> List[str]:
        """Ask multiple questions in batch"""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
        
        answers = []
        for query in queries:
            answer = self.ask(query, temperature, max_new_tokens, n_resources)
            answers.append(answer)
        
        return answers
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "num_text_chunks": len(self.text_chunks) if self.text_chunks else 0,
            "embedding_stats": self.embedding_manager.get_embedding_stats(),
            "llm_info": self.llm_manager.get_model_info()
        }
        
        return info
    
    def save_pipeline_state(self, file_path: Optional[Path] = None) -> str:
        """Save the current pipeline state"""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        if file_path is None:
            file_path = EMBEDDINGS_CSV_PATH
        
        return self.embedding_manager.save_embeddings(file_path)
    
    def load_pipeline_state(self, file_path: Optional[Path] = None) -> bool:
        """Load pipeline state from file"""
        try:
            self.embeddings, self.text_chunks = self.embedding_manager.load_embeddings(file_path)
            
            # Reinitialize retrieval system
            self.retrieval_system = RetrievalSystem(
                embedding_model=self.embedding_manager,
                embeddings=self.embeddings,
                text_chunks=self.text_chunks
            )
            
            # Load LLM if not already loaded
            if self.llm_manager.model is None:
                self.llm_manager.load_model()
            
            self.is_initialized = True
            print("[INFO] Pipeline state loaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load pipeline state: {e}")
            return False
    
    def interactive_mode(self):
        """Run the pipeline in interactive mode"""
        if not self.is_initialized:
            print("[ERROR] Pipeline not initialized. Call setup_pipeline() first.")
            return
        
        print("\n" + "="*50)
        print("RAG Pipeline - Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*50)
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'info':
                    self._print_pipeline_info()
                    continue
                elif not query:
                    continue
                
                # Ask question with streaming
                try:
                    print(f"\n🤖 Bot: ", end='', flush=True)
                    answer = self.ask(query, stream=True)
                    print()  # New line after streaming completes
                except Exception as e:
                    print(f"[ERROR] Exception in ask(): {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
    
    def _get_unprocessed_pdfs(self) -> List[Path]:
        """Get list of PDFs that haven't been processed yet"""
        from pathlib import Path
        import json
        
        # Find all PDFs in uploaded_pdfs directory
        uploaded_pdfs_dir = Path("data/uploaded_pdfs")
        if not uploaded_pdfs_dir.exists():
            return []
        
        pdf_files = list(uploaded_pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            return []
        
        # Load processed PDFs tracking file
        processed_pdfs_file = Path("data/processed_pdfs.json")
        processed_pdfs = set()
        
        if processed_pdfs_file.exists():
            try:
                with open(processed_pdfs_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                    processed_pdfs = set(processed_data.get('processed_files', []))
            except (json.JSONDecodeError, KeyError):
                processed_pdfs = set()
        
        # Find unprocessed PDFs
        unprocessed_pdfs = []
        for pdf_file in pdf_files:
            pdf_key = f"{pdf_file.name}_{pdf_file.stat().st_mtime}"
            if pdf_key not in processed_pdfs:
                unprocessed_pdfs.append(pdf_file)
        
        return unprocessed_pdfs
    
    def _mark_pdf_as_processed(self, pdf_path: Path):
        """Mark a PDF as processed"""
        import json
        
        processed_pdfs_file = Path("data/processed_pdfs.json")
        processed_data = {'processed_files': []}
        
        # Load existing data
        if processed_pdfs_file.exists():
            try:
                with open(processed_pdfs_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                processed_data = {'processed_files': []}
        
        # Add new PDF
        pdf_key = f"{pdf_path.name}_{pdf_path.stat().st_mtime}"
        if pdf_key not in processed_data['processed_files']:
            processed_data['processed_files'].append(pdf_key)
        
        # Save updated data
        processed_pdfs_file.parent.mkdir(exist_ok=True)
        with open(processed_pdfs_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    def _process_and_append_pdf(self, pdf_path: Path):
        """Process a single PDF and append its embeddings to existing ones"""
        import numpy as np
        
        try:
            # Create temporary PDF processor for this specific PDF
            temp_pdf_processor = PDFProcessor(pdf_path)
            
            # Process PDF
            pages_and_texts = temp_pdf_processor.process_pdf()
            
            # Process text
            new_text_chunks = self.text_processor.process_text(pages_and_texts)
            
            # Create embeddings for new chunks
            new_embeddings = self.embedding_manager.create_embeddings(new_text_chunks)
            
            # Append to existing data
            if self.embeddings is not None and len(self.text_chunks) > 0:
                # Combine embeddings
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
                # Combine text chunks
                self.text_chunks.extend(new_text_chunks)
                print(f"[INFO] Appended {len(new_text_chunks)} chunks to existing {len(self.text_chunks) - len(new_text_chunks)} chunks")
            else:
                # First PDF
                self.embeddings = new_embeddings
                self.text_chunks = new_text_chunks
                print(f"[INFO] Created initial embeddings with {len(new_text_chunks)} chunks")
            
            # Save updated embeddings
            self.embedding_manager.embeddings = self.embeddings
            self.embedding_manager.text_chunks = self.text_chunks
            self.embedding_manager.save_embeddings()
            
            # Mark PDF as processed
            self._mark_pdf_as_processed(pdf_path)
            
            print(f"[INFO] Successfully processed and saved: {pdf_path.name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process PDF {pdf_path.name}: {e}")
            raise
    
    def _print_help(self):
        """Print help information"""
        print("\nAvailable commands:")
        print("  help  - Show this help message")
        print("  info  - Show pipeline information")
        print("  quit  - Exit the program")
        print("\nJust type your question to get an answer!")
    
    def _print_pipeline_info(self):
        """Print pipeline information"""
        info = self.get_pipeline_info()
        print(f"\nPipeline Status: {info['status']}")
        print(f"Text Chunks: {info['num_text_chunks']}")
        
        if 'embedding_stats' in info:
            stats = info['embedding_stats']
            print(f"Embeddings: {stats.get('num_embeddings', 0)} vectors")
            print(f"Embedding Dimension: {stats.get('embedding_dim', 0)}")
        
        if 'llm_info' in info:
            llm_info = info['llm_info']
            print(f"LLM Model: {llm_info.get('model_id', 'Unknown')}")
            print(f"Parameters: {llm_info.get('num_parameters', 0):,}")
    
    def demo_questions(self):
        """Run a demo with predefined questions"""
        demo_questions = [
            "What are the macronutrients and what are their functions in the body?",
            "How do vitamins and minerals differ in their roles and importance for health?",
            "What role does fibre play in digestion? Name five fibre containing foods.",
            "Explain the concept of energy balance and its importance in weight management.",
            "What are symptoms of pellagra?",
            "How does saliva help with digestion?",
            "What is the RDI for protein per day?",
            "What are water soluble vitamins?"
        ]
        
        print("\n" + "="*60)
        print("RAG Pipeline - Demo Mode")
        print("="*60)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 60)
            
            try:
                answer = self.ask(question)
                print(f"Answer: {answer}")
            except Exception as e:
                print(f"Error: {e}")
            
            if i < len(demo_questions):
                input("\nPress Enter to continue to next question...")
        
        print("\nDemo completed!")
