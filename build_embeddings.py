"""
Script to build and save embeddings for the RAG pipeline
"""
import pandas as pd
from pathlib import Path
import logging

from processors.pdf_processor import PDFProcessor
from processors.text_processor import TextProcessor
from models.embedding_manager import EmbeddingManager
from config.config import DATA_DIR, OUTPUTS_DIR, PDF_FILENAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to build and save embeddings"""
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        text_processor = TextProcessor()
        embedding_manager = EmbeddingManager()
        
        # Process PDF
        logger.info("Processing PDF document...")
        pages_and_texts = pdf_processor.process_pdf()
        
        # Split into sentences and create chunks
        pages_with_sentences = text_processor.split_into_sentences(pages_and_texts)
        pages_with_chunks = text_processor.create_sentence_chunks(pages_with_sentences)
        
        # Create text chunks for embedding
        chunks_data = []
        for page in pages_with_chunks:
            page_num = page['page_number']
            for chunk_idx, chunk in enumerate(page['sentence_chunks']):
                chunk_text = " ".join(chunk)
                chunks_data.append({
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'sentence_chunk': chunk_text
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(chunks_data)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = df['sentence_chunk'].tolist()
        embeddings = embedding_manager.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=False  # Return numpy array
        )
        
        # Add embeddings to DataFrame
        df['embedding'] = embeddings.tolist()
        
        # Save to CSV
        output_path = OUTPUTS_DIR / "text_chunks_and_embeddings_df.csv"
        logger.info(f"Saving embeddings to {output_path}...")
        df.to_csv(output_path, index=False)
        
        logger.info("Embeddings built and saved successfully!")
        logger.info(f"Total chunks processed: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error building embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main()