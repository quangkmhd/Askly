import streamlit as st
import os
import json
import shutil
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
from rag_pipeline import RAGPipeline
from processors.pdf_processor import PDFProcessor
from config.config import UPLOADED_PDFS_DIR, EMBEDDINGS_INDEX_FILE, EMBEDDINGS_DATA_FILE, TEXT_CHUNKS_FILE

def check_data_exists():
    """Check if there is any data to load"""
    return (EMBEDDINGS_INDEX_FILE.exists() and 
            EMBEDDINGS_DATA_FILE.exists() and 
            TEXT_CHUNKS_FILE.exists())

def load_existing_data():
    """Load existing embeddings and text chunks if available"""
    try:
        if check_data_exists():
            # Load embeddings data with allow_pickle=True
            embeddings_data = np.load(EMBEDDINGS_DATA_FILE, allow_pickle=True)
            embeddings = embeddings_data['embeddings']
            
            # Ensure embeddings is a numpy array with correct dtype
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Load text chunks with proper encoding
            with open(TEXT_CHUNKS_FILE, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
                
            # Load index mapping
            with open(EMBEDDINGS_INDEX_FILE, 'r', encoding='utf-8') as f:
                embeddings_index = json.load(f)
                
            return embeddings, text_chunks, embeddings_index
    except Exception as e:
        st.warning(f"Could not load existing data: {str(e)}")
        raise  # Re-raise to see full traceback
    return None, None, {}

def save_data(embeddings, text_chunks, embeddings_index):
    """Save embeddings and text chunks to disk"""
    try:
        # Ensure directory exists
        EMBEDDINGS_DIR = EMBEDDINGS_DATA_FILE.parent
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Convert embeddings to numpy array if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        
        # Save embeddings data with compression
        np.savez_compressed(EMBEDDINGS_DATA_FILE, embeddings=embeddings)
        
        # Save text chunks with proper encoding
        with open(TEXT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)
            
        # Save index mapping with proper encoding
        with open(EMBEDDINGS_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_index, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        raise  # Re-raise to see full traceback

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
    if check_data_exists():
        try:
            embeddings, text_chunks, _ = load_existing_data()
            if embeddings is not None and text_chunks is not None:
                st.session_state.rag_pipeline.embeddings = embeddings
                st.session_state.rag_pipeline.text_chunks = text_chunks
                st.session_state.rag_pipeline.setup_pipeline(load_existing_embeddings=True)
        except Exception as e:
            st.error(f"Error loading existing data: {str(e)}")

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and update the RAG pipeline"""
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = uploaded_file.name.replace(" ", "_")
    pdf_filename = f"{timestamp}_{safe_filename}"
    pdf_path = UPLOADED_PDFS_DIR / pdf_filename
    
    try:
        # Create directory if it doesn't exist
        UPLOADED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        with st.spinner('Processing PDF...'):
            if not st.session_state.rag_pipeline.is_initialized:
                # First PDF - initialize the pipeline
                st.session_state.rag_pipeline = RAGPipeline(pdf_path)
                st.session_state.rag_pipeline.setup_pipeline(load_existing_embeddings=True)
                
                # Update index with new file
                embeddings_index = {
                    str(pdf_path): {
                        'original_name': uploaded_file.name,
                        'timestamp': timestamp
                    }
                }
                
            else:
                # Additional PDF - update existing knowledge base
                pdf_processor = PDFProcessor(pdf_path)
                pages_and_texts = pdf_processor.process_pdf()
                text_chunks = st.session_state.rag_pipeline.text_processor.process_text(pages_and_texts)
                new_embeddings = st.session_state.rag_pipeline.embedding_manager.create_embeddings(text_chunks)
                
                # Load index or create new one
                try:
                    with open(EMBEDDINGS_INDEX_FILE, 'r', encoding='utf-8') as f:
                        embeddings_index = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    embeddings_index = {}
                
                # Combine with existing embeddings
                if isinstance(st.session_state.rag_pipeline.embeddings, np.ndarray):
                    combined_embeddings = np.vstack([st.session_state.rag_pipeline.embeddings, new_embeddings])
                else:
                    combined_embeddings = new_embeddings
                    
                st.session_state.rag_pipeline.embeddings = combined_embeddings
                st.session_state.rag_pipeline.text_chunks.extend(text_chunks)
                
                # Update retrieval system
                st.session_state.rag_pipeline.retrieval_system.update_embeddings(
                    st.session_state.rag_pipeline.embeddings,
                    st.session_state.rag_pipeline.text_chunks
                )
                
                # Update index
                embeddings_index[str(pdf_path)] = {
                    'original_name': uploaded_file.name,
                    'timestamp': timestamp
                }
            
            # Save updated data
            save_data(
                st.session_state.rag_pipeline.embeddings,
                st.session_state.rag_pipeline.text_chunks,
                embeddings_index
            )
            
            st.success(f"PDF '{uploaded_file.name}' successfully processed and added to the knowledge base!")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if pdf_path.exists():
            pdf_path.unlink()  # Delete the file if processing failed
        raise  # Re-raise to see full traceback

def get_answer(question):
    """Get answer from RAG pipeline"""
    try:
        if not st.session_state.rag_pipeline.is_initialized:
            if not check_data_exists():
                return "Vui lòng tải lên tài liệu PDF trước khi đặt câu hỏi."
            
            try:
                embeddings, text_chunks, _ = load_existing_data()
                if embeddings is not None and text_chunks is not None:
                    st.session_state.rag_pipeline.embeddings = embeddings
                    st.session_state.rag_pipeline.text_chunks = text_chunks
                    st.session_state.rag_pipeline.setup_pipeline(load_existing_embeddings=True)
                else:
                    return "Có lỗi khi tải dữ liệu hiện có. Vui lòng tải lên tài liệu PDF mới."
            except Exception as e:
                return f"Lỗi khi khởi tạo hệ thống: {str(e)}"
        
        answer = st.session_state.rag_pipeline.ask(question)
        return answer
    except Exception as e:
        if "Pipeline not initialized" in str(e):
            return "Hệ thống chưa sẵn sàng. Vui lòng tải lên tài liệu PDF và đợi xử lý hoàn tất."
        return f"Lỗi khi sinh câu trả lời: {str(e)}"

# Streamlit UI
st.title("RAG Chatbot with PDF Upload")

# PDF upload section
st.subheader("Upload New PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    if st.button("Process PDF"):
        process_pdf(uploaded_file)

# Chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about the uploaded documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and display assistant response
    with st.chat_message("assistant"):
        answer = get_answer(question)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG (Retrieval Augmented Generation) chatbot that can:
    - Process and learn from uploaded PDF documents
    - Answer questions based on the document content
    - Maintain chat history during the session
    - Save documents and knowledge for future use
    """)
    
    # Show loaded documents
    st.subheader("Loaded Documents")
    try:
        if EMBEDDINGS_INDEX_FILE.exists():
            with open(EMBEDDINGS_INDEX_FILE, 'r', encoding='utf-8') as f:
                embeddings_index = json.load(f)
            if embeddings_index:
                for path, info in embeddings_index.items():
                    st.write(f"📄 {info['original_name']}")
                    st.caption(f"Added: {info['timestamp']}")
            else:
                st.write("No documents loaded yet")
        else:
            st.write("No documents loaded yet")
    except Exception as e:
        st.write(f"Error loading document list: {str(e)}")