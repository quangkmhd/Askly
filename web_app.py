"""
Askly Web Application - Modern UI for Vietnamese RAG Chatbot
Based on the design specifications from the PDF document
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from rag_pipeline import RAGPipeline
from processors.pdf_processor import PDFProcessor
from config.config import UPLOADED_PDFS_DIR, EMBEDDINGS_INDEX_FILE, EMBEDDINGS_DATA_FILE, TEXT_CHUNKS_FILE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(UPLOADED_PDFS_DIR)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()
chat_history = []

def check_data_exists():
    """Check if there is any data to load"""
    return (EMBEDDINGS_INDEX_FILE.exists() and 
            EMBEDDINGS_DATA_FILE.exists() and 
            TEXT_CHUNKS_FILE.exists())

def load_existing_data():
    """Load existing embeddings and text chunks if available"""
    try:
        if check_data_exists():
            embeddings_data = np.load(EMBEDDINGS_DATA_FILE, allow_pickle=True)
            embeddings = np.array(embeddings_data['embeddings'], dtype=np.float32)
            with open(TEXT_CHUNKS_FILE, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            with open(EMBEDDINGS_INDEX_FILE, 'r', encoding='utf-8') as f:
                embeddings_index = json.load(f)
            return embeddings, text_chunks, embeddings_index
    except Exception as e:
        print(f"Error loading existing data: {e}")
    return None, None, {}

def save_data(embeddings, text_chunks, embeddings_index):
    """Save embeddings and text chunks to disk"""
    try:
        EMBEDDINGS_DIR = EMBEDDINGS_DATA_FILE.parent
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        np.savez_compressed(EMBEDDINGS_DATA_FILE, embeddings=embeddings)
        with open(TEXT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)
        with open(EMBEDDINGS_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def initialize_pipeline():
    """Initialize or load RAG pipeline"""
    global rag_pipeline
    if not rag_pipeline.is_initialized and check_data_exists():
        print("Loading existing data into pipeline...")
        try:
            embeddings, text_chunks, _ = load_existing_data()
            if embeddings is not None and text_chunks is not None:
                rag_pipeline.embeddings = embeddings
                rag_pipeline.text_chunks = text_chunks
                rag_pipeline.setup_pipeline(load_existing_embeddings=True)
                print("Pipeline initialized successfully with existing data.")
        except Exception as e:
            print(f"Error initializing pipeline: {e}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of loaded documents"""
    _, _, embeddings_index = load_existing_data()
    documents = []
    for path, info in embeddings_index.items():
        documents.append({
            'name': info['original_name'],
            'timestamp': info['timestamp'],
            'path': path
        })
    return jsonify({'documents': documents})

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save the file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = secure_filename(file.filename)
        pdf_filename = f"{timestamp}_{safe_filename}"
        pdf_path = Path(app.config['UPLOAD_FOLDER']) / pdf_filename
        
        UPLOADED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
        file.save(str(pdf_path))
        
        # Process the PDF
        if not rag_pipeline.is_initialized:
            rag_pipeline.setup_pipeline(load_existing_embeddings=True)
        
        pdf_processor = PDFProcessor(pdf_path)
        pages_and_texts = pdf_processor.process_pdf()
        text_chunks = rag_pipeline.text_processor.process_text(pages_and_texts)
        new_embeddings = rag_pipeline.embedding_manager.create_embeddings(text_chunks)
        
        embeddings, _, embeddings_index = load_existing_data()
        
        if embeddings is not None and len(embeddings) > 0:
            combined_embeddings = np.vstack([embeddings, new_embeddings])
            rag_pipeline.text_chunks.extend(text_chunks)
        else:
            combined_embeddings = new_embeddings
            rag_pipeline.text_chunks = text_chunks
        
        rag_pipeline.embeddings = combined_embeddings
        rag_pipeline.retrieval_system.update_embeddings(rag_pipeline.embeddings, rag_pipeline.text_chunks)
        
        embeddings_index[str(pdf_path)] = {
            'original_name': file.filename,
            'timestamp': timestamp
        }
        save_data(rag_pipeline.embeddings, rag_pipeline.text_chunks, embeddings_index)
        
        return jsonify({
            'success': True,
            'message': f'PDF "{file.filename}" đã được xử lý thành công!',
            'filename': file.filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global chat_history
    
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if not rag_pipeline.is_initialized:
        if not check_data_exists():
            return jsonify({
                'response': 'Vui lòng tải lên tài liệu PDF trước khi đặt câu hỏi.',
                'sources': []
            })
        
        try:
            initialize_pipeline()
        except Exception as e:
            return jsonify({'error': f'Error initializing system: {str(e)}'}), 500
    
    try:
        # Add user message to history
        chat_history.append({'role': 'user', 'content': message})
        
        # Get response from RAG pipeline
        response = rag_pipeline.ask(message, chat_history=chat_history)
        
        # Add assistant response to history
        chat_history.append({'role': 'assistant', 'content': response})
        
        # Extract sources (mock for now, you can enhance this)
        sources = []
        if 'trang' in response.lower() or 'page' in response.lower():
            sources.append({
                'document': 'Tài liệu đã tải lên',
                'page': 'Nhiều trang',
                'snippet': 'Thông tin được tổng hợp từ tài liệu'
            })
        
        return jsonify({
            'response': response,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete a document from the knowledge base"""
    # This is a simplified version - in production you'd need to properly remove embeddings
    try:
        _, _, embeddings_index = load_existing_data()
        
        # Find and remove the document
        for path in list(embeddings_index.keys()):
            if embeddings_index[path]['original_name'] == filename:
                del embeddings_index[path]
                # In a real implementation, you'd also need to remove the corresponding embeddings
                save_data(rag_pipeline.embeddings, rag_pipeline.text_chunks, embeddings_index)
                return jsonify({'success': True, 'message': f'Đã xóa {filename}'})
        
        return jsonify({'error': 'Document not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_pipeline()
    app.run(debug=True, port=5001)
