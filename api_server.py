"""
Flask API Server để kết nối React frontend với RAG pipeline
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import RAGPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize RAG pipeline
pipeline = None

def init_pipeline():
    """Initialize RAG pipeline on first request"""
    global pipeline
    if pipeline is None:
        try:
            logger.info("Initializing RAG pipeline...")
            pipeline = RAGPipeline()
            success = pipeline.setup_pipeline(load_existing_embeddings=True)
            if success:
                logger.info("✅ RAG pipeline initialized successfully!")
                return True
            else:
                logger.error("❌ Failed to load embeddings")
                return False
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}")
            return False
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "pipeline_loaded": pipeline is not None
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main endpoint to ask questions
    Expected JSON body:
    {
        "query": "câu hỏi của bạn",
        "n_resources": 5  # optional
    }
    """
    try:
        # Initialize pipeline if not already done
        if not init_pipeline():
            return jsonify({
                "error": "Failed to initialize RAG pipeline",
                "answer": "⚠️ Không thể khởi động RAG pipeline. Vui lòng kiểm tra embeddings."
            }), 500
        
        # Get request data
        data = request.get_json()
        query = data.get('query', '')
        n_resources = data.get('n_resources', 5)
        
        if not query:
            return jsonify({
                "error": "Query is required",
                "answer": "❌ Vui lòng nhập câu hỏi"
            }), 400
        
        logger.info(f"Processing query: {query}")
        
        # Call RAG pipeline
        answer = pipeline.ask(
            query=query,
            n_resources=n_resources,
            return_context=False
        )
        
        # Format response
        response = {
            "answer": answer,
            "query": query,
            "sources": []  # Can be extended to return source documents
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "error": str(e),
            "answer": f"❌ Lỗi khi xử lý câu hỏi: {e}"
        }), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear chat history (if needed)"""
    return jsonify({"status": "ok", "message": "History cleared"})

if __name__ == '__main__':
    print("🚀 Starting FPTU Chatbot API Server...")
    print("📍 Server will run on http://localhost:8000")
    print("💡 Make sure embeddings are built: python run.py --build")
    print("\n⚡ Loading embeddings on startup for faster responses...")
    
    # Pre-load embeddings on startup
    if init_pipeline():
        print("✅ Embeddings loaded successfully!")
    else:
        print("⚠️  Failed to load embeddings. Will retry on first request.")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
