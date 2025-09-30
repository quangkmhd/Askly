"""
Askly Web Application Demo - Modern UI for Vietnamese RAG Chatbot
Standalone demo version without heavy dependencies
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Mock data for demo
demo_documents = []
demo_chat_history = []

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of loaded documents"""
    return jsonify({'documents': demo_documents})

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload (demo version)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Add to demo documents
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_documents.append({
        'name': file.filename,
        'timestamp': timestamp,
        'path': f'/demo/{file.filename}'
    })
    
    return jsonify({
        'success': True,
        'message': f'PDF "{file.filename}" đã được xử lý thành công!',
        'filename': file.filename
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages (demo version)"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Demo responses based on keywords
    response = generate_demo_response(message)
    
    # Add to history
    demo_chat_history.append({'role': 'user', 'content': message})
    demo_chat_history.append({'role': 'assistant', 'content': response})
    
    # Mock sources
    sources = []
    if any(keyword in message.lower() for keyword in ['tài liệu', 'pdf', 'file']):
        sources.append({
            'document': 'Tài liệu mẫu',
            'page': '1-3',
            'snippet': 'Thông tin được tổng hợp từ tài liệu'
        })
    
    return jsonify({
        'response': response,
        'sources': sources
    })

@app.route('/api/delete/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete a document from the knowledge base (demo)"""
    global demo_documents
    
    # Find and remove the document
    for i, doc in enumerate(demo_documents):
        if doc['name'] == filename:
            del demo_documents[i]
            return jsonify({'success': True, 'message': f'Đã xóa {filename}'})
    
    return jsonify({'error': 'Document not found'}), 404

def generate_demo_response(message):
    """Generate demo responses based on message content"""
    message_lower = message.lower()
    
    if 'xin chào' in message_lower or 'hello' in message_lower:
        return "Xin chào! Tôi là Askly, trợ lý AI giúp bạn tìm hiểu thông tin từ tài liệu PDF. Bạn có thể tải lên tài liệu và đặt câu hỏi về nội dung của chúng."
    
    elif 'askly là gì' in message_lower or 'giới thiệu' in message_lower:
        return """Askly là một hệ thống chatbot RAG (Retrieval-Augmented Generation) tiếng Việt được thiết kế để:
        
• **Xử lý tài liệu PDF**: Tự động phân tích và lưu trữ nội dung từ các file PDF bạn tải lên
• **Trả lời câu hỏi**: Tìm kiếm thông tin chính xác từ tài liệu để trả lời câu hỏi của bạn
• **Hỗ trợ tiếng Việt**: Được tối ưu hóa cho ngôn ngữ tiếng Việt
• **Trích dẫn nguồn**: Cung cấp thông tin về nguồn gốc của câu trả lời

Hãy tải lên tài liệu PDF và bắt đầu đặt câu hỏi!"""
    
    elif 'cách sử dụng' in message_lower or 'hướng dẫn' in message_lower:
        return """**Hướng dẫn sử dụng Askly:**

1. **Tải lên tài liệu**: Nhấn vào khu vực "Tải lên PDF mới" hoặc kéo thả file PDF vào
2. **Chờ xử lý**: Hệ thống sẽ phân tích và lưu trữ nội dung tài liệu
3. **Đặt câu hỏi**: Gõ câu hỏi của bạn vào ô chat và nhấn Enter
4. **Nhận câu trả lời**: Askly sẽ tìm kiếm thông tin từ tài liệu và trả lời

💡 *Mẹo: Câu hỏi càng cụ thể, câu trả lời càng chính xác!*"""
    
    elif 'tính năng' in message_lower:
        return """**Các tính năng chính của Askly:**

✅ **Xử lý PDF thông minh**: Tự động trích xuất và phân tích nội dung
✅ **Tìm kiếm ngữ nghĩa**: Hiểu ngữ cảnh để tìm thông tin chính xác
✅ **Lưu trữ tri thức**: Giữ lại tất cả tài liệu đã tải lên để tra cứu lâu dài
✅ **Giao diện hiện đại**: Thiết kế đẹp mắt với hiệu ứng mượt mà
✅ **Trích dẫn nguồn**: Chỉ rõ thông tin đến từ trang nào trong tài liệu"""
    
    elif any(word in message_lower for word in ['pdf', 'tài liệu', 'file']):
        if not demo_documents:
            return "Hiện tại chưa có tài liệu nào được tải lên. Vui lòng tải lên file PDF để tôi có thể trả lời câu hỏi về nội dung của chúng."
        else:
            doc_list = '\n'.join([f"• {doc['name']}" for doc in demo_documents])
            return f"Các tài liệu hiện có trong hệ thống:\n\n{doc_list}\n\nBạn có thể đặt câu hỏi về nội dung của các tài liệu này."
    
    else:
        return f"""Tôi đã nhận được câu hỏi của bạn: "{message}"

Trong phiên bản demo này, tôi chưa thể xử lý câu hỏi cụ thể về nội dung tài liệu. Trong phiên bản đầy đủ, Askly sẽ:
1. Tìm kiếm thông tin liên quan trong tài liệu đã tải lên
2. Tổng hợp và trả lời dựa trên nội dung tìm được
3. Cung cấp trích dẫn từ tài liệu gốc

Hãy thử tải lên một file PDF và đặt câu hỏi về nội dung của nó!"""

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Askly Web Demo Server")
    print("="*50)
    print("📍 Truy cập: http://localhost:5001")
    print("📝 Đây là phiên bản demo với các chức năng mô phỏng")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5001)
