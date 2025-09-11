#!/usr/bin/env python3
"""
Test script for Askly chatbot functionality.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from askly import AsklyBot


def test_basic_functionality():
    """Test basic chatbot functionality."""
    print("🧪 Bắt đầu kiểm tra chức năng cơ bản...")
    
    # Initialize chatbot
    bot = AsklyBot()
    
    # Test 1: Load a document
    print("\n1️⃣ Kiểm tra tải tài liệu...")
    test_file = "examples/documents/ai_education.txt"
    
    if not os.path.exists(test_file):
        print(f"❌ File kiểm tra không tồn tại: {test_file}")
        return False
    
    result = bot.load_document(test_file)
    if result['status'] != 'success':
        print(f"❌ Lỗi khi tải tài liệu: {result['message']}")
        return False
    
    print(f"✅ Đã tải thành công: {result['document_info']['filename']}")
    print(f"   Số từ: {result['document_info']['word_count']}")
    
    # Test 2: Ask a question
    print("\n2️⃣ Kiểm tra hỏi đáp...")
    question = "AI trong giáo dục có lợi ích gì?"
    result = bot.ask_question(question)
    
    if result['status'] != 'success':
        print(f"❌ Lỗi khi hỏi đáp: {result.get('answer', 'Không có thông tin')}")
        return False
    
    print(f"✅ Câu hỏi: {question}")
    print(f"   Trả lời: {result['answer']}")
    print(f"   Độ tin cậy: {result['confidence']:.2f}")
    
    # Test 3: Extract information
    print("\n3️⃣ Kiểm tra trích xuất thông tin...")
    query = "chatbot"
    result = bot.extract_information(query)
    
    if result['status'] != 'success':
        print(f"❌ Lỗi khi trích xuất: {result['message']}")
        return False
    
    print(f"✅ Truy vấn: {query}")
    print(f"   Tìm thấy: {result['total_results']} kết quả")
    
    if result['results']:
        print(f"   Kết quả đầu tiên: {result['results'][0]['text'][:100]}...")
    
    # Test 4: Get summary
    print("\n4️⃣ Kiểm tra tóm tắt...")
    result = bot.get_document_summary()
    
    if result['status'] != 'success':
        print(f"❌ Lỗi khi lấy tóm tắt: {result['message']}")
        return False
    
    summary = result['summary']
    print(f"✅ Tổng số tài liệu: {summary['total_documents']}")
    print(f"   Tổng số từ: {summary['total_words']}")
    
    print("\n🎉 Tất cả kiểm tra đều thành công!")
    return True


def test_document_loading():
    """Test loading multiple documents."""
    print("\n🧪 Kiểm tra tải nhiều tài liệu...")
    
    bot = AsklyBot()
    
    # Test loading directory
    docs_dir = "examples/documents"
    if os.path.exists(docs_dir):
        result = bot.load_documents_from_directory(docs_dir)
        
        if result['status'] == 'success':
            print(f"✅ Đã tải {len(result['loaded_files'])} tài liệu từ thư mục")
            print(f"   Các file: {', '.join(result['loaded_files'])}")
            
            # Test with multiple documents
            question = "Phát triển bền vững là gì?"
            result = bot.ask_question(question)
            
            if result['status'] == 'success':
                print(f"✅ Câu hỏi với nhiều tài liệu thành công")
                print(f"   Trả lời: {result['answer']}")
            else:
                print(f"❌ Lỗi khi hỏi với nhiều tài liệu: {result['answer']}")
        else:
            print(f"❌ Lỗi khi tải thư mục: {result['message']}")
    else:
        print(f"❌ Thư mục tài liệu không tồn tại: {docs_dir}")


def test_error_handling():
    """Test error handling."""
    print("\n🧪 Kiểm tra xử lý lỗi...")
    
    bot = AsklyBot()
    
    # Test asking question without documents
    result = bot.ask_question("Test question")
    if "tài liệu" in result['answer'].lower():
        print("✅ Xử lý lỗi khi chưa tải tài liệu: OK")
    else:
        print("❌ Xử lý lỗi không đúng")
    
    # Test loading non-existent file
    result = bot.load_document("non_existent_file.txt")
    if result['status'] == 'error':
        print("✅ Xử lý lỗi file không tồn tại: OK")
    else:
        print("❌ Không phát hiện file không tồn tại")


def main():
    """Run all tests."""
    print("🚀 BẮT ĐẦU KIỂM TRA ASKLY CHATBOT")
    print("=" * 50)
    
    try:
        # Run basic tests
        if not test_basic_functionality():
            print("❌ Kiểm tra cơ bản thất bại")
            return False
        
        # Test multiple documents
        test_document_loading()
        
        # Test error handling
        test_error_handling()
        
        print("\n🎊 TẤT CẢ KIỂM TRA HOÀN THÀNH!")
        return True
        
    except Exception as e:
        print(f"\n💥 Lỗi trong quá trình kiểm tra: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)