#!/usr/bin/env python3
"""
Command-line interface for Askly chatbot.
"""

import sys
import os
import argparse
from askly import AsklyBot


def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("🤖 ASKLY - Chatbot Trích Xuất Thông Tin & Hỏi Đáp")
    print("=" * 60)
    print("Hỗ trợ các định dạng: .txt, .pdf, .docx")
    print("Nhập 'help' để xem danh sách lệnh, 'quit' để thoát")
    print()


def print_help():
    """Print available commands."""
    print("\n📋 DANH SÁCH LỆNH:")
    print("-" * 40)
    print("load <đường_dẫn_file>     - Tải một tài liệu")
    print("load_dir <đường_dẫn_thư_mục> - Tải tất cả tài liệu từ thư mục")
    print("ask <câu_hỏi>             - Đặt câu hỏi")
    print("extract <truy_vấn>        - Trích xuất thông tin")
    print("summary                   - Xem tóm tắt tài liệu đã tải")
    print("status                    - Xem trạng thái chatbot")
    print("clear                     - Xóa tất cả tài liệu")
    print("help                      - Hiển thị hướng dẫn")
    print("quit                      - Thoát chương trình")
    print()


def format_response(response):
    """Format response for better display."""
    if response['status'] == 'error':
        print(f"❌ LỖI: {response.get('message', 'Có lỗi xảy ra')}")
        return
    
    if 'answer' in response:
        # Q&A response
        print(f"💭 CÂU HỎI: {response['question']}")
        print(f"🤖 TRẢ LỜI: {response['answer']}")
        print(f"📊 ĐỘ TIN CẬY: {response['confidence']:.2f}")
        if response['sources']:
            print(f"📄 NGUỒN: {', '.join(response['sources'])}")
    
    elif 'results' in response:
        # Information extraction response
        print(f"🔍 TRUY VẤN: {response['query']}")
        print(f"📊 TÌM THẤY: {response['total_results']} kết quả")
        
        for i, result in enumerate(response['results'][:3], 1):  # Show top 3
            print(f"\n{i}. 📄 {result['source']} (Điểm: {result['relevance_score']:.2f})")
            print(f"   📝 {result['text'][:200]}...")
    
    elif 'summary' in response:
        # Summary response
        summary = response['summary']
        if 'total_documents' in summary:
            print(f"📚 TỔNG SỐ TÀI LIỆU: {summary['total_documents']}")
            print(f"📄 TỔNG SỐ ĐOẠN VĂN: {summary['total_chunks']}")
            print(f"📝 TỔNG SỐ TỪ: {summary['total_words']}")
            print("\n📋 DANH SÁCH TÀI LIỆU:")
            for doc in summary['loaded_documents']:
                print(f"  - {doc['filename']} ({doc['file_type']}, {doc['word_count']} từ)")
    
    else:
        # Generic response
        print(f"✅ {response.get('message', 'Thành công')}")


def interactive_mode():
    """Run chatbot in interactive mode."""
    global bot
    print_header()
    
    while True:
        try:
            user_input = input("Askly> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            # Parse command
            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == 'load':
                if not args:
                    print("❌ Vui lòng cung cấp đường dẫn file: load <đường_dẫn_file>")
                    continue
                
                print("⏳ Đang tải tài liệu...")
                response = bot.load_document(args)
                format_response(response)
            
            elif command == 'load_dir':
                if not args:
                    print("❌ Vui lòng cung cấp đường dẫn thư mục: load_dir <đường_dẫn_thư_mục>")
                    continue
                
                print("⏳ Đang tải tài liệu từ thư mục...")
                response = bot.load_documents_from_directory(args)
                format_response(response)
            
            elif command == 'ask':
                if not args:
                    print("❌ Vui lòng nhập câu hỏi: ask <câu_hỏi>")
                    continue
                
                print("🤔 Đang xử lý câu hỏi...")
                response = bot.ask_question(args)
                format_response(response)
            
            elif command == 'extract':
                if not args:
                    print("❌ Vui lòng nhập truy vấn: extract <truy_vấn>")
                    continue
                
                print("🔍 Đang trích xuất thông tin...")
                response = bot.extract_information(args)
                format_response(response)
            
            elif command == 'summary':
                response = bot.get_document_summary()
                format_response(response)
            
            elif command == 'status':
                status = bot.get_status()
                print(f"🟢 TRẠNG THÁI: {'Sẵn sàng' if status['is_ready'] else 'Chưa sẵn sàng'}")
                print(f"📚 SỐ TÀI LIỆU: {status['total_documents']}")
                print(f"📄 ĐỊNH DẠNG HỖ TRỢ: {', '.join(status['supported_formats'])}")
                if status['loaded_documents']:
                    print(f"📋 TÀI LIỆU ĐÃ TẢI: {', '.join(status['loaded_documents'])}")
            
            elif command == 'clear':
                response = bot.clear_documents()
                format_response(response)
            
            else:
                print(f"❌ Lệnh không hợp lệ: {command}")
                print("Nhập 'help' để xem danh sách lệnh")
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Askly - Chatbot trích xuất thông tin và hỏi đáp')
    parser.add_argument('--version', action='version', version='Askly 0.1.0')
    parser.add_argument('--load', help='Tải tài liệu ngay khi khởi động')
    parser.add_argument('--load-dir', help='Tải tất cả tài liệu từ thư mục')
    
    args = parser.parse_args()
    
    # Global bot instance
    global bot
    bot = AsklyBot()
    
    # Load documents if specified
    if args.load:
        print(f"⏳ Đang tải tài liệu: {args.load}")
        response = bot.load_document(args.load)
        format_response(response)
        print()
    
    if args.load_dir:
        print(f"⏳ Đang tải tài liệu từ thư mục: {args.load_dir}")
        response = bot.load_documents_from_directory(args.load_dir)
        format_response(response)
        print()
    
    # Start interactive mode
    interactive_mode()


if __name__ == '__main__':
    main()