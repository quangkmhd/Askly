# 🤖 Askly - Chatbot Trích Xuất Thông Tin & Hỏi Đáp

Askly là một chatbot thông minh được thiết kế để trích xuất thông tin từ tài liệu và trả lời các câu hỏi dựa trên nội dung tài liệu. Hệ thống hỗ trợ nhiều định dạng tài liệu và cung cấp khả năng tìm kiếm, hỏi đáp thông minh.

## ✨ Tính năng chính

- 📄 **Hỗ trợ đa định dạng**: TXT, PDF, DOCX
- 🔍 **Trích xuất thông tin**: Tìm kiếm thông tin cụ thể trong tài liệu
- 💬 **Hỏi đáp thông minh**: Trả lời câu hỏi dựa trên nội dung tài liệu
- 🎯 **Tìm kiếm chính xác**: Sử dụng thuật toán TF-IDF và cosine similarity
- 🖥️ **Giao diện dòng lệnh**: CLI đơn giản, dễ sử dụng
- 🇻🇳 **Hỗ trợ tiếng Việt**: Giao diện và xử lý tiếng Việt

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- pip (Python package installer)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/quangkmhd/Askly.git
cd Askly

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### 1. Sử dụng giao diện dòng lệnh (CLI)

```bash
# Khởi động chatbot
python cli.py

# Hoặc tải sẵn một tài liệu
python cli.py --load examples/documents/ai_education.txt

# Hoặc tải tất cả tài liệu từ thư mục
python cli.py --load-dir examples/documents/
```

### 2. Các lệnh cơ bản trong CLI

```
load <đường_dẫn_file>           # Tải một tài liệu
load_dir <đường_dẫn_thư_mục>    # Tải tất cả tài liệu từ thư mục
ask <câu_hỏi>                   # Đặt câu hỏi
extract <truy_vấn>              # Trích xuất thông tin
summary                         # Xem tóm tắt tài liệu đã tải
status                          # Xem trạng thái chatbot
clear                           # Xóa tất cả tài liệu
help                            # Hiển thị hướng dẫn
quit                            # Thoát chương trình
```

### 3. Ví dụ sử dụng

```bash
Askly> load examples/documents/ai_education.txt
✅ Đã tải thành công tài liệu: ai_education.txt

Askly> ask AI trong giáo dục có lợi ích gì?
💭 CÂU HỎI: AI trong giáo dục có lợi ích gì?
🤖 TRẢ LỜI: AI cho phép tạo ra các chương trình học tập được cá nhân hóa cho từng học sinh
📊 ĐỘ TIN CẬY: 0.75
📄 NGUỒN: ai_education.txt

Askly> extract chatbot giáo dục
🔍 TRUY VẤN: chatbot giáo dục
📊 TÌM THẤY: 2 kết quả

1. 📄 ai_education.txt (Điểm: 0.85)
   📝 Chatbot AI có thể trả lời câu hỏi của học sinh 24/7, cung cấp hỗ trợ học tập ngay lập tức...
```

### 4. Sử dụng trong Python

```python
from askly import AsklyBot

# Khởi tạo chatbot
bot = AsklyBot()

# Tải tài liệu
result = bot.load_document("path/to/document.txt")
print(result['message'])

# Đặt câu hỏi
response = bot.ask_question("Câu hỏi của bạn?")
print(f"Trả lời: {response['answer']}")
print(f"Độ tin cậy: {response['confidence']}")

# Trích xuất thông tin
info = bot.extract_information("từ khóa tìm kiếm")
for result in info['results']:
    print(f"Tìm thấy: {result['text']}")
```

## 🧪 Kiểm tra

Chạy script kiểm tra để đảm bảo mọi thứ hoạt động chính xác:

```bash
python test_askly.py
```

## 📁 Cấu trúc project

```
Askly/
├── askly/                      # Package chính
│   ├── __init__.py            # Khởi tạo package
│   ├── chatbot.py             # Lớp chatbot chính
│   ├── document_processor.py   # Xử lý tài liệu
│   └── qa_engine.py           # Engine hỏi đáp
├── examples/                  # Ví dụ và tài liệu mẫu
│   └── documents/
│       ├── ai_education.txt
│       └── sustainable_development.txt
├── cli.py                     # Giao diện dòng lệnh
├── test_askly.py             # Script kiểm tra
├── requirements.txt          # Dependencies
└── README.md                 # Tài liệu này
```

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ lập trình chính
- **scikit-learn**: TF-IDF vectorization và cosine similarity
- **PyPDF2**: Xử lý file PDF
- **python-docx**: Xử lý file DOCX
- **NumPy**: Tính toán số học
- **NLTK**: Xử lý ngôn ngữ tự nhiên (tùy chọn)

## 📈 Tính năng sắp tới

- [ ] Hỗ trợ thêm định dạng tài liệu (Excel, PowerPoint)
- [ ] Giao diện web với Flask/FastAPI
- [ ] Tích hợp mô hình ngôn ngữ lớn (LLM)
- [ ] Hỗ trợ tìm kiếm semantic với embeddings
- [ ] API REST cho tích hợp
- [ ] Database để lưu trữ tài liệu
- [ ] Xử lý hình ảnh trong tài liệu

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch cho tính năng mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên hệ

- **GitHub**: [quangkmhd/Askly](https://github.com/quangkmhd/Askly)
- **Issues**: [GitHub Issues](https://github.com/quangkmhd/Askly/issues)

---

💡 **Gợi ý**: Bắt đầu bằng cách thử nghiệm với các tài liệu mẫu trong thư mục `examples/documents/` để làm quen với hệ thống!