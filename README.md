# Askly - Hệ thống RAG Tiếng Việt

> Chatbot thông minh cho tài liệu tiếng Việt sử dụng công nghệ RAG (Retrieval-Augmented Generation)

## 📋 Tổng quan

Askly là hệ thống hỏi đáp thông minh dựa trên tài liệu PDF tiếng Việt. Hệ thống sử dụng công nghệ RAG để truy xuất thông tin từ tài liệu và tạo câu trả lời chính xác, có nguồn trích dẫn.

### ✨ Tính năng chính

- **🔍 Tìm kiếm ngữ nghĩa**: Sử dụng TensorFlow Universal Sentence Encoder để tìm kiếm theo ngữ nghĩa
- **🤖 Trả lời thông minh**: Tích hợp Gemini AI để tạo câu trả lời tự nhiên và chính xác
- **📄 Xử lý PDF**: Hỗ trợ upload và xử lý nhiều file PDF tiếng Việt
- **💬 Giao diện web hiện đại**: React frontend với TailwindCSS + DaisyUI
- **🔌 API RESTful**: Flask backend với CORS support
- **📊 Đánh giá chất lượng**: Framework đánh giá với BERT Score
- **💾 Lưu trữ hiệu quả**: Cập nhật embeddings tăng dần, không cần xây dựng lại toàn bộ

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────┐
│  React Frontend     │  ← Giao diện người dùng (Vite + TailwindCSS)
│  (Port 5173)        │
└──────────┬──────────┘
           │ REST API
           ↓
┌─────────────────────┐
│  Flask API Server   │  ← API Layer + CORS
│  (Port 5000)        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   RAG Pipeline      │  ← Xử lý chính
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Semantic Search     │  ← TF Hub embeddings
│ (Cosine Similarity) │     Query expansion
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   LLM (Gemini)      │  ← Tạo câu trả lời
└─────────────────────┘
```

## 📁 Cấu trúc thư mục

```
askly/
├── api_server.py              # Flask API server
├── rag_pipeline.py            # RAG pipeline chính
├── build_embeddings_multi.py  # Xây dựng embeddings
├── run.py                     # Entry point chính
├── run_fast.py                # Chạy nhanh (skip checks)
│
├── config/                    # Cấu hình
│   └── config.py              # File cấu hình chính
│
├── models/                    # AI models
│   ├── embedding_manager.py   # Quản lý embeddings
│   ├── llm_manager.py         # Quản lý LLM
│   └── retrieval_system.py    # Hệ thống truy xuất
│
├── processors/                # Xử lý tài liệu
│   ├── pdf_processor.py       # Xử lý PDF
│   ├── text_processor.py      # Xử lý văn bản
│   └── document_chunker.py    # Chia nhỏ tài liệu
│
├── evaluation/                # Đánh giá
│   └── bert_score_evaluator.py
│
├── utils/                     # Tiện ích
│
├── data/                      # Dữ liệu
│   ├── uploaded_pdfs/         # PDF đã upload
│   ├── extracted_texts/       # Text đã trích xuất
│   ├── embeddings/            # Embeddings
│   └── processed_pdfs.json    # Metadata
│
├── streamlit_app/             # Streamlit UI (legacy)
│   └── front-end/             # React frontend
│
├── docs/                      # Tài liệu
│
├── requirements.txt           # Dependencies Python
├── .env                       # Biến môi trường
└── .gitignore                 # Git ignore
```

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống

- **Python**: 3.9 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: ~2GB cho models và cache
- **OS**: Linux, macOS, hoặc Windows

### 2. Cài đặt môi trường

```bash
# Clone repository
cd /path/to/askly

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Tải spaCy model cho tiếng Việt
python -m spacy download vi_core_news_lg
```

### 3. Cấu hình API Keys

Tạo file `.env` trong thư mục gốc:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration (optional)
EMBEDDING_MODEL=https://tfhub.dev/google/universal-sentence-encoder/4
LLM_MODEL=gemini-1.5-flash

# API Settings (optional)
API_HOST=0.0.0.0
API_PORT=5000
```

**Lấy Gemini API Key:**
1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập với Google account
3. Tạo API key mới
4. Copy và paste vào file `.env`

### 4. Chuẩn bị dữ liệu

```bash
# Tạo thư mục data nếu chưa có
mkdir -p data/uploaded_pdfs

# Copy các file PDF vào thư mục
cp /path/to/your/*.pdf data/uploaded_pdfs/

# Xây dựng embeddings
python run.py --build
```

## 💻 Sử dụng

### Cách 1: Chạy Full Stack (Khuyến nghị)

**Terminal 1 - Backend:**
```bash
bash start_backend.sh
# API sẽ chạy tại: http://localhost:5000
```

**Terminal 2 - Frontend:**
```bash
cd streamlit_app/front-end
npm install  # Chỉ cần chạy lần đầu
npm run dev
# UI sẽ chạy tại: http://localhost:5173
```

**Hoặc dùng tmux để chạy cả hai cùng lúc:**
```bash
bash start_all.sh
```

### Cách 2: Chế độ CLI (Command Line)

```bash
python run.py

# Hoặc chạy nhanh (bỏ qua checks):
python run_fast.py
```

Sau đó nhập câu hỏi trực tiếp trong terminal:
```
> Học phí là bao nhiêu?
> Điều kiện tốt nghiệp là gì?
> quit  # để thoát
```

### Cách 3: Chỉ chạy API Server

```bash
python api_server.py
# API docs: http://localhost:5000/health
```

## 📖 Ví dụ sử dụng

### Python API

```python
from rag_pipeline import RAGPipeline

# Khởi tạo pipeline
pipeline = RAGPipeline()
pipeline.setup_pipeline(load_existing_embeddings=True)

# Hỏi câu hỏi
answer = pipeline.ask(
    query="Học phí của trường là bao nhiêu?",
    n_resources=5,      # Số tài liệu truy xuất
    temperature=0.2,    # Độ sáng tạo của LLM (0-1)
    max_new_tokens=250  # Độ dài tối đa câu trả lời
)

print(answer)

# Chỉ tìm kiếm (không tạo câu trả lời)
results = pipeline.search("Học phí", n_results=5)
for result in results:
    print(f"Điểm số: {result['score']:.4f}")
    print(f"Nội dung: {result['sentence_chunk'][:100]}...")
    print(f"Nguồn: {result['page_number']}")
```

### REST API

```bash
# Health check
curl http://localhost:5000/health

# Hỏi câu hỏi (POST request)
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Học phí là bao nhiêu?",
    "n_resources": 5,
    "temperature": 0.2
  }'

# Tìm kiếm tài liệu
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Học phí",
    "n_results": 5
  }'
```

### JavaScript/React

```javascript
// Trong React component
const askQuestion = async (question) => {
  const response = await fetch('http://localhost:5000/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: question,
      n_resources: 5,
      temperature: 0.2
    })
  });
  
  const data = await response.json();
  console.log(data.answer);
  console.log(data.sources);
};
```

## 🔧 Cấu hình nâng cao

### Tùy chỉnh tham số truy xuất

```python
# Trong rag_pipeline.py hoặc khi gọi API

# Tăng số tài liệu truy xuất để có nhiều context hơn
answer = pipeline.ask(query, n_resources=10)  # Mặc định: 5

# Điều chỉnh temperature để thay đổi độ sáng tạo
answer = pipeline.ask(query, temperature=0.7)  # Mặc định: 0.2
# 0.0 = rất chính xác, ít sáng tạo
# 1.0 = rất sáng tạo, có thể không chính xác

# Giới hạn độ dài câu trả lời
answer = pipeline.ask(query, max_new_tokens=500)  # Mặc định: 250
```

### Tùy chỉnh prompt template

Chỉnh sửa trong `models/llm_manager.py`:

```python
PROMPT_TEMPLATE = """Bạn là trợ lý AI thông minh...
[Tùy chỉnh prompt của bạn ở đây]
"""
```

### Thay đổi model embeddings

Trong file `.env`:
```bash
# Sử dụng model khác từ TensorFlow Hub
EMBEDDING_MODEL=https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
```

## 📊 Đánh giá hệ thống

### Sử dụng BERT Score Evaluator

```python
from evaluation.bert_score_evaluator import RAGEvaluator
from rag_pipeline import RAGPipeline

# Khởi tạo
pipeline = RAGPipeline()
pipeline.setup_pipeline(load_existing_embeddings=True)
evaluator = RAGEvaluator(pipeline)

# Chuẩn bị test data
test_data = [
    {
        "question": "Học phí là bao nhiêu?",
        "reference_answer": "Học phí từ 10-15 triệu đồng mỗi năm..."
    },
    {
        "question": "Điều kiện tốt nghiệp?",
        "reference_answer": "Sinh viên cần đạt 120 tín chỉ..."
    }
]

# Đánh giá
results = evaluator.evaluate_generation(test_data)

# In kết quả
evaluator.bert_evaluator.print_summary(results)

# Xuất ra JSON
evaluator.bert_evaluator.export_results(results, "evaluation_results.json")
```

### Metrics

- **BERT Score**: Đánh giá độ tương đồng ngữ nghĩa (0-1)
- **Precision@k**: Độ chính xác của top-k kết quả
- **Recall@k**: Độ phủ của top-k kết quả
- **MRR**: Mean Reciprocal Rank

## 🐛 Xử lý lỗi

### Lỗi: "Failed to load embeddings"

```bash
# Xây dựng lại embeddings
python run.py --build

# Kiểm tra xem file embeddings có tồn tại không
ls -la data/embeddings/
```

### Lỗi: "GEMINI_API_KEY not found"

```bash
# Kiểm tra file .env
cat .env

# Đảm bảo có dòng:
# GEMINI_API_KEY=your_key_here

# Reload environment
source venv/bin/activate
```

### Lỗi: Backend không khởi động được

```bash
# Kiểm tra port có bị chiếm không
lsof -i :5000

# Kill process nếu cần
kill -9 <PID>

# Cài đặt lại dependencies
pip install -r requirements.txt --upgrade
```

### Lỗi: Out of Memory

```python
# Giảm số tài liệu truy xuất
answer = pipeline.ask(query, n_resources=3)  # Thay vì 5

# Hoặc giảm max_new_tokens
answer = pipeline.ask(query, max_new_tokens=150)
```

### Lỗi: Frontend không kết nối được Backend

1. Kiểm tra Backend đang chạy: `curl http://localhost:5000/health`
2. Kiểm tra CORS settings trong `api_server.py`
3. Kiểm tra API URL trong frontend code

## 🧪 Testing

```bash
# Cài đặt test dependencies
pip install pytest pytest-asyncio

# Chạy tests
pytest tests/ -v

# Test thủ công
python -c "from rag_pipeline import RAGPipeline; p = RAGPipeline(); p.setup_pipeline()"
```

## 📈 Hiệu năng

### Benchmark

- **Tìm kiếm ngữ nghĩa**: ~50-100ms
- **Tạo câu trả lời (Gemini)**: ~2-5 giây
- **Tổng thời gian**: ~2-5 giây/câu hỏi
- **Throughput**: ~10-15 requests/giây (chỉ search)

### Tối ưu hóa

1. **Cache embeddings**: Embeddings được cache tự động
2. **Batch processing**: Xử lý nhiều câu hỏi cùng lúc
3. **GPU acceleration**: Tự động sử dụng GPU nếu có
4. **Incremental updates**: Chỉ xử lý PDF mới

## 🔒 Bảo mật

- ✅ API keys được lưu trong `.env` (không commit lên Git)
- ✅ CORS được cấu hình cho frontend
- ✅ Input validation cho tất cả API endpoints
- ✅ Rate limiting (có thể thêm nếu cần)

## 📦 Dependencies chính

### Backend
- **TensorFlow** >= 2.13.0 - Deep learning framework
- **TensorFlow Hub** >= 0.14.0 - Pre-trained models
- **Transformers** >= 4.46.0 - Hugging Face models
- **Flask** >= 2.0.0 - Web framework
- **PyMuPDF** == 1.23.26 - PDF processing
- **spaCy** - NLP toolkit

### Frontend
- **React** 18 - UI framework
- **Vite** 4 - Build tool
- **TailwindCSS** 3 - Styling
- **DaisyUI** 2 - Component library

Xem đầy đủ trong `requirements.txt`

## 🛠️ Development

### Setup môi trường phát triển

```bash
# Clone và cài đặt
git clone <repository>
cd askly
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Cài đặt pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Style

- **Python**: Follow PEP 8
- **Type hints**: Sử dụng type hints cho functions
- **Docstrings**: Google style docstrings
- **Comments**: Tiếng Việt hoặc tiếng Anh

### Thêm tính năng mới

1. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
2. Implement changes
3. Thêm tests
4. Update documentation
5. Submit pull request

## 🗺️ Roadmap

### Phiên bản hiện tại (v2.0)
- ✅ Tìm kiếm ngữ nghĩa với TF Hub
- ✅ Tích hợp Gemini AI
- ✅ Xử lý PDF tiếng Việt
- ✅ React frontend
- ✅ Flask API backend
- ✅ Framework đánh giá (BERT Score)

### Kế hoạch tương lai
- [ ] Hỗ trợ nhiều loại file (Word, Excel, TXT)
- [ ] Upload file qua UI
- [ ] Multi-user support với authentication
- [ ] Vector database (Pinecone, Weaviate)
- [ ] Advanced reranking (Cross-encoder)
- [ ] Docker deployment
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Hỗ trợ thêm LLM providers (OpenAI, Anthropic)
- [ ] Chat history persistence
- [ ] Export conversation

## 📝 License

MIT License - Xem file LICENSE để biết chi tiết

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Hỗ trợ

- **Issues**: Mở issue trên GitHub
- **Documentation**: Xem thêm trong thư mục `docs/`
- **Email**: [Thêm email nếu có]

## 🙏 Acknowledgments

- **TensorFlow Hub**: Universal Sentence Encoder
- **Google**: Gemini AI API
- **Hugging Face**: Transformers library
- **spaCy**: Vietnamese NLP models

---

**Askly v2.0** - Hệ thống RAG Tiếng Việt  
Được xây dựng với ❤️ cho cộng đồng Việt Nam

