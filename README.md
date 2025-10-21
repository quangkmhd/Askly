# Askly - Hệ thống RAG Tiếng Việt

> Chatbot thông minh cho tài liệu tiếng Việt sử dụng công nghệ RAG (Retrieval-Augmented Generation)

## 📋 Tổng quan

Askly là hệ thống hỏi đáp thông minh dựa trên tài liệu PDF tiếng Việt. Hệ thống sử dụng công nghệ RAG để truy xuất thông tin từ tài liệu và tạo câu trả lời chính xác, có nguồn trích dẫn.

### ✨ Tính năng chính

#### Xử lý tài liệu
- **📄 Xử lý PDF nâng cao**: Hỗ trợ cả PDF text và PDF scan (OCR)
- **✂️ Semantic Chunking**: Chia tài liệu thành chunks 2000 tokens với bảo toàn cấu trúc
- **📚 Văn bản pháp lý**: Tối ưu cho tài liệu có cấu trúc (Điều, Chương, Mục, Khoản)
- **💾 Cập nhật tăng dần**: Chỉ xử lý file PDF mới, không rebuild toàn bộ

#### Tìm kiếm & Truy xuất
- **🔍 Hybrid Search**: Kết hợp tìm kiếm semantic và keyword
- **🎯 Intent Classification**: Tự động phân loại loại câu hỏi (học phí, tuyển sinh, điểm số...)
- **📊 Smart Reranking**: Xếp hạng lại kết quả dựa trên độ liên quan
- **🔗 Legal Anchors**: Trích dẫn chính xác vị trí trong tài liệu pháp lý

#### AI & LLM
- **🤖 Dual LLM Support**: Gemini API (cloud) hoặc Qwen2.5-3B (local)
- **💡 Dynamic Prompts**: Prompts tùy chỉnh theo intent với few-shot examples
- **💬 Multi-turn Conversation**: Hiểu ngữ cảnh từ lịch sử chat
- **🎛️ 4-bit Quantization**: Chạy local LLM trên GPU 4GB VRAM

#### Giao diện & API
- **🌐 React Frontend**: Giao diện chat hiện đại với TailwindCSS + DaisyUI
- **🔌 RESTful API**: Flask backend với CORS support
- **📊 Evaluation Framework**: Đánh giá chất lượng với BERT Score

## 🎯 Use Cases

Hệ thống Askly phù hợp cho nhiều lĩnh vực:

### Giáo dục
- **Trường học/Đại học**: Chatbot trả lời về quy chế, học phí, tuyển sinh, điều kiện tốt nghiệp
- **E-learning**: Trợ lý học tập từ giáo trình PDF
- **Hỏi đáp tài liệu học thuật**: Luận văn, báo cáo nghiên cứu

### Doanh nghiệp
- **Knowledge base**: Tra cứu tài liệu nội bộ, quy trình vận hành
- **Onboarding**: Hỗ trợ nhân viên mới tìm hiểu quy định công ty
- **Customer support**: Trả lời câu hỏi từ hướng dẫn sử dụng sản phẩm

### Pháp lý
- **Tra cứu văn bản pháp luật**: Nghị định, thông tư, quy định
- **Tư vấn pháp lý**: Tìm kiếm điều khoản liên quan
- **Compliance**: Kiểm tra tuân thủ quy định

### Y tế
- **Hỏi đáp y khoa**: Tra cứu thông tin từ tài liệu y học
- **Hướng dẫn sức khỏe**: Thông tin từ sổ tay chăm sóc sức khỏe

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
│  (Port 8000)        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   RAG Pipeline      │  ← Orchestrator chính
└──────────┬──────────┘
           │
           ├──→ Query Processor ──→ Intent Classification
           │                        Standalone Questions
           ↓
┌─────────────────────┐
│ Retrieval System    │  ← Hybrid Search (Semantic + Keyword)
│ + Reranker          │     Cosine Similarity + Heuristic Boost
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   LLM Manager       │  ← Dynamic Prompts
│ (Gemini/Qwen)       │     Tạo câu trả lời
└─────────────────────┘
```

## 📁 Cấu trúc thư mục

```
askly/
├── api_server.py              # Flask API server
├── rag_pipeline.py            # RAG pipeline chính
├── run.py                     # Entry point chính
├── rebuild_embeddings_semantic.py  # Rebuild embeddings với semantic chunking
│
├── config/                    # Cấu hình hệ thống
│   └── config.py              # File cấu hình chính
│
├── models/                    # AI models & retrieval
│   ├── embedding_manager.py   # Quản lý embeddings (TF-Hub)
│   ├── llm_manager.py         # Quản lý LLM (Gemini/Qwen)
│   ├── retrieval_system.py    # Hệ thống tìm kiếm
│   └── reranker.py            # Heuristic reranking
│
├── processors/                # Xử lý tài liệu
│   ├── pdf_processor.py       # Xử lý PDF (text + OCR)
│   ├── text_processor.py      # Xử lý văn bản
│   ├── document_chunker.py    # Chunking engine
│   └── semantic_chunker.py    # Semantic chunking
│
├── prompts/                   # Prompt templates
│   └── dynamic_prompts.py     # Dynamic prompt generator
│
├── utils/                     # Utilities
│   ├── query_processor.py     # Intent classification
│   └── utils.py               # Helper functions
│
├── evaluation/                # Evaluation framework
│   └── bert_score_evaluator.py
│
├── data/                      # Dữ liệu
│   ├── uploaded_pdfs/         # PDF files
│   ├── extracted_texts/       # Extracted text
│   ├── embeddings/            # Embeddings & chunks
│   └── processed_pdfs.json    # Tracking metadata
│
├── streamlit_app/             
│   └── front-end/             # React frontend (Vite + TailwindCSS)
│
├── docs/                      # Documentation
│
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── start_all.sh               # Start backend + frontend
└── start_backend.sh           # Start backend only
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

**Cách nhanh nhất (Khuyến nghị):**
```bash
bash start_all.sh
# Tự động khởi động cả Backend (port 8000) và Frontend (port 5173)
```

**Hoặc chạy riêng lẻ:**

**Terminal 1 - Backend:**
```bash
bash start_backend.sh
# API sẽ chạy tại: http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd streamlit_app/front-end
npm install  # Chỉ cần chạy lần đầu
npm run dev
# UI sẽ chạy tại: http://localhost:5173
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
# API docs: http://localhost:8000/health
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
curl http://localhost:8000/health

# Hỏi câu hỏi (POST request)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Học phí là bao nhiêu?",
    "n_resources": 5
  }'

# Clear chat history
curl -X POST http://localhost:8000/clear
```

### JavaScript/React

```javascript
// Trong React component
const askQuestion = async (question) => {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: question,
      n_resources: 5
    })
  });
  
  const data = await response.json();
  console.log(data.answer);
  console.log(data.sources);
};
```

## 🔧 Cấu hình nâng cao

### Tùy chỉnh tham số RAG

Chỉnh sửa trong `config/config.py`:

```python
# Retrieval settings
DEFAULT_N_RESOURCES_TO_RETURN = 10  # Số chunks truy xuất (mặc định: 10)

# Generation settings
DEFAULT_TEMPERATURE = 0.1           # Độ ngẫu nhiên (0.0-1.0)
DEFAULT_MAX_NEW_TOKENS = 100        # Độ dài câu trả lời tối đa

# Chunking settings
NUM_SENTENCE_CHUNK_SIZE = 25        # Số câu mỗi chunk (sentence-based)
CHUNK_OVERLAP = 5                   # Overlap giữa các chunks
```

### Tùy chỉnh khi gọi API

```python
# Python API
answer = pipeline.ask(
    query="Học phí là bao nhiêu?",
    n_resources=10,      # Số chunks truy xuất
    temperature=0.1,     # Độ chính xác (0.0 = chính xác, 1.0 = sáng tạo)
    max_new_tokens=150,  # Độ dài tối đa
    return_context=True  # Trả về context items
)

# REST API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Học phí?", "n_resources": 10}'
```

### Chuyển đổi giữa LLM models

Trong `config/config.py`:

```python
# Dùng Gemini API (cloud)
USE_REMOTE = True
GEMINI_API_KEY = "your_api_key"  # Trong .env

# Dùng Qwen local (GPU 4GB+)
USE_REMOTE = False
LLM_MODEL_ID = "models/model"    # LoRA adapter path
USE_QUANTIZATION = True          # 4-bit quantization
```

### Tùy chỉnh prompts theo intent

Chỉnh sửa trong `prompts/dynamic_prompts.py`:

```python
INTENT_PROMPTS = {
    "tuition_fee": "Bạn là chuyên gia tư vấn học phí...",
    "admission": "Bạn là chuyên viên tuyển sinh...",
    # Thêm intent mới của bạn
}
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
lsof -i :8000

# Kill process nếu cần
kill -9 <PID>

# Hoặc dùng script stop
bash stop_all.sh

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

```bash
# 1. Kiểm tra Backend đang chạy
curl http://localhost:8000/health

# 2. Kiểm tra CORS settings trong api_server.py
# Đảm bảo có: CORS(app)

# 3. Kiểm tra API URL trong frontend
# File: streamlit_app/front-end/src/...
# Đảm bảo dùng: http://localhost:8000
```

## 🧪 Testing

```bash
# Cài đặt test dependencies
pip install pytest pytest-asyncio

# Chạy tests
pytest tests/ -v

# Test thủ công
python -c "from rag_pipeline import RAGPipeline; p = RAGPipeline(); p.setup_pipeline()"
```

## 📈 Hiệu năng & Metrics

### Benchmark thực tế

| Thao tác | Thời gian | Ghi chú |
|----------|-----------|---------|
| **Semantic Search** | 50-100ms | Cosine similarity trên embeddings |
| **Reranking** | 10-20ms | Heuristic-based |
| **LLM Generation** (Gemini) | 2-5s | Tùy độ dài câu trả lời |
| **LLM Generation** (Local) | 5-15s | Qwen 4-bit quantized |
| **Total Response Time** | 2-5s | End-to-end (cloud) |
| **Throughput** | 10-15 req/s | Chỉ search, không LLM |

### Độ chính xác

- **Retrieval accuracy**: ~80-85% (top-10)
- **Answer quality**: ~85%+ (human evaluation)
- **Multi-turn understanding**: ~85%+
- **Hallucination rate**: 10-15%

### Tối ưu hóa có sẵn

✅ **Automatic caching**: Embeddings được cache và reuse  
✅ **Incremental updates**: Chỉ xử lý PDF mới  
✅ **GPU acceleration**: Tự động detect CUDA  
✅ **4-bit quantization**: Local LLM chạy trên 4GB VRAM  
✅ **Batch processing**: Hỗ trợ xử lý nhiều queries  
✅ **Smart chunking**: Semantic chunking 2000 tokens

## 🔒 Bảo mật

- ✅ API keys được lưu trong `.env` (không commit lên Git)
- ✅ CORS được cấu hình cho frontend
- ✅ Input validation cho tất cả API endpoints
- ✅ Rate limiting (có thể thêm nếu cần)

## 📦 Dependencies chính

### Backend (Python)
```txt
# Deep Learning & Embeddings
tensorflow >= 2.13.0
tensorflow-hub >= 0.14.0
transformers >= 4.46.0
torch >= 2.0.0

# PDF Processing
PyMuPDF == 1.23.26
pytesseract >= 0.3.10       # OCR support
Pillow >= 10.0.0

# NLP
spacy
tiktoken >= 0.5.0           # Accurate token counting

# LLM Optimization
accelerate
bitsandbytes                # 4-bit quantization
peft >= 0.16.0              # LoRA adapters

# API Server
fastapi >= 0.104.0
uvicorn >= 0.24.0
flask
flask-cors

# Evaluation
bert-score >= 0.3.13
rouge-score >= 0.1.2

# Utilities
pandas, numpy, tqdm, python-dotenv
```

### Frontend (JavaScript)
```json
{
  "react": "^18.x",
  "vite": "^4.x",
  "tailwindcss": "^3.x",
  "daisyui": "^2.x"
}
```

📝 Xem đầy đủ: `requirements.txt` và `streamlit_app/front-end/package.json`

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

## 🗺️ Tính năng hiện có & Roadmap

### ✅ Đã triển khai

**Core RAG:**
- ✅ Semantic search với Universal Sentence Encoder
- ✅ Hybrid retrieval (semantic + keyword)
- ✅ Intent classification & dynamic prompts
- ✅ Multi-turn conversation với standalone questions
- ✅ Heuristic reranking

**Document Processing:**
- ✅ PDF text extraction (PyMuPDF)
- ✅ OCR support cho PDF scan (Tesseract)
- ✅ Semantic chunking (2000 tokens)
- ✅ Legal document support (Điều, Chương, Mục)
- ✅ Incremental embeddings updates

**AI Models:**
- ✅ Dual LLM support (Gemini API + Qwen local)
- ✅ 4-bit quantization cho local LLM
- ✅ LoRA adapters với PEFT

**Interface:**
- ✅ React frontend (Vite + TailwindCSS + DaisyUI)
- ✅ Flask REST API với CORS
- ✅ CLI interactive mode

**Quality:**
- ✅ BERT Score evaluation framework
- ✅ Accurate token counting (tiktoken)

### 🚀 Kế hoạch tương lai

**Tính năng mới:**
- [ ] Upload PDF qua web UI (drag & drop)
- [ ] Multi-user authentication & sessions
- [ ] Chat history persistence & export
- [ ] Document summarization
- [ ] Multi-language support (English)

**Tối ưu hóa:**
- [ ] Cross-encoder reranking (sentence-transformers)
- [ ] Vector database integration (Pinecone/Weaviate/Qdrant)
- [ ] Query expansion với synonyms
- [ ] Caching cho frequent queries
- [ ] Rate limiting & monitoring

**Deployment:**
- [ ] Docker containerization
- [ ] Docker Compose cho full stack
- [ ] Cloud deployment guide (AWS/GCP/Azure)
- [ ] Kubernetes manifests

**Integrations:**
- [ ] Hỗ trợ thêm file formats (Word, Excel, TXT, Markdown)
- [ ] OpenAI API support (GPT-4/GPT-3.5)
- [ ] Anthropic Claude support
- [ ] Webhook notifications

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

Dự án này sử dụng các công nghệ và thư viện mã nguồn mở:

- **[TensorFlow Hub](https://tfhub.dev/)** - Universal Sentence Encoder cho embeddings
- **[Google Gemini](https://ai.google.dev/)** - Gemini AI API cho LLM
- **[Hugging Face](https://huggingface.co/)** - Transformers library & Qwen models
- **[spaCy](https://spacy.io/)** - Vietnamese NLP models
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF processing library
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** - OCR engine
- **[React](https://react.dev/)** & **[Vite](https://vitejs.dev/)** - Modern web development

## 📚 Tài liệu thêm

- 📖 **[CHANGELOG.md](CHANGELOG.md)** - Lịch sử thay đổi và cải tiến
- 🐛 **[BUGFIX_CHANGELOG.md](BUGFIX_CHANGELOG.md)** - Chi tiết các bug fixes
- 📜 **[SCRIPTS.md](SCRIPTS.md)** - Hướng dẫn sử dụng scripts
- 📁 **[docs/](docs/)** - Documentation chi tiết

---

<div align="center">

**Askly** - Hệ thống RAG Tiếng Việt  
Được xây dựng với ❤️ cho cộng đồng Việt Nam

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

