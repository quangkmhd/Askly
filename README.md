# Askly - Trợ Lý Học Tập Thông Minh

> Hệ thống RAG hỗ trợ sinh viên tiếp cận thông tin từ tài liệu chính thống của trường

## 🎯 Tổng Quan

Askly giúp sinh viên dễ dàng tra cứu thông tin từ giáo trình, quy định, thông báo của trường. Hệ thống tự động tìm kiếm, tích hợp nội dung và trả lời câu hỏi chính xác.

**Ví dụ câu hỏi:**
- "Học phí năm 2024 là bao nhiêu?"
- "Quy định về điểm danh?"
- "Điều kiện xét học bổng?"

### ✨ Tính Năng

- 📄 **OCR 100%**: Xử lý PDF scan tiếng Việt chính xác
- ✂️ **Semantic Chunking**: Tìm kiếm thông minh, bảo toàn ngữ cảnh
- 🤖 **LLM Finetuned**: Qwen2.5-3B + LoRA cho câu trả lời tự nhiên
- 💬 **Multi-turn Chat**: Hiểu ngữ cảnh hội thoại
- 🌐 **Giao diện đẹp**: React + TailwindCSS

---

## 🚀 Cài Đặt Nhanh

### 🪟 Windows

```cmd
REM 1. Cài Python 3.11 + Node.js 18+
REM 2. Clone & setup
git clone <repo>
cd askly
setup_windows.bat

REM 3. Start
start_all.bat
```

### 🐧 Linux/macOS

```bash
# 1. Setup
conda create -n rag311 python=3.11
conda activate rag311
pip install -r requirements.txt

# 2. Build database
python scripts/rebuild_clean_database.py

# 3. Start
bash start_all.sh
```

**Truy cập:** http://localhost:5173

📖 Chi tiết: [QUICK_START.md](QUICK_START.md) | [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)

---

## 💻 Sử Dụng

### Web UI

```bash
bash start_all.sh
# Mở: http://localhost:5173
```

### CLI

```bash
python run.py
> Học phí kỳ 1 năm 2024 là bao nhiêu?
```

### REST API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Học phí là bao nhiêu?", "n_resources": 5}'
```

---

## 📁 Cấu Trúc

```
askly/
├── config/          # Cấu hình
├── models/          # LLM (Qwen2.5-3B + LoRA)
├── processors/      # PDF, OCR, chunking
├── prompts/         # Prompt engineering
├── scripts/         # Rebuild database, test
├── data/            # PDFs & embeddings
├── docs/            # Documentation
├── streamlit_app/   # React frontend
├── api_server.py    # Flask API
├── rag_pipeline.py  # RAG pipeline
└── run.py           # CLI
```

---

## 🔧 Cấu Hình

### Model (config/config.py)

```python
LLM_MODEL_PATH = "models/model"  # Qwen2.5-3B + LoRA
USE_QUANTIZATION = True          # 4-bit (1GB VRAM)
SEMANTIC_MAX_TOKENS = 2000       # Chunk size
```

### Thêm Tài Liệu Mới

```bash
# Copy PDFs
cp *.pdf data/uploaded_pdfs/

# Rebuild
python scripts/rebuild_clean_database.py
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Database** | 204 chunks, 18 PDFs |
| **OCR** | 100% chính xác |
| **Search** | ~50-100ms |
| **Model** | Qwen2.5-3B + LoRA (4-bit, 1GB VRAM) |

---

## 🛠️ Scripts

```bash
# Rebuild database
python scripts/rebuild_clean_database.py

# Test OCR
python scripts/test_ocr_quality.py

# Stop services
bash stop_all.sh  # hoặc stop_all.bat
```

---

## 📖 Documentation

| File | Mô Tả |
|------|-------|
| [QUICK_START.md](QUICK_START.md) | Quick reference |
| [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md) | Hướng dẫn Windows |
| [docs/OCR_IMPROVEMENT_GUIDE.md](docs/OCR_IMPROVEMENT_GUIDE.md) | Kỹ thuật OCR |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Cấu trúc chi tiết |

---

## 🔒 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Qwen2.5-3B-Instruct finetuned + LoRA (4-bit) |
| **Embeddings** | Universal Sentence Encoder |
| **OCR** | Tesseract (vie+eng, 300 DPI) |
| **Backend** | Python 3.11, Flask, PyTorch |
| **Frontend** | React 18, Vite, TailwindCSS |
| **OS** | ✅ Windows, Linux, macOS |

---

## 🎓 Dành Cho Sinh Viên

**Hỗ trợ tra cứu:**
- Quy định, quy chế (điểm danh, thi cử, tốt nghiệp)
- Học phí, lịch học, đăng ký môn
- Điều kiện học bổng, miễn giảm
- Nội dung giáo trình

**Cách đặt câu hỏi:**
- ✅ "Học phí năm 2024-2025 là bao nhiêu?"
- ✅ "Quy định về điểm danh ở FPT?"
- ❌ "Học phí?" (quá ngắn)

---

## 📝 License

MIT License

---

<div align="center">

**Askly** - Trợ Lý Học Tập Thông Minh  
Built with ❤️ for Vietnamese students

</div>
