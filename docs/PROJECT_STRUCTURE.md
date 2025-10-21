# Cấu Trúc Project Askly

## 📁 Tổng Quan

```
askly/
├── config/                   # Cấu hình hệ thống
│   └── config.py             # File cấu hình chính
│
├── models/                   # AI Models & Retrieval
│   ├── embedding_manager.py  # Quản lý embeddings
│   ├── llm_manager.py        # Quản lý LLM (Vi-Qwen2)
│   ├── retrieval_system.py   # Hệ thống tìm kiếm
│   ├── reranker.py           # Reranking kết quả
│   └── model/                # Pretrained model (1.5B params)
│
├── processors/               # Xử Lý Tài Liệu
│   ├── pdf_processor.py      # Extract text từ PDF
│   ├── document_chunker.py   # Chia văn bản thành chunks
│   ├── semantic_chunker.py   # Semantic chunking (2000 tokens)
│   └── text_processor.py     # Xử lý text
│
├── prompts/                  # Prompt Templates
│   └── dynamic_prompts.py    # Dynamic prompts theo intent
│
├── utils/                    # Utilities
│   ├── query_processor.py    # Intent classification
│   └── utils.py              # Helper functions
│
├── evaluation/               # Đánh Giá
│   └── bert_score_evaluator.py  # BERT Score
│
├── scripts/                  # Utility Scripts
│   ├── rebuild_clean_database.py  # Rebuild embeddings (OCR 100%)
│   ├── test_ocr_quality.py        # Test OCR
│   └── rebuild_embeddings_semantic.py  # Rebuild với semantic chunking
│
├── data/                     # Data Storage
│   ├── uploaded_pdfs/        # PDF files (18 files)
│   ├── embeddings/           # [Deprecated] Embeddings cũ
│   ├── extracted_texts/      # [Deprecated] Text cũ
│   └── processed_pdfs.json   # Tracking metadata
│
├── outputs/                  # Embeddings & Chunks MỚI
│   ├── text_chunks_and_embeddings_df.npy    # Embeddings (204 chunks)
│   ├── text_chunks_and_embeddings_df.csv    # Backup CSV
│   ├── text_chunks_and_embeddings_df_chunks.json  # Metadata
│   └── text_chunks_clean.json               # Chunks sạch
│
├── backups/                  # Backups
│   ├── outputs_old_21oct2025/      # Files cũ đã dọn
│   └── embeddings_old_21oct2025/   # Embeddings cũ (lỗi OCR)
│
├── docs/                     # Documentation
│   ├── PROJECT_STRUCTURE.md        # (This file)
│   ├── HOW_TO_USE_MODEL.md         # Hướng dẫn model
│   ├── OCR_IMPROVEMENT_GUIDE.md    # Kỹ thuật OCR
│   ├── OCR_REBUILD_SUMMARY_21OCT2025.md  # Summary rebuild
│   └── FIXES_21OCT2025.md          # Changelog fixes
│
├── streamlit_app/front-end/  # React Frontend
│   ├── src/                   # React source code
│   ├── public/                # Static assets
│   └── package.json           # NPM dependencies
│
├── api_server.py             # Flask API Server (Port 8000)
├── rag_pipeline.py           # RAG Pipeline chính
├── run.py                    # CLI Entry Point
│
├── start_all.sh              # Start backend + frontend
├── start_backend.sh          # Start backend only
├── stop_all.sh               # Stop all services
│
├── requirements.txt          # Python dependencies
├── requirements_api.txt      # API server dependencies
├── .env                      # Environment variables (API keys)
├── .gitignore                # Git ignore rules
└── README.md                 # Main documentation
```

## 📂 Chi Tiết Các Thư Mục

### `config/`
Chứa cấu hình toàn bộ hệ thống:
- Paths, models, embedding settings
- LLM configuration
- Retrieval parameters
- Chunking settings

### `models/`
Các module liên quan đến AI models:
- **embedding_manager.py**: Universal Sentence Encoder
- **llm_manager.py**: Load & manage Vi-Qwen2-1.5B-RAG
- **retrieval_system.py**: Semantic search (cosine similarity)
- **reranker.py**: Heuristic reranking

### `processors/`
Xử lý tài liệu:
- **pdf_processor.py**: PyMuPDF + Tesseract OCR
- **semantic_chunker.py**: Chunk 2000 tokens, bảo toàn cấu trúc
- **document_chunker.py**: Orchestrator cho chunking
- **text_processor.py**: Clean & normalize text

### `prompts/`
Quản lý prompts:
- **dynamic_prompts.py**: Tạo prompts động theo intent
- Few-shot examples
- Intent-specific instructions

### `utils/`
Utilities:
- **query_processor.py**: Phân loại intent, standalone questions
- **utils.py**: Helper functions

### `scripts/`
Scripts tiện ích:
- **rebuild_clean_database.py**: ⭐ Rebuild embeddings với OCR 100%
- **test_ocr_quality.py**: Test chất lượng OCR
- **rebuild_embeddings_semantic.py**: Rebuild với semantic chunking

### `data/` & `outputs/`
**`outputs/` (MỚI - Đang dùng):**
- `text_chunks_and_embeddings_df.npy`: Embeddings chính (RAG dùng)
- 204 chunks, 0 lỗi OCR, 100% chính xác

**`data/` (CŨ - Deprecated):**
- `embeddings/`: Embeddings cũ (13.3% lỗi OCR)
- `extracted_texts/`: Text cũ (không dùng)

### `docs/`
Documentation:
- Project structure, guides, summaries

### `streamlit_app/front-end/`
React UI (Vite + TailwindCSS)

## 🔧 Files Quan Trọng

| File | Mô Tả | Port/Path |
|------|-------|-----------|
| `api_server.py` | Flask API server | 8000 |
| `rag_pipeline.py` | RAG pipeline chính | - |
| `run.py` | CLI entry point | - |
| `start_all.sh` | Start backend + frontend | - |
| `config/config.py` | Cấu hình toàn bộ | - |
| `models/llm_manager.py` | LLM management | - |
| `outputs/*.npy` | Embeddings database | - |

## 📊 Data Flow

```
1. PDF Upload
   └─→ data/uploaded_pdfs/

2. OCR & Processing
   └─→ scripts/rebuild_clean_database.py
       ├─→ Tesseract OCR (300 DPI)
       ├─→ Image preprocessing
       └─→ Text extraction

3. Chunking
   └─→ processors/semantic_chunker.py
       └─→ 204 chunks (avg 834 tokens)

4. Embeddings
   └─→ models/embedding_manager.py
       └─→ Universal Sentence Encoder
           └─→ outputs/text_chunks_and_embeddings_df.npy

5. Search & Retrieval
   └─→ rag_pipeline.py
       ├─→ Query → Intent classification
       ├─→ Search → Hybrid (semantic + keyword)
       ├─→ Rerank → Top-k results
       └─→ LLM → Generate answer
```

## 🚀 Workflow

### Thêm PDF Mới

```bash
1. Copy PDF
   cp new.pdf data/uploaded_pdfs/

2. Rebuild embeddings
   python scripts/rebuild_clean_database.py
   # → Force OCR lại toàn bộ
   # → Tạo embeddings mới

3. Restart
   bash start_all.sh
```

### Update Model

```bash
1. Copy model mới
   rm -rf models/model/*
   cp -r /path/to/new/model/* models/model/

2. Restart
   bash start_all.sh
```

### Debug

```bash
# Test OCR
python scripts/test_ocr_quality.py

# Test RAG pipeline
python run.py

# Check embeddings
python -c "from models.embedding_manager import EmbeddingManager; em = EmbeddingManager(); em.load_embeddings()"
```

## 📝 Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

## 🔒 Important Notes

- ⚠️ **KHÔNG commit `models/model/`** (3GB model files)
- ⚠️ **KHÔNG commit `data/uploaded_pdfs/`** (PDF files)
- ⚠️ **KHÔNG commit `.env`** (API keys)
- ✅ Chỉ commit code, requirements, docs

---

**Last Updated:** 21 October 2025  
**Version:** 1.0

