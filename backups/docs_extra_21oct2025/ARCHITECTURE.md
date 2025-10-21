# Kiến trúc Hệ thống - Askly

## Tổng quan

Askly là hệ thống RAG (Retrieval-Augmented Generation) được thiết kế theo kiến trúc modular, dễ mở rộng và bảo trì.

## 🏗️ Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ React Web UI │  │  CLI Client  │  │  API Client  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                     API LAYER                                 │
│                   ┌────────▼─────────┐                        │
│                   │  Flask API       │                        │
│                   │  - CORS          │                        │
│                   │  - Routing       │                        │
│                   │  - Validation    │                        │
│                   └────────┬─────────┘                        │
└────────────────────────────┼─────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                       │
│                   ┌────────▼─────────┐                        │
│                   │   RAG Pipeline   │                        │
│                   │  - Orchestration │                        │
│                   │  - Flow Control  │                        │
│                   └────────┬─────────┘                        │
│                            │                                  │
│         ┌──────────────────┼──────────────────┐              │
│         │                  │                  │              │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼──────┐      │
│  │  Retrieval  │  │  LLM Manager    │  │ Processors │      │
│  │   System    │  │  - Gemini API   │  │ - PDF      │      │
│  │  - Search   │  │  - Prompting    │  │ - Text     │      │
│  │  - Ranking  │  │  - Generation   │  │ - Chunking │      │
│  └──────┬──────┘  └─────────────────┘  └─────┬──────┘      │
└─────────┼──────────────────────────────────────┼─────────────┘
          │                                      │
┌─────────┼──────────────────────────────────────┼─────────────┐
│                     DATA LAYER                                │
│  ┌──────▼──────┐                        ┌──────▼──────┐      │
│  │  Embeddings │                        │   PDF Data  │      │
│  │  - Vectors  │                        │  - Raw PDFs │      │
│  │  - Metadata │                        │  - Texts    │      │
│  └─────────────┘                        └─────────────┘      │
└───────────────────────────────────────────────────────────────┘
```

## 📦 Components chi tiết

### 1. API Layer (`api_server.py`)

**Trách nhiệm**:
- Nhận HTTP requests từ clients
- Validate input data
- Route requests đến RAG Pipeline
- Format và trả về responses
- Handle errors

**Endpoints**:
- `GET /health` - Health check
- `POST /ask` - Hỏi câu hỏi
- `POST /search` - Tìm kiếm tài liệu

**Technologies**:
- Flask: Web framework
- Flask-CORS: Cross-origin support
- Pydantic: Data validation (optional)

### 2. RAG Pipeline (`rag_pipeline.py`)

**Trách nhiệm**:
- Orchestrate toàn bộ RAG workflow
- Coordinate giữa các components
- Manage state và cache
- Error handling và logging

**Workflow**:
```python
def ask(query):
    # 1. Retrieve relevant documents
    docs = retrieval_system.search(query)
    
    # 2. Prepare context
    context = prepare_context(docs)
    
    # 3. Generate answer
    answer = llm_manager.generate(query, context)
    
    # 4. Format response
    return format_response(answer, docs)
```

**Key Methods**:
- `setup_pipeline()` - Initialize components
- `ask()` - Main Q&A method
- `search()` - Document search only
- `build_embeddings()` - Build/update embeddings

### 3. Retrieval System (`models/retrieval_system.py`)

**Trách nhiệm**:
- Semantic search với embeddings
- Ranking documents by relevance
- Query expansion
- Filtering và post-processing

**Architecture**:
```
Query → Embedding → Similarity Search → Ranking → Top-K Results
```

**Components**:
- **Embedding Manager**: Tạo và quản lý embeddings
- **Similarity Calculator**: Tính cosine similarity
- **Query Expander**: Mở rộng query với synonyms
- **Result Ranker**: Sắp xếp kết quả

**Code Structure**:
```python
class RetrievalSystem:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.embeddings_df = None
        
    def load_embeddings(self):
        """Load pre-computed embeddings"""
        
    def search(self, query, n_results=5):
        """Search for relevant documents"""
        # 1. Embed query
        query_embedding = self.embedding_manager.embed([query])
        
        # 2. Calculate similarities
        similarities = cosine_similarity(
            query_embedding, 
            self.embeddings_df['embeddings']
        )
        
        # 3. Rank and return top-k
        top_indices = similarities.argsort()[-n_results:]
        return self.embeddings_df.iloc[top_indices]
```

### 4. LLM Manager (`models/llm_manager.py`)

**Trách nhiệm**:
- Tích hợp với Gemini API
- Prompt engineering
- Response generation
- Error handling và retries

**Prompt Template**:
```python
PROMPT_TEMPLATE = """
Bạn là trợ lý AI thông minh, chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

NGUYÊN TẮC:
1. Chỉ trả lời dựa trên thông tin trong tài liệu
2. Nếu không tìm thấy thông tin, nói rõ "Tôi không tìm thấy thông tin..."
3. Trích dẫn nguồn khi có thể
4. Trả lời ngắn gọn, súc tích

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {query}

TRẢ LỜI:
"""
```

**Key Methods**:
```python
class LLMManager:
    def generate(self, query, context, temperature=0.2):
        """Generate answer from query and context"""
        
    def format_prompt(self, query, context):
        """Format prompt with template"""
        
    def parse_response(self, response):
        """Parse and clean LLM response"""
```

### 5. Embedding Manager (`models/embedding_manager.py`)

**Trách nhiệm**:
- Load TensorFlow Hub model
- Generate embeddings cho text
- Cache embeddings
- Batch processing

**Model**: Universal Sentence Encoder (TF Hub)
- Dimension: 512
- Language: Multilingual (including Vietnamese)
- Speed: ~50ms per query

**Code**:
```python
class EmbeddingManager:
    def __init__(self):
        self.model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )
        
    def embed(self, texts):
        """Generate embeddings for texts"""
        return self.model(texts).numpy()
        
    def embed_batch(self, texts, batch_size=32):
        """Batch processing for large datasets"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings.append(self.embed(batch))
        return np.vstack(embeddings)
```

### 6. Document Processors (`processors/`)

#### PDF Processor (`pdf_processor.py`)
```python
class PDFProcessor:
    def extract_text(self, pdf_path):
        """Extract text from PDF"""
        
    def extract_metadata(self, pdf_path):
        """Extract metadata (pages, title, etc.)"""
        
    def process_incremental(self, pdf_path):
        """Process only if not already processed"""
```

#### Text Processor (`text_processor.py`)
```python
class TextProcessor:
    def clean_text(self, text):
        """Clean and normalize text"""
        
    def remove_special_chars(self, text):
        """Remove unwanted characters"""
        
    def normalize_vietnamese(self, text):
        """Normalize Vietnamese text"""
```

#### Document Chunker (`document_chunker.py`)
```python
class DocumentChunker:
    def chunk_by_sentences(self, text, chunk_size=5):
        """Chunk text by sentences"""
        
    def chunk_with_overlap(self, text, size=500, overlap=50):
        """Chunk with sliding window"""
        
    def smart_chunk(self, text):
        """Smart chunking based on structure"""
```

## 🔄 Data Flow

### 1. Indexing Flow (Build Embeddings)

```
PDF Files → PDF Processor → Text Extraction
                                  ↓
                          Text Processor → Cleaning
                                  ↓
                        Document Chunker → Chunks
                                  ↓
                      Embedding Manager → Embeddings
                                  ↓
                          Save to Disk → JSON/CSV
```

### 2. Query Flow (Ask Question)

```
User Query → API Server → RAG Pipeline
                              ↓
                    Retrieval System → Search
                              ↓
                    Top-K Documents → Context
                              ↓
                      LLM Manager → Generate Answer
                              ↓
                    Format Response → Return to User
```

### 3. Search Flow (Document Search)

```
Search Query → API Server → RAG Pipeline
                                 ↓
                       Retrieval System → Search
                                 ↓
                       Top-K Documents → Format
                                 ↓
                       Return Results → User
```

## 💾 Data Storage

### File Structure

```
data/
├── uploaded_pdfs/              # Raw PDF files
│   ├── document1.pdf
│   └── document2.pdf
│
├── extracted_texts/            # Extracted text
│   ├── document1.txt
│   └── document2.txt
│
├── embeddings/                 # Embeddings storage
│   ├── text_chunks.json       # Chunks + metadata
│   └── embeddings.npy         # Numpy array of vectors
│
└── processed_pdfs.json        # Processing metadata
```

### Embeddings Format

**text_chunks.json**:
```json
[
  {
    "chunk_id": "doc1_chunk1",
    "text": "Học phí được quy định...",
    "file_name": "quy_che_dao_tao.pdf",
    "page_number": 5,
    "chunk_index": 0
  }
]
```

**embeddings.npy**:
- NumPy array shape: (N, 512)
- N = number of chunks
- 512 = embedding dimension

### Metadata Storage

**processed_pdfs.json**:
```json
{
  "document1.pdf": {
    "processed_date": "2024-01-15T10:30:00",
    "num_pages": 50,
    "num_chunks": 245,
    "file_hash": "abc123..."
  }
}
```

## 🔧 Configuration Management

### Config Structure (`config/config.py`)

```python
class Config:
    # API Settings
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    
    # Model Settings
    EMBEDDING_MODEL = os.getenv(
        'EMBEDDING_MODEL',
        'https://tfhub.dev/google/universal-sentence-encoder/4'
    )
    LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-1.5-flash')
    
    # RAG Settings
    DEFAULT_N_RESOURCES = int(os.getenv('DEFAULT_N_RESOURCES', 5))
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', 0.2))
    DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', 250))
    
    # Paths
    DATA_DIR = Path('data')
    EMBEDDINGS_DIR = DATA_DIR / 'embeddings'
    PDF_DIR = DATA_DIR / 'uploaded_pdfs'
```

## 🎯 Design Patterns

### 1. Singleton Pattern (Pipeline)

```python
class RAGPipeline:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 2. Factory Pattern (Model Loading)

```python
class ModelFactory:
    @staticmethod
    def create_embedding_model(model_type):
        if model_type == 'tfhub':
            return TFHubEmbedding()
        elif model_type == 'sentence-transformers':
            return SentenceTransformerEmbedding()
```

### 3. Strategy Pattern (Chunking)

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text):
        pass

class SentenceChunking(ChunkingStrategy):
    def chunk(self, text):
        # Implementation

class OverlapChunking(ChunkingStrategy):
    def chunk(self, text):
        # Implementation
```

## 🔍 Error Handling

### Error Hierarchy

```python
class AsklyError(Exception):
    """Base exception"""

class EmbeddingError(AsklyError):
    """Embedding related errors"""

class LLMError(AsklyError):
    """LLM related errors"""

class DocumentError(AsklyError):
    """Document processing errors"""
```

### Error Handling Strategy

```python
try:
    result = pipeline.ask(query)
except EmbeddingError as e:
    logger.error(f"Embedding error: {e}")
    return {"error": "Failed to process query"}
except LLMError as e:
    logger.error(f"LLM error: {e}")
    return {"error": "Failed to generate answer"}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"error": "Internal server error"}
```

## 📊 Performance Considerations

### 1. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_embedding(text):
    return embedding_manager.embed([text])
```

### 2. Lazy Loading

```python
class RAGPipeline:
    def __init__(self):
        self._retrieval_system = None
        
    @property
    def retrieval_system(self):
        if self._retrieval_system is None:
            self._retrieval_system = RetrievalSystem()
        return self._retrieval_system
```

### 3. Batch Processing

```python
def process_pdfs_batch(pdf_paths, batch_size=10):
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i+batch_size]
        process_batch(batch)
```

## 🔐 Security Architecture

### 1. API Security
- Rate limiting
- Input validation
- CORS configuration
- API key authentication (future)

### 2. Data Security
- Environment variables for secrets
- No hardcoded credentials
- Secure file permissions
- Data encryption at rest (future)

## 🚀 Scalability

### Horizontal Scaling
- Stateless API design
- Load balancer ready
- Shared storage for embeddings

### Vertical Scaling
- Efficient memory usage
- GPU acceleration support
- Batch processing optimization

---

**Note**: Kiến trúc này được thiết kế để dễ mở rộng và bảo trì. Mỗi component có trách nhiệm rõ ràng và có thể thay thế độc lập.
