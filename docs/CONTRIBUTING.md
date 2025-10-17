# Hướng dẫn Đóng góp - Askly

## 🤝 Chào mừng!

Cảm ơn bạn đã quan tâm đến việc đóng góp cho Askly! Tài liệu này sẽ hướng dẫn bạn cách đóng góp hiệu quả.

## 📋 Các cách đóng góp

### 1. Báo cáo Bug
- Kiểm tra xem bug đã được báo cáo chưa trong Issues
- Tạo issue mới với template bug report
- Mô tả chi tiết: steps to reproduce, expected vs actual behavior
- Attach logs, screenshots nếu có

### 2. Đề xuất tính năng mới
- Tạo issue với label "enhancement"
- Giải thích use case và lợi ích
- Thảo luận với maintainers trước khi implement

### 3. Cải thiện Documentation
- Fix typos, grammar
- Thêm examples
- Cập nhật outdated info
- Dịch sang ngôn ngữ khác

### 4. Submit Code
- Fix bugs
- Implement features
- Optimize performance
- Add tests

## 🚀 Getting Started

### 1. Fork và Clone

```bash
# Fork repository trên GitHub
# Clone fork của bạn
git clone https://github.com/YOUR_USERNAME/askly.git
cd askly

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/askly.git
```

### 2. Setup môi trường

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Setup pre-commit hooks
pre-commit install
```

### 3. Tạo branch mới

```bash
# Sync với upstream
git fetch upstream
git checkout main
git merge upstream/main

# Tạo feature branch
git checkout -b feature/ten-tinh-nang
# hoặc
git checkout -b fix/ten-bug
```

## 💻 Development Workflow

### 1. Code Style

**Python**: Follow PEP 8

```bash
# Format code với Black
black .

# Check style với flake8
flake8 .

# Type checking với mypy
mypy .
```

**Conventions**:
- Sử dụng type hints
- Docstrings theo Google style
- Comments bằng tiếng Việt hoặc tiếng Anh
- Tên biến/hàm descriptive và clear

**Example**:
```python
def calculate_similarity(
    query_embedding: np.ndarray, 
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Tính cosine similarity giữa query và documents.
    
    Args:
        query_embedding: Query embedding vector (1, 512)
        doc_embeddings: Document embeddings matrix (N, 512)
        
    Returns:
        Similarity scores array (N,)
        
    Raises:
        ValueError: Nếu dimensions không match
    """
    if query_embedding.shape[1] != doc_embeddings.shape[1]:
        raise ValueError("Embedding dimensions must match")
        
    return cosine_similarity(query_embedding, doc_embeddings)[0]
```

### 2. Testing

**Viết tests cho code mới**:

```python
# tests/test_retrieval.py
import pytest
from models.retrieval_system import RetrievalSystem

def test_search_returns_results():
    """Test search returns correct number of results"""
    retrieval = RetrievalSystem()
    retrieval.load_embeddings()
    
    results = retrieval.search("test query", n_results=5)
    
    assert len(results) == 5
    assert all('score' in r for r in results)

def test_search_with_empty_query():
    """Test search handles empty query"""
    retrieval = RetrievalSystem()
    
    with pytest.raises(ValueError):
        retrieval.search("")
```

**Chạy tests**:
```bash
# Chạy tất cả tests
pytest

# Chạy với coverage
pytest --cov=. --cov-report=html

# Chạy specific test file
pytest tests/test_retrieval.py

# Chạy specific test
pytest tests/test_retrieval.py::test_search_returns_results
```

### 3. Commit Messages

**Format**: `<type>(<scope>): <subject>`

**Types**:
- `feat`: Tính năng mới
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Add/update tests
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat(retrieval): add query expansion for better recall"
git commit -m "fix(api): handle empty query parameter"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(pipeline): add integration tests"
```

### 4. Pull Request Process

#### Trước khi submit PR:

```bash
# 1. Sync với upstream
git fetch upstream
git rebase upstream/main

# 2. Run tests
pytest

# 3. Check code style
black .
flake8 .

# 4. Update documentation nếu cần

# 5. Push to your fork
git push origin feature/ten-tinh-nang
```

#### Tạo Pull Request:

1. Vào GitHub và tạo PR từ fork của bạn
2. Điền template:

```markdown
## Description
Mô tả ngắn gọn về changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Has This Been Tested?
Mô tả tests đã chạy

## Checklist:
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
```

3. Request review từ maintainers
4. Địa chỉ feedback nếu có
5. Đợi approval và merge

## 📝 Documentation Guidelines

### README Updates

- Giữ README concise và clear
- Thêm examples cho features mới
- Update table of contents nếu cần
- Check links không bị broken

### Code Documentation

```python
class RetrievalSystem:
    """
    Hệ thống truy xuất tài liệu dựa trên semantic search.
    
    Attributes:
        embedding_manager: Manager để tạo embeddings
        embeddings_df: DataFrame chứa embeddings và metadata
        
    Example:
        >>> retrieval = RetrievalSystem()
        >>> retrieval.load_embeddings()
        >>> results = retrieval.search("học phí", n_results=5)
    """
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Tìm kiếm tài liệu liên quan đến query.
        
        Args:
            query: Câu query từ user
            n_results: Số kết quả trả về
            
        Returns:
            List of dictionaries chứa kết quả search
            
        Raises:
            ValueError: Nếu query rỗng
            EmbeddingError: Nếu không load được embeddings
        """
        pass
```

## 🐛 Bug Fix Guidelines

### 1. Reproduce Bug

```python
# Viết test case reproduce bug
def test_bug_reproduction():
    """Test reproducing bug #123"""
    # Setup
    pipeline = RAGPipeline()
    
    # Action that causes bug
    result = pipeline.ask("")  # Empty query
    
    # Assert expected behavior
    assert "error" in result
```

### 2. Fix Bug

```python
# Implement fix
def ask(self, query: str):
    # Add validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Rest of implementation
    ...
```

### 3. Verify Fix

```bash
# Run test
pytest tests/test_bug_fix.py

# Manual testing
python run.py
```

## ✨ Feature Development Guidelines

### 1. Design

- Viết design doc cho features lớn
- Thảo luận với maintainers
- Consider backward compatibility
- Plan for testing

### 2. Implementation

```python
# Implement incrementally
# Step 1: Basic functionality
def new_feature_v1():
    pass

# Step 2: Add options
def new_feature_v2(option1=None):
    pass

# Step 3: Optimize
def new_feature_v3(option1=None, option2=None):
    pass
```

### 3. Testing

- Unit tests cho individual components
- Integration tests cho end-to-end flow
- Performance tests nếu relevant

### 4. Documentation

- Update README
- Add docstrings
- Create examples
- Update API docs

## 🔍 Code Review Guidelines

### Cho Reviewers:

- Review code trong 24-48 hours
- Provide constructive feedback
- Check for:
  - Correctness
  - Code style
  - Tests
  - Documentation
  - Performance
  - Security

### Cho Authors:

- Respond to feedback promptly
- Don't take feedback personally
- Ask questions nếu unclear
- Make requested changes
- Re-request review after updates

## 📊 Performance Guidelines

### Benchmarking

```python
import time

def benchmark_function():
    start = time.time()
    
    # Your code
    result = some_function()
    
    end = time.time()
    print(f"Execution time: {end - start:.4f}s")
    
    return result
```

### Profiling

```bash
# Profile với cProfile
python -m cProfile -o profile.stats run.py

# Visualize với snakeviz
pip install snakeviz
snakeviz profile.stats
```

### Optimization Tips

- Use caching cho expensive operations
- Batch processing cho multiple items
- Lazy loading cho large data
- Profile before optimizing

## 🔒 Security Guidelines

### 1. Secrets Management

```python
# ❌ BAD
API_KEY = "sk-1234567890"

# ✅ GOOD
import os
API_KEY = os.getenv('GEMINI_API_KEY')
```

### 2. Input Validation

```python
# Validate user input
def validate_query(query: str) -> str:
    if not query:
        raise ValueError("Query cannot be empty")
    
    if len(query) > 1000:
        raise ValueError("Query too long")
    
    # Sanitize input
    return query.strip()
```

### 3. Error Messages

```python
# ❌ BAD - Exposes internal details
raise Exception(f"Database connection failed: {db_password}")

# ✅ GOOD - Generic message
raise Exception("Failed to connect to database")
```

## 📞 Getting Help

- **Issues**: Tạo issue với label "question"
- **Discussions**: Sử dụng GitHub Discussions
- **Email**: [Thêm email nếu có]

## 🎉 Recognition

Contributors sẽ được:
- Thêm vào CONTRIBUTORS.md
- Mentioned trong release notes
- Credit trong documentation

## 📜 License

Bằng việc contribute, bạn đồng ý rằng contributions của bạn sẽ được licensed theo MIT License.

---

**Cảm ơn bạn đã đóng góp cho Askly! 🚀**
