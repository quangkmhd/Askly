# API Reference - Askly

## Tổng quan

Askly cung cấp RESTful API để tích hợp vào các ứng dụng khác. API được xây dựng bằng Flask và hỗ trợ CORS.

**Base URL**: `http://localhost:8000`

## Authentication

Hiện tại API không yêu cầu authentication. Trong production, nên thêm API key hoặc OAuth.

## Endpoints

### 1. Health Check

Kiểm tra trạng thái của API server và RAG pipeline.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "pipeline_loaded": true
}
```

**Status Codes**:
- `200 OK`: Server đang hoạt động
- `500 Internal Server Error`: Server gặp lỗi

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 2. Ask Question

Hỏi câu hỏi và nhận câu trả lời từ RAG system.

**Endpoint**: `POST /ask`

**Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "question": "Học phí là bao nhiêu?",
  "n_resources": 5,
  "temperature": 0.2,
  "max_new_tokens": 250
}
```

**Parameters**:
- `question` (string, required): Câu hỏi của người dùng
- `n_resources` (integer, optional): Số tài liệu truy xuất (default: 5, range: 1-20)
- `temperature` (float, optional): Độ sáng tạo của LLM (default: 0.2, range: 0.0-1.0)
- `max_new_tokens` (integer, optional): Độ dài tối đa câu trả lời (default: 250)

**Response**:
```json
{
  "answer": "Học phí của trường dao động từ 10-15 triệu đồng mỗi năm...",
  "sources": [
    {
      "page_number": 5,
      "file_name": "quy_che_dao_tao.pdf",
      "content": "Học phí được quy định...",
      "score": 0.8523
    }
  ],
  "query_time": 2.34
}
```

**Status Codes**:
- `200 OK`: Thành công
- `400 Bad Request`: Thiếu hoặc sai parameters
- `500 Internal Server Error`: Lỗi xử lý

**Example**:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Học phí là bao nhiêu?",
    "n_resources": 5,
    "temperature": 0.2
  }'
```

```python
import requests

response = requests.post(
    'http://localhost:8000/ask',
    json={
        'question': 'Học phí là bao nhiêu?',
        'n_resources': 5,
        'temperature': 0.2
    }
)

data = response.json()
print(data['answer'])
```

```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'Học phí là bao nhiêu?',
    n_resources: 5,
    temperature: 0.2
  })
});

const data = await response.json();
console.log(data.answer);
```

---

### 3. Search Documents

Tìm kiếm tài liệu liên quan mà không tạo câu trả lời.

**Endpoint**: `POST /search`

**Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "Học phí",
  "n_results": 5
}
```

**Parameters**:
- `query` (string, required): Từ khóa tìm kiếm
- `n_results` (integer, optional): Số kết quả trả về (default: 5, range: 1-50)

**Response**:
```json
{
  "results": [
    {
      "sentence_chunk": "Học phí được quy định theo từng ngành...",
      "page_number": 5,
      "file_name": "quy_che_dao_tao.pdf",
      "score": 0.8523
    }
  ],
  "query_time": 0.05
}
```

**Status Codes**:
- `200 OK`: Thành công
- `400 Bad Request`: Thiếu hoặc sai parameters
- `500 Internal Server Error`: Lỗi xử lý

**Example**:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Học phí",
    "n_results": 5
  }'
```

---

## Error Handling

Tất cả errors đều trả về format:

```json
{
  "error": "Error message description"
}
```

**Common Error Codes**:
- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Endpoint không tồn tại
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Pipeline chưa được khởi tạo

---

## Rate Limiting

Hiện tại không có rate limiting. Trong production nên thêm:
- Rate limit: 100 requests/minute per IP
- Burst: 20 requests

---

## CORS

API hỗ trợ CORS cho tất cả origins. Trong production nên giới hạn:

```python
CORS(app, origins=['http://localhost:5173', 'https://yourdomain.com'])
```

---

## Best Practices

### 1. Caching

Cache kết quả cho các câu hỏi phổ biến:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_answer(question):
    return pipeline.ask(question)
```

### 2. Async Processing

Sử dụng async cho multiple requests:

```python
import asyncio
import aiohttp

async def ask_multiple(questions):
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post('http://localhost:8000/ask', json={'question': q})
            for q in questions
        ]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

### 3. Error Handling

Luôn handle errors:

```python
try:
    response = requests.post('http://localhost:8000/ask', json=data, timeout=30)
    response.raise_for_status()
    return response.json()
except requests.exceptions.Timeout:
    print("Request timeout")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

### 4. Retry Logic

Implement retry cho transient errors:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def ask_with_retry(question):
    response = requests.post('http://localhost:8000/ask', json={'question': question})
    response.raise_for_status()
    return response.json()
```

---

## Examples

### Complete Python Client

```python
import requests
from typing import Dict, List, Optional

class AsklyClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def ask(
        self, 
        question: str,
        n_resources: int = 5,
        temperature: float = 0.2,
        max_new_tokens: int = 250
    ) -> Dict:
        """Ask a question"""
        response = requests.post(
            f"{self.base_url}/ask",
            json={
                "question": question,
                "n_resources": n_resources,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search documents"""
        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "n_results": n_results
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# Usage
client = AsklyClient()

# Check health
health = client.health_check()
print(f"Status: {health['status']}")

# Ask question
result = client.ask("Học phí là bao nhiêu?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")

# Search
results = client.search("Học phí")
print(f"Found {len(results['results'])} documents")
```

### React Hook

```javascript
import { useState } from 'react';

export const useAskly = (baseUrl = 'http://localhost:8000') => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const ask = async (question, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${baseUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          n_resources: options.nResources || 5,
          temperature: options.temperature || 0.2,
          max_new_tokens: options.maxTokens || 250
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get answer');
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const search = async (query, nResults = 5) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, n_results: nResults })
      });
      
      if (!response.ok) {
        throw new Error('Search failed');
      }
      
      return await response.json();
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { ask, search, loading, error };
};

// Usage in component
function ChatComponent() {
  const { ask, loading, error } = useAskly();
  
  const handleAsk = async (question) => {
    try {
      const result = await ask(question);
      console.log(result.answer);
    } catch (err) {
      console.error('Error:', err);
    }
  };
  
  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
      {/* Your UI here */}
    </div>
  );
}
```

---

## API Features

### Hiện có
- ✅ Flask API với CORS support
- ✅ Health check endpoint (`/health`)
- ✅ Ask endpoint (`/ask`) - Hỏi câu hỏi với RAG
- ✅ Search endpoint (`/search`) - Tìm kiếm tài liệu
- ✅ Clear endpoint (`/clear`) - Xóa lịch sử chat
- ✅ Error handling và validation
- ✅ JSON response format chuẩn

### Roadmap
- [ ] Authentication (API keys, OAuth)
- [ ] Rate limiting (100 req/min)
- [ ] WebSocket support cho streaming real-time
- [ ] Batch processing endpoint
- [ ] File upload endpoint (PDF upload qua API)
- [ ] Conversation history API (lưu/load chat history)
- [ ] Analytics endpoint (usage stats)

