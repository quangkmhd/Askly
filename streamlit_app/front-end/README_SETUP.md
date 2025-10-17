# FPTU Chatbot Frontend - Setup Guide

## 🎯 Tổng quan
Frontend React cho FPTU Chatbot với tích hợp RAG backend.

## 📋 Yêu cầu
- Node.js >= 16
- Python 3.8+
- RAG pipeline đã được build

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies
```bash
cd /home/tuanhq/project/askly2/askly/streamlit_app/front-end
npm install
```

### 2. Khởi động Backend API (Terminal 1)
```bash
cd /home/tuanhq/project/askly2/askly
python api_server.py
```
Backend sẽ chạy tại: `http://localhost:8000`

### 3. Khởi động Frontend (Terminal 2)
```bash
cd /home/tuanhq/project/askly2/askly/streamlit_app/front-end
npm run dev
```
Frontend sẽ chạy tại: `http://localhost:5173`

## 🔧 Cấu hình

### API Endpoint
File: `src/components/ChatBot.jsx`
```javascript
fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    query: promptInput,
    n_resources: 5
  }),
})
```

## 📝 Thay đổi đã thực hiện

### 1. Branding
- ✅ Thay "NTTU Chatbot" → "FPTU Chatbot"
- ✅ Thay "Đại học Nguyễn Tất Thành" → "Đại học FPT"
- ✅ Cập nhật URL: `https://daihoc.fpt.edu.vn/`

### 2. API Integration
- ✅ Thay ngrok endpoint → Local RAG API
- ✅ Method: GET → POST
- ✅ Response format: `result.result` → `result.answer`

### 3. Source Data
- ✅ Default source: `nttu` → `fptu`

## 🏗️ Cấu trúc Project

```
front-end/
├── src/
│   ├── components/
│   │   ├── NavBar.jsx       # Navigation bar (FPTU Chatbot)
│   │   └── ChatBot.jsx      # Main chat interface
│   ├── pages/
│   │   ├── HomePage.jsx     # Landing page
│   │   ├── FAQPage.jsx      # FAQs
│   │   └── IssuePage.jsx    # Feedback form
│   └── App.jsx              # Main app component
├── package.json
└── vite.config.js
```

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối với server"
- Kiểm tra backend API đang chạy: `http://localhost:8000/health`
- Kiểm tra CORS đã được enable trong `api_server.py`

### Lỗi: "Failed to initialize RAG pipeline"
- Build embeddings trước: `cd /home/tuanhq/project/askly2/askly && python run.py --build`
- Kiểm tra file embeddings tồn tại trong `data/embeddings/`

### Lỗi: Port 8000 đã được sử dụng
```bash
# Tìm và kill process
lsof -ti:8000 | xargs kill -9
```

## 📚 API Endpoints

### POST /ask
Gửi câu hỏi đến RAG pipeline
```json
{
  "query": "Điều kiện nhận học bổng?",
  "n_resources": 5
}
```

Response:
```json
{
  "answer": "Câu trả lời từ RAG...",
  "query": "Điều kiện nhận học bổng?",
  "sources": []
}
```

### GET /health
Kiểm tra trạng thái server
```json
{
  "status": "ok",
  "pipeline_loaded": true
}
```

## 🎨 Features
- ✅ Chat interface với typing animation
- ✅ Lịch sử trò chuyện (sidebar trái)
- ✅ Câu hỏi gợi ý (sidebar phải)
- ✅ Nguồn tham khảo (Wikipedia / FPTU)
- ✅ FAQs page
- ✅ Feedback form
- ✅ Responsive design

## 📞 Support
Nếu gặp vấn đề, kiểm tra logs:
- Backend: Terminal chạy `api_server.py`
- Frontend: Browser Console (F12)
