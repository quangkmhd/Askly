#!/bin/bash

# Script để khởi động cả Backend và Frontend cùng lúc

echo "🚀 Starting FPTU Chatbot (Backend + Frontend)..."
echo ""

# Màu sắc
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kiểm tra embeddings
echo -e "${YELLOW}📋 Checking embeddings...${NC}"
if [ ! -f "outputs/text_chunks_and_embeddings_df.csv" ] && [ ! -f "outputs/text_chunks_and_embeddings_df.npy" ]; then
    echo -e "${RED}❌ Embeddings not found!${NC}"
    echo ""
    echo -e "${YELLOW}Please build embeddings first:${NC}"
    echo -e "   ${BLUE}python scripts/rebuild_clean_database.py${NC}"
    echo ""
    echo "This needs to be done once, or when you add new PDFs."
    echo ""
    exit 1
fi
echo -e "${GREEN}✅ Embeddings ready!${NC}"
echo ""

# Kiểm tra flask-cors
echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"
python -c "import flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing flask-cors...${NC}"
    pip install flask-cors
fi
echo -e "${GREEN}✅ Python dependencies ready!${NC}"
echo ""

# Load nvm và kiểm tra Node.js
echo -e "${YELLOW}📦 Checking Node.js...${NC}"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

if ! command -v nvm &> /dev/null; then
    echo -e "${RED}❌ nvm not found. Installing nvm...${NC}"
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install 18
fi

nvm use 18 > /dev/null 2>&1
echo -e "${GREEN}✅ Node.js $(node --version) ready!${NC}"
echo ""

# Khởi động Backend
echo -e "${BLUE}🔧 Starting Backend API (port 8000)...${NC}"
python api_server.py > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✅ Backend PID: $BACKEND_PID${NC}"

# Đợi backend khởi động (local LLM cần thời gian load model)
echo -e "${YELLOW}⏳ Waiting for backend to start (loading local LLM may take 1-2 minutes)...${NC}"
sleep 10

# Kiểm tra backend health với timeout dài hơn
for i in {1..60}; do
    curl -s http://localhost:8000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backend is healthy!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}❌ Backend failed to start after 2 minutes. Check backend.log${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    # Show progress every 10 seconds
    if [ $((i % 5)) -eq 0 ]; then
        echo -e "${YELLOW}   Still loading... ($((i * 2))s elapsed)${NC}"
    fi
    sleep 2
done

echo ""

# Khởi động Frontend
echo -e "${BLUE}🎨 Starting Frontend (port 5173)...${NC}"
cd streamlit_app/front-end

# Cài dependencies nếu chưa có
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm dependencies (first time, ~2-3 minutes)...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install npm dependencies${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
fi

# Chạy frontend
npm run dev > ../../frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✅ Frontend PID: $FRONTEND_PID${NC}"

# Đợi frontend khởi động
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
sleep 5

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 FPTU Chatbot is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📍 Frontend:${NC}     http://localhost:5173"
echo -e "${BLUE}📍 Backend API:${NC}  http://localhost:8000"
echo -e "${BLUE}📍 Health Check:${NC} http://localhost:8000/health"
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "   Backend:  tail -f backend.log"
echo -e "   Frontend: tail -f frontend.log"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop both servers${NC}"
echo ""

# Hàm cleanup khi tắt
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Kill tất cả process con
    pkill -P $BACKEND_PID 2>/dev/null
    pkill -P $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Servers stopped${NC}"
    exit 0
}

# Bắt signal Ctrl+C
trap cleanup INT TERM

# Đợi và hiển thị logs
echo -e "${BLUE}📊 Showing backend logs (Ctrl+C to stop):${NC}"
echo ""

# Đợi log file được tạo
sleep 2

# Quay về thư mục gốc (sử dụng biến động thay vì hardcoded path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -f "backend.log" ]; then
    tail -f backend.log &
    TAIL_PID=$!
else
    echo -e "${YELLOW}⚠️  Backend log not ready yet. Check manually: tail -f backend.log${NC}"
fi

# Đợi
wait $BACKEND_PID $FRONTEND_PID

# Cleanup nếu process tự tắt
cleanup
