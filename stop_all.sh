#!/bin/bash

# Script để dừng tất cả servers

echo "🛑 Stopping FPTU Chatbot servers..."
echo ""

# Màu sắc
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kill processes trên port 8000 (Backend)
echo -e "${YELLOW}Stopping Backend (port 8000)...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Backend stopped${NC}"
else
    echo -e "${YELLOW}⚠️  No backend process found${NC}"
fi

# Kill processes trên port 5173 (Frontend)
echo -e "${YELLOW}Stopping Frontend (port 5173)...${NC}"
lsof -ti:5173 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Frontend stopped${NC}"
else
    echo -e "${YELLOW}⚠️  No frontend process found${NC}"
fi

# Xóa log files
if [ -f "backend.log" ]; then
    rm backend.log
    echo -e "${GREEN}✅ Cleaned backend.log${NC}"
fi

if [ -f "frontend.log" ]; then
    rm frontend.log
    echo -e "${GREEN}✅ Cleaned frontend.log${NC}"
fi

echo ""
echo -e "${GREEN}✅ All servers stopped!${NC}"
