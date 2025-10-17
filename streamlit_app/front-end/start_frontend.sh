#!/bin/bash

# Script để khởi động Frontend

echo "🎨 Starting FPTU Chatbot Frontend..."
echo ""

# Màu sắc
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kiểm tra node_modules
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm dependencies...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Dependencies ready!${NC}"
echo ""

# Khởi động Frontend
echo -e "${BLUE}🚀 Starting Frontend on port 5173...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📍 Frontend:${NC}     http://localhost:5173"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop${NC}"
echo ""

npm run dev
