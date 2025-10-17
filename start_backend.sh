#!/bin/bash

# Script để khởi động Backend API

echo "🚀 Starting FPTU Chatbot Backend API..."
echo ""

# Màu sắc
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kiểm tra embeddings
echo -e "${YELLOW}📋 Checking embeddings...${NC}"
if [ ! -d "data/embeddings" ] || [ -z "$(ls -A data/embeddings 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  Embeddings not found. Building embeddings...${NC}"
    python run.py --build
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to build embeddings${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Embeddings ready!${NC}"
echo ""

# Kiểm tra flask-cors
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
python -c "import flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing flask-cors...${NC}"
    pip install flask-cors
fi

echo -e "${GREEN}✅ Dependencies ready!${NC}"
echo ""

# Khởi động Backend API
echo -e "${BLUE}🔧 Starting Backend API on port 8000...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📍 Backend API:${NC}  http://localhost:8000"
echo -e "${BLUE}📍 Health Check:${NC} http://localhost:8000/health"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop${NC}"
echo ""

python api_server.py
