#!/bin/bash

# Script để cài đặt và chạy frontend với Node.js từ WSL

echo "🎨 FPTU Chatbot Frontend Setup & Run"
echo ""

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Use Node 18
nvm use 18

echo "📦 Node version: $(node --version)"
echo "📦 NPM version: $(npm --version)"
echo ""

# Cài dependencies nếu chưa có
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed!"
    echo ""
fi

# Chạy dev server
echo "🚀 Starting development server..."
echo "📍 Frontend will run on: http://localhost:5173"
echo ""
echo "💡 Press Ctrl+C to stop"
echo ""

npm run dev
