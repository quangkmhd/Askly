#!/bin/bash

# Script to install Tesseract OCR for Vietnamese PDF support

echo "🔧 Installing Tesseract OCR for Vietnamese PDF support..."
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Detected Linux - Installing via apt..."
    
    # Update package list
    sudo apt-get update
    
    # Install Tesseract and Vietnamese language pack
    sudo apt-get install -y tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng
    
    echo "✅ Tesseract installed!"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Detected macOS - Installing via Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install Tesseract
    brew install tesseract tesseract-lang
    
    echo "✅ Tesseract installed!"
    
else
    echo "❌ Unsupported OS: $OSTYPE"
    echo "Please install Tesseract manually:"
    echo "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki"
    echo "  - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-vie"
    echo "  - macOS: brew install tesseract tesseract-lang"
    exit 1
fi

echo ""
echo "🔍 Verifying installation..."
tesseract --version

echo ""
echo "📋 Available languages:"
tesseract --list-langs

echo ""
echo "✅ Tesseract OCR setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Install Python dependencies: pip install pytesseract Pillow"
echo "  2. Rebuild embeddings: python run.py --build"
echo "  3. Test with scanned PDFs"
