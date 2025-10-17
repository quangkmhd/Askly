#!/bin/bash
# Cleanup script for FPTU Chatbot project

echo "🧹 Cleaning up project..."

# 1. Python cache
echo "  → Removing Python cache..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyo" -delete

# 2. Logs
echo "  → Removing logs..."
rm -f *.log backend.log frontend.log 2>/dev/null

# 3. Temporary files
echo "  → Removing temporary files..."
rm -f *.tmp *.bak *~ 2>/dev/null
rm -rf .pytest_cache/ .coverage htmlcov/ 2>/dev/null

# 4. Test files
echo "  → Removing test/debug files..."
rm -f test_*.py debug_*.py 2>/dev/null

# 5. Node modules (optional - uncomment if needed)
# echo "  → Removing node_modules (can reinstall with npm install)..."
# rm -rf streamlit_app/front-end/node_modules/

# 6. Build artifacts
echo "  → Removing build artifacts..."
rm -rf streamlit_app/front-end/dist/ 2>/dev/null
rm -rf streamlit_app/front-end/.vite/ 2>/dev/null

# 7. OS metadata
echo "  → Removing OS metadata..."
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "Thumbs.db" -delete 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Project size:"
du -sh . 2>/dev/null
echo ""
echo "💡 To clean node_modules (saves ~100MB), uncomment line 24 in cleanup.sh"
