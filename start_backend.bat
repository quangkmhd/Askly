@echo off
REM ============================================================================
REM START BACKEND API - WINDOWS
REM ============================================================================

echo.
echo 🚀 Starting Askly Backend API...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Run setup_windows.bat first
    pause
    exit /b 1
)

REM Check embeddings
echo 📋 Checking embeddings...
if not exist "outputs\text_chunks_and_embeddings_df.npy" (
    echo ❌ Embeddings not found!
    echo.
    echo Please run setup_windows.bat first, or:
    echo    python scripts\rebuild_clean_database.py
    pause
    exit /b 1
)
echo ✅ Embeddings ready
echo.

REM Check dependencies
echo 📦 Checking dependencies...
python -c "import flask_cors" 2>nul
if %errorlevel% neq 0 (
    echo Installing flask-cors...
    pip install flask-cors
)
echo ✅ Dependencies ready
echo.

REM Start Backend
echo 🔧 Starting Backend API on port 8000...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📍 Backend API:  http://localhost:8000
echo 📍 Health Check: http://localhost:8000/health
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 Press Ctrl+C to stop
echo.

python api_server.py

