@echo off
REM ============================================================================
REM START BACKEND + FRONTEND - WINDOWS
REM ============================================================================

echo.
echo 🚀 Starting Askly (Backend + Frontend)...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Run setup_windows.bat first
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! Run setup_windows.bat first
    pause
    exit /b 1
)

REM Check embeddings
echo 📋 Checking embeddings...
if not exist "outputs\text_chunks_and_embeddings_df.npy" (
    echo ❌ Embeddings not found!
    echo.
    echo Building embeddings (this will take ~5 minutes)...
    python scripts\rebuild_clean_database.py
    if %errorlevel% neq 0 (
        echo ❌ Failed to build embeddings
        pause
        exit /b 1
    )
)
echo ✅ Embeddings ready
echo.

REM Check Python dependencies
echo 📦 Checking Python dependencies...
python -c "import flask_cors" 2>nul
if %errorlevel% neq 0 (
    pip install -q flask-cors
)
echo ✅ Python dependencies ready
echo.

REM Check Frontend dependencies
echo 📦 Checking Frontend dependencies...
if not exist "streamlit_app\front-end\node_modules" (
    echo Installing npm packages (first time only)...
    cd streamlit_app\front-end
    call npm install
    cd ..\..
)
echo ✅ Frontend dependencies ready
echo.

REM Start Backend in new window
echo 🔧 Starting Backend API...
start "Askly Backend" cmd /k "python api_server.py"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

REM Start Frontend
echo 🌐 Starting Frontend...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📍 Backend:  http://localhost:8000
echo 📍 Frontend: http://localhost:5173
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 Press Ctrl+C to stop Frontend
echo    Close "Askly Backend" window to stop Backend
echo.

cd streamlit_app\front-end
call npm run dev

