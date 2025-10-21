@echo off
REM ============================================================================
REM SETUP ASKLY RAG - WINDOWS AUTO INSTALLER
REM Git clone về -> chạy file này -> xong!
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║              🚀 ASKLY RAG - WINDOWS AUTO SETUP                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM ============================================================================
REM 1. CHECK PYTHON
REM ============================================================================
echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found!
    echo.
    echo 📥 Please install Python 3.11 from:
    echo    https://www.python.org/downloads/
    echo.
    echo ⚠️  IMPORTANT: Check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

python --version
echo ✅ Python OK
echo.

REM ============================================================================
REM 2. CHECK CONDA (Optional but recommended)
REM ============================================================================
echo [2/6] Checking Conda...
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Conda not found (optional)
    echo    Continuing with system Python...
    set USE_CONDA=0
) else (
    conda --version
    echo ✅ Conda OK
    set USE_CONDA=1
)
echo.

REM ============================================================================
REM 3. INSTALL PYTHON DEPENDENCIES
REM ============================================================================
echo [3/6] Installing Python dependencies...

if %USE_CONDA%==1 (
    echo Using Conda environment...
    call conda create -n rag311 python=3.11 -y
    call conda activate rag311
)

echo Installing packages (this may take 5-10 minutes)...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo ✅ Python dependencies installed
echo.

REM ============================================================================
REM 4. CHECK NODE.JS
REM ============================================================================
echo [4/6] Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found!
    echo.
    echo 📥 Please install Node.js 18+ from:
    echo    https://nodejs.org/
    pause
    exit /b 1
)

node --version
npm --version
echo ✅ Node.js OK
echo.

REM ============================================================================
REM 5. INSTALL FRONTEND DEPENDENCIES
REM ============================================================================
echo [5/6] Installing Frontend dependencies...
cd streamlit_app\front-end

if not exist "node_modules" (
    echo Installing npm packages (this may take 3-5 minutes)...
    call npm install
    if %errorlevel% neq 0 (
        echo ❌ Failed to install npm packages
        cd ..\..
        pause
        exit /b 1
    )
)

cd ..\..
echo ✅ Frontend dependencies installed
echo.

REM ============================================================================
REM 6. BUILD EMBEDDINGS DATABASE
REM ============================================================================
echo [6/6] Building embeddings database...

REM Check if embeddings exist
if exist "outputs\text_chunks_and_embeddings_df.npy" (
    echo ✅ Embeddings already exist, skipping...
) else (
    echo This will take ~5-10 minutes...
    echo.
    python scripts\rebuild_clean_database.py
    if %errorlevel% neq 0 (
        echo ❌ Failed to build embeddings
        pause
        exit /b 1
    )
    echo ✅ Embeddings built successfully
)
echo.

REM ============================================================================
REM SETUP COMPLETE!
REM ============================================================================
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                      ✅ SETUP COMPLETE!                                      ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo 🎉 Askly RAG is ready to use!
echo.
echo 🚀 To start the system:
echo    Option 1: start_all.bat        (Backend + Frontend)
echo    Option 2: start_backend.bat    (Backend only)
echo.
echo 📖 For more info:
echo    - README.md
echo    - docs\WINDOWS_SETUP.md
echo.
pause

