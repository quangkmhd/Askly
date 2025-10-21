# ⚡ Quick Start - Askly RAG

> Git clone → 1 command → Done!

---

## 🪟 Windows

### Lần Đầu (Setup)

```cmd
REM 1. Cài Python 3.11 + Node.js 18+
REM    https://python.org
REM    https://nodejs.org

REM 2. Clone
git clone <repo>
cd askly

REM 3. Setup (tự động, ~15 phút)
setup_windows.bat
```

### Mỗi Lần Dùng

```cmd
REM Start
start_all.bat

REM Stop
stop_all.bat
```

---

## 🐧 Linux / macOS

### Lần Đầu (Setup)

```bash
# 1. Clone
git clone <repo>
cd askly

# 2. Setup environment
conda create -n rag311 python=3.11
conda activate rag311
pip install -r requirements.txt

# 3. Build embeddings
python scripts/rebuild_clean_database.py
```

### Mỗi Lần Dùng

```bash
# Start
conda activate rag311
bash start_all.sh

# Stop
bash stop_all.sh
```

---

## 📍 URLs

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173
- **Health Check**: http://localhost:8000/health

---

## 🛠️ Common Commands

### Windows

```cmd
REM Rebuild embeddings
python scripts\rebuild_clean_database.py

REM Test OCR
python scripts\test_ocr_quality.py

REM CLI mode
python run.py
```

### Linux/macOS

```bash
# Rebuild embeddings
python scripts/rebuild_clean_database.py

# Test OCR
python scripts/test_ocr_quality.py

# CLI mode
python run.py
```

---

## ❓ Troubleshooting

### Windows

```cmd
REM Module not found
pip install -r requirements.txt

REM Port 8000 busy
stop_all.bat
start_all.bat

REM Embeddings missing
python scripts\rebuild_clean_database.py
```

### Linux/macOS

```bash
# Module not found
pip install -r requirements.txt

# Port 8000 busy
bash stop_all.sh
bash start_all.sh

# Embeddings missing
python scripts/rebuild_clean_database.py
```

---

## 📖 More Info

- **README.md** - Full documentation
- **docs/WINDOWS_SETUP.md** - Windows guide
- **docs/PROJECT_STRUCTURE.md** - Project structure

---

**That's it! 🚀**

