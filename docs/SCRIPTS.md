# Scripts Reference

Quick reference for all scripts in the project.

---

## 🚀 Main Scripts

### **start_all.sh** (4.8K)
Start both backend and frontend servers.

```bash
bash start_all.sh
```

**What it does:**
- Starts Flask API server (port 8000)
- Starts React frontend (port 5173)
- Checks health endpoint
- Shows logs in real-time

**Access:**
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Health: http://localhost:8000/health

---

### **stop_all.sh** (1.0K)
Stop all running servers.

```bash
bash stop_all.sh
```

**What it does:**
- Kills Flask API server
- Kills React frontend
- Cleans up processes

---

### **start_backend.sh** (1.6K)
Start only the backend server (for development).

```bash
bash start_backend.sh
```

**Use case:** When you only need API server without frontend.

---

## 📦 Setup Scripts

### **setup_ocr.sh** (1.7K)
Install Tesseract OCR for scanned PDFs.

```bash
bash setup_ocr.sh
```

**What it does:**
- Detects OS (Ubuntu/Debian/CentOS/macOS)
- Installs Tesseract OCR
- Installs Vietnamese language pack
- Verifies installation

**Requirements:** sudo access

---

### **cleanup.sh** (1.3K)
Clean up project files (cache, logs, temp files).

```bash
bash cleanup.sh
```

**What it does:**
- Removes Python cache (`__pycache__`, `*.pyc`)
- Removes logs (`*.log`)
- Removes temporary files (`*.tmp`, `*.bak`)
- Removes test files (`test_*.py`, `debug_*.py`)
- Removes build artifacts

**Optional:** Uncomment line 24 to remove `node_modules` (saves ~100MB)

---

## 🔧 Build Scripts

### **run.py** (3.3K)
Main entry point for building embeddings and running the system.

```bash
# Build embeddings
python run.py --build

# Run with existing embeddings
python run.py
```

**Options:**
- `--build`: Build embeddings from PDFs
- `--pdf PATH`: Specify PDF path
- No args: Load existing embeddings

---

### **rebuild_embeddings_semantic.py** (2.8K)
Rebuild embeddings with semantic chunking (recommended).

```bash
python rebuild_embeddings_semantic.py
```

**What it does:**
- Uses semantic chunking (2000 tokens/chunk)
- Preserves headers in chunks
- Processes all PDFs in `data/uploaded_pdfs/`
- Saves to `outputs/`

**When to use:**
- After adding new PDFs
- After changing chunking parameters
- After updating document processing logic

**Output:**
- `outputs/text_chunks_and_embeddings_df.npy` - Embeddings (fast load)
- `outputs/text_chunks_and_embeddings_df.csv` - Embeddings (human-readable)
- `outputs/text_chunks_and_embeddings_df_chunks.json` - Text chunks

---

## 📊 Comparison

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `start_all.sh` | Start servers | Development & Production |
| `stop_all.sh` | Stop servers | When done working |
| `start_backend.sh` | Backend only | API development |
| `setup_ocr.sh` | Install OCR | First time setup (if using scanned PDFs) |
| `cleanup.sh` | Clean project | Before commit, when disk full |
| `run.py` | Build & run | Initial setup, testing |
| `rebuild_embeddings_semantic.py` | Rebuild embeddings | After adding PDFs, changing chunking |

---

## 🔄 Typical Workflow

### **First Time Setup:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install OCR (optional)
bash setup_ocr.sh

# 3. Build embeddings
python rebuild_embeddings_semantic.py

# 4. Start servers
bash start_all.sh
```

### **Daily Development:**
```bash
# Start servers
bash start_all.sh

# ... work on code ...

# Stop servers
bash stop_all.sh
```

### **After Adding New PDFs:**
```bash
# 1. Stop servers
bash stop_all.sh

# 2. Rebuild embeddings
python rebuild_embeddings_semantic.py

# 3. Restart servers
bash start_all.sh
```

### **Before Committing:**
```bash
# Clean up
bash cleanup.sh

# Check git status
git status
```

---

## 🐛 Troubleshooting

### **Port already in use**
```bash
# Stop all servers first
bash stop_all.sh

# Or manually kill processes
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:5173 | xargs kill -9  # Frontend
```

### **Embeddings not loading**
```bash
# Rebuild embeddings
python rebuild_embeddings_semantic.py

# Check outputs exist
ls -lh outputs/
```

### **OCR not working**
```bash
# Reinstall OCR
bash setup_ocr.sh

# Verify installation
tesseract --version
tesseract --list-langs | grep vie
```

---

## 💡 Tips

1. **Always stop servers before rebuilding embeddings**
   ```bash
   bash stop_all.sh
   python rebuild_embeddings_semantic.py
   bash start_all.sh
   ```

2. **Check logs if something fails**
   ```bash
   tail -f backend.log
   tail -f frontend.log
   ```

3. **Clean up regularly to save disk space**
   ```bash
   bash cleanup.sh
   ```

4. **Use semantic chunking for better results**
   ```bash
   python rebuild_embeddings_semantic.py  # Better than run.py --build
   ```

---

**Last Updated**: October 17, 2025
