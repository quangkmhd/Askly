# 🪟 Hướng Dẫn Cài Đặt Askly Trên Windows

## 🚀 Quick Start (3 Bước)

### 1. Cài Đặt Prerequisites

**Python 3.11:**
- Download: https://www.python.org/downloads/
- ⚠️ **QUAN TRỌNG**: Tick ✅ "Add Python to PATH"

**Node.js 18+:**
- Download: https://nodejs.org/
- Chọn phiên bản LTS (Long Term Support)

**Git:**
- Download: https://git-scm.com/download/win
- Dùng default settings

### 2. Clone Project

```cmd
git clone <repository-url>
cd askly
```

### 3. Chạy Setup (1 lần duy nhất)

```cmd
setup_windows.bat
```

Script này sẽ tự động:
- ✅ Kiểm tra Python, Node.js
- ✅ Cài đặt Python dependencies (5-10 phút)
- ✅ Cài đặt Frontend dependencies (3-5 phút)
- ✅ Build embeddings database (~5 phút)

**Tổng thời gian:** ~15-20 phút (chạy 1 lần duy nhất)

---

## 🎯 Sử Dụng

### Start Hệ Thống

**Option 1: Backend + Frontend (Full)**
```cmd
start_all.bat
```
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

**Option 2: Chỉ Backend**
```cmd
start_backend.bat
```
- API: http://localhost:8000

### Stop Hệ Thống

```cmd
stop_all.bat
```

---

## 📁 Scripts Cho Windows

| File | Mô Tả | Chạy Bao Nhiêu Lần |
|------|-------|-------------------|
| `setup_windows.bat` | Setup ban đầu | **1 lần** (lần đầu) |
| `start_all.bat` | Start backend + frontend | Mỗi lần dùng |
| `start_backend.bat` | Start backend only | Mỗi lần dùng |
| `stop_all.bat` | Stop tất cả services | Khi cần stop |

---

## 🔧 Cài Đặt Chi Tiết

### Bước 1: Kiểm Tra Prerequisites

**Mở Command Prompt (cmd) và test:**

```cmd
REM Test Python
python --version
REM → Phải hiện: Python 3.11.x

REM Test Node.js
node --version
REM → Phải hiện: v18.x.x hoặc cao hơn

REM Test Git
git --version
REM → Phải hiện: git version x.x.x
```

**Nếu lỗi "command not found":**
- Python: Cài lại và tick "Add to PATH"
- Node.js: Restart Command Prompt sau khi cài
- Git: Restart Command Prompt sau khi cài

### Bước 2: Clone Repository

```cmd
REM Clone project
git clone <repository-url>
cd askly

REM Kiểm tra files
dir
REM → Phải thấy: setup_windows.bat, start_all.bat, etc.
```

### Bước 3: Chạy Setup

```cmd
setup_windows.bat
```

**Quá trình setup:**
```
[1/6] Checking Python...           ✅ OK
[2/6] Checking Conda...             ⚠️ Optional  
[3/6] Installing Python deps...     ⏳ 5-10 min
[4/6] Checking Node.js...           ✅ OK
[5/6] Installing Frontend deps...   ⏳ 3-5 min
[6/6] Building embeddings...        ⏳ ~5 min
```

**Tổng thời gian:** ~15-20 phút

### Bước 4: Start Hệ Thống

```cmd
REM Start backend + frontend
start_all.bat

REM Hoặc chỉ backend
start_backend.bat
```

**Kiểm tra:**
- Backend API: http://localhost:8000/health
- Frontend UI: http://localhost:5173

---

## ❓ Troubleshooting

### Lỗi: "Python not found"

**Nguyên nhân:** Python chưa được thêm vào PATH

**Giải pháp:**
1. Mở "Environment Variables"
2. Thêm vào PATH: `C:\Users\<username>\AppData\Local\Programs\Python\Python311`
3. Restart Command Prompt

### Lỗi: "pip install failed"

**Nguyên nhân:** Thiếu Microsoft Visual C++ Build Tools

**Giải pháp:**
```cmd
REM Download và cài:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

REM Hoặc dùng Conda (khuyến nghị):
conda create -n rag311 python=3.11
conda activate rag311
pip install -r requirements.txt
```

### Lỗi: "Port 8000 already in use"

**Giải pháp:**
```cmd
REM Kill process đang dùng port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

REM Hoặc dùng stop_all.bat
stop_all.bat
```

### Lỗi: "npm install failed"

**Giải pháp:**
```cmd
REM Xóa node_modules và cài lại
cd streamlit_app\front-end
rmdir /s /q node_modules
npm cache clean --force
npm install
```

### Lỗi: "Embeddings not found"

**Giải pháp:**
```cmd
REM Build lại embeddings
python scripts\rebuild_clean_database.py
```

---

## 🔥 Tips & Tricks

### 1. Dùng Conda (Khuyến nghị)

**Tại sao?**
- ✅ Quản lý dependencies tốt hơn
- ✅ Tránh conflict với Python system
- ✅ Dễ dàng switch giữa các projects

**Cài đặt Miniconda:**
```cmd
REM Download:
https://docs.conda.io/en/latest/miniconda.html

REM Sau khi cài:
conda create -n rag311 python=3.11
conda activate rag311
pip install -r requirements.txt
```

**Khi dùng Conda:**
```cmd
REM Mỗi lần mở Command Prompt:
conda activate rag311
start_all.bat
```

### 2. Dùng Git Bash (Alternative)

**Nếu quen Linux commands:**
```bash
# Mở Git Bash terminal
bash setup_windows.sh  # Tạo file này (copy từ .bat)
bash start_all.sh
```

### 3. Windows Terminal (Recommended)

**Cài Windows Terminal:**
- Microsoft Store → "Windows Terminal"
- Hoặc: https://aka.ms/terminal

**Ưu điểm:**
- ✅ Đẹp hơn cmd
- ✅ Tabs support
- ✅ Copy/paste dễ hơn

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **RAM** | 8GB | 16GB |
| **Disk** | 5GB free | 10GB free |
| **CPU** | 4 cores | 8 cores |
| **GPU** | None | NVIDIA GPU (CUDA) |

---

## 🚀 Next Steps

Sau khi setup xong:

1. **Thêm PDFs mới:**
   ```cmd
   REM Copy PDFs
   copy your-pdfs\*.pdf data\uploaded_pdfs\
   
   REM Rebuild embeddings
   python scripts\rebuild_clean_database.py
   ```

2. **Update model:**
   ```cmd
   REM Copy model mới vào models\model\
   rmdir /s /q models\model
   xcopy /e /i new-model models\model
   
   REM Restart
   start_all.bat
   ```

3. **Xem docs:**
   - `README.md` - Overview
   - `docs\PROJECT_STRUCTURE.md` - Cấu trúc
   - `docs\OCR_IMPROVEMENT_GUIDE.md` - Kỹ thuật OCR

---

## 📞 Support

**Gặp vấn đề?**
1. Đọc phần Troubleshooting ở trên
2. Check `docs\` folder
3. Mở issue trên GitHub

---

**Last Updated:** 21 October 2025  
**Tested On:** Windows 10, Windows 11

