# OCR Support cho PDF ảnh và PDF 2 lớp

## 📋 Tổng quan

Askly giờ đây hỗ trợ **3 loại PDF**:

### 1. **PDF text thông thường** ✅
- PDF có text layer (có thể copy/paste text)
- Được xử lý bằng PyMuPDF thông thường
- **Không cần OCR**

### 2. **PDF ảnh (Scanned PDF)** 🖼️
- PDF được scan từ tài liệu giấy
- Không có text layer (không copy được text)
- **CẦN OCR** để extract text

### 3. **PDF 2 lớp (Hybrid PDF)** 📄
- Có cả text layer và image layer
- Một số phần có text, một số phần là ảnh
- **Tự động detect** và dùng OCR khi cần

## 🔧 Cài đặt

### Bước 1: Install Tesseract OCR

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng
```

#### macOS:
```bash
brew install tesseract tesseract-lang
```

#### Windows:
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install và thêm vào PATH
3. Download Vietnamese language pack

#### Hoặc dùng script tự động:
```bash
bash setup_ocr.sh
```

### Bước 2: Install Python dependencies

```bash
pip install pytesseract Pillow
```

Hoặc:
```bash
pip install -r requirements.txt
```

### Bước 3: Verify installation

```bash
tesseract --version
tesseract --list-langs  # Phải có 'vie' và 'eng'
```

## 🚀 Sử dụng

### Automatic OCR (Khuyến nghị)

OCR được **tự động bật** mặc định:

```python
from processors.document_chunker import DocumentChunker

# OCR tự động kích hoạt khi phát hiện PDF ảnh
chunker = DocumentChunker(
    use_ocr=True,           # Bật OCR (default)
    ocr_lang='vie+eng'      # Tiếng Việt + Tiếng Anh
)

chunks = chunker.process_pdf_to_chunks("scanned.pdf")
```

### Manual Control

```python
# Tắt OCR hoàn toàn
chunker = DocumentChunker(use_ocr=False)

# Chỉ dùng tiếng Việt
chunker = DocumentChunker(ocr_lang='vie')

# Chỉ dùng tiếng Anh
chunker = DocumentChunker(ocr_lang='eng')

# Nhiều ngôn ngữ
chunker = DocumentChunker(ocr_lang='vie+eng+chi_sim')
```

## 🔍 Cách hoạt động

### 1. **Auto-detection**

Code tự động phát hiện PDF ảnh:

```python
def _is_scanned_pdf(self, page: fitz.Page, threshold: int = 50) -> bool:
    """Kiểm tra xem page có phải là ảnh không"""
    text = page.get_text().strip()
    return len(text) < threshold  # < 50 chars → là ảnh
```

### 2. **OCR Processing**

Nếu phát hiện PDF ảnh:

```python
# Convert page → image
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
img = Image.open(io.BytesIO(pix.tobytes("png")))

# Run Tesseract OCR
text = pytesseract.image_to_string(img, lang='vie+eng')
```

### 3. **Hybrid Handling**

Với PDF 2 lớp:
- Pages có text → dùng PyMuPDF
- Pages là ảnh → dùng OCR
- **Tự động switch** giữa 2 phương pháp

## 📊 Testing

### Test OCR trên PDFs

```bash
python test_ocr.py
```

Output:
```
🔍 Testing OCR functionality for PDFs

✅ Tesseract version: 5.3.0

📁 Found 4 PDF files

================================================================================
Testing OCR on: document.pdf
================================================================================

🔍 Extracting with OCR enabled...
[INFO] Page 1 appears to be scanned, using OCR...
[INFO] Page 2 appears to be scanned, using OCR...

📊 Results:

With OCR:
  - Total pages: 2
  - Total characters: 2,543
  - Total words: 412

Without OCR:
  - Total pages: 2
  - Total characters: 45
  - Total words: 8

✅ OCR extracted 2,498 additional characters!
   This PDF appears to be scanned or image-based.
```

### Test với specific PDF

```python
from processors.document_chunker import DocumentChunker

chunker = DocumentChunker(use_ocr=True)
pages = chunker.extract_text_from_pdf("scanned.pdf")

for page in pages:
    print(f"Page {page['page_number']}: {page['page_word_count']} words")
```

## ⚙️ Configuration

### Điều chỉnh OCR threshold

```python
# Threshold thấp hơn → nhạy hơn với PDF ảnh
chunker = DocumentChunker(use_ocr=True)
chunker._is_scanned_pdf(page, threshold=30)  # Default: 50

# Threshold cao hơn → ít OCR hơn
chunker._is_scanned_pdf(page, threshold=100)
```

### Tăng chất lượng OCR

```python
# Tăng resolution khi convert page → image
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom (chậm hơn)
```

### Tesseract config

```python
# Custom Tesseract config
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, lang='vie+eng', config=custom_config)
```

## 🎯 Best Practices

### 1. **Language Selection**

```python
# Tiếng Việt + Tiếng Anh (khuyến nghị)
ocr_lang='vie+eng'

# Chỉ tiếng Việt (nhanh hơn)
ocr_lang='vie'

# Nhiều ngôn ngữ (chậm hơn)
ocr_lang='vie+eng+chi_sim+jpn'
```

### 2. **Performance**

- OCR **chậm hơn** 10-50x so với text extraction thông thường
- Chỉ dùng khi cần thiết
- Auto-detection giúp optimize performance

### 3. **Accuracy**

- PDF scan chất lượng cao → OCR chính xác hơn
- Font rõ ràng, không bị mờ → tốt hơn
- Tiếng Việt có dấu → cần language pack 'vie'

## 🐛 Troubleshooting

### Lỗi: "tesseract is not installed"

```bash
# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify
which tesseract
```

### Lỗi: "Failed to load language 'vie'"

```bash
# Linux
sudo apt-get install tesseract-ocr-vie

# macOS
brew install tesseract-lang

# Verify
tesseract --list-langs | grep vie
```

### OCR không chính xác

1. **Tăng resolution**:
   ```python
   pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
   ```

2. **Pre-process image**:
   ```python
   from PIL import ImageEnhance
   
   # Tăng contrast
   enhancer = ImageEnhance.Contrast(img)
   img = enhancer.enhance(2.0)
   ```

3. **Thử PSM modes khác**:
   ```python
   # PSM 6: Uniform block of text (default)
   # PSM 3: Fully automatic page segmentation
   # PSM 11: Sparse text
   config = '--psm 3'
   ```

### OCR quá chậm

1. **Giảm resolution**:
   ```python
   pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
   ```

2. **Tắt OCR cho PDFs không cần**:
   ```python
   chunker = DocumentChunker(use_ocr=False)
   ```

3. **Parallel processing** (advanced):
   ```python
   from multiprocessing import Pool
   
   with Pool() as pool:
       results = pool.map(process_page, pages)
   ```

## 📈 Performance Comparison

| PDF Type | Method | Speed | Accuracy |
|----------|--------|-------|----------|
| Text PDF | PyMuPDF | ⚡⚡⚡⚡⚡ Fast | ✅ 100% |
| Scanned PDF | OCR | 🐌 Slow | ⚠️ 85-95% |
| Hybrid PDF | Auto | ⚡⚡⚡ Mixed | ✅ 95-100% |

## 🔄 Rebuild Embeddings

Sau khi cài đặt OCR, rebuild embeddings:

```bash
# Rebuild với OCR enabled
python run.py --build

# Kiểm tra số lượng chunks
# Nên tăng lên nếu có scanned PDFs
```

## 📝 Examples

### Example 1: Process scanned thesis

```python
from processors.document_chunker import DocumentChunker

chunker = DocumentChunker(
    use_ocr=True,
    ocr_lang='vie+eng',
    chunk_size=25,
    chunk_overlap=5
)

chunks = chunker.process_pdf_to_chunks("thesis_scan.pdf")
print(f"Extracted {len(chunks)} chunks from scanned thesis")
```

### Example 2: Batch process mixed PDFs

```python
from pathlib import Path

pdf_dir = Path("data/uploaded_pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

chunker = DocumentChunker(use_ocr=True)  # Auto-detect

all_chunks = []
for pdf_file in pdf_files:
    chunks = chunker.process_pdf_to_chunks(str(pdf_file))
    all_chunks.extend(chunks)
    print(f"{pdf_file.name}: {len(chunks)} chunks")

print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
```

### Example 3: Compare with/without OCR

```python
# Without OCR
chunker_no_ocr = DocumentChunker(use_ocr=False)
chunks_no_ocr = chunker_no_ocr.process_pdf_to_chunks("scan.pdf")

# With OCR
chunker_ocr = DocumentChunker(use_ocr=True)
chunks_ocr = chunker_ocr.process_pdf_to_chunks("scan.pdf")

print(f"Without OCR: {len(chunks_no_ocr)} chunks")
print(f"With OCR: {len(chunks_ocr)} chunks")
print(f"Gained: {len(chunks_ocr) - len(chunks_no_ocr)} chunks")
```

## 🎓 Summary

- ✅ **Auto-detection**: Tự động phát hiện PDF ảnh
- ✅ **Multi-language**: Hỗ trợ tiếng Việt + Anh
- ✅ **Hybrid support**: Xử lý cả PDF 2 lớp
- ✅ **Easy setup**: Script tự động cài đặt
- ✅ **Testing tools**: Script test OCR functionality

**Khuyến nghị**: Luôn bật `use_ocr=True` để tự động xử lý mọi loại PDF!
