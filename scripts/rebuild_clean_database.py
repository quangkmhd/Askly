#!/usr/bin/env python3
"""
Rebuild toàn bộ database từ PDFs - Extract text sạch, không lỗi OCR
"""
import sys
import fitz
import pytesseract
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import io
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up to askly/ directory

from processors.document_chunker import DocumentChunker
from models.embedding_manager import EmbeddingManager
from config.config import DATA_DIR, OUTPUTS_DIR

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Cải thiện chất lượng ảnh trước khi OCR"""
    # Chuyển sang grayscale
    img = img.convert('L')
    
    # Tăng contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Denoise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

def force_ocr_pdf(pdf_path: str) -> list:
    """
    Force OCR toàn bộ PDF - BỎ text layer cũ
    Dùng Tesseract với config tối ưu cho tiếng Việt
    """
    doc = fitz.open(pdf_path)
    pages_data = []
    
    print(f"\n📄 Processing: {Path(pdf_path).name}")
    print(f"   Total pages: {len(doc)}")
    
    for page_num in tqdm(range(len(doc)), desc="   OCR"):
        page = doc[page_num]
        
        # Render page thành ảnh với DPI cao
        mat = fitz.Matrix(3, 3)  # 3x zoom = 300 DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess
        img = preprocess_image_for_ocr(img)
        
        # OCR với config tốt nhất
        custom_config = r'--oem 3 --psm 6 -l vie+eng'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        # Clean text
        text = text.strip()
        
        if text:
            pages_data.append({
                'page_number': page_num,
                'page_text': text,
                'page_char_count': len(text),
                'page_word_count': len(text.split()),
                'source_file': Path(pdf_path).name
            })
    
    doc.close()
    return pages_data

def main():
    print("="*80)
    print("REBUILD CLEAN DATABASE - FORCE OCR TOÀN BỘ")
    print("="*80)
    print()
    print("⚠️  WARNING: Quá trình này sẽ:")
    print("   1. Force OCR lại TOÀN BỘ PDFs (bỏ text layer cũ)")
    print("   2. Sử dụng Tesseract với config tốt nhất cho tiếng Việt")
    print("   3. Rebuild embeddings hoàn toàn")
    print("   4. Mất khoảng 10-30 phút tùy số lượng trang")
    print()
    
    response = input("Tiếp tục? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Get all PDFs
    pdf_dir = DATA_DIR / "uploaded_pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDFs found in {pdf_dir}")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDFs")
    print()
    
    # Process all PDFs with FORCE OCR
    all_pages = []
    for pdf_path in pdf_files:
        try:
            pages = force_ocr_pdf(str(pdf_path))
            all_pages.extend(pages)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print()
    print(f"✅ Extracted {len(all_pages)} pages from {len(pdf_files)} PDFs")
    print()
    
    # Create chunks using DocumentChunker
    print("📊 Creating chunks...")
    chunker = DocumentChunker(
        use_semantic_chunking=True,
        semantic_max_tokens=2000,
        min_token_length=100
    )
    
    # Add metadata to each page (source_file is already there)
    for page in all_pages:
        # Ensure we have the metadata structure
        if 'metadata' not in page:
            page['metadata'] = {}
        page['metadata']['source_file'] = page.get('source_file', 'Unknown')
    
    # Use semantic chunker directly
    chunks = chunker.semantic_chunker.chunk_document(all_pages, metadata=None)
    
    # Debug: Print some stats about chunks creation
    print(f"[DEBUG] Processed {len(all_pages)} pages")
    non_empty_pages = [p for p in all_pages if len(p.get('page_text', '').strip()) > 0]
    print(f"[DEBUG] Non-empty pages: {len(non_empty_pages)}")
    if len(chunks) == 0 and len(non_empty_pages) > 0:
        # Check first non-empty page
        sample_page = non_empty_pages[0]
        print(f"[DEBUG] Sample page text (first 200 chars): {sample_page['page_text'][:200]}")
        print(f"[DEBUG] Sample page tokens: {chunker.semantic_chunker.count_tokens(sample_page['page_text'])}")
    
    print(f"✅ Created {len(chunks)} chunks")
    print()
    
    # Show statistics
    if chunks:
        stats = chunker.get_chunk_statistics(chunks)
        print("📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.1f}")
            else:
                print(f"   {key}: {value}")
        print()
    else:
        print("⚠️  No chunks created!")
        print()
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    embedding_manager = EmbeddingManager()
    embedding_manager.text_chunks = chunks
    
    embeddings = embedding_manager.create_embeddings(chunks)
    print(f"✅ Created {len(embeddings)} embeddings")
    print()
    
    # Save
    print("💾 Saving...")
    embedding_manager.save_embeddings()  # This already saves chunks to _chunks.json
    
    print()
    print("="*80)
    print(" REBUILD COMPLETE!")
    print("="*80)
    print()
    print(f"📊 Results:")
    print(f"   PDFs processed: {len(pdf_files)}")
    print(f"   Pages extracted: {len(all_pages)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Embeddings: {len(embeddings)}")
    print()
    print(" Next steps:")
    print("   1. Test search: python run.py")
    print("   2. Start system: bash start_all.sh")

if __name__ == "__main__":
    main()

