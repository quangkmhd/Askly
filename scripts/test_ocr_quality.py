#!/usr/bin/env python3
"""
Test OCR quality với 1 PDF
"""
import sys
import fitz
import pytesseract
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

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

def test_pdf_ocr(pdf_path: str, max_pages: int = 3):
    """Test OCR trên vài trang đầu"""
    doc = fitz.open(pdf_path)
    
    print("="*80)
    print(f"TEST OCR QUALITY")
    print("="*80)
    print(f"File: {Path(pdf_path).name}")
    print(f"Total pages: {len(doc)}")
    print(f"Testing first {min(max_pages, len(doc))} pages")
    print()
    
    results = []
    
    for page_num in range(min(max_pages, len(doc))):
        print(f"\n{'='*80}")
        print(f"PAGE {page_num + 1}")
        print(f"{'='*80}")
        
        page = doc[page_num]
        
        # 1. Get original text layer
        original_text = page.get_text().strip()
        
        # 2. Force OCR với config tối ưu
        mat = fitz.Matrix(3, 3)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = preprocess_image_for_ocr(img)
        
        custom_config = r'--oem 3 --psm 6 -l vie+eng'
        ocr_text = pytesseract.image_to_string(img, config=custom_config).strip()
        
        # Compare
        print(f"\n📄 ORIGINAL TEXT (first 300 chars):")
        print("-"*80)
        print(original_text[:300] if original_text else "[EMPTY]")
        
        print(f"\n✨ OCR TEXT (first 300 chars):")
        print("-"*80)
        print(ocr_text[:300] if ocr_text else "[EMPTY]")
        
        # Check for common OCR errors
        original_errors = sum(1 for pattern in ['DVC', 'CIIU', 'TRUbNG', 'ltrc', 'trtremg', 'Quart', 'Chang', 'Truang'] 
                             if pattern in original_text)
        ocr_errors = sum(1 for pattern in ['DVC', 'CIIU', 'TRUbNG', 'ltrc', 'trtremg', 'Quart', 'Chang', 'Truang'] 
                        if pattern in ocr_text)
        
        print(f"\n📊 QUALITY CHECK:")
        print(f"   Original errors: {original_errors} patterns")
        print(f"   OCR errors: {ocr_errors} patterns")
        print(f"   Improvement: {'✅ BETTER' if ocr_errors < original_errors else '❌ WORSE' if ocr_errors > original_errors else '➡️ SAME'}")
        
        results.append({
            'page': page_num + 1,
            'original_quality': 'BAD' if original_errors > 0 else 'GOOD',
            'ocr_quality': 'BAD' if ocr_errors > 0 else 'GOOD',
            'improved': ocr_errors < original_errors
        })
    
    doc.close()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        status = '✅' if r['improved'] else '➡️'
        print(f"Page {r['page']}: {r['original_quality']} → {r['ocr_quality']} {status}")
    
    improved_count = sum(1 for r in results if r['improved'])
    print(f"\nImproved pages: {improved_count}/{len(results)}")
    
    return results

if __name__ == "__main__":
    # Test PDF có lỗi OCR
    test_pdf = "data/uploaded_pdfs/Ban hành Quy chế đào tạo trình độ Thạc sĩ.pdf"
    
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    
    print(f"\n🧪 Testing OCR quality on: {test_pdf}\n")
    
    results = test_pdf_ocr(test_pdf, max_pages=3)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print()
    print("If OCR quality is better, run:")
    print("  python rebuild_clean_database.py")

