#!/usr/bin/env python3
"""
Easy installation script for Askly chatbot
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main installation function."""
    print("🚀 Cài đặt Askly Chatbot")
    print("=" * 40)
    
    # Core dependencies
    core_packages = ["numpy>=1.21.0", "scikit-learn>=1.1.0"]
    
    print("📦 Cài đặt các thư viện cốt lõi...")
    for package in core_packages:
        print(f"   Installing {package}...")
        if install_package(package):
            print(f"   ✅ {package} cài đặt thành công")
        else:
            print(f"   ❌ Lỗi khi cài đặt {package}")
            return False
    
    # Optional dependencies
    optional_packages = {
        "PyPDF2>=3.0.0": "Hỗ trợ đọc file PDF",
        "python-docx>=0.8.11": "Hỗ trợ đọc file DOCX"
    }
    
    print("\n📄 Cài đặt hỗ trợ định dạng tài liệu (tùy chọn)...")
    for package, description in optional_packages.items():
        response = input(f"   Cài đặt {description}? (y/n): ").lower().strip()
        if response in ['y', 'yes', 'có']:
            if install_package(package):
                print(f"   ✅ {package} cài đặt thành công")
            else:
                print(f"   ⚠️  Không thể cài đặt {package} (bỏ qua)")
        else:
            print(f"   ⏭️  Bỏ qua {package}")
    
    print("\n🎉 Cài đặt hoàn tất!")
    print("\n📖 Hướng dẫn sử dụng:")
    print("   python cli.py --help")
    print("   python cli.py --load examples/documents/ai_education.txt")
    print("   python test_askly.py")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        print("❌ Cài đặt thất bại")
        sys.exit(1)