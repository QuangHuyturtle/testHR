#!/usr/bin/env python3
"""
Quick setup script for HR DSS
Script khởi tạo nhanh cho hệ thống
"""

import os
import sys

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = ['data', 'models', 'uploads', 'logs', 'templates']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")

def download_nltk_data():
    """Download NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK data already exists")
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("✓ NLTK data downloaded")
            
    except ImportError:
        print("⚠ NLTK not installed, skipping...")

def main():
    """Main setup"""
    print("=== HR DSS Quick Setup ===")
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Download NLTK data
    print("\n2. Setting up NLTK...")
    download_nltk_data()
    
    # Create sample data
    print("\n3. Creating sample data...")
    try:
        from create_sample_data import main as create_data
        create_data()
    except Exception as e:
        print(f"⚠ Could not create sample data: {e}")
        print("You can create it manually later by running: python create_sample_data.py")
    
    print("\n🎉 Quick setup completed!")
    print("\nNext steps:")
    print("1. Run: python test_demo.py")
    print("2. Run: python app.py")
    print("3. Open: http://localhost:5000")

if __name__ == "__main__":
    main()