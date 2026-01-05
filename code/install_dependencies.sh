#!/usr/bin/env bash

# Installation script for Image Fact Checker dependencies
# Run this script to install the required packages

echo "Installing Image Fact Checker dependencies..."
echo "=============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one for better dependency management."
fi

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install easyocr>=1.7.0
pip install pytesseract>=0.3.10
pip install opencv-python>=4.8.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.0
pip install imagehash>=4.3.1
pip install "numpy<2.0.0"
pip install scikit-learn>=1.3.0

echo ""
echo "Installing Tesseract OCR engine..."
echo "=================================="

# Detect OS and install Tesseract
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS. Installing Tesseract via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install tesseract
        brew install tesseract-lang
    else
        echo "⚠️  Homebrew not found. Please install Tesseract manually:"
        echo "   brew install tesseract tesseract-lang"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux. Installing Tesseract via package manager..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng
    elif command -v yum &> /dev/null; then
        sudo yum install -y tesseract tesseract-langpack-ben tesseract-langpack-eng
    else
        echo "⚠️  Package manager not found. Please install Tesseract manually."
    fi
else
    echo "⚠️  Unsupported OS. Please install Tesseract manually for your system."
fi

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "To test the installation, run:"
echo "  python test_image_verification.py"
echo ""
echo "To start the application, run:"
echo "  python app.py"
echo ""
echo "Note: First run may take longer as EasyOCR downloads language models."
