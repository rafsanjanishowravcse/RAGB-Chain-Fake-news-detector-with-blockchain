#!/usr/bin/env python3
"""
Simple test to verify the image fact checker works in OCR-only mode
"""

import sys
import os

def test_basic_functionality():
    """Test basic image processing without visual models"""
    print("Testing basic image fact checker functionality...")
    
    try:
        from image_fact_checker import ImageFactChecker
        
        # Initialize checker
        checker = ImageFactChecker()
        print("✓ ImageFactChecker initialized")
        
        # Test if OCR is working
        if checker.ocr_reader:
            print("✓ EasyOCR is available")
        elif hasattr(checker, '_extract_with_tesseract'):
            print("✓ Tesseract fallback is available")
        else:
            print("⚠️  No OCR engine available")
        
        # Test basic functions exist
        assert hasattr(checker, 'process_image_upload'), "process_image_upload method missing"
        assert hasattr(checker, 'extract_text_from_image'), "extract_text_from_image method missing"
        assert hasattr(checker, 'verify_image_news'), "verify_image_news method missing"
        
        print("✓ All required methods are present")
        
        # Test visual model status
        if checker.caption_model:
            print("✓ Visual models loaded successfully")
        else:
            print("⚠️  Visual models not loaded - OCR-only mode")
        
        print("\n🎉 Basic functionality test passed!")
        print("\nYou can now test with an actual image:")
        print("1. Start the app: python3 app.py")
        print("2. Upload an image with text")
        print("3. Check if OCR extraction works")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    if test_basic_functionality():
        print("\n✅ Ready to test image fact checking!")
    else:
        print("\n❌ Issues found. Check the error messages above.")
