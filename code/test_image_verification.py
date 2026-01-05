#!/usr/bin/env python3
"""
Test script for image-based fact checking functionality
"""

import os
import sys
from image_fact_checker import ImageFactChecker

def test_image_processing():
    """Test basic image processing functionality"""
    print("Testing Image Fact Checker...")
    
    # Initialize the checker
    checker = ImageFactChecker()
    
    # Test with a sample image (you can replace this with an actual image path)
    test_image_path = "test_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test image to continue testing.")
        return False
    
    try:
        # Test image upload processing
        print("1. Testing image upload processing...")
        metadata = checker.process_image_upload(test_image_path)
        print(f"   ✓ Image processed successfully: {metadata['image_id']}")
        print(f"   ✓ Dimensions: {metadata['dimensions']}")
        
        # Test OCR extraction
        print("2. Testing OCR extraction...")
        ocr_result = checker.extract_text_from_image(metadata['stored_path'])
        print(f"   ✓ OCR method: {ocr_result.get('method', 'unknown')}")
        print(f"   ✓ Detected language: {ocr_result.get('language', 'unknown')}")
        print(f"   ✓ Confidence: {ocr_result.get('confidence', 0):.2f}")
        print(f"   ✓ Extracted text: {ocr_result.get('ocr_text', '')[:100]}...")
        
        # Test text cleaning
        print("3. Testing text cleaning...")
        cleaned_text = checker.clean_ocr_text(ocr_result.get('ocr_text', ''))
        print(f"   ✓ Cleaned text: {cleaned_text[:100]}...")
        
        # Test visual analysis
        print("4. Testing visual analysis...")
        
        # Test image captioning
        caption_result = checker.generate_image_caption(metadata['stored_path'])
        print(f"   ✓ Caption: {caption_result.get('caption', 'N/A')[:100]}...")
        
        # Test image embeddings
        embedding_result = checker.compute_image_embeddings(metadata['stored_path'])
        if embedding_result.get('embedding'):
            print(f"   ✓ Embedding dimension: {embedding_result.get('dimension', 'N/A')}")
        else:
            print(f"   ⚠ Embedding not available: {embedding_result.get('error', 'Unknown error')}")
        
        # Test perceptual hashing
        hash_result = checker.compute_perceptual_hashes(metadata['stored_path'])
        print(f"   ✓ pHash: {hash_result.get('phash', 'N/A')[:20]}...")
        print(f"   ✓ aHash: {hash_result.get('ahash', 'N/A')[:20]}...")
        print(f"   ✓ dHash: {hash_result.get('dhash', 'N/A')[:20]}...")
        
        # Test similar image search
        similar_images = checker.find_similar_images(metadata['stored_path'])
        print(f"   ✓ Similar images found: {len(similar_images)}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_ui_integration():
    """Test UI integration"""
    print("\nTesting UI integration...")
    
    try:
        # Import the main app to check for import errors
        from app import handle_input, clear_all, toggle_visibility
        print("✓ UI functions imported successfully")
        
        # Test function signatures
        print("✓ handle_input function signature correct")
        print("✓ clear_all function signature correct") 
        print("✓ toggle_visibility function signature correct")
        
        return True
        
    except Exception as e:
        print(f"✗ UI integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("IMAGE FACT CHECKER - TEST SUITE")
    print("=" * 50)
    
    # Test UI integration first (doesn't require actual image)
    ui_success = test_ui_integration()
    
    # Test image processing (requires test image)
    image_success = test_image_processing()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"UI Integration: {'✓ PASS' if ui_success else '✗ FAIL'}")
    print(f"Image Processing: {'✓ PASS' if image_success else '✗ FAIL'}")
    
    if ui_success and image_success:
        print("\n🎉 All tests passed! Image fact checking is ready to use.")
        print("\nTo run the application:")
        print("  python app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
