#!/usr/bin/env python3
"""
Quick fix script to resolve NumPy compatibility issues and test visual models
"""

import subprocess
import sys
import os

def fix_numpy_compatibility():
    """Fix NumPy compatibility issue"""
    print("Fixing NumPy compatibility issue...")
    
    try:
        # Downgrade NumPy to compatible version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0.0", "--force-reinstall"])
        print("✓ NumPy downgraded successfully")
        
        # Upgrade PyTorch to fix security vulnerability
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.6.0", "--upgrade"])
        print("✓ PyTorch upgraded successfully")
        
        # Test imports
        print("\nTesting imports...")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("✓ BLIP models imported successfully")
        
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers imported successfully")
        
        import imagehash
        print("✓ ImageHash imported successfully")
        
        print("\n🎉 All visual analysis dependencies are working!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_visual_models():
    """Test if visual models can be loaded"""
    print("\nTesting visual model loading...")
    
    try:
        # Test BLIP model loading
        print("Loading BLIP captioning model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✓ BLIP model loaded successfully")
        
        # Test CLIP model loading
        print("Loading CLIP embedding model...")
        from sentence_transformers import SentenceTransformer
        clip_model = SentenceTransformer('clip-ViT-B-32')
        print("✓ CLIP model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Visual model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("VISUAL ANALYSIS DEPENDENCY FIX")
    print("=" * 50)
    
    # Fix NumPy compatibility
    if fix_numpy_compatibility():
        # Test visual models
        if test_visual_models():
            print("\n✅ All visual analysis components are ready!")
            print("\nYou can now restart your app.py and test image captioning.")
        else:
            print("\n⚠️  Visual models failed to load. Check your internet connection for model downloads.")
    else:
        print("\n❌ Failed to fix dependencies. Please check the error messages above.")
