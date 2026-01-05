import os
import json
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image, ExifTags
import cv2
import numpy as np
import imagehash
from sklearn.metrics.pairwise import cosine_similarity

# OCR imports with fallback handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Will use Tesseract fallback.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR functionality will be limited.")

# Visual analysis imports with fallback handling
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    VISUAL_MODELS_AVAILABLE = True
except ImportError:
    VISUAL_MODELS_AVAILABLE = False
    print("Warning: Visual analysis models not available. Will skip captioning and embeddings.")

from fact_check_llm import verify_news

class ImageFactChecker:
    """Handles image-based fact checking with OCR and verification"""
    
    def __init__(self, upload_dir: str = "uploaded_images", metadata_dir: str = "image_metadata"):
        self.upload_dir = upload_dir
        self.metadata_dir = metadata_dir
        self.ocr_reader = None
        self.caption_model = None
        self.caption_processor = None
        self.embedding_model = None
        # Ensure directories exist
        import os
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Initialize OCR reader if available
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['bn', 'en'], gpu=False)
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                self.ocr_reader = None
        
        # Initialize visual analysis models if available
        if VISUAL_MODELS_AVAILABLE:
            try:
                print("Initializing visual analysis models...")
                
                # Try to load models with safetensors to avoid security issue
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
                
                # Initialize BLIP-2 for image captioning
                print("Loading BLIP processor...")
                self.caption_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_safetensors=True
                )
                print("Loading BLIP model...")
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_safetensors=True
                )
                print("✓ BLIP captioning model initialized successfully")
                
                # Initialize CLIP for image embeddings
                print("Loading CLIP model...")
                self.embedding_model = SentenceTransformer('clip-ViT-B-32')
                print("✓ CLIP embedding model initialized successfully")
                
            except Exception as e:
                print(f"✗ Failed to initialize visual models: {e}")
                print(f"Error type: {type(e).__name__}")
                print("⚠️  Visual analysis will be disabled - OCR-only mode")
                import traceback
                traceback.print_exc()
                self.caption_model = None
                self.caption_processor = None
                self.embedding_model = None
        else:
            print("⚠️  Visual models not available - skipping initialization")
    
    def process_image_upload(self, image_path_or_url: str) -> Dict:
        """
        Process image upload from file path or URL
        Returns metadata dict with file_path, timestamp, dimensions
        """
        try:
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Handle URL vs file path
            if image_path_or_url.startswith(('http://', 'https://')):
                # Download from URL
                response = requests.get(image_path_or_url, timeout=30)
                response.raise_for_status()
                
                # Determine file extension from URL or content type
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'webp' in content_type:
                    ext = '.webp'
                else:
                    ext = '.jpg'  # Default fallback
                
                filename = f"{image_id}{ext}"
                file_path = os.path.join(self.upload_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                # Local file path
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                
                # Get file extension
                _, ext = os.path.splitext(image_path_or_url)
                if not ext:
                    ext = '.jpg'  # Default fallback
                
                filename = f"{image_id}{ext}"
                file_path = os.path.join(self.upload_dir, filename)
                
                # Copy file to upload directory
                with open(image_path_or_url, 'rb') as src, open(file_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Validate and normalize image
            normalized_path = self._normalize_image_orientation(file_path)
            
            # Get image metadata
            with Image.open(normalized_path) as img:
                width, height = img.size
                format_name = img.format
            
            metadata = {
                'image_id': image_id,
                'original_path': image_path_or_url,
                'stored_path': normalized_path,
                'filename': filename,
                'upload_timestamp': datetime.now().isoformat(),
                'dimensions': {'width': width, 'height': height},
                'format': format_name,
                'file_size': os.path.getsize(normalized_path)
            }
            
            # Save metadata
            self._save_image_metadata(metadata, image_id)
            
            return metadata
            
        except Exception as e:
            print(f"Error processing image upload: {e}")
            raise
    
    def _normalize_image_orientation(self, image_path: str) -> str:
        """
        Normalize image orientation based on EXIF data
        Returns path to normalized image
        """
        try:
            with Image.open(image_path) as img:
                # Check for EXIF orientation
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = dict(img._getexif().items())
                    orientation = exif.get(ExifTags.Orientation, 1)
                    
                    # Rotate image based on orientation
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
                
                # Save normalized image
                normalized_path = image_path.replace('.', '_normalized.')
                img.save(normalized_path, quality=95, optimize=True)
                
                # Replace original with normalized version
                os.replace(normalized_path, image_path)
                
                return image_path
                
        except Exception as e:
            print(f"Warning: Could not normalize image orientation: {e}")
            return image_path
    
    def _save_image_metadata(self, metadata: Dict, image_id: str):
        """Save image metadata to JSON file"""
        metadata_path = os.path.join(self.metadata_dir, f"{image_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from image using EasyOCR (primary) or Tesseract (fallback)
        Returns dict with ocr_text, language, confidence, segments
        """
        try:
            # Try EasyOCR first
            if self.ocr_reader:
                return self._extract_with_easyocr(image_path)
            
            # Fallback to Tesseract
            if TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(image_path)
            
            raise Exception("No OCR engine available")
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return {
                'ocr_text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'segments': [],
                'error': str(e)
            }
    
    def _extract_with_easyocr(self, image_path: str) -> Dict:
        """Extract text using EasyOCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Run OCR
            results = self.ocr_reader.readtext(image)
            
            # Process results
            segments = []
            full_text = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence results
                    segments.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    full_text.append(text)
            
            ocr_text = ' '.join(full_text)
            
            # Detect language
            language = self._detect_language(ocr_text)
            
            return {
                'ocr_text': ocr_text,
                'language': language,
                'confidence': np.mean([s['confidence'] for s in segments]) if segments else 0.0,
                'segments': segments,
                'method': 'easyocr'
            }
            
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}")
            raise
    
    def _extract_with_tesseract(self, image_path: str) -> Dict:
        """Extract text using Tesseract as fallback"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Convert to RGB for Tesseract
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run OCR with Bengali and English
            custom_config = r'--oem 3 --psm 6 -l ben+eng'
            text = pytesseract.image_to_string(image_rgb, config=custom_config)
            
            # Get detailed data for confidence
            data = pytesseract.image_to_data(image_rgb, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            # Detect language
            language = self._detect_language(text)
            
            return {
                'ocr_text': text.strip(),
                'language': language,
                'confidence': avg_confidence,
                'segments': [],  # Tesseract doesn't provide detailed segments easily
                'method': 'tesseract'
            }
            
        except Exception as e:
            print(f"Tesseract extraction failed: {e}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text (simple heuristic)"""
        if not text.strip():
            return 'unknown'
        
        # Check for Bengali characters
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0 and bengali_chars / total_chars > 0.3:
            return 'bengali'
        else:
            return 'english'
    
    def clean_ocr_text(self, raw_text: str) -> str:
        """Clean and normalize OCR text"""
        if not raw_text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(raw_text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common OCR mistake
        text = text.replace('0', 'O')  # In some contexts
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()
    
    def generate_image_caption(self, image_path: str) -> Dict:
        """
        Generate image caption using BLIP model
        Returns dict with caption and confidence
        """
        print(f"Generating caption for image: {image_path}")
        
        if not self.caption_model or not self.caption_processor:
            print("Captioning model not available")
            return {'caption': '', 'confidence': 0.0, 'error': 'Captioning model not available'}
        
        try:
            print("Loading image...")
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            print(f"Image loaded: {image.size}")
            
            print("Processing image with BLIP...")
            # Generate caption
            inputs = self.caption_processor(image, return_tensors="pt")
            print("Generating caption...")
            out = self.caption_model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            print(f"Generated caption: {caption}")
            
            return {
                'caption': caption,
                'confidence': 0.8,  # BLIP doesn't provide confidence scores
                'method': 'blip'
            }
            
        except Exception as e:
            print(f"Caption generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'caption': '', 'confidence': 0.0, 'error': str(e)}
    
    def compute_image_embeddings(self, image_path: str) -> Dict:
        """
        Compute image embeddings using CLIP
        Returns dict with embedding vector
        """
        if not self.embedding_model:
            return {'embedding': None, 'error': 'Embedding model not available'}
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Compute embedding
            embedding = self.embedding_model.encode([image])
            
            return {
                'embedding': embedding[0].tolist(),  # Convert to list for JSON serialization
                'dimension': len(embedding[0]),
                'method': 'clip'
            }
            
        except Exception as e:
            print(f"Embedding computation failed: {e}")
            return {'embedding': None, 'error': str(e)}
    
    def compute_perceptual_hashes(self, image_path: str) -> Dict:
        """
        Compute perceptual hashes for duplicate detection
        Returns dict with pHash, aHash, dHash
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Compute different types of hashes
            phash = imagehash.phash(image)
            ahash = imagehash.average_hash(image)
            dhash = imagehash.dhash(image)
            
            return {
                'phash': str(phash),
                'ahash': str(ahash),
                'dhash': str(dhash),
                'phash_int': int(str(phash), 16),
                'ahash_int': int(str(ahash), 16),
                'dhash_int': int(str(dhash), 16)
            }
            
        except Exception as e:
            print(f"Hash computation failed: {e}")
            return {'phash': '', 'ahash': '', 'dhash': '', 'error': str(e)}
    
    def find_similar_images(self, image_path: str, threshold: float = 0.8) -> List[Dict]:
        """
        Find similar images in the metadata directory using perceptual hashes
        Returns list of similar image metadata
        """
        try:
            # Compute hashes for input image
            input_hashes = self.compute_perceptual_hashes(image_path)
            if 'error' in input_hashes:
                return []
            
            similar_images = []
            
            # Check all metadata files
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    metadata_path = os.path.join(self.metadata_dir, filename)
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Check if this image has hashes
                        if 'perceptual_hashes' in metadata:
                            stored_hashes = metadata['perceptual_hashes']
                            
                            # Compare pHash (most reliable for near-duplicates)
                            if 'phash_int' in stored_hashes and 'phash_int' in input_hashes:
                                hamming_distance = bin(input_hashes['phash_int'] ^ stored_hashes['phash_int']).count('1')
                                similarity = 1.0 - (hamming_distance / 64.0)  # pHash is 64-bit
                                
                                if similarity >= threshold:
                                    similar_images.append({
                                        'image_id': metadata['image_id'],
                                        'similarity': similarity,
                                        'stored_path': metadata['stored_path'],
                                        'upload_timestamp': metadata['upload_timestamp']
                                    })
                    
                    except Exception as e:
                        print(f"Error reading metadata file {filename}: {e}")
                        continue
            
            # Sort by similarity
            similar_images.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_images
            
        except Exception as e:
            print(f"Similar image search failed: {e}")
            return []
    
    def verify_image_news(self, image_input) -> Tuple[str, str, str, str, str, str, int, str]:
        """
        Enhanced function to verify news from image with visual analysis
        Returns: (claim, verdict_english, verdict_original, ocr_text, caption, visual_summary, credibility_score)
        """
        try:
            # Process image upload
            metadata = self.process_image_upload(image_input)
            
            # Extract OCR text
            ocr_result = self.extract_text_from_image(metadata['stored_path'])
            ocr_text = self.clean_ocr_text(ocr_result['ocr_text'])
            
            # Generate image caption
            caption_result = self.generate_image_caption(metadata['stored_path'])
            caption = caption_result.get('caption', '')
            
            # Compute image embeddings
            embedding_result = self.compute_image_embeddings(metadata['stored_path'])
            
            # Compute perceptual hashes
            hash_result = self.compute_perceptual_hashes(metadata['stored_path'])
            
            # Find similar images
            similar_images = self.find_similar_images(metadata['stored_path'])
            
            # Create visual summary
            visual_summary = self._create_visual_summary(caption, similar_images, ocr_text)
            
            credibility_score = 0
            classification = 'UNSURE'
            
            # Verify the extracted text using existing pipeline
            if ocr_text.strip():
                claim, verdict_english, verdict_original, credibility_score, classification, _, _ = verify_news(ocr_text)
            else:
                # If no OCR text, try using caption for verification
                if caption.strip():
                    claim, verdict_english, verdict_original, credibility_score, classification, _, _ = verify_news(caption)
                    ocr_text = "No text detected in image"
                else:
                    error_msg = "No text or visual content could be extracted from the image."
                    return error_msg, error_msg, error_msg, "No text detected", "No caption generated", visual_summary, 0, 'UNSURE'
            
            # Update metadata with all results
            metadata['ocr_result'] = ocr_result
            metadata['caption_result'] = caption_result
            metadata['embedding_result'] = embedding_result
            metadata['perceptual_hashes'] = hash_result
            metadata['similar_images'] = similar_images
            metadata['credibility_score'] = credibility_score
            metadata['classification'] = classification
            self._save_image_metadata(metadata, metadata['image_id'])
            
            return claim, verdict_english, verdict_original, ocr_text, caption, visual_summary, credibility_score, classification
            
        except Exception as e:
            print(f"Error in image verification: {e}")
            error_msg = f"Error processing image: {str(e)}"
            return error_msg, error_msg, error_msg, "Error occurred", "Error occurred", "Error occurred", 0, 'UNSURE'
    
    def _create_visual_summary(self, caption: str, similar_images: List[Dict], ocr_text: str) -> str:
        """Create a summary of visual analysis results"""
        summary_parts = []
        
        if caption:
            summary_parts.append(f"Visual Description: {caption}")
        
        if similar_images:
            summary_parts.append(f"Found {len(similar_images)} similar images")
            if len(similar_images) > 0:
                best_match = similar_images[0]
                summary_parts.append(f"Best match similarity: {best_match['similarity']:.2f}")
        
        if ocr_text and len(ocr_text) > 10:
            summary_parts.append(f"Text extracted: {len(ocr_text)} characters")
        
        return " | ".join(summary_parts) if summary_parts else "No visual analysis available"

# Global instance
image_fact_checker = ImageFactChecker()

def verify_image_news(image_input):
    """
    Wrapper function for Gradio integration
    Returns: (claim, verdict_english, verdict_original, ocr_text, caption, visual_summary, credibility_score, classification) - 8-tuple
    """
    return image_fact_checker.verify_image_news(image_input)
