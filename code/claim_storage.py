import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
import json as json_lib


class ClaimStorageManager:
    """Manages persistent storage for verified claims (with optional embeddings)"""
    
    def __init__(
        self,
        storage_dir: str = None,
        embedding_model: SentenceTransformer = None,
        snapshot_dir: str = None,
        flagged_sources_dir: str = None,
    ):
        """
        Initialize the claim storage manager
        
        Args:
            storage_dir: Directory to store claim metadata JSON files (defaults to claim_metadata relative to module)
            embedding_model: Optional pre-initialized SentenceTransformer model to reuse
        """
        base_dir = os.path.dirname(__file__)
        if storage_dir is None:
            # Use path relative to module file
            storage_dir = os.path.join(base_dir, "claim_metadata")
        self.storage_dir = storage_dir
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        if snapshot_dir is None:
            snapshot_dir = os.path.join(base_dir, "claim_snapshots")
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

        if flagged_sources_dir is None:
            flagged_sources_dir = os.path.join(base_dir, "flagged_sources")
        self.flagged_sources_dir = flagged_sources_dir
        os.makedirs(flagged_sources_dir, exist_ok=True)
        
        # Initialize or reuse embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
            print("✓ Reusing provided embedding model")
        else:
            try:
                print("Loading SentenceTransformer model for claim embeddings...")
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ Embedding model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def _generate_claim_id(self) -> str:
        """Generate a unique UUID for a new claim"""
        return str(uuid.uuid4())
    
    def _normalize_claim_text(self, text: str) -> str:
        """
        Clean and normalize claim text for comparison
        
        Args:
            text: Raw claim text
            
        Returns:
            Normalized claim text
        """
        if not text:
            return ""
        
        # Lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(text.lower().strip().split())
        
        return normalized
    
    def compute_claim_embedding(self, claim_text: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a claim using SentenceTransformer
        
        Args:
            claim_text: The claim text to embed
            
        Returns:
            Embedding vector as numpy array, or None if model unavailable
        """
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode([claim_text])
            return embedding[0]
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None
    
    def save_claim_record(
        self,
        claim_text: str,
        claim_text_original: str,
        classification: str,
        credibility_score: int,
        explanation: str,
        evidence_sources: List[Dict],
        language: str = 'unknown',
        submitted_url: str = None,
        article_text: str = None,
        url_snapshot_html: str = None,
        flagged_sources: List[Dict] = None,
        onchain_metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Save claim verification results to JSON file
        
        Args:
            claim_text: Normalized claim text
            claim_text_original: Original claim text as submitted
            classification: REAL, FAKE, MISINFORMATION, or UNSURE
            credibility_score: 0-100 credibility score
            explanation: Explanation text
            evidence_sources: List of dicts with url, title, snippet
            language: Language code (e.g., 'en', 'bn', 'hi')
            submitted_url: Optional URL submitted with the claim
            article_text: Optional article text submitted alongside URL
            url_snapshot_html: Raw HTML content of the submitted URL (saved only for FAKE/MISINFORMATION)
            flagged_sources: Optional list of flagged sources dicts (url/title/snippet/similarity/stance/reason)
            onchain_metadata: Optional dict with blockchain registration/lookup info
            
        Returns:
            claim_id if successful, None otherwise
        """
        try:
            claim_id = self._generate_claim_id()
            normalized_claim = self._normalize_claim_text(claim_text)
            
            # Compute embedding
            embedding = self.compute_claim_embedding(normalized_claim)
            embedding_list = embedding.tolist() if embedding is not None else None
            
            # Create metadata record
            metadata = {
                'claim_id': claim_id,
                'claim_text': normalized_claim,
                'claim_text_original': claim_text_original,
                'classification': classification,
                'credibility_score': credibility_score,
                'explanation': explanation,
                'evidence_sources': evidence_sources,
                'language': language,
                'timestamp': datetime.now().isoformat(),
                'embedding': embedding_list,
                'submitted_url': submitted_url,
                'article_text': article_text,
                'url_snapshot_path': None,
                'flagged_sources': flagged_sources or [],
                'onchain': onchain_metadata or {},
            }

            # Save snapshot for flagged claims
            if url_snapshot_html and classification in {'FAKE', 'MISINFORMATION'}:
                snapshot_path = os.path.join(self.snapshot_dir, f"{claim_id}.html")
                with open(snapshot_path, 'w', encoding='utf-8') as f:
                    f.write(url_snapshot_html)
                metadata['url_snapshot_path'] = snapshot_path
            
            # Save to JSON file
            metadata_path = os.path.join(self.storage_dir, f"{claim_id}.json")
            self._save_claim_metadata(metadata, metadata_path)
            
            print(f"Saved claim record: {claim_id}")
            return claim_id
        
        except Exception as e:
            print(f"Error saving claim record: {e}")
            return None
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Dict]:
        """
        Load claim record from JSON file by claim_id
        
        Args:
            claim_id: UUID of the claim
            
        Returns:
            Claim metadata dict, or None if not found
        """
        try:
            metadata_path = os.path.join(self.storage_dir, f"{claim_id}.json")
            if not os.path.exists(metadata_path):
                return None
            
            return self._load_claim_metadata(metadata_path)
        
        except Exception as e:
            print(f"Error loading claim by ID: {e}")
            return None
    
    def get_all_fake_claims(self) -> List[Dict]:
        """
        Return list of all claims classified as FAKE
        
        Returns:
            List of claim metadata dicts
        """
        fake_claims = []
        
        try:
            if not os.path.exists(self.storage_dir):
                return fake_claims
            
            for filename in os.listdir(self.storage_dir):
                if not filename.endswith('.json'):
                    continue
                
                metadata_path = os.path.join(self.storage_dir, filename)
                try:
                    record = self._load_claim_metadata(metadata_path)
                    if record and record.get('classification') == 'FAKE':
                        fake_claims.append(record)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error retrieving fake claims: {e}")
        
        return fake_claims
    
    def _load_claim_metadata(self, file_path: str) -> Optional[Dict]:
        """
        Read JSON file and return dict
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Metadata dict, or None on error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {file_path}: {e}")
            return None
    
    def _save_claim_metadata(self, metadata: Dict, file_path: str):
        """
        Write dict to JSON file with proper encoding
        
        Args:
            metadata: Metadata dict to save
            file_path: Path to save JSON file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata to {file_path}: {e}")
            raise

    def list_recent_claims(self, limit: int = 10) -> List[Dict]:
        """
        Return a list of recent claims sorted by timestamp desc.

        Each entry: {claim_id, classification, claim_text_original, timestamp, language}
        """
        claims = []
        try:
            pattern = os.path.join(self.storage_dir, "*.json")
            files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)[:limit]
            for path in files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json_lib.load(f)
                        claims.append({
                            "claim_id": data.get("claim_id"),
                            "classification": data.get("classification"),
                            "claim_text_original": data.get("claim_text_original", "")[:180],
                            "timestamp": data.get("timestamp"),
                            "language": data.get("language", "unknown")
                        })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
        except Exception as outer:
            print(f"Error listing recent claims: {outer}")
        return claims
