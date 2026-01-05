#!/usr/bin/env python3
"""
Test suite for claim storage functionality
"""

import sys
import os
import json
import shutil

def test_storage_initialization():
    """Test storage initialization"""
    print("Testing storage initialization...")
    
    try:
        from claim_storage import ClaimStorageManager
        
        # Use a test directory
        test_dir = "test_claim_metadata"
        storage = ClaimStorageManager(storage_dir=test_dir)
        
        assert os.path.exists(test_dir), "Storage directory not created"
        print("✓ Storage directory created")
        
        assert storage.embedding_model is not None, "Embedding model not loaded"
        print("✓ Embedding model loaded")
        
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        snapshot_dir = os.path.join(os.path.dirname(__file__), "claim_snapshots")
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        flagged_dir = os.path.join(os.path.dirname(__file__), "flagged_sources")
        if os.path.exists(flagged_dir):
            shutil.rmtree(flagged_dir)
        snapshot_dir = os.path.join(os.path.dirname(__file__), "claim_snapshots")
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        flagged_dir = os.path.join(os.path.dirname(__file__), "flagged_sources")
        if os.path.exists(flagged_dir):
            shutil.rmtree(flagged_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_claim_storage():
    """Test storing a claim record"""
    print("\nTesting claim storage...")
    
    try:
        from claim_storage import ClaimStorageManager
        
        test_dir = "test_claim_metadata"
        storage = ClaimStorageManager(storage_dir=test_dir)
        
        # Create sample claim record
        claim_id = storage.save_claim_record(
            claim_text="test claim normalized",
            claim_text_original="Test Claim Original",
            classification="FAKE",
            credibility_score=85,
            explanation="This is a test explanation",
            evidence_sources=[
                {"url": "https://example.com", "title": "Test Source", "snippet": "Test snippet"}
            ],
            language="en",
            submitted_url="https://example.com/article",
            article_text="Submitted article body",
            url_snapshot_html="<html>snapshot</html>",
            flagged_sources=[{"url": "https://example.com/bad", "title": "Bad Source", "snippet": "Bad info", "similarity": 0.9, "stance": "assert"}],
        )
        
        assert claim_id is not None, "Claim ID not returned"
        print(f"✓ Claim saved with ID: {claim_id}")
        
        # Verify file exists
        claim_file = os.path.join(test_dir, f"{claim_id}.json")
        assert os.path.exists(claim_file), "Claim JSON file not created"
        print("✓ Claim JSON file created")
        
        # Verify contents
        with open(claim_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data['classification'] == 'FAKE', "Classification mismatch"
        assert data['credibility_score'] == 85, "Credibility score mismatch"
        assert data['claim_text_original'] == "Test Claim Original", "Original claim text mismatch"
        assert len(data['evidence_sources']) == 1, "Evidence sources not saved"
        assert data['embedding'] is not None, "Embedding not saved"
        assert data['submitted_url'] == "https://example.com/article", "Submitted URL mismatch"
        assert data['article_text'] == "Submitted article body", "Article text mismatch"
        assert data['url_snapshot_path'] is not None, "Snapshot path not saved"
        assert os.path.exists(data['url_snapshot_path']), "Snapshot file not created"
        assert len(data['flagged_sources']) == 1, "Flagged sources not saved"
        assert "onchain" in data, "On-chain metadata missing"
        print("✓ Claim data verified")
        
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        snapshot_dir = os.path.join(os.path.dirname(__file__), "claim_snapshots")
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        flagged_dir = os.path.join(os.path.dirname(__file__), "flagged_sources")
        if os.path.exists(flagged_dir):
            shutil.rmtree(flagged_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_computation():
    """Test embedding computation"""
    print("\nTesting embedding computation...")
    
    try:
        from claim_storage import ClaimStorageManager
        
        test_dir = "test_claim_metadata"
        storage = ClaimStorageManager(storage_dir=test_dir)
        
        # Test English claim
        english_claim = "This is a test claim in English"
        embedding_en = storage.compute_claim_embedding(english_claim)
        assert embedding_en is not None, "English embedding not computed"
        assert len(embedding_en) > 0, "English embedding is empty"
        print(f"✓ English embedding computed (dimension: {len(embedding_en)})")
        
        # Test Bengali claim (if possible)
        bengali_claim = "এটি একটি পরীক্ষামূলক দাবি"
        embedding_bn = storage.compute_claim_embedding(bengali_claim)
        assert embedding_bn is not None, "Bengali embedding not computed"
        assert len(embedding_bn) > 0, "Bengali embedding is empty"
        print(f"✓ Bengali embedding computed (dimension: {len(embedding_bn)})")
        
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        snapshot_dir = os.path.join(os.path.dirname(__file__), "claim_snapshots")
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        flagged_dir = os.path.join(os.path.dirname(__file__), "flagged_sources")
        if os.path.exists(flagged_dir):
            shutil.rmtree(flagged_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_methods():
    """Test retrieval methods"""
    print("\nTesting retrieval methods...")
    
    try:
        from claim_storage import ClaimStorageManager
        
        test_dir = "test_claim_metadata"
        storage = ClaimStorageManager(storage_dir=test_dir)
        
        # Store multiple claims
        fake_id1 = storage.save_claim_record(
            claim_text="claim one fake",
            claim_text_original="Claim One Fake",
            classification="FAKE",
            credibility_score=80,
            explanation="Explanation 1",
            evidence_sources=[],
            language="en"
        )
        
        fake_id2 = storage.save_claim_record(
            claim_text="claim two fake",
            claim_text_original="Claim Two Fake",
            classification="FAKE",
            credibility_score=75,
            explanation="Explanation 2",
            evidence_sources=[],
            language="en"
        )
        
        real_id = storage.save_claim_record(
            claim_text="claim three real",
            claim_text_original="Claim Three Real",
            classification="REAL",
            credibility_score=0,
            explanation="Explanation 3",
            evidence_sources=[],
            language="en"
        )
        
        # Test get_claim_by_id
        retrieved = storage.get_claim_by_id(fake_id1)
        assert retrieved is not None, "Claim not retrieved by ID"
        assert retrieved['classification'] == 'FAKE', "Classification mismatch"
        print("✓ get_claim_by_id works")
        
        # Test get_all_fake_claims
        fake_claims = storage.get_all_fake_claims()
        assert len(fake_claims) == 2, f"Expected 2 fake claims, got {len(fake_claims)}"
        assert all(c['classification'] == 'FAKE' for c in fake_claims), "Not all claims are FAKE"
        print(f"✓ get_all_fake_claims works (found {len(fake_claims)} fake claims)")
        
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        snapshot_dir = os.path.join(os.path.dirname(__file__), "claim_snapshots")
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("CLAIM STORAGE TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Storage Initialization", test_storage_initialization),
        ("Claim Storage", test_claim_storage),
        ("Embedding Computation", test_embedding_computation),
        ("Retrieval Methods", test_retrieval_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
