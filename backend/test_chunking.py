#!/usr/bin/env python3
"""
Test script for chunking service to validate fixes and check for issues.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.chunking_service import chunking_service
from utils.logger import chat_logger

def test_basic_chunking():
    """Test basic chunking functionality"""
    print("\n" + "=" * 60)
    print("TESTING BASIC CHUNKING")
    print("=" * 60)
    
    test_text = """
    This is a test document. It has multiple sentences.
    Each sentence should be properly handled by the chunking algorithm.
    
    This paragraph has more content to test the chunking behavior.
    The algorithm should split at appropriate points like sentences or paragraphs.
    """
    
    start_time = time.time()
    
    try:
        chunks = chunking_service.chunk_text(test_text, chunk_size=100, chunk_overlap=20)
        
        end_time = time.time()
        
        print(f"✓ Chunking completed in {end_time - start_time:.4f} seconds")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.1f} characters")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk)}")
            print(f"  Content: {chunk[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic chunking failed: {e}")
        return False

def test_chunking_without_breaks():
    """Test chunking with text that has no natural break points"""
    print("\n" + "=" * 60)
    print("TESTING CHUNKING WITHOUT BREAK POINTS")
    print("=" * 60)
    
    # Text without periods or newlines
    test_text = "This is a long sentence without any natural break points that should test the chunking algorithm's ability to handle such cases properly and ensure it doesn't loop infinitely"
    
    start_time = time.time()
    
    try:
        chunks = chunking_service.chunk_text(test_text, chunk_size=50, chunk_overlap=10)
        
        end_time = time.time()
        
        print(f"✓ Chunking completed in {end_time - start_time:.4f} seconds")
        print(f"Number of chunks: {len(chunks)}")
        
        if len(chunks) > 1:
            print("✓ Successfully split long sentence into multiple chunks")
        else:
            print("⚠ Only one chunk created - may indicate issue with break point detection")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk)}")
            print(f"  Content: {chunk}")
        
        return True
        
    except Exception as e:
        print(f"✗ Chunking without breaks failed: {e}")
        return False

def test_large_text_chunking():
    """Test chunking with large text to check performance and infinite loops"""
    print("\n" + "=" * 60)
    print("TESTING LARGE TEXT CHUNKING")
    print("=" * 60)
    
    # Generate large text
    large_text = "This is a sentence. " * 500  # About 10,000 characters
    
    start_time = time.time()
    
    try:
        chunks = chunking_service.chunk_text(large_text, chunk_size=1000, chunk_overlap=200)
        
        end_time = time.time()
        
        print(f"✓ Large text chunking completed in {end_time - start_time:.4f} seconds")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Total text length: {len(large_text)}")
        print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.1f} characters")
        
        # Check for reasonable number of chunks
        expected_chunks = len(large_text) // (1000 - 200) + 1  # Approximate
        if len(chunks) <= expected_chunks + 5:  # Allow some margin
            print("✓ Reasonable number of chunks generated")
        else:
            print(f"⚠ Potentially too many chunks: {len(chunks)} (expected ~{expected_chunks})")
        
        return True
        
    except Exception as e:
        print(f"✗ Large text chunking failed: {e}")
        return False

def test_chunking_edge_cases():
    """Test edge cases for chunking"""
    print("\n" + "=" * 60)
    print("TESTING CHUNKING EDGE CASES")
    print("=" * 60)
    
    test_cases = [
        ("Empty text", ""),
        ("Very short text", "Short"),
        ("Text with only spaces", "   "),
        ("Text smaller than chunk size", "This is a short text that is smaller than the default chunk size of 1000 characters."),
        ("Text exactly at chunk size", "A" * 1000),
    ]
    
    all_passed = True
    
    for name, text in test_cases:
        print(f"\nTesting: {name}")
        try:
            chunks = chunking_service.chunk_text(text, chunk_size=1000, chunk_overlap=200)
            print(f"  ✓ Passed - Generated {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all chunking tests"""
    print("Starting Chunking Service Tests")
    print("=" * 60)
    
    tests = [
        test_basic_chunking,
        test_chunking_without_breaks,
        test_large_text_chunking,
        test_chunking_edge_cases,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"ERROR: Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("CHUNKING TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed / total * 100:.1f}%")
    
    if passed == total:
        print("✓ All chunking tests passed!")
    else:
        print(f"⚠ {total - passed} tests failed")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)