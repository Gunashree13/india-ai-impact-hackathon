"""
Test Script for AI Voice Detection API
Tests the API with sample audio files.
"""

import base64
import json
import requests
import sys
import os
from pathlib import Path


API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "sk_test_123456789")


def test_health():
    """Test health endpoint."""
    resp = requests.get(f"{API_URL}/health")
    print(f"Health: {resp.json()}")
    return resp.status_code == 200


def test_voice_detection(audio_path, language="English"):
    """Test voice detection with a file."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    print(f"\nTesting: {audio_path}")
    print(f"  Language: {language}")
    print(f"  Audio size: {len(audio_bytes)} bytes")
    print(f"  Base64 size: {len(audio_base64)} chars")

    resp = requests.post(
        f"{API_URL}/api/voice-detection",
        json=payload,
        headers=headers,
        timeout=30
    )

    result = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(result, indent=2)}")
    return result


def test_invalid_api_key():
    """Test that invalid API key is rejected."""
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "dGVzdA=="
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key"
    }

    resp = requests.post(
        f"{API_URL}/api/voice-detection",
        json=payload,
        headers=headers
    )

    print(f"\nInvalid API key test: Status {resp.status_code}")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  PASSED!")


def test_invalid_language():
    """Test that invalid language is rejected."""
    payload = {
        "language": "French",
        "audioFormat": "mp3",
        "audioBase64": "dGVzdA=="
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    resp = requests.post(
        f"{API_URL}/api/voice-detection",
        json=payload,
        headers=headers
    )

    print(f"\nInvalid language test: Status {resp.status_code}")
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    print("  PASSED!")


def test_all_files_in_directory(directory, label="unknown"):
    """Test all audio files in a directory."""
    results = {"correct": 0, "total": 0, "details": []}

    for filepath in sorted(Path(directory).rglob("*.mp3")):
        # Detect language from filename
        lang = "English"
        for l in ["tamil", "english", "hindi", "malayalam", "telugu"]:
            if l in str(filepath).lower():
                lang = l.capitalize()
                break

        try:
            result = test_voice_detection(str(filepath), lang)
            results["total"] += 1

            if label == "ai" and result.get("classification") == "AI_GENERATED":
                results["correct"] += 1
            elif label == "human" and result.get("classification") == "HUMAN":
                results["correct"] += 1

            results["details"].append({
                "file": str(filepath),
                "expected": label.upper(),
                "predicted": result.get("classification"),
                "confidence": result.get("confidenceScore")
            })
        except Exception as e:
            print(f"  Error: {e}")

    if results["total"] > 0:
        accuracy = results["correct"] / results["total"]
        print(f"\n{'='*60}")
        print(f"Results for {label.upper()} samples:")
        print(f"  Accuracy: {results['correct']}/{results['total']} ({accuracy:.1%})")
        print(f"{'='*60}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("AI Voice Detection API - Test Suite")
    print("=" * 60)

    # Health check
    print("\n--- Health Check ---")
    test_health()

    # Error handling tests
    print("\n--- Error Handling Tests ---")
    test_invalid_api_key()
    test_invalid_language()

    # Test with audio files if provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "English"
        test_voice_detection(audio_path, language)

    # Test directories if they exist
    if os.path.exists("data/ai"):
        print("\n\n--- Testing AI Samples ---")
        test_all_files_in_directory("data/ai", label="ai")

    if os.path.exists("data/human"):
        print("\n\n--- Testing Human Samples ---")
        test_all_files_in_directory("data/human", label="human")
