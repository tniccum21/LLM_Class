#!/usr/bin/env python3
"""
Test LM Studio API endpoints to see what's available
"""

import requests
import json

def test_endpoints():
    """Test various LM Studio endpoints."""
    base_url = "http://localhost:1234"

    endpoints = [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/models",
        "/api/v1/generate",
        "/completions",
        "/chat/completions"
    ]

    print("Testing LM Studio endpoints...")
    print("-" * 50)

    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"\nTesting: {url}")

        try:
            # Try GET first (for models endpoint)
            response = requests.get(url, timeout=2)
            print(f"  GET: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  Response: {json.dumps(data, indent=2)[:200]}...")

        except Exception as e:
            print(f"  GET failed: {e}")

        # Try POST for completion endpoints
        if "completion" in endpoint or "generate" in endpoint:
            try:
                test_payload = {
                    "prompt": "Test",
                    "max_tokens": 10,
                    "temperature": 0.1
                }

                response = requests.post(url, json=test_payload, timeout=5)
                print(f"  POST (prompt): {response.status_code}")

                if response.status_code == 400:
                    print(f"  Error: {response.text[:100]}")

            except Exception as e:
                print(f"  POST failed: {e}")

def test_completions():
    """Test the completions endpoint specifically."""
    print("\n" + "=" * 50)
    print("Testing /v1/completions with actual request")
    print("=" * 50)

    url = "http://localhost:1234/v1/completions"

    payload = {
        "prompt": "Extract problems from: Server is down. Return JSON: {\"problems\": [], \"products\": []}",
        "max_tokens": 100,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response structure: {list(data.keys())}")

            if 'choices' in data and len(data['choices']) > 0:
                text = data['choices'][0].get('text', '')
                print(f"Generated text: {text[:200]}")
        else:
            print(f"Error: {response.text[:200]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_endpoints()
    test_completions()