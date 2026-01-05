#!/usr/bin/env python3
"""
Test API connectivity for Groq and Serper
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

grok_api_key = os.getenv('GROQ_API_KEY', '').strip()
serp_dev_api_key = os.getenv('SERP_DEV_API_KEY', '').strip()

print("="*60)
print("TESTING API CONNECTIVITY")
print("="*60)

# Test Serper API
print("\n1. Testing Serper API...")
if not serp_dev_api_key:
    print("   ❌ SERP_DEV_API_KEY not set")
else:
    try:
        _SERPER_SEARCH_URL = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": serp_dev_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": "test query",
            "num": 1
        }
        
        response = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✓ Serper API is working!")
        elif response.status_code == 403:
            print("   ❌ 403 Unauthorized - API key is invalid or expired")
            print(f"   Response: {response.text}")
        elif response.status_code == 401:
            print("   ❌ 401 Unauthorized - API key is missing or malformed")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test Groq API
print("\n2. Testing Groq API...")
if not grok_api_key:
    print("   ❌ GROQ_API_KEY not set")
else:
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(api_key=grok_api_key, model_name='llama-3.1-8b-instant')
        response = llm.invoke("Say 'test' if you can read this.")
        print(f"   ✓ Groq API is working!")
        print(f"   Response: {response.content[:100]}")
    except Exception as e:
        error_str = str(e)
        print(f"   ❌ Error: {error_str[:200]}")
        
        if 'organization_restricted' in error_str.lower():
            print("\n   ⚠️  ORGANIZATION RESTRICTED ERROR")
            print("   This means your Groq account/organization has been restricted.")
            print("   Possible causes:")
            print("   - Account suspended or flagged")
            print("   - Rate limit exceeded")
            print("   - Terms of service violation")
            print("   - Billing issue")
            print("\n   Solutions:")
            print("   1. Check your Groq account status at https://console.groq.com")
            print("   2. Contact Groq support: support@groq.com")
            print("   3. Try creating a new API key")
            print("   4. Check if you have any outstanding invoices")
        elif 'unauthorized' in error_str.lower() or '401' in error_str:
            print("\n   ⚠️  UNAUTHORIZED ERROR")
            print("   Your API key is invalid or expired.")
            print("   Solutions:")
            print("   1. Generate a new API key at https://console.groq.com/keys")
            print("   2. Make sure you copied the full key (no spaces, no truncation)")
            print("   3. Check if the key has expired")
        elif 'rate limit' in error_str.lower():
            print("\n   ⚠️  RATE LIMIT ERROR")
            print("   You've exceeded the API rate limit.")
            print("   Solutions:")
            print("   1. Wait a few minutes and try again")
            print("   2. Upgrade your Groq plan for higher limits")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nIf both APIs are failing:")
print("1. Verify your API keys are correct and active")
print("2. Check your account status on the provider websites")
print("3. Ensure you have sufficient credits/quota")
print("4. Try regenerating the API keys")

