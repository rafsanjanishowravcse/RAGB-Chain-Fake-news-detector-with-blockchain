#!/usr/bin/env python3
"""
Diagnostic script to check API key configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
print(f"Script directory: {script_dir}")
print(f"Current working directory: {os.getcwd()}")

# Try loading .env from script directory
env_path = script_dir / '.env'
print(f"\nLooking for .env file at: {env_path}")
print(f".env file exists: {env_path.exists()}")

# Load .env explicitly from script directory
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✓ Loaded .env from script directory")
else:
    # Try loading from current directory
    load_dotenv()
    print("⚠️  Tried loading .env from current directory")

# Check for .env in parent directory
parent_env = script_dir.parent / '.env'
if parent_env.exists():
    print(f"Found .env in parent directory: {parent_env}")
    load_dotenv(dotenv_path=parent_env, override=True)
    print("✓ Loaded .env from parent directory")

print("\n" + "="*60)
print("API KEY STATUS")
print("="*60)

# Check each API key
grok_api_key = os.getenv('GROQ_API_KEY')
serp_dev_api_key = os.getenv('SERP_DEV_API_KEY')
sarvam_api_key = os.getenv('SARVAM_API_KEY')

print(f"\nGROQ_API_KEY:")
if not grok_api_key:
    print("  ❌ NOT SET")
elif grok_api_key == 'your_grok_api_key_here':
    print("  ⚠️  Still has placeholder value")
else:
    print(f"  ✓ SET (length: {len(grok_api_key)} chars)")
    print(f"  First 10 chars: {grok_api_key[:10]}...")
    print(f"  Last 10 chars: ...{grok_api_key[-10:]}")

print(f"\nSERP_DEV_API_KEY:")
if not serp_dev_api_key:
    print("  ❌ NOT SET")
elif serp_dev_api_key == 'your_serper_dev_api_key_here':
    print("  ⚠️  Still has placeholder value")
else:
    print(f"  ✓ SET (length: {len(serp_dev_api_key)} chars)")
    print(f"  First 10 chars: {serp_dev_api_key[:10]}...")
    print(f"  Last 10 chars: ...{serp_dev_api_key[-10:]}")

print(f"\nSARVAM_API_KEY:")
if not sarvam_api_key:
    print("  ❌ NOT SET")
elif sarvam_api_key == 'your_sarvam_api_key_here':
    print("  ⚠️  Still has placeholder value")
else:
    print(f"  ✓ SET (length: {len(sarvam_api_key)} chars)")

# Check for common issues
print("\n" + "="*60)
print("COMMON ISSUES CHECK")
print("="*60)

if grok_api_key:
    if grok_api_key.startswith(' ') or grok_api_key.endswith(' '):
        print("⚠️  GROQ_API_KEY has leading/trailing spaces")
    if '\n' in grok_api_key or '\r' in grok_api_key:
        print("⚠️  GROQ_API_KEY contains newlines")

if serp_dev_api_key:
    if serp_dev_api_key.startswith(' ') or serp_dev_api_key.endswith(' '):
        print("⚠️  SERP_DEV_API_KEY has leading/trailing spaces")
    if '\n' in serp_dev_api_key or '\r' in serp_dev_api_key:
        print("⚠️  SERP_DEV_API_KEY contains newlines")

# Test API connectivity (optional)
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if not grok_api_key or grok_api_key == 'your_grok_api_key_here':
    print("1. Set GROQ_API_KEY in .env file")
    print("   Get key from: https://console.groq.com/keys")

if not serp_dev_api_key or serp_dev_api_key == 'your_serper_dev_api_key_here':
    print("2. Set SERP_DEV_API_KEY in .env file")
    print("   Get key from: https://serper.dev/api-key")

if not (grok_api_key and serp_dev_api_key):
    print("\n3. Create .env file in the 'code' directory with:")
    print("   GROQ_API_KEY=your_actual_key_here")
    print("   SERP_DEV_API_KEY=your_actual_key_here")
    print("   SARVAM_API_KEY=your_actual_key_here")

