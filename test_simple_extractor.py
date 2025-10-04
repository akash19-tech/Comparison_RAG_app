"""
Test Simple Extraction (No LLM)
"""

from src.scrapers.jina_reader import fetch_url_content
import re

url = "https://www.apple.com/iphone-15/"

print("="*70)
print("🧪 Simple Extraction Test (No LLM)")
print("="*70)

# Fetch
print(f"\n📥 Fetching: {url}")
content = fetch_url_content(url)

if not content.success:
    print(f"❌ Failed: {content.error}")
    exit(1)

print(f"✅ Fetched: {content.title}")
print(f"📊 Length: {len(content.content)} chars")

# Simple extraction
print("\n🔍 Extracting information...")

# Extract price
price_patterns = [
    (r'\$\s*(\d+(?:,\d{3})*)', '$'),
    (r'From\s+\$(\d+)', '$'),
]

price = None
currency = None
for pattern, curr in price_patterns:
    match = re.search(pattern, content.content)
    if match:
        price = match.group(1).replace(',', '')
        currency = curr
        break

# Extract features (bullet points)
features = []
lines = content.content.split('\n')
for line in lines:
    line = line.strip()
    if line.startswith(('* ', '- ', '• ')):
        feature = line[2:].strip()
        if 10 < len(feature) < 100 and 'Image' not in feature:
            features.append(feature)
            if len(features) >= 10:
                break

# Results
print("\n" + "="*70)
print("📦 EXTRACTED INFORMATION")
print("="*70)
print(f"Title: {content.title}")
print(f"Price: {currency}{price}" if price else "Price: Not found")
print(f"Features found: {len(features)}")

if features:
    print("\n✨ Features:")
    for i, feat in enumerate(features[:5], 1):
        print(f"   {i}. {feat}")

print("\n" + "="*70)
