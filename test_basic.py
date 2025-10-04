"""
Test Basic Components Without LLM
"""

from src.scrapers.jina_reader import fetch_url_content

print("Testing Jina Reader...")
print("="*70)

url = "https://www.apple.com/iphone-15/"
print(f"Fetching: {url}\n")

content = fetch_url_content(url)

if content.success:
    print("✅ SUCCESS!")
    print(f"\nTitle: {content.title}")
    print(f"Length: {len(content.content)} chars")
    print(f"\nFirst 1000 chars:")
    print("="*70)
    print(content.content[:1000])
    print("="*70)
else:
    print(f"❌ FAILED: {content.error}")
