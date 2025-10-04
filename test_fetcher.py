"""
Test the Jina Reader URL fetcher
"""

from src.scrapers.jina_reader import JinaReader, fetch_url_content

def test_single_url():
    """Test fetching a single URL"""
    print("=" * 60)
    print("🧪 TEST 1: Single URL Fetch")
    print("=" * 60)
    
    # Test with a product page
    test_url = "https://www.apple.com/iphone-15/"
    
    print(f"\n📥 Fetching: {test_url}\n")
    
    content = fetch_url_content(test_url)
    
    if content.success:
        print("✅ SUCCESS!")
        print(f"\n📄 Title: {content.title}")
        print(f"⏱️  Fetch Time: {content.fetch_time:.2f}s")
        print(f"📊 Content Length: {len(content.content)} characters")
        print(f"\n📝 First 500 characters:\n")
        print("-" * 60)
        print(content.content[:500])
        print("-" * 60)
    else:
        print(f"❌ FAILED: {content.error}")


def test_multiple_urls():
    """Test fetching multiple URLs"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Multiple URLs Fetch")
    print("=" * 60 + "\n")
    
    test_urls = [
        "https://www.apple.com/iphone-15/",
        "https://www.samsung.com/us/smartphones/galaxy-s24/",
    ]
    
    reader = JinaReader()
    results = reader.fetch_multiple(test_urls)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    for url, content in results.items():
        if content.success:
            print(f"\n✅ {content.title}")
            print(f"   📏 {len(content.content)} chars")
        else:
            print(f"\n❌ {url}")
            print(f"   ⚠️  {content.error}")


def test_invalid_url():
    """Test with invalid URL"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Invalid URL Handling")
    print("=" * 60 + "\n")
    
    invalid_url = "not-a-valid-url"
    
    content = fetch_url_content(invalid_url)
    
    if not content.success:
        print(f"✅ Correctly handled invalid URL")
        print(f"   Error message: {content.error}")
    else:
        print(f"❌ Should have failed for invalid URL")


if __name__ == "__main__":
    print("\n🚀 Starting URL Fetcher Tests...\n")
    
    # Run tests
    test_single_url()
    test_multiple_urls()
    test_invalid_url()
    
    print("\n" + "=" * 60)
    print("✅ All Tests Complete!")
    print("=" * 60 + "\n")
