"""
Test the Product Extractor
"""

from src.scrapers.jina_reader import fetch_url_content
from src.extractors.product_extractor import ProductExtractor
import json


def test_single_product():
    """Test extracting a single product"""
    print("=" * 70)
    print("🧪 TEST: Product Information Extraction")
    print("=" * 70)
    
    # Fetch content
    test_url = "https://www.apple.com/iphone-15/"
    
    print(f"\n📥 Step 1: Fetching URL...")
    url_content = fetch_url_content(test_url)
    
    if not url_content.success:
        print(f"❌ Failed to fetch URL: {url_content.error}")
        return
    
    print(f"✅ Fetched: {url_content.title}")
    print(f"📊 Content length: {len(url_content.content)} chars")
    
    # Extract product info
    print(f"\n🔍 Step 2: Extracting product information...")
    extractor = ProductExtractor()
    product = extractor.extract(test_url, url_content.content)
    
    print(f"\n✅ Extraction Complete!")
    print("=" * 70)
    print("📦 PRODUCT INFORMATION")
    print("=" * 70)
    
    print(f"\n📱 Title: {product.title}")
    if product.brand:
        print(f"🏢 Brand: {product.brand}")
    if product.price:
        print(f"💰 Price: {product.currency}{product.price}")
    if product.category:
        print(f"📂 Category: {product.category}")
    if product.rating:
        print(f"⭐ Rating: {product.rating}/5 ({product.review_count} reviews)")
    if product.availability:
        print(f"📦 Availability: {product.availability}")
    
    if product.description:
        print(f"\n📝 Description:")
        print(f"   {product.description[:200]}...")
    
    if product.key_features:
        print(f"\n✨ Key Features ({len(product.key_features)}):")
        for i, feature in enumerate(product.key_features[:5], 1):
            print(f"   {i}. {feature}")
        if len(product.key_features) > 5:
            print(f"   ... and {len(product.key_features) - 5} more")
    
    if product.specifications:
        print(f"\n🔧 Specifications ({len(product.specifications)}):")
        for spec, value in list(product.specifications.items())[:5]:
            print(f"   • {spec}: {value}")
        if len(product.specifications) > 5:
            print(f"   ... and {len(product.specifications) - 5} more")
    
    if product.pros:
        print(f"\n👍 Pros ({len(product.pros)}):")
        for pro in product.pros[:3]:
            print(f"   ✅ {pro}")
    
    if product.cons:
        print(f"\n👎 Cons ({len(product.cons)}):")
        for con in product.cons[:3]:
            print(f"   ❌ {con}")
    
    # Show RAG-ready text format
    print("\n" + "=" * 70)
    print("📄 RAG-READY TEXT FORMAT")
    print("=" * 70)
    print(product.to_text()[:800])
    print("\n... (truncated)")
    
    # Show JSON format
    print("\n" + "=" * 70)
    print("🗂️  JSON FORMAT")
    print("=" * 70)
    print(json.dumps(product.to_dict(), indent=2)[:800])
    print("\n... (truncated)")


def test_multiple_products():
    """Test extracting multiple products"""
    print("\n" + "=" * 70)
    print("🧪 TEST: Multiple Products Extraction")
    print("=" * 70)
    
    test_urls = [
        "https://www.apple.com/iphone-15/",
        "https://www.samsung.com/us/smartphones/galaxy-s24/",
    ]
    
    print(f"\n📥 Fetching {len(test_urls)} URLs...\n")
    
    from src.scrapers.jina_reader import JinaReader
    reader = JinaReader()
    url_contents = reader.fetch_multiple(test_urls)
    
    # Prepare content map
    content_map = {
        url: content.content 
        for url, content in url_contents.items() 
        if content.success
    }
    
    print(f"\n🔍 Extracting products...")
    extractor = ProductExtractor()
    products = extractor.extract_multiple(content_map)
    
    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    
    for url, product in products.items():
        print(f"\n{'='*70}")
        print(f"📱 {product.title}")
        print(f"{'='*70}")
        print(f"🏢 Brand: {product.brand or 'N/A'}")
        print(f"💰 Price: {product.currency or ''}{product.price or 'N/A'}")
        print(f"✨ Features: {len(product.key_features)}")
        print(f"🔧 Specs: {len(product.specifications)}")
        print(f"⭐ Rating: {product.rating or 'N/A'}")


if __name__ == "__main__":
    print("\n🚀 Starting Product Extractor Tests...\n")
    
    # Test 1: Single product
    test_single_product()
    
    # Test 2: Multiple products (optional - takes longer)
    proceed = input("\n\n🤔 Run multiple products test? (y/n): ")
    if proceed.lower() == 'y':
        test_multiple_products()
    
    print("\n" + "=" * 70)
    print("✅ All Tests Complete!")
    print("=" * 70 + "\n")
