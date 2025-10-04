"""
Test the Vector Store
"""

from src.scrapers.jina_reader import JinaReader
from src.extractors.product_extractor import ProductExtractor
from src.rag.vector_store import VectorStore
import json


def test_basic_operations():
    """Test basic vector store operations"""
    print("=" * 70)
    print("🧪 TEST 1: Basic Vector Store Operations")
    print("=" * 70)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    reader = JinaReader()
    extractor = ProductExtractor()
    vector_store = VectorStore(persist_directory="./data/test_chroma_db")
    
    # Clear any existing data
    vector_store.clear()
    print(f"📊 Current count: {vector_store.count()}")
    
    # Fetch and extract products
    test_urls = [
        "https://www.apple.com/iphone-15/",
        "https://www.samsung.com/us/smartphones/galaxy-s24/",
    ]
    
    print(f"\n📥 Fetching {len(test_urls)} products...")
    url_contents = reader.fetch_multiple(test_urls)
    
    content_map = {
        url: content.content 
        for url, content in url_contents.items() 
        if content.success
    }
    
    print(f"\n🔍 Extracting product information...")
    products_dict = extractor.extract_multiple(content_map)
    products = list(products_dict.values())
    
    # Add to vector store
    print(f"\n💾 Adding products to vector store...")
    doc_ids = vector_store.add_products(products)
    
    print(f"\n✅ Added {len(doc_ids)} products")
    print(f"📊 Total count: {vector_store.count()}")
    
    return vector_store, products


def test_semantic_search(vector_store: VectorStore):
    """Test semantic search"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Semantic Search")
    print("=" * 70)
    
    test_queries = [
        "Which phone has better camera?",
        "Tell me about the display",
        "What's the price?",
        "Battery life comparison",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")
        
        results = vector_store.query(query, n_results=2)
        
        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"], results["metadatas"], results["distances"]),
            1
        ):
            print(f"\n🏆 Result {i} (Distance: {distance:.4f})")
            print(f"�� {metadata['title']}")
            print(f"🏢 {metadata['brand']}")
            print(f"💰 {metadata['currency']}{metadata['price']}")
            print(f"\n📄 Relevant content:")
            print("-" * 70)
            print(doc[:300])
            print("...")


def test_metadata_filtering(vector_store: VectorStore):
    """Test filtering by metadata"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Metadata Filtering")
    print("=" * 70)
    
    # Filter by brand
    print("\n🔍 Filter: brand = 'Apple'")
    results = vector_store.query(
        "Tell me about this phone",
        n_results=5,
        filter_metadata={"brand": "Apple"}
    )
    
    print(f"\n✅ Found {len(results['ids'])} Apple products:")
    for metadata in results["metadatas"]:
        print(f"   • {metadata['title']} - {metadata['brand']}")


def test_get_all_products(vector_store: VectorStore):
    """Test getting all products"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Get All Products")
    print("=" * 70)
    
    all_products = vector_store.get_all_products()
    
    print(f"\n📦 Total products in store: {len(all_products['ids'])}")
    print("\n📋 Product List:")
    
    for metadata in all_products["metadatas"]:
        print(f"\n   📱 {metadata['title']}")
        print(f"      🏢 Brand: {metadata['brand']}")
        print(f"      �� Price: {metadata['currency']}{metadata['price']}")
        print(f"      ⭐ Rating: {metadata['rating']}")
        print(f"      🔗 URL: {metadata['url'][:50]}...")


def test_comparison_query(vector_store: VectorStore):
    """Test comparison-style queries"""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: Comparison Queries")
    print("=" * 70)
    
    comparison_queries = [
        "Compare the cameras",
        "Which one is cheaper?",
        "Display size comparison",
        "Performance and speed",
    ]
    
    for query in comparison_queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")
        
        # Get all products for comparison
        results = vector_store.query(query, n_results=10)
        
        print(f"\n📊 Found {len(results['ids'])} relevant results")
        
        # Show top 2
        for i in range(min(2, len(results["ids"]))):
            print(f"\n🏆 Result {i+1}")
            print(f"   📱 {results['metadatas'][i]['title']}")
            print(f"   📏 Distance: {results['distances'][i]:.4f}")


if __name__ == "__main__":
    print("\n🚀 Starting Vector Store Tests...\n")
    
    # Test 1: Basic operations
    vector_store, products = test_basic_operations()
    
    # Test 2: Semantic search
    test_semantic_search(vector_store)
    
    # Test 3: Metadata filtering
    test_metadata_filtering(vector_store)
    
    # Test 4: Get all products
    test_get_all_products(vector_store)
    
    # Test 5: Comparison queries
    test_comparison_query(vector_store)
    
    print("\n" + "=" * 70)
    print("✅ All Vector Store Tests Complete!")
    print("=" * 70)
    
    # Cleanup
    cleanup = input("\n🗑️  Clean up test database? (y/n): ")
    if cleanup.lower() == 'y':
        vector_store.clear()
        print("✅ Cleaned up!")
    
    print()
