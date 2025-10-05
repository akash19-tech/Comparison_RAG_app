"""
Test the Comparison Synthesizer
"""

from src.rag.synthesizer import get_synthesizer
from src.rag.vector_store import get_vector_store
import json


def print_result(result):
    """Pretty print a comparison result"""
    print(f"\n{'='*70}")
    print(f"�� COMPARISON RESULT")
    print(f"{'='*70}")
    
    print(f"\n❓ Query: {result.query}")
    
    if result.products_compared:
        print(f"\n📱 Products Compared ({len(result.products_compared)}):")
        for product in result.products_compared:
            print(f"   • {product}")
    
    print(f"\n💬 Answer:")
    print("─" * 70)
    print(result.answer)
    print("─" * 70)
    
    if result.comparison_table:
        print(f"\n📋 Quick Comparison Table:")
        print(json.dumps(result.comparison_table, indent=2))
    
    if result.citations:
        print(f"\n🔗 Sources ({len(result.citations)}):")
        for i, citation in enumerate(result.citations[:3], 1):
            print(f"   {i}. {citation[:60]}...")


def test_basic_comparison():
    """Test basic comparison"""
    print("=" * 70)
    print("🧪 TEST 1: Basic Comparison")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        print("   Run: python test_vector_store.py first")
        return
    
    synthesizer = get_synthesizer()
    
    query = "Which phone has better camera quality?"
    
    print(f"\n🔍 Testing query: {query}")
    result = synthesizer.compare(query)
    
    print_result(result)


def test_price_comparison():
    """Test price comparison"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Price Comparison")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    synthesizer = get_synthesizer()
    
    query = "Compare the prices. Which one offers better value for money?"
    
    print(f"\n🔍 Testing query: {query}")
    result = synthesizer.compare(query)
    
    print_result(result)


def test_feature_comparison():
    """Test feature comparison"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Feature Comparison")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    synthesizer = get_synthesizer()
    
    query = "What are the key differences in features and specifications?"
    
    print(f"\n🔍 Testing query: {query}")
    result = synthesizer.compare(query)
    
    print_result(result)


def test_specific_question():
    """Test specific (non-comparative) question"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Specific Question")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    synthesizer = get_synthesizer()
    
    query = "What is the battery capacity?"
    
    print(f"\n🔍 Testing query: {query}")
    result = synthesizer.compare(query)
    
    print_result(result)


def test_multi_aspect_comparison():
    """Test multi-aspect comparison"""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: Multi-Aspect Comparison")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    # Get all product URLs
    all_products = vector_store.get_all_products()
    product_urls = [meta["url"] for meta in all_products["metadatas"]]
    
    if len(product_urls) < 2:
        print("\n⚠️  Need at least 2 products for comparison!")
        return
    
    synthesizer = get_synthesizer()
    
    aspects = [
        "price and value",
        "camera quality",
        "battery life"
    ]
    
    print(f"\n🔬 Comparing across {len(aspects)} aspects...")
    print(f"📱 Products: {len(product_urls)}")
    
    results = synthesizer.compare_with_aspects(
        product_urls[:2],  # Compare first 2 products
        aspects
    )
    
    for aspect, result in results.items():
        print(f"\n{'─'*70}")
        print(f"📊 Aspect: {aspect}")
        print(f"{'─'*70}")
        print(result.answer[:400])
        print("...\n")


def interactive_mode():
    """Interactive query mode"""
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE MODE")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        print("   Run: python test_vector_store.py first")
        return
    
    print(f"\n📦 Products available: {vector_store.count()}")
    
    all_products = vector_store.get_all_products()
    print("\n📱 Products in database:")
    for i, meta in enumerate(all_products["metadatas"], 1):
        print(f"   {i}. {meta['title']} - {meta['brand']}")
    
    synthesizer = get_synthesizer()
    
    print("\n" + "─" * 70)
    print("💡 Example queries:")
    print("   • Which phone has better camera?")
    print("   • Compare the prices")
    print("   • What are the main differences?")
    print("   • Which one has longer battery life?")
    print("   • Tell me about the display")
    print("\nType 'quit' to exit")
    print("─" * 70)
    
    while True:
        query = input("\n🔍 Your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        result = synthesizer.compare(query)
        print_result(result)
        
        print("\n" + "─" * 70)


if __name__ == "__main__":
    print("\n🚀 Starting Comparison Synthesizer Tests...\n")
    
    # Automated tests
    test_basic_comparison()
    test_price_comparison()
    test_feature_comparison()
    test_specific_question()
    
    # Multi-aspect test
    proceed = input("\n\n🤔 Run multi-aspect comparison? (y/n): ")
    if proceed.lower() == 'y':
        test_multi_aspect_comparison()
    
    print("\n" + "=" * 70)
    print("✅ Automated Tests Complete!")
    print("=" * 70)
    
    # Interactive mode
    proceed = input("\n🎮 Enter interactive mode? (y/n): ")
    if proceed.lower() == 'y':
        interactive_mode()
    
    print("\n✅ All tests complete!\n")
