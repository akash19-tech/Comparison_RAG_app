"""
Test the Smart Retrieval System
"""

from src.rag.retriever import get_retriever
from src.rag.vector_store import get_vector_store


def test_query_analysis():
    """Test query analysis"""
    print("=" * 70)
    print("🧪 TEST 1: Query Analysis")
    print("=" * 70)
    
    retriever = get_retriever()
    
    test_queries = [
        "Which phone has better camera?",
        "Tell me about the iPhone 15",
        "Compare the prices",
        "What's the battery life of these phones?",
        "Which one should I buy?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        analysis = retriever.query_analyzer.analyze(query)
        
        print(f"   🔍 Type: {analysis.query_type}")
        print(f"   📊 Aspects: {analysis.aspects}")
        print(f"   🏷️  Entities: {analysis.entities}")
        print(f"   ⚖️  Comparative: {analysis.is_comparative}")
        print(f"   🔄 Expanded: {analysis.expanded_query}")


def test_basic_retrieval():
    """Test basic retrieval"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Basic Retrieval")
    print("=" * 70)
    
    # Check if we have products
    vector_store = get_vector_store()
    count = vector_store.count()
    
    if count == 0:
        print("\n⚠️  No products in vector store!")
        print("   Run: python test_vector_store.py first")
        return
    
    print(f"\n📦 Products in store: {count}")
    
    retriever = get_retriever()
    
    test_queries = [
        "Which phone has better camera?",
        "Compare the prices",
        "Tell me about battery life",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")
        
        context = retriever.retrieve(query, max_results=5)
        
        print(f"\n✅ Retrieved {context.total_retrieved} products:")
        
        for i, result in enumerate(context.results[:3], 1):
            print(f"\n   {i}. {result.product_title}")
            print(f"      🏢 {result.brand}")
            print(f"      📊 Relevance: {result.relevance_score:.2%}")
            print(f"      💰 Price: {result.metadata['currency']}{result.metadata['price']}")


def test_comparison_retrieval():
    """Test comparison-specific retrieval"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Comparison Retrieval")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    retriever = get_retriever()
    
    query = "Compare iPhone 15 and Galaxy S24 cameras"
    
    print(f"\n🔍 Comparison Query: {query}")
    
    context = retriever.retrieve_for_comparison(query)
    
    print(f"\n✅ Retrieved {context.total_retrieved} products for comparison:")
    print(f"\n📱 Products found:")
    for product in context.get_all_products():
        print(f"   • {product}")
    
    print(f"\n📊 Top Results:")
    for i, result in enumerate(context.get_top_k(3), 1):
        print(f"\n   {i}. {result.product_title}")
        print(f"      Relevance: {result.relevance_score:.2%}")


def test_context_formatting():
    """Test context formatting for LLM"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Context Formatting for LLM")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    retriever = get_retriever()
    
    query = "Which phone has better value for money?"
    
    print(f"\n🔍 Query: {query}")
    
    context = retriever.retrieve(query, max_results=3)
    formatted = retriever.format_context_for_llm(context, max_products=2)
    
    print(f"\n📄 Formatted Context for LLM:")
    print("=" * 70)
    print(formatted[:1000])
    print("\n... (truncated)")
    print(f"\n📏 Total length: {len(formatted)} characters")


def test_aspect_specific_retrieval():
    """Test retrieval for specific aspects"""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: Aspect-Specific Retrieval")
    print("=" * 70)
    
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        print("\n⚠️  No products in vector store!")
        return
    
    retriever = get_retriever()
    
    aspect_queries = [
        ("Price Comparison", "How much does each phone cost? Compare prices"),
        ("Camera Quality", "Which has the best camera system and photo quality?"),
        ("Battery Life", "Compare battery capacity and charging speeds"),
        ("Display", "Tell me about the screen quality and display features"),
    ]
    
    for aspect, query in aspect_queries:
        print(f"\n{'='*70}")
        print(f"📊 Aspect: {aspect}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")
        
        context = retriever.retrieve(query, max_results=3)
        
        print(f"\n✅ Analysis:")
        print(f"   Type: {context.query_analysis.query_type}")
        print(f"   Aspects: {context.query_analysis.aspects}")
        
        print(f"\n📱 Top Products:")
        for result in context.get_top_k(2):
            print(f"   • {result.product_title} (Relevance: {result.relevance_score:.2%})")


if __name__ == "__main__":
    print("\n🚀 Starting Smart Retrieval Tests...\n")
    
    # Test 1: Query analysis
    test_query_analysis()
    
    # Test 2: Basic retrieval
    test_basic_retrieval()
    
    # Test 3: Comparison retrieval
    test_comparison_retrieval()
    
    # Test 4: Context formatting
    test_context_formatting()
    
    # Test 5: Aspect-specific
    test_aspect_specific_retrieval()
    
    print("\n" + "=" * 70)
    print("✅ All Retrieval Tests Complete!")
    print("=" * 70 + "\n")
