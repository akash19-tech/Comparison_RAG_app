"""
Quick Demo of the Complete RAG System
"""

from src.scrapers.jina_reader import JinaReader
from src.extractors.product_extractor import ProductExtractor
from src.rag.vector_store import VectorStore
from src.rag.synthesizer import get_synthesizer


def quick_demo():
    """Quick end-to-end demo"""
    
    print("=" * 70)
    print("🎯 PRODUCT COMPARISON RAG - QUICK DEMO")
    print("=" * 70)
    
    # Step 1: Add products
    print("\n📥 STEP 1: Adding Products")
    print("─" * 70)
    
    reader = JinaReader()
    extractor = ProductExtractor()
    vector_store = VectorStore(persist_directory="./data/demo_db")
    
    # Clear previous data
    vector_store.clear()
    
    urls = [
        "https://www.apple.com/iphone-15/",
        "https://www.samsung.com/us/smartphones/galaxy-s24/",
    ]
    
    print(f"\n📡 Fetching {len(urls)} products...")
    url_contents = reader.fetch_multiple(urls)
    
    content_map = {
        url: content.content 
        for url, content in url_contents.items() 
        if content.success
    }
    
    print(f"\n🔍 Extracting product information...")
    products_dict = extractor.extract_multiple(content_map)
    products = list(products_dict.values())
    
    print(f"\n💾 Storing in vector database...")
    vector_store.add_products(products)
    
    print(f"\n✅ Added {len(products)} products to database")
    
    # Step 2: Compare products
    print("\n\n🔬 STEP 2: Running Comparisons")
    print("─" * 70)
    
    synthesizer = get_synthesizer()
    
    queries = [
        "Which phone has better camera?",
        "Compare the prices",
        "What are the main differences?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")
        print("─" * 70)
        
        result = synthesizer.compare(query)
        
        print(f"\n�� Comparing: {', '.join(result.products_compared)}")
        print(f"\n💬 Answer:\n")
        print(result.answer[:500])
        print("...\n")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("  • Run: python test_comparison.py (for interactive mode)")
    print("  • Or continue building the Streamlit UI")
    print()


if __name__ == "__main__":
    quick_demo()
