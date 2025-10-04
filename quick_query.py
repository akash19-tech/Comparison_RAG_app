"""Quick query test"""
from src.rag.vector_store import get_vector_store

vector_store = get_vector_store()

print(f"\n📊 Products in store: {vector_store.count()}\n")

if vector_store.count() > 0:
    query = input("🔍 Enter your query: ")
    results = vector_store.query(query, n_results=2)
    
    print(f"\n{'='*70}")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"], results["metadatas"], results["distances"]),
        1
    ):
        print(f"\n🏆 Result {i}")
        print(f"�� {meta['title']}")
        print(f"📏 Similarity: {1 - dist:.2%}")
        print(f"\n{doc[:200]}...")
else:
    print("⚠️  No products in store. Run test_vector_store.py first!")
