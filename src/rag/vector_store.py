"""
Vector Store using ChromaDB
Stores product information with embeddings for semantic search
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from src.rag.embeddings import get_embedding_generator
from src.extractors.product_extractor import ProductInfo


class VectorStore:
    """ChromaDB-based vector store for product information"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Where to store the database
        """
        print(f"📁 Initializing ChromaDB at: {persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection_name = "products"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"✅ Loaded existing collection: {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Product information for comparison"}
            )
            print(f"✅ Created new collection")
        
        # Initialize embedding generator
        self.embedder = get_embedding_generator()
    
    def add_product(self, product: ProductInfo) -> str:
        """
        Add a product to the vector store
        
        Args:
            product: ProductInfo object
            
        Returns:
            Document ID
        """
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Convert product to text for embedding
        product_text = product.to_text()
        
        # Generate embedding
        embedding = self.embedder.encode(product_text)
        
        # Prepare metadata
        metadata = {
            "url": product.url,
            "title": product.title,
            "brand": product.brand or "Unknown",
            "price": product.price or "N/A",
            "currency": product.currency or "",
            "category": product.category or "Unknown",
            "rating": product.rating or 0.0,
            "review_count": product.review_count or 0,
        }
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[product_text],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_products(self, products: List[ProductInfo]) -> List[str]:
        """
        Add multiple products to the vector store
        
        Args:
            products: List of ProductInfo objects
            
        Returns:
            List of document IDs
        """
        print(f"\n📦 Adding {len(products)} products to vector store...")
        
        doc_ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for product in products:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            # Convert to text
            product_text = product.to_text()
            documents.append(product_text)
            
            # Prepare metadata
            metadata = {
                "url": product.url,
                "title": product.title,
                "brand": product.brand or "Unknown",
                "price": product.price or "N/A",
                "currency": product.currency or "",
                "category": product.category or "Unknown",
                "rating": float(product.rating) if product.rating else 0.0,
                "review_count": int(product.review_count) if product.review_count else 0,
            }
            metadatas.append(metadata)
        
        # Generate embeddings in batch
        print("🔄 Generating embeddings...")
        embeddings = self.embedder.encode_batch(documents)
        
        # Add to collection
        print("💾 Storing in ChromaDB...")
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(products)} products successfully!")
        return doc_ids
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Query the vector store
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            filter_metadata: Filter by metadata (e.g., {"brand": "Apple"})
            
        Returns:
            Dictionary with results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_metadata
        )
        
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def get_all_products(self) -> Dict:
        """Get all products in the store"""
        results = self.collection.get()
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }
    
    def count(self) -> int:
        """Get number of products in store"""
        return self.collection.count()
    
    def delete_by_url(self, url: str) -> int:
        """
        Delete products by URL
        
        Args:
            url: Product URL
            
        Returns:
            Number of deleted items
        """
        results = self.collection.get(where={"url": url})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    
    def clear(self):
        """Clear all data from the store"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Product information for comparison"}
        )
        print("🗑️  Vector store cleared")
    
    def reset(self):
        """Reset the entire database"""
        self.client.reset()
        print("🔄 Database reset")


# Global instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create global vector store"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
