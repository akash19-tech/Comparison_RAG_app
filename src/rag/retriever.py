"""
Smart Retriever
Retrieves relevant product information for queries
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from src.rag.vector_store import get_vector_store
from src.rag.query_analyzer import QueryAnalyzer, QueryAnalysis


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    product_title: str
    brand: str
    content: str
    metadata: Dict
    relevance_score: float  # 0-1, higher is better
    
    def __repr__(self):
        return f"RetrievalResult({self.product_title}, score={self.relevance_score:.3f})"


@dataclass
class RetrievalContext:
    """Complete retrieval context for a query"""
    query: str
    query_analysis: QueryAnalysis
    results: List[RetrievalResult]
    total_retrieved: int
    
    def get_top_k(self, k: int) -> List[RetrievalResult]:
        """Get top k results"""
        return self.results[:k]
    
    def get_by_product(self, product_title: str) -> Optional[RetrievalResult]:
        """Get result for specific product"""
        for result in self.results:
            if result.product_title == product_title:
                return result
        return None
    
    def get_all_products(self) -> List[str]:
        """Get list of all product titles"""
        return [r.product_title for r in self.results]


class SmartRetriever:
    """Intelligent retrieval system for product comparison"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.query_analyzer = QueryAnalyzer()
    
    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        min_relevance: float = 0.0
    ) -> RetrievalContext:
        """
        Retrieve relevant products for a query
        
        Args:
            query: User's question
            max_results: Maximum number of results
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            RetrievalContext with results
        """
        
        # Step 1: Analyze query
        print(f"\n🔍 Analyzing query: '{query}'")
        analysis = self.query_analyzer.analyze(query)
        print(f"   Type: {analysis.query_type}")
        print(f"   Aspects: {analysis.aspects}")
        print(f"   Comparative: {analysis.is_comparative}")
        
        # Step 2: Retrieve using expanded query
        search_query = analysis.expanded_query
        print(f"\n🔎 Searching with: '{search_query}'")
        
        raw_results = self.vector_store.query(
            search_query,
            n_results=max_results
        )
        
        # Step 3: Process results
        results = []
        for i in range(len(raw_results["ids"])):
            # Convert distance to relevance score (0-1)
            # Lower distance = higher relevance
            distance = raw_results["distances"][i]
            relevance = max(0.0, 1.0 - distance)
            
            # Skip if below threshold
            if relevance < min_relevance:
                continue
            
            result = RetrievalResult(
                product_title=raw_results["metadatas"][i]["title"],
                brand=raw_results["metadatas"][i]["brand"],
                content=raw_results["documents"][i],
                metadata=raw_results["metadatas"][i],
                relevance_score=relevance
            )
            results.append(result)
        
        print(f"✅ Retrieved {len(results)} relevant products")
        
        # Step 4: If comparative, ensure we have multiple products
        if analysis.is_comparative and len(results) > 0:
            results = self._ensure_diversity(results)
        
        return RetrievalContext(
            query=query,
            query_analysis=analysis,
            results=results,
            total_retrieved=len(results)
        )
    
    def _ensure_diversity(
        self,
        results: List[RetrievalResult],
        max_per_product: int = 1
    ) -> List[RetrievalResult]:
        """
        Ensure diversity in results (don't repeat same product)
        
        Args:
            results: Original results
            max_per_product: Maximum results per product
            
        Returns:
            Filtered results
        """
        seen_products = set()
        diverse_results = []
        
        for result in results:
            if result.product_title not in seen_products:
                diverse_results.append(result)
                seen_products.add(result.product_title)
            elif len([r for r in diverse_results if r.product_title == result.product_title]) < max_per_product:
                diverse_results.append(result)
        
        return diverse_results
    
    def retrieve_for_comparison(
        self,
        query: str,
        product_urls: Optional[List[str]] = None
    ) -> RetrievalContext:
        """
        Specialized retrieval for comparison queries
        
        Args:
            query: Comparison question
            product_urls: Specific products to compare (optional)
            
        Returns:
            RetrievalContext optimized for comparison
        """
        
        if product_urls:
            # Retrieve specific products
            print(f"\n🎯 Targeted comparison for {len(product_urls)} products")
            results = []
            
            for url in product_urls:
                # Search for this specific product
                raw = self.vector_store.query(
                    query,
                    n_results=1,
                    filter_metadata={"url": url}
                )
                
                if raw["ids"]:
                    result = RetrievalResult(
                        product_title=raw["metadatas"][0]["title"],
                        brand=raw["metadatas"][0]["brand"],
                        content=raw["documents"][0],
                        metadata=raw["metadatas"][0],
                        relevance_score=1.0 - raw["distances"][0]
                    )
                    results.append(result)
            
            analysis = self.query_analyzer.analyze(query)
            
            return RetrievalContext(
                query=query,
                query_analysis=analysis,
                results=results,
                total_retrieved=len(results)
            )
        
        else:
            # General comparison - retrieve all products
            return self.retrieve(query, max_results=20)
    
    def format_context_for_llm(
        self,
        context: RetrievalContext,
        max_products: int = 5
    ) -> str:
        """
        Format retrieval context for LLM consumption
        
        Args:
            context: RetrievalContext
            max_products: Maximum products to include
            
        Returns:
            Formatted string
        """
        
        lines = [
            "# Product Information",
            "",
            f"Query: {context.query}",
            f"Query Type: {context.query_analysis.query_type}",
            ""
        ]
        
        if context.query_analysis.aspects:
            lines.append(f"Focus Areas: {', '.join(context.query_analysis.aspects)}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        for i, result in enumerate(context.get_top_k(max_products), 1):
            lines.append(f"## Product {i}: {result.product_title}")
            lines.append(f"Brand: {result.brand}")
            lines.append(f"Relevance: {result.relevance_score:.2%}")
            lines.append("")
            lines.append(result.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


# Global instance
_retriever = None


def get_retriever() -> SmartRetriever:
    """Get or create global retriever"""
    global _retriever
    if _retriever is None:
        _retriever = SmartRetriever()
    return _retriever
