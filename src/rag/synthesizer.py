"""
Comparison Synthesizer
Uses LLM to create intelligent product comparisons from retrieved context
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from src.utils.llm_client import get_llm_client
from src.rag.retriever import RetrievalContext, get_retriever
import json


@dataclass
class ComparisonResult:
    """Result of a comparison"""
    query: str
    answer: str
    products_compared: List[str]
    comparison_table: Optional[Dict] = None
    summary: Optional[str] = None
    citations: List[str] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "products_compared": self.products_compared,
            "comparison_table": self.comparison_table,
            "summary": self.summary,
            "citations": self.citations
        }


class ComparisonSynthesizer:
    """Synthesizes intelligent comparisons from product information"""
    
    def __init__(self):
        self.llm = get_llm_client()
        self.retriever = get_retriever()
    
    def compare(
        self,
        query: str,
        product_urls: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare products based on user query
        
        Args:
            query: User's comparison question
            product_urls: Specific products to compare (optional)
            
        Returns:
            ComparisonResult
        """
        
        print(f"\n{'='*70}")
        print(f"🔬 Synthesizing Comparison")
        print(f"{'='*70}")
        print(f"Query: {query}")
        
        # Step 1: Retrieve relevant context
        if product_urls:
            context = self.retriever.retrieve_for_comparison(query, product_urls)
        else:
            context = self.retriever.retrieve(query, max_results=10)
        
        if not context.results:
            return ComparisonResult(
                query=query,
                answer="I couldn't find any products to compare. Please add products first.",
                products_compared=[],
                summary="No products found."
            )
        
        products = context.get_all_products()
        print(f"\n📦 Comparing {len(products)} products:")
        for p in products:
            print(f"   • {p}")
        
        # Step 2: Generate comparison
        if context.query_analysis.is_comparative:
            result = self._generate_comparison(query, context)
        else:
            result = self._generate_answer(query, context)
        
        return result
    
    def _generate_comparison(
        self,
        query: str,
        context: RetrievalContext
    ) -> ComparisonResult:
        """Generate a comparative analysis"""
        
        print(f"\n🔄 Generating comparative analysis...")
        
        # Format context for LLM
        formatted_context = self.retriever.format_context_for_llm(context)
        
        system_prompt = """You are an expert product comparison analyst. 
Your job is to provide clear, objective comparisons based on the given product information.

Guidelines:
1. Be factual and cite specific information from the products
2. Organize comparisons clearly (use tables/bullet points when helpful)
3. Highlight key differences and similarities
4. Provide a balanced view - mention both pros and cons
5. If specific information is not available, say so clearly
6. Don't make up specifications or features
7. Focus on the aspects the user asked about
8. End with a brief summary/recommendation if appropriate

Format your response in a clear, structured way."""

        user_prompt = f"""Based on the product information below, answer this comparison query:

Query: {query}

Product Information:
{formatted_context}

Provide a comprehensive comparison addressing the user's query.
Structure your answer with:
1. Direct answer to the query
2. Detailed comparison of relevant aspects
3. Key differences highlighted
4. Brief summary/recommendation

Your response:"""

        try:
            answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            print(f"✅ Comparison generated!")
            
            # Try to extract structured data
            comparison_table = self._extract_comparison_table(answer, context)
            
            return ComparisonResult(
                query=query,
                answer=answer,
                products_compared=context.get_all_products(),
                comparison_table=comparison_table,
                citations=[r.metadata["url"] for r in context.results]
            )
            
        except Exception as e:
            print(f"❌ Comparison generation failed: {e}")
            return ComparisonResult(
                query=query,
                answer=f"Error generating comparison: {str(e)}",
                products_compared=context.get_all_products()
            )
    
    def _generate_answer(
        self,
        query: str,
        context: RetrievalContext
    ) -> ComparisonResult:
        """Generate a direct answer (non-comparative)"""
        
        print(f"\n🔄 Generating answer...")
        
        formatted_context = self.retriever.format_context_for_llm(context, max_products=3)
        
        system_prompt = """You are a helpful product information assistant.
Answer the user's question based on the provided product information.

Guidelines:
1. Be accurate and cite specific details
2. If information is not available, say so
3. Be concise but complete
4. Use bullet points or lists when appropriate
5. Don't make up information"""

        user_prompt = f"""Answer this question based on the product information:

Query: {query}

Product Information:
{formatted_context}

Your answer:"""

        try:
            answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            print(f"✅ Answer generated!")
            
            return ComparisonResult(
                query=query,
                answer=answer,
                products_compared=context.get_all_products(),
                citations=[r.metadata["url"] for r in context.results]
            )
            
        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return ComparisonResult(
                query=query,
                answer=f"Error generating answer: {str(e)}",
                products_compared=context.get_all_products()
            )
    
    def _extract_comparison_table(
        self,
        answer: str,
        context: RetrievalContext
    ) -> Optional[Dict]:
        """
        Attempt to extract structured comparison data
        This is a simple version - can be enhanced
        """
        
        # Try to extract key specs from products
        if len(context.results) < 2:
            return None
        
        try:
            # Get metadata from products
            table = {}
            for result in context.results[:5]:  # Top 5 products
                table[result.product_title] = {
                    "brand": result.metadata["brand"],
                    "price": f"{result.metadata['currency']}{result.metadata['price']}",
                    "rating": result.metadata.get("rating", "N/A"),
                }
            
            return table
            
        except Exception as e:
            return None
    
    def generate_summary_comparison(
        self,
        queries: List[str],
        product_urls: List[str]
    ) -> Dict[str, ComparisonResult]:
        """
        Generate multiple comparisons for the same products
        Useful for comprehensive product analysis
        
        Args:
            queries: List of comparison questions
            product_urls: Products to compare
            
        Returns:
            Dictionary mapping query to ComparisonResult
        """
        
        print(f"\n🔬 Multi-Aspect Comparison")
        print(f"{'='*70}")
        print(f"Products: {len(product_urls)}")
        print(f"Aspects: {len(queries)}")
        
        results = {}
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query}")
            result = self.compare(query, product_urls)
            results[query] = result
        
        return results
    
    def compare_with_aspects(
        self,
        product_urls: List[str],
        aspects: Optional[List[str]] = None
    ) -> Dict[str, ComparisonResult]:
        """
        Compare products across multiple aspects
        
        Args:
            product_urls: Products to compare
            aspects: Aspects to compare (or use default)
            
        Returns:
            Dictionary of comparisons by aspect
        """
        
        if aspects is None:
            aspects = [
                "price and value for money",
                "features and specifications",
                "performance and speed",
                "design and build quality",
                "user reviews and ratings"
            ]
        
        queries = [
            f"Compare the {aspect} of these products"
            for aspect in aspects
        ]
        
        return self.generate_summary_comparison(queries, product_urls)


# Global instance
_synthesizer = None


def get_synthesizer() -> ComparisonSynthesizer:
    """Get or create global synthesizer"""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ComparisonSynthesizer()
    return _synthesizer
