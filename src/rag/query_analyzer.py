"""
Query Analyzer
Analyzes user queries to understand intent and extract key aspects
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.llm_client import get_llm_client
import json


@dataclass
class QueryAnalysis:
    """Analysis of a user query"""
    original_query: str
    query_type: str  # "comparison", "specific", "general"
    aspects: List[str]  # What to compare: ["price", "camera", "battery"]
    entities: List[str]  # Specific products/brands mentioned
    is_comparative: bool
    expanded_query: str  # Expanded version for better retrieval
    
    def __repr__(self):
        return f"QueryAnalysis(type={self.query_type}, aspects={self.aspects})"


class QueryAnalyzer:
    """Analyzes queries to improve retrieval"""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query
        
        Args:
            query: User's question
            
        Returns:
            QueryAnalysis object
        """
        
        system_prompt = """You are a query analysis expert for product comparison systems.
Analyze the user's query and return ONLY valid JSON with this structure:

{
    "query_type": "comparison" or "specific" or "general",
    "aspects": ["aspect1", "aspect2"],
    "entities": ["brand1", "product1"],
    "is_comparative": true or false,
    "expanded_query": "expanded version of query"
}

Guidelines:
- query_type: 
  * "comparison" if comparing multiple products
  * "specific" if asking about one product feature
  * "general" for broad questions
  
- aspects: Extract what user wants to know about (price, camera, battery, display, performance, design, reviews, etc.)

- entities: Extract specific brands, product names, or models mentioned

- is_comparative: true if query compares or asks "which/better/best"

- expanded_query: Rewrite query with more keywords for better search
  Example: "which is cheaper?" → "price comparison cost budget affordable"

Return ONLY the JSON object."""

        user_prompt = f"""Analyze this query:

Query: {query}

Return the JSON analysis."""

        try:
            response = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            data = json.loads(response)
            
            return QueryAnalysis(
                original_query=query,
                query_type=data.get("query_type", "general"),
                aspects=data.get("aspects", []),
                entities=data.get("entities", []),
                is_comparative=data.get("is_comparative", False),
                expanded_query=data.get("expanded_query", query)
            )
            
        except Exception as e:
            # Fallback: basic analysis
            print(f"⚠️  Query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Simple fallback analysis without LLM"""
        query_lower = query.lower()
        
        # Check if comparative
        comparative_keywords = [
            "compare", "versus", "vs", "which", "better", "best",
            "difference", "between", "or"
        ]
        is_comparative = any(kw in query_lower for kw in comparative_keywords)
        
        # Determine type
        if is_comparative:
            query_type = "comparison"
        elif any(word in query_lower for word in ["tell me", "what is", "how"]):
            query_type = "specific"
        else:
            query_type = "general"
        
        # Extract common aspects
        aspect_keywords = {
            "price": ["price", "cost", "expensive", "cheap", "budget"],
            "camera": ["camera", "photo", "picture", "video"],
            "battery": ["battery", "charge", "power"],
            "display": ["display", "screen", "resolution"],
            "performance": ["performance", "speed", "fast", "processor"],
            "design": ["design", "look", "style", "build"],
        }
        
        aspects = []
        for aspect, keywords in aspect_keywords.items():
            if any(kw in query_lower for kw in keywords):
                aspects.append(aspect)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            aspects=aspects,
            entities=[],
            is_comparative=is_comparative,
            expanded_query=query
        )
