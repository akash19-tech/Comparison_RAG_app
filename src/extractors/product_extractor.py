"""
Product Information Extractor
Uses LLM to extract structured product data from raw content
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from src.utils.llm_client import get_llm_client


@dataclass
class ProductInfo:
    """Structured product information"""
    url: str
    title: str
    brand: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    
    # Specifications
    specifications: Dict[str, str] = None
    
    # Features & Description
    key_features: List[str] = None
    description: Optional[str] = None
    
    # Reviews & Ratings
    rating: Optional[float] = None
    review_count: Optional[int] = None
    pros: List[str] = None
    cons: List[str] = None
    
    # Additional
    availability: Optional[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.specifications is None:
            self.specifications = {}
        if self.key_features is None:
            self.key_features = []
        if self.pros is None:
            self.pros = []
        if self.cons is None:
            self.cons = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_text(self) -> str:
        """Convert to readable text for RAG"""
        lines = [
            f"# {self.title}",
            f"URL: {self.url}",
            ""
        ]
        
        if self.brand:
            lines.append(f"Brand: {self.brand}")
        
        if self.price:
            lines.append(f"Price: {self.currency or ''}{self.price}")
        
        if self.category:
            lines.append(f"Category: {self.category}")
        
        if self.rating:
            lines.append(f"Rating: {self.rating}/5 ({self.review_count or 0} reviews)")
        
        if self.availability:
            lines.append(f"Availability: {self.availability}")
        
        lines.append("")
        
        if self.description:
            lines.append("## Description")
            lines.append(self.description)
            lines.append("")
        
        if self.key_features:
            lines.append("## Key Features")
            for feature in self.key_features:
                lines.append(f"- {feature}")
            lines.append("")
        
        if self.specifications:
            lines.append("## Specifications")
            for spec, value in self.specifications.items():
                lines.append(f"- {spec}: {value}")
            lines.append("")
        
        if self.pros:
            lines.append("## Pros")
            for pro in self.pros:
                lines.append(f"- {pro}")
            lines.append("")
        
        if self.cons:
            lines.append("## Cons")
            for con in self.cons:
                lines.append(f"- {con}")
        
        return "\n".join(lines)


class ProductExtractor:
    """Extracts structured product information using LLM"""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def extract(self, url: str, content: str) -> ProductInfo:
        """
        Extract product information from content
        
        Args:
            url: Source URL
            content: Raw content (markdown)
            
        Returns:
            ProductInfo object
        """
        
        print(f"🔍 Extracting from {len(content)} characters of content...")
        
        # First, try basic extraction for title
        basic_title = self._extract_title_fallback(content)
        print(f"   Basic title: {basic_title}")
        
        # Create extraction prompt
        system_prompt = """You are an expert at extracting product information from web content.

Extract structured information and return ONLY valid JSON (no markdown, no code blocks).

CRITICAL: Return raw JSON only - do NOT wrap it in ```json``` or any other formatting.

JSON Schema:
{
    "title": "exact product name",
    "brand": "brand name or null",
    "price": "numeric price only or null",
    "currency": "$ or ₹ or € or null",
    "category": "product category or null",
    "specifications": {"key": "value"},
    "key_features": ["feature1", "feature2"],
    "description": "brief description or null",
    "rating": null or number,
    "review_count": null or number,
    "pros": [],
    "cons": [],
    "availability": "availability status or null"
}

Extract what you can find. Use null for missing fields."""

        # Take first 12000 chars to avoid token limits
        content_sample = content[:12000]
        
        user_prompt = f"""Extract product information from this content:

URL: {url}

CONTENT:
{content_sample}

Return ONLY the JSON object with no markdown formatting."""

        try:
            print(f"   🤖 Calling LLM for extraction...")
            
            # Get LLM response
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            
            print(f"   📄 LLM response received ({len(response)} chars)")
            
            # Clean the response - remove markdown code blocks
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON
            data = json.loads(cleaned_response)
            
            print(f"   ✅ Parsed JSON successfully")
            print(f"   📱 Extracted: {data.get('title', 'Unknown')}")
            print(f"   💰 Price: {data.get('currency', '')}{data.get('price', 'N/A')}")
            print(f"   ✨ Features: {len(data.get('key_features', []))}")
            
            # Create ProductInfo object
            product = ProductInfo(
                url=url,
                title=data.get("title") or basic_title or "Unknown Product",
                brand=data.get("brand"),
                price=str(data.get("price")) if data.get("price") else None,
                currency=data.get("currency"),
                category=data.get("category"),
                specifications=data.get("specifications", {}),
                key_features=data.get("key_features", []),
                description=data.get("description"),
                rating=float(data.get("rating")) if data.get("rating") else None,
                review_count=int(data.get("review_count")) if data.get("review_count") else None,
                pros=data.get("pros", []),
                cons=data.get("cons", []),
                availability=data.get("availability")
            )
            
            return product
            
        except json.JSONDecodeError as e:
            # Fallback: create basic product info
            print(f"   ⚠️  JSON parsing failed: {e}")
            print(f"   📝 Response preview: {response[:300]}")
            
            # Try to extract from the partial JSON
            fallback_data = self._extract_from_partial_json(response)
            
            if fallback_data:
                print(f"   🔄 Using partial extraction")
                return ProductInfo(
                    url=url,
                    title=fallback_data.get('title') or basic_title,
                    brand=fallback_data.get('brand'),
                    price=fallback_data.get('price'),
                    currency=fallback_data.get('currency'),
                    description=content[:500],
                    key_features=fallback_data.get('key_features', [])
                )
            
            # Last resort: basic extraction
            fallback_price = self._extract_price_fallback(content)
            fallback_features = self._extract_features_fallback(content)
            
            print(f"   🔄 Using regex fallback extraction")
            
            return ProductInfo(
                url=url,
                title=basic_title,
                price=fallback_price.get('price') if fallback_price else None,
                currency=fallback_price.get('currency') if fallback_price else None,
                description=content[:500],
                key_features=fallback_features
            )
        
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return ProductInfo(
                url=url,
                title=self._extract_title_fallback(content),
                description=f"Extraction error: {str(e)}"
            )
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM"""
        # Remove markdown code blocks
        response = response.strip()
        
        # Remove ```json or ``` at start
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        # Remove ``` at end
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        # If response is truncated (doesn't end with }), try to fix it
        if not response.endswith('}'):
            # Find last complete field
            last_brace = response.rfind('}')
            if last_brace > 0:
                response = response[:last_brace + 1]
        
        return response
    
    def _extract_from_partial_json(self, response: str) -> Optional[Dict]:
        """Try to extract data from partial/broken JSON"""
        try:
            # Try to find and extract key fields with regex
            data = {}
            
            # Extract title
            title_match = re.search(r'"title":\s*"([^"]+)"', response)
            if title_match:
                data['title'] = title_match.group(1)
            
            # Extract brand
            brand_match = re.search(r'"brand":\s*"([^"]+)"', response)
            if brand_match:
                data['brand'] = brand_match.group(1)
            
            # Extract price
            price_match = re.search(r'"price":\s*"?(\d+)"?', response)
            if price_match:
                data['price'] = price_match.group(1)
            
            # Extract currency
            currency_match = re.search(r'"currency":\s*"([^"]+)"', response)
            if currency_match:
                data['currency'] = currency_match.group(1)
            
            return data if data else None
            
        except:
            return None
    
    def _extract_title_fallback(self, content: str) -> str:
        """Fallback title extraction from content"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Look for markdown heading
            if line.startswith('# '):
                title = line[2:].strip()
                # Clean up common artifacts
                title = re.sub(r'\[.*?\]\(.*?\)', '', title)
                title = title.strip()
                if len(title) > 3 and 'Image' not in title:
                    return title
        
        # If no heading, look for first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith(('http', 'www', '#', '-', '*', '!')):
                return line[:100]
        
        return "Unknown Product"
    
    def _extract_price_fallback(self, content: str) -> Optional[Dict]:
        """Fallback price extraction"""
        price_patterns = [
            (r'From\s+\$(\d+(?:,\d{3})*)', '$'),
            (r'\$\s*(\d+(?:,\d{3})*)', '$'),
            (r'₹\s*(\d+(?:,\d{3})*)', '₹'),
            (r'€\s*(\d+(?:,\d{3})*)', '€'),
        ]
        
        for pattern, currency in price_patterns:
            matches = re.findall(pattern, content)
            if matches:
                price = matches[0].replace(',', '')
                return {'price': price, 'currency': currency}
        
        return None
    
    def _extract_features_fallback(self, content: str) -> List[str]:
        """Fallback feature extraction"""
        features = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered lists
            if line.startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line):
                feature = re.sub(r'^[-*•\d.]+\s*', '', line).strip()
                if 5 < len(feature) < 100 and 'Image' not in feature:
                    features.append(feature)
                    if len(features) >= 10:
                        break
        
        return features
    
    def extract_multiple(self, url_content_map: Dict[str, str]) -> Dict[str, ProductInfo]:
        """Extract product info from multiple URLs"""
        results = {}
        
        for i, (url, content) in enumerate(url_content_map.items(), 1):
            print(f"\n{'='*70}")
            print(f"🔍 Extracting product {i}/{len(url_content_map)}")
            print(f"{'='*70}")
            product = self.extract(url, content)
            results[url] = product
            print(f"✅ Extracted: {product.title}")
        
        return results
