"""
Jina Reader API Integration
Fetches clean, markdown-formatted content from any URL
"""

import requests
from typing import Dict, Optional
from dataclasses import dataclass
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import re


@dataclass
class URLContent:
    """Structured content from a URL"""
    url: str
    title: str
    content: str
    markdown: str
    success: bool
    error: Optional[str] = None
    fetch_time: float = 0.0


class JinaReader:
    """
    Fetches and cleans content from URLs using Jina Reader API
    Free tier: No API key needed!
    """
    
    BASE_URL = "https://r.jina.ai/"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize Jina Reader
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Product-Comparison-RAG/1.0)'
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def fetch_url(self, url: str) -> URLContent:
        """
        Fetch content from a URL
        
        Args:
            url: The URL to fetch
            
        Returns:
            URLContent object with extracted data
        """
        start_time = time.time()
        
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return URLContent(
                    url=url,
                    title="",
                    content="",
                    markdown="",
                    success=False,
                    error="Invalid URL: Must start with http:// or https://"
                )
            
            # Make request to Jina Reader
            jina_url = f"{self.BASE_URL}{url}"
            response = self.session.get(jina_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Get markdown content
            markdown_content = response.text
            
            # Extract title from Jina's format
            title = self._extract_title_from_jina(markdown_content)
            
            # Clean content (remove Jina metadata)
            clean_content = self._clean_jina_content(markdown_content)
            
            fetch_time = time.time() - start_time
            
            return URLContent(
                url=url,
                title=title,
                content=clean_content,
                markdown=markdown_content,
                success=True,
                fetch_time=fetch_time
            )
            
        except requests.exceptions.Timeout:
            return URLContent(
                url=url,
                title="",
                content="",
                markdown="",
                success=False,
                error=f"Timeout: Could not fetch URL within {self.timeout} seconds",
                fetch_time=time.time() - start_time
            )
            
        except requests.exceptions.RequestException as e:
            return URLContent(
                url=url,
                title="",
                content="",
                markdown="",
                success=False,
                error=f"Request error: {str(e)}",
                fetch_time=time.time() - start_time
            )
            
        except Exception as e:
            return URLContent(
                url=url,
                title="",
                content="",
                markdown="",
                success=False,
                error=f"Unexpected error: {str(e)}",
                fetch_time=time.time() - start_time
            )
    
    def _extract_title_from_jina(self, content: str) -> str:
        """Extract title from Jina's format"""
        # Jina format: "Title: <title>"
        title_match = re.search(r'^Title:\s*(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
            if title and title.lower() != 'untitled':
                return title
        
        # Fallback: Look for first # heading in markdown
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not line.startswith('# Image'):
                title = line[2:].strip()
                # Remove markdown links
                title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
                # Remove images
                title = re.sub(r'!\[.*?\]\(.*?\)', '', title)
                title = title.strip()
                if title and len(title) > 2:
                    return title
        
        # Last resort: Look for URL Source
        url_match = re.search(r'URL Source:\s*https?://(?:www\.)?([^/]+)', content)
        if url_match:
            domain = url_match.group(1)
            if 'apple.com' in domain:
                return 'Apple Product'
            elif 'samsung.com' in domain:
                return 'Samsung Product'
        
        return "Product Page"
    
    def _clean_jina_content(self, markdown: str) -> str:
        """
        Clean Jina Reader output
        - Remove metadata headers
        - Remove navigation items
        - Keep main content
        """
        lines = markdown.split('\n')
        cleaned_lines = []
        
        skip_until_markdown = True
        
        for line in lines:
            # Skip metadata section
            if 'Markdown Content:' in line:
                skip_until_markdown = False
                continue
            
            if skip_until_markdown:
                continue
            
            line = line.strip()
            
            # Skip empty lines at start
            if not cleaned_lines and not line:
                continue
            
            # Skip navigation/menu items
            if self._is_navigation(line):
                continue
            
            # Skip image-only lines
            if line.startswith('![Image') and '](' in line and line.endswith(')'):
                continue
            
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # Remove multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _is_navigation(self, line: str) -> bool:
        """Check if line is likely navigation/menu content"""
        if not line:
            return False
        
        nav_keywords = [
            'menu', 'navigation', 'navbar', 'footer', 
            'cookie', 'subscribe', 'sign in', 'log in',
            'skip to', 'breadcrumb'
        ]
        line_lower = line.lower()
        
        # Check for nav keywords
        if any(keyword in line_lower for keyword in nav_keywords):
            return True
        
        # Check for navigation patterns like "* [Link](url)"
        if re.match(r'^\*\s*\[.*?\]\(.*?\)$', line):
            return True
        
        return False
    
    def fetch_multiple(self, urls: list[str]) -> Dict[str, URLContent]:
        """
        Fetch content from multiple URLs
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            Dictionary mapping URL to URLContent
        """
        results = {}
        
        for url in urls:
            print(f"🔄 Fetching: {url}")
            content = self.fetch_url(url)
            results[url] = content
            
            if content.success:
                print(f"✅ Success: {content.title} ({content.fetch_time:.2f}s)")
            else:
                print(f"❌ Failed: {content.error}")
            
            # Be nice to the API - small delay between requests
            time.sleep(1)
        
        return results


# Convenience function
def fetch_url_content(url: str) -> URLContent:
    """Quick function to fetch a single URL"""
    reader = JinaReader()
    return reader.fetch_url(url)
