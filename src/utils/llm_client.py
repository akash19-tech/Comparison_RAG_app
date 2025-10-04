"""
LLM Client for Groq API
Handles all LLM interactions
"""

import os
from groq import Groq
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient:
    """Wrapper for Groq LLM API"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize LLM client
        
        Args:
            model: Groq model to use
                   Options: llama-3.1-70b-versatile (best quality)
                           llama-3.1-8b-instant (fastest)
                           llama3-70b-8192
                           mixtral-8x7b-32768
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it in your .env file"
            )
        
        try:
            # Try new Groq client initialization
            self.client = Groq(api_key=api_key)
        except TypeError:
            # Fallback for older versions
            self.client = Groq(
                api_key=api_key,
            )
        
        self.model = model
        print(f"✅ LLM Client initialized with model: {model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Creativity (0.0-1.0, lower = more focused)
            max_tokens: Maximum response length
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """
        Generate JSON output
        Forces the model to return valid JSON
        """
        if system_prompt:
            system_prompt += "\n\nYou MUST respond with valid JSON only. No other text."
        else:
            system_prompt = "You MUST respond with valid JSON only. No other text."
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=3000
        )


# Global instance for easy access
_llm_client = None

def get_llm_client() -> LLMClient:
    """Get or create global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
