"""
Embedding Generation using Sentence Transformers
Converts text into dense vectors for semantic search
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model
                       - all-MiniLM-L6-v2: Fast, good quality (default)
                       - all-mpnet-base-v2: Better quality, slower
                       - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        """
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Dimension: {self.dimension}")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single string
        if isinstance(text, str):
            return self.model.encode([text])[0]
        
        # Handle list of strings
        return self.model.encode(text)
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for large batches efficiently
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


# Global instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global embedding generator"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
