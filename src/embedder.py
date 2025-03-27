"""
Document and query embedding using SentenceTransformers.
"""

import os
from typing import List, Dict, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Handles embedding of documents and queries using SentenceTransformer models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, documents: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and optional metadata
            
        Returns:
            Dictionary mapping document IDs to their embeddings
        """
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create a dictionary mapping document IDs to embeddings
        doc_embeddings = {}
        for i, doc in enumerate(documents):
            doc_embeddings[doc['id']] = embeddings[i]
            
        return doc_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        
        Args:
            query: The query text to embed
            
        Returns:
            Embedding vector for the query
        """
        return self.model.encode(query)
    
    def batch_embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed multiple queries at once.
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of embedding vectors for the queries
        """
        return self.model.encode(queries, show_progress_bar=True)
