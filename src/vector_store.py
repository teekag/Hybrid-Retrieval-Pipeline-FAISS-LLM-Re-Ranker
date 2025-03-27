"""
FAISS-based vector database for storing document embeddings and searching similar content.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Any, Optional

class VectorStore:
    """
    A vector store implementation using FAISS for efficient similarity search.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimensionality of the embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.doc_ids = []
        self.documents = {}
        
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and optional metadata
            embeddings: Dictionary mapping document IDs to their embeddings
        """
        # Store documents for retrieval
        for doc in documents:
            doc_id = doc['id']
            self.documents[doc_id] = doc
            self.doc_ids.append(doc_id)
        
        # Add embeddings to FAISS index
        vectors = np.array([embeddings[doc_id] for doc_id in self.doc_ids], dtype=np.float32)
        self.index.add(vectors)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document info and similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query embedding is 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_ids):  # Ensure index is valid
                doc_id = self.doc_ids[idx]
                doc = self.documents[doc_id]
                
                results.append({
                    'id': doc_id,
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {}),
                    'score': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                })
                
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save document mappings
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump({
                "doc_ids": self.doc_ids,
                "documents": self.documents
            }, f)
            
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing the saved vector store
            
        Returns:
            Loaded VectorStore instance
        """
        # Load document mappings
        with open(os.path.join(directory, "documents.json"), "r") as f:
            data = json.load(f)
            
        # Load FAISS index
        index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        # Create and populate instance
        instance = cls(dimension=index.d)
        instance.index = index
        instance.doc_ids = data["doc_ids"]
        instance.documents = data["documents"]
        
        return instance
