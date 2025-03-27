"""
Orchestrates embedding, retrieval, reranking, and outputs the top documents for a query.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from .embedder import Embedder
from .vector_store import VectorStore
from .reranker import Reranker

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline that combines embedding, 
    vector search, and reranking to retrieve relevant documents.
    """
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the SentenceTransformer model to use
            vector_store: Optional existing VectorStore instance
            embedder: Optional existing Embedder instance
            reranker: Optional existing Reranker instance
        """
        # Initialize components
        self.embedder = embedder or Embedder(model_name=embedding_model)
        self.vector_store = vector_store or VectorStore(dimension=384)  # Default dimension for all-MiniLM-L6-v2
        self.reranker = reranker or Reranker()
        
        # Storage for documents
        self.documents = []
        
    def load_documents(self, document_path: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Load documents from a file or directory.
        
        Args:
            document_path: Path to a file, directory, or list of document texts
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        if isinstance(document_path, list):
            # Handle case where document_path is a list of strings
            for i, text in enumerate(document_path):
                documents.append({
                    'id': f"doc_{i}",
                    'content': text,
                    'metadata': {'source': 'inline'}
                })
        else:
            path = Path(document_path)
            
            if path.is_file():
                # Load a single file
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'id': path.stem,
                        'content': content,
                        'metadata': {'source': str(path)}
                    })
            elif path.is_dir():
                # Load all text files in directory
                for file_path in path.glob('**/*.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'id': file_path.stem,
                            'content': content,
                            'metadata': {'source': str(file_path)}
                        })
        
        self.documents = documents
        return documents
    
    def index_documents(self) -> None:
        """
        Embed and index the loaded documents in the vector store.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
            
        # Embed documents
        doc_embeddings = self.embedder.embed_documents(self.documents)
        
        # Add to vector store
        self.vector_store.add_documents(self.documents, doc_embeddings)
        
    def query(
        self, 
        query_text: str, 
        top_k: int = 5, 
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            rerank: Whether to apply reranking
            
        Returns:
            Dictionary with query results and timing information
        """
        start_time = time.time()
        
        # Embed query
        query_embedding_time = time.time()
        query_embedding = self.embedder.embed_query(query_text)
        query_embedding_time = time.time() - query_embedding_time
        
        # Vector search
        vector_search_time = time.time()
        search_results = self.vector_store.search(query_embedding, top_k=top_k * 2 if rerank else top_k)
        vector_search_time = time.time() - vector_search_time
        
        # Reranking (if enabled)
        rerank_time = 0
        if rerank and search_results:
            rerank_time = time.time()
            search_results = self.reranker.rerank(query_text, search_results, top_k=top_k)
            rerank_time = time.time() - rerank_time
        
        total_time = time.time() - start_time
        
        # Prepare response
        response = {
            'query': query_text,
            'results': search_results,
            'timing': {
                'query_embedding_ms': round(query_embedding_time * 1000, 2),
                'vector_search_ms': round(vector_search_time * 1000, 2),
                'rerank_ms': round(rerank_time * 1000, 2),
                'total_ms': round(total_time * 1000, 2)
            },
            'metadata': {
                'embedding_model': self.embedder.model_name,
                'num_documents': len(self.documents),
                'top_k': top_k,
                'reranking_applied': rerank
            }
        }
        
        return response
    
    def save(self, directory: str) -> None:
        """
        Save the pipeline components to disk.
        
        Args:
            directory: Directory to save the pipeline
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save vector store
        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.save(vector_store_dir)
        
        # Save pipeline config
        config = {
            'embedding_model': self.embedder.model_name,
            'num_documents': len(self.documents)
        }
        
        with open(os.path.join(directory, "pipeline_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, directory: str) -> 'RAGPipeline':
        """
        Load a pipeline from disk.
        
        Args:
            directory: Directory containing the saved pipeline
            
        Returns:
            Loaded RAGPipeline instance
        """
        # Load pipeline config
        with open(os.path.join(directory, "pipeline_config.json"), "r") as f:
            config = json.load(f)
            
        # Load vector store
        vector_store_dir = os.path.join(directory, "vector_store")
        vector_store = VectorStore.load(vector_store_dir)
        
        # Create pipeline with loaded components
        pipeline = cls(
            embedding_model=config['embedding_model'],
            vector_store=vector_store
        )
        
        return pipeline
