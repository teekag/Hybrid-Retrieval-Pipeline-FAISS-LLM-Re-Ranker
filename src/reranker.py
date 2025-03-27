"""
TF-IDF + cosine similarity re-ranker to improve top-K document accuracy.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Reranker:
    """
    Reranks retrieved documents using TF-IDF and cosine similarity.
    """
    
    def __init__(self):
        """
        Initialize the reranker with a TF-IDF vectorizer.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english',
            max_features=10000
        )
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on TF-IDF similarity to the query.
        
        Args:
            query: The query string
            documents: List of document dictionaries with 'content' and other fields
            top_k: Number of top results to return (None returns all reranked)
            
        Returns:
            Reranked list of document dictionaries with updated scores
        """
        if not documents:
            return []
            
        # Extract document contents
        doc_contents = [doc['content'] for doc in documents]
        
        # Fit and transform documents and query
        try:
            tfidf_matrix = self.vectorizer.fit_transform([query] + doc_contents)
            
            # Calculate cosine similarity between query and documents
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Create reranked results
            reranked_results = []
            for i, score in enumerate(similarities):
                doc = documents[i].copy()
                doc['tfidf_score'] = float(score)
                
                # Combine original score with TF-IDF score (weighted average)
                original_score = doc.get('score', 0.0)
                doc['score'] = 0.4 * original_score + 0.6 * score
                
                reranked_results.append(doc)
                
            # Sort by combined score in descending order
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k if specified
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
                
            return reranked_results
            
        except ValueError as e:
            # Handle case where vectorizer fails (e.g., empty documents)
            print(f"Reranking failed: {e}")
            return documents[:top_k] if top_k is not None else documents
