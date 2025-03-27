"""
Main entry point for running the RAG pipeline as a module.
"""

import os
import sys
import json
import argparse
from pathlib import Path

from .rag_pipeline import RAGPipeline

def main():
    """
    Main function to run the RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description='Run the Hybrid Retrieval Pipeline')
    parser.add_argument('--docs', type=str, default='./data/sample_docs',
                        help='Path to documents directory')
    parser.add_argument('--query', type=str, 
                        default='What are the applications of NLP?',
                        help='Query to search for')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top results to return')
    parser.add_argument('--no-rerank', action='store_true',
                        help='Disable reranking')
    parser.add_argument('--output', type=str, default='./outputs/result.json',
                        help='Path to save the output JSON')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print(f"Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Load documents
    print(f"Loading documents from {args.docs}...")
    documents = pipeline.load_documents(args.docs)
    print(f"Loaded {len(documents)} documents")
    
    # Index documents
    print("Indexing documents...")
    pipeline.index_documents()
    
    # Process query
    print(f"Processing query: '{args.query}'")
    result = pipeline.query(
        args.query,
        top_k=args.top_k,
        rerank=not args.no_rerank
    )
    
    # Print results
    print("\nResults:")
    for i, doc in enumerate(result['results']):
        print(f"{i+1}. {doc['id']} (Score: {doc['score']:.3f})")
        print(f"   Source: {doc['metadata'].get('source', 'unknown')}")
        print(f"   Content preview: {doc['content'][:100]}...")
    
    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {args.output}")
    
    # Print timing info
    print("\nTiming information:")
    for key, value in result['timing'].items():
        print(f"  {key}: {value}ms")

if __name__ == "__main__":
    main()
