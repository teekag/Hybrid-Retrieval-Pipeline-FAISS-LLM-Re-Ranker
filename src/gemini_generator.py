"""
Gemini LLM integration for generating answers based on retrieved documents.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai

class GeminiGenerator:
    """
    Handles the generation of answers using Google's Gemini model based on retrieved documents.
    """
    
    def __init__(
        self, 
        model: str = "gemini-pro",
        temperature: float = 0.3,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Gemini generator.
        
        Args:
            model: The Gemini model to use (e.g., "gemini-pro")
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens in the response
            api_key: Google API key (if None, uses environment variable)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided and not found in environment variables")
            genai.configure(api_key=api_key)
    
    def generate_answer(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer to the query based on the retrieved documents.
        
        Args:
            query: The user's query
            documents: List of retrieved documents with content and metadata
            prompt_template: Optional custom prompt template
            
        Returns:
            Dictionary with generated answer and metadata
        """
        # Use default prompt template if none provided
        if not prompt_template:
            prompt_template = self._get_default_prompt_template()
        
        # Format context from documents
        context = self._format_context(documents)
        
        # Create the prompt
        prompt = prompt_template.format(
            query=query,
            context=context
        )
        
        # Generate response from Gemini
        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            system_instruction = "You are a helpful assistant that provides accurate information based on the given context."
            
            response = model.generate_content(
                [
                    {"role": "system", "parts": [system_instruction]},
                    {"role": "user", "parts": [prompt]}
                ]
            )
            
            answer = response.text.strip()
            
            # Prepare result
            result = {
                "query": query,
                "answer": answer,
                "sources": [doc.get("id", "unknown") for doc in documents],
                "model": self.model
            }
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "sources": [doc.get("id", "unknown") for doc in documents]
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Extract source information
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            
            # Format document content
            content = doc.get("content", "").strip()
            
            # Add to context parts
            context_parts.append(f"[Document {i+1}] Source: {source}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _get_default_prompt_template(self) -> str:
        """
        Get the default prompt template for answer generation.
        
        Returns:
            Default prompt template string
        """
        return """
Given the following context information, please answer the question accurately and concisely.
If the answer cannot be determined from the context, please state that clearly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    def compare_with_without_retrieval(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]],
        random_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare answers generated with retrieved documents vs. random documents.
        
        Args:
            query: The user's query
            retrieved_docs: Documents retrieved by the pipeline
            random_docs: Random documents (if None, no documents will be used)
            
        Returns:
            Dictionary with both answers for comparison
        """
        # Generate answer with retrieved documents
        retrieved_answer = self.generate_answer(query, retrieved_docs)
        
        # Generate answer with random documents or no context
        if random_docs:
            random_answer = self.generate_answer(query, random_docs)
        else:
            # No context prompt
            no_context_prompt = "Please answer the following question based on your knowledge: {query}"
            
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            system_instruction = "You are a helpful assistant."
            
            response = model.generate_content(
                [
                    {"role": "system", "parts": [system_instruction]},
                    {"role": "user", "parts": [no_context_prompt.format(query=query)]}
                ]
            )
            
            random_answer = {
                "query": query,
                "answer": response.text.strip(),
                "sources": ["No context provided"],
                "model": self.model
            }
        
        # Return comparison
        return {
            "query": query,
            "with_retrieval": retrieved_answer,
            "without_retrieval": random_answer
        }
