# RAG Pipeline System Diagram

This document provides instructions for creating a comprehensive system diagram for the Hybrid Retrieval Pipeline using draw.io (diagrams.net).

## Diagram Structure

The system diagram should illustrate the complete flow of the RAG (Retrieval-Augmented Generation) pipeline, including:

1. **Document Processing**
   - Raw document ingestion
   - Text cleaning and preprocessing
   - Chunking strategy
   - Metadata enrichment

2. **Embedding & Indexing**
   - Document embedding generation
   - Vector database indexing
   - Metadata storage

3. **Query Processing**
   - Query embedding
   - Vector similarity search
   - TF-IDF reranking

4. **Answer Generation**
   - Context preparation
   - Prompt engineering
   - LLM integration
   - Response formatting

## Creating the Diagram

1. Go to [diagrams.net](https://app.diagrams.net/) (draw.io)
2. Create a new diagram
3. Use the following components:
   - Rectangles for processing steps
   - Cylinders for data storage
   - Arrows for data flow
   - Colors to distinguish different pipeline stages

## Example Structure

```
[Raw Documents] → [Preprocessing] → [Chunking] → [Embedding] → [Vector DB]
                                                                    ↑
[User Query] → [Query Processing] → [Query Embedding] → [Vector Search]
                                                            ↓
                                                     [Reranking]
                                                            ↓
                                                  [Context Preparation]
                                                            ↓
                                                  [LLM Integration]
                                                            ↓
                                                  [Generated Answer]
```

## Color Scheme

- **Blue**: Document processing steps
- **Green**: Embedding and indexing
- **Orange**: Query processing
- **Purple**: Answer generation
- **Gray**: Data storage components

## Additional Elements

- Include metrics calculation and evaluation components
- Show feedback loops for improvement
- Highlight the hybrid nature of the retrieval (dense + sparse)
- Include visualization components

## Export Instructions

1. After creating the diagram, export it as PNG and SVG formats
2. Save both files in the `diagrams` directory
3. Reference the diagram in the project README

## Sample Diagram

For reference, a sample RAG pipeline diagram might look like this:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Raw Documents│     │  Preprocessing │     │    Chunking   │     │   Embedding   │
│               │────▶│   & Cleaning   │────▶│   Strategy    │────▶│  Generation   │
└───────────────┘     └───────────────┘     └───────────────┘     └───────┬───────┘
                                                                          │
                                                                          ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   User Query  │     │Query Processing│     │Query Embedding│     │  Vector DB    │
│               │────▶│  & Analysis    │────▶│  Generation   │────▶│   (FAISS)     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────┬───────┘
                                                                          │
                                                                          ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Generated    │     │ LLM Integration│     │   Context     │     │   TF-IDF      │
│    Answer     │◀────│  (OpenAI API)  │◀────│  Preparation  │◀────│   Reranking   │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

This is a simplified ASCII representation. The actual diagram should be more detailed and visually appealing.
