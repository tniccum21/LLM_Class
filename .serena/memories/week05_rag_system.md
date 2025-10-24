# Week 05 - Ticket RAG System

## Overview
Advanced Retrieval-Augmented Generation (RAG) system for ticket resolution using historical ticket data.

## Implementation: Week05/ticket_rag_system.ipynb

### Core Features
1. **Data Management**
   - CSV loading with 80/20 train/test split
   - 72K+ tickets from dataset-tickets-multi-lang3-4k.csv
   - Automated data cleaning (removes entries with missing subject/body/answer)

2. **Vector Database (ChromaDB)**
   - Embeddings: subject + body + answer fields combined
   - Model: all-MiniLM-L6-v2 (384 dimensions)
   - Metadata: type, queue, priority, business_type
   - Persistent storage in ./chroma_ticket_db/

3. **Query System**
   - JSON input format with optional metadata filters
   - Top-K retrieval (configurable, default: 5)
   - Cosine similarity search
   - Metadata filtering support

4. **Response Generation**
   - LM Studio LLM integration (gpt-oss-20b)
   - Context-aware answers using similar tickets
   - **Rich responses include:**
     - Generated answer text
     - Confidence score (weighted similarity)
     - Number of references
     - Similar ticket details with previews
     - Timestamp

5. **LLM Evaluation System**
   - Compares generated vs original answers
   - Scores: accuracy, completeness, clarity, helpfulness
   - Overall rating (1-5 scale)
   - Detailed feedback from LLM judge
   - Batch evaluation capabilities

### Architecture
```
CSV → Split (80/20) → Embed → ChromaDB
                                  ↓
New Ticket (JSON) → Embed → Search → Retrieve Top-K
                                         ↓
                    Context → LLM → Answer + Confidence + Refs
```

### Configuration (All Centralised)
- csv_path, chroma_db_path, random_seed
- train_test_split (0.8), top_k (5)
- embedding_model, embedding_fields
- metadata_fields, similarity_threshold
- lm_studio_url, llm_model, temperature, max_tokens
- enable_reranking (placeholder for future)

### Key Functions
1. **load_and_split_data()**: CSV loading with train/test split
2. **embed_tickets()**: Generate embeddings with progress tracking
3. **load_vector_db()**: Initialize ChromaDB with batching
4. **search_similar_tickets()**: Vector search with metadata filtering
5. **generate_answer()**: LLM answer generation with context
6. **calculate_confidence()**: Weighted similarity scoring
7. **answer_ticket()**: Complete RAG pipeline (main interface)
8. **evaluate_answer()**: LLM-based answer quality assessment
9. **run_batch_evaluation()**: Bulk testing with statistics

### Usage Example
```python
ticket = {
    "subject": "VPN connection issue",
    "body": "Cannot connect to VPN from home",
    "type": "Incident",
    "priority": "high",
    "queue": "Technical Support"
}

response = answer_ticket(ticket)
# Returns: answer, confidence, references
```

### Modular Design
- Separate functions for each pipeline stage
- Configurable parameters (no hardcoded values)
- Reusable components for loading, embedding, querying
- Easy to extend with re-ranking or hybrid search

### Evaluation Capabilities
- Test set evaluation (20% of data)
- LLM-as-judge for quality assessment
- Batch processing with statistics
- Export results to CSV for analysis

### Future Extensions (Prepared)
- Re-ranking: enable_reranking flag ready
- Hybrid search: keyword + vector combination
- Multi-language: already handles multilingual data
- Feedback loop: structure ready for user feedback
- Advanced filters: complex metadata queries

## Dependencies
- chromadb, sentence-transformers, pandas
- requests, python-dotenv, openai
- LM Studio running on localhost:1234

## Differences from Week04
- CSV input vs PDF input
- Metadata-rich search vs simple similarity
- Train/test split for proper evaluation
- JSON I/O vs interactive queries
- LLM evaluation system
- Confidence scoring
- Reference tracking with previews
