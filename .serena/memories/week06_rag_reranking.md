# Week 06 - Advanced RAG with Re-ranking

## Overview
Enhanced RAG system building on Week05 with two-stage retrieval: vector similarity + cross-encoder re-ranking for improved answer quality.

## Implementation: Week06/ticket_rag_system.ipynb

### Key Enhancements from Week05
1. **Two-Stage Retrieval**
   - Stage 1: Vector search retrieves top-20 candidates (cosine similarity)
   - Stage 2: Cross-encoder re-ranks to top-5 most relevant (semantic understanding)
   
2. **Re-ranking Model**
   - Model: mixedbread-ai/mxbai-rerank-base-v1 (278M parameters, ~1GB memory)
   - Evaluates query-document pairs for deeper semantic matching
   - Significantly improves relevance over pure vector similarity

3. **Dual RAG Modes**
   - **Strict Mode**: LLM constrained to ONLY use historical ticket context
   - **Augmented Mode**: LLM can supplement with general IT knowledge
   - Configurable via `rag_mode` in CONFIG

4. **Enhanced Evaluation**
   - LLM-as-judge rates answers 1-5 on: accuracy, completeness, clarity, actionability
   - JSON-based structured evaluation with fallback handling
   - Handles LM Studio special tokens in responses

### Architecture
```
CSV → Split (80/20) → Embed → ChromaDB
                                  ↓
New Ticket → Embed → Vector Search (top-20)
                           ↓
                    Re-rank (top-5) → LLM Context
                                          ↓
                              Strict/Augmented → Answer + Evaluation
```

### Configuration (Cell 2)
```python
CONFIG = {
    'top_k_initial': 20,          # Initial vector retrieval
    'top_k_reranked': 5,          # After re-ranking
    'reranker_model': 'mixedbread-ai/mxbai-rerank-base-v1',
    'rag_mode': 'strict',         # 'strict' or 'augmented'
    'lm_studio_url': 'http://192.168.7.171:1234',
    'llm_model': 'gpt-oss-20b',
    'max_tokens': 6000,
    # ... embedding and ChromaDB config from Week05
}
```

### Core Functions (Cell 9)
1. **search_similar_tickets()**: Vector search for initial candidates
2. **rerank_tickets()**: Cross-encoder re-ranking of candidates
3. **generate_answer()**: Context-aware LLM answer generation with mode support
4. **calculate_confidence()**: Weighted confidence from re-rank scores
5. **evaluate_answer()**: LLM-based quality assessment with JSON parsing

### User Interface (Cell 11)
- Interactive ticket input
- Step-by-step pipeline execution display:
  1. Retrieve 20 similar tickets (vector search)
  2. Re-rank to top 5 (cross-encoder)
  3. Generate answer (LLM with context)
  4. Evaluate quality (LLM-as-judge)
- Complete output with answer, scores, and full ticket references

### Technical Details

#### Re-ranking Process
```python
pairs = [[query_text, ticket['document']] for ticket in tickets]
scores = reranker.predict(pairs)  # Cross-encoder scores
reranked = sorted(tickets, key=lambda x: x['rerank_score'], reverse=True)[:5]
```

#### Evaluation JSON Parsing
- Handles LM Studio special tokens: `<|channel|>`, `<|constrain|>`, `<|message|>`
- Extracts JSON from markdown code blocks
- Fallback to default scores if parsing fails
- Error messages preserved in response

#### Memory Constraints
- Attempted models:
  - ❌ Qwen3-Reranker-8B: 30GB+ memory (OOM on MPS)
  - ✅ mxbai-rerank-base-v1: ~1GB memory (works on MPS)

### Dataset Information
- **Primary**: dataset-tickets-multi-lang3-4k-translated-all.csv (3,601 tickets)
- **Train/Test Split**: 80/20 (2,880 train, 721 test)
- **Random Seed**: 42 (reproducible splits)

### Data Quality Issue: Near-Duplicates
Investigation revealed dataset contains near-duplicate tickets:
- Example: "Issue with Jira Software 8.20" appears 4 times with slight variations
- Train ticket [29] has 78.57% word overlap with test ticket [555]
- **NOT data leakage** - duplicates exist in original CSV before splitting
- Realistic for production ticket systems (users submit similar issues)

#### Duplicate Analysis Results
- Test ticket [555] at shuffled index 3435
- Train indices: 0-2879, Test indices: 2880-3600
- No exact matches between train/test (split is correct)
- 3 tickets with subject "Issue with Jira Software 8.20" in training set
- Only train[29] is near-duplicate (minor wording differences: "urgently" vs "ASAP")

### Performance
- Re-ranking: ~0.2-0.5s for 20 tickets
- LLM generation: Variable (depends on server)
- Total pipeline: 2-5 seconds per query

### LM Studio Configuration
- **Working URL**: http://192.168.7.171:1234
- **Previous issue**: Wrong IP (169.254.233.17) caused APIConnectionError
- **Model**: gpt-oss-20b
- **Timeout**: 60 seconds
- **Connection test**: Added diagnostic cell (Cell 10) for troubleshooting

### Key Improvements Over Week05
1. **Better relevance**: Re-ranking significantly improves top-5 results
2. **Dual modes**: Strict mode tests context-only performance, augmented allows knowledge supplement
3. **Production-ready**: Handles real-world dataset issues (duplicates, parsing errors)
4. **Comprehensive evaluation**: Structured quality assessment with error handling
5. **Full transparency**: Shows all 5 references with complete content and re-rank scores

### Future Extensions
- Compare strict vs augmented mode performance
- Implement deduplication preprocessing
- Add more sophisticated confidence scoring
- Experiment with other re-ranking models
- Track duplicate impact on evaluation metrics

## Dependencies (Updated)
- chromadb, sentence-transformers, pandas
- requests, python-dotenv, openai
- **New**: CrossEncoder from sentence-transformers for re-ranking
- LM Studio running on 192.168.7.171:1234
