# Week 05 - RAG System for Ticket Support

## Learning Summary Presentation

---

# What is RAG?

## Retrieval-Augmented Generation

- **Combine search with AI generation** to provide context-aware answers
- **Pattern**: `Document → Embed → Store → Query → Retrieve Similar → LLM Answer`
- **Advantage**: LLM answers grounded in actual historical data, not just general knowledge

### Why RAG Matters
- Traditional LLM: General knowledge, may hallucinate
- RAG: Grounded in real historical tickets, traceable sources
- Best of both: Search precision + LLM comprehension

---

# Vector Embeddings & Semantic Search

## How Text Becomes Searchable

- **Embeddings convert text to 384-dimensional vectors** using `all-MiniLM-L6-v2` model
- **Semantic similarity** via cosine distance (similar problems = similar vectors)
- **Combined fields** (subject + body + answer) capture full problem→solution pattern
- **ChromaDB vector database** enables fast similarity search across thousands of tickets

### Example
```
"VPN connection failed" → [0.12, -0.34, 0.56, ...]
"Cannot connect to VPN" → [0.11, -0.33, 0.58, ...]  ← Very similar!
"Printer not working" → [-0.45, 0.78, -0.12, ...]  ← Different
```

---

# Train/Test Split Methodology

## Proper ML Evaluation

- **80/20 split** (2,880 train / 721 test) for unbiased evaluation
- **Training set**: Build knowledge base (embed & store in vector DB)
- **Test set**: Evaluate system on unseen tickets
- **Random seed (42)** ensures reproducible experiments

### Why This Matters
- **No data leakage**: Test tickets never seen during training
- **Unbiased metrics**: Realistic performance on new tickets
- **Scientific rigor**: Same as ML best practices

---

# RAG Pipeline Architecture

## End-to-End Flow

```
Pre-translated CSV → Train/Test Split → Generate Embeddings → ChromaDB Storage
                                                                     ↓
New Ticket (JSON) → Embed Query → Vector Search → Top-K Retrieval
                                                        ↓
                        Context + Question → LM Studio LLM → Answer + Confidence
```

### Pipeline Stages
1. **Data Preparation**: Load & split tickets
2. **Embedding**: Convert to vectors
3. **Storage**: ChromaDB vector database
4. **Query**: New ticket arrives
5. **Retrieval**: Find top-K similar tickets
6. **Generation**: LLM creates context-aware answer

---

# Dual RAG Modes

## Strict vs Augmented

### Strict Mode
- **LLM uses ONLY historical ticket context** (traceable, compliance-friendly)
- Will say "I don't have enough information" if context insufficient
- **Use case**: QA testing retrieval quality, compliance environments

### Augmented Mode
- **LLM uses context + general knowledge** (helpful, flexible)
- Always provides helpful answer, supplements with IT knowledge
- **Use case**: Production support, maximize helpfulness

### Trade-off
**Strict** = Traceability + Coverage gaps
**Augmented** = Helpfulness + Mixed sources

---

# Confidence Scoring

## How Certain Is The System?

- **Weighted similarity scores** from retrieved tickets
- **Top results weighted more heavily** (1.0, 0.8, 0.6, 0.4, 0.2)
- **Range 0-1**: Higher confidence = more relevant historical tickets found
- **Helps users trust system answers** (e.g., 90% confidence vs 40%)

### Example Calculation
```
Top-5 similarities: [0.91, 0.88, 0.85, 0.82, 0.79]
Weighted average: (0.91×1.0 + 0.88×0.8 + 0.85×0.6 + 0.82×0.4 + 0.79×0.2) / 3.0
Confidence: 87%
```

---

# LLM-as-Judge Evaluation

## AI Judging AI Quality

### Evaluation Criteria (1-5 scale)
- **Accuracy**: Correctness of information
- **Completeness**: Coverage of important points
- **Clarity**: Ease of understanding
- **Helpfulness**: Actionability of solution

### Process
1. RAG generates answer for test ticket
2. LLM compares generated vs. original answer
3. Scores each criterion + provides feedback
4. Overall score (1-5) + detailed explanation

### Why This Works
- **Consistent**: Same evaluation logic for all tickets
- **Scalable**: Batch evaluate hundreds of tickets
- **Detailed**: Specific feedback for improvements

---

# Building on Previous Weeks

## Evolution Timeline

### Week 02/03 → Week 05
- **Week 02**: Individual ticket translation + classification (one-at-a-time)
- **Week 03**: Optimized LLM calls, JSON output, token tracking
- **Week 05**: Pre-translated data + RAG (leverage historical knowledge base)

### Week 04 → Week 05
- **Week 04**: RAG with PDF documents (arbitrary chunks)
- **Week 05**: RAG with structured tickets (subject/body/answer fields)
- **Key difference**: PDFs need chunking strategy, tickets use structured fields directly

---

# Design Patterns Reinforced

## Code Organization Principles

### Configuration-First
- **Centralized `CONFIG` dictionary** - all parameters in one place
- Easy experimentation: change `top_k` once, not in multiple functions

### Modular Architecture
- **Reusable components** (load, embed, search, generate)
- Each function does one thing well

### Consistency Across Weeks
- **Helper function pattern** - consistent with Week 02/03 (`show_error`, `call_llm_sdk`)
- **OpenAI SDK integration** - same interface across weeks

---

# Data Management Strategy

## Pre-Translation Efficiency

### One-Time Translation
- **`translate_tickets.py`** runs once (75 min for 4K tickets on GROQ, $1.24 cost)
- **Separate `_english` columns** preserve originals while enabling fast loading
- **Efficiency gain**: No runtime translation overhead, consistent translations

### CSV Structure
- **Original columns**: `subject`, `body`, `answer` (multilingual: es/fr/de/pt/en)
- **Translation columns**: `subject_english`, `body_english`, `answer_english`
- **Metadata**: `type`, `queue`, `priority`, `business_type`, `original_language`
- **Data cleaning**: Remove 397 tickets with missing translations (3,601 valid)

---

# Performance & Optimization

## Speed Metrics

### Test Mode for Learning
- **`test_mode: True`** loads only first 100 tickets (fast experimentation)
- **`test_mode: False`** processes full 3,601 ticket dataset

### Timing Breakdown
- **Embedding speed**: ~5 seconds for 100 tickets, ~2 minutes for full dataset
- **Query speed**: <1 second per ticket (vector search is fast!)
- **Evaluation speed**: ~10 seconds per ticket (includes LLM calls)

### Key Parameters to Tune
- **`top_k` (default: 5)**: Number of similar tickets to retrieve (try 3-10)
- **`temperature` (default: 0.2)**: LLM creativity (0.1=factual, 0.9=creative)
- **`embedding_model`**: Trade speed vs. quality (all-MiniLM-L6-v2 is balanced)

---

# Technical Implementation

## ChromaDB Integration

- **Persistent storage** at `./chroma_ticket_db/`
- **Cosine similarity** metric for vector search
- **Batch loading** (1,000 tickets per batch) handles large datasets
- **Metadata filtering** (not used yet, prepared for future enhancements)

## LM Studio LLM Integration

- **Local LLM server** at http://169.254.233.17:1234
- **Model**: gpt-oss-20b (open-source 20B parameter model)
- **OpenAI-compatible API** enables easy integration
- **Timeout handling**: 60-second limit for reliability
- **Max tokens**: 2,000 for comprehensive answers

---

# Embedding Strategy

## Combining Ticket Fields

### Combined Text Approach
```python
subject_english: "VPN connection failed"
body_english: "Cannot connect from home, timeout error..."
answer_english: "Check firewall settings, restart VPN client..."
```

### Why Combine?
- **Subject**: Main issue keywords
- **Body**: Detailed context and symptoms
- **Answer**: Solution keywords and actions
- **Result**: Embedding captures full problem→solution pattern

### Model Details
- **sentence-transformers all-MiniLM-L6-v2** (384 dimensions)
- **Progress tracking**: Shows batch processing status
- **Field labeling**: Each field prefixed for clarity

---

# Quality Metrics & Results

## Batch Evaluation Performance

### Evaluation Results (5 test tickets)
- **Overall score**: 5.00/5 average (excellent quality)
- **Accuracy**: 5.00/5 (correct information)
- **Completeness**: 4.80/5 (comprehensive coverage)
- **Clarity**: 5.00/5 (easy to understand)
- **Helpfulness**: 5.00/5 (actionable solutions)
- **Average confidence**: 87% (high similarity to historical tickets)

### What These Numbers Mean
- **5/5 scores**: RAG answers match original quality
- **87% confidence**: System finds highly relevant historical tickets
- **Consistent performance**: All test tickets scored well

---

# Troubleshooting Signals

## When to Worry

### Low Confidence (<50%)
- **Issue**: Not finding relevant historical tickets
- **Fix**: Check retrieval quality, increase `top_k`, verify embeddings

### Low Overall Scores (<3)
- **Issue**: Generated answers not meeting quality bar
- **Fix**: Review LLM prompts, adjust temperature, increase `top_k`

### High Confidence + Low Scores
- **Issue**: Finding similar tickets but generating poor answers
- **Fix**: Training data quality issues, check LLM prompt engineering

---

# Week 04 vs Week 05

## Key Differences

| Aspect | Week 04 (PDF RAG) | Week 05 (Ticket RAG) |
|--------|-------------------|----------------------|
| **Input** | PDF documents | Pre-translated CSV |
| **Structure** | Unstructured text chunks | Structured fields (subject/body/answer) |
| **Chunking** | Required (arbitrary splits) | Not needed (use fields directly) |
| **Metadata** | Minimal | Rich (type, queue, priority, language) |
| **Evaluation** | Manual inspection | LLM-as-judge with metrics |
| **Split** | None | Train/test (80/20) |
| **I/O** | Interactive queries | JSON input format |

---

# Practical Applications

## Real-World Use Cases

### IT Support Automation
- Answer common support tickets using historical solutions
- Reduce response time from hours to seconds
- Consistent quality across all agents

### Knowledge Base Search
- Find similar past issues for technicians
- Discover solutions from months/years ago
- Institutional knowledge retention

### Quality Assurance
- Test if answers match documented solutions
- Identify gaps in knowledge base
- Validate new support agent responses

### Training Data
- Generate examples for support agent training
- Show best-practice responses
- Onboard new team members faster

---

# Production Considerations

## Deployment Factors

### Performance
- **Response time**: <1 second for search, variable for LLM generation
- **Scalability**: ChromaDB handles thousands of tickets efficiently
- **Cost**: Local LLM = no API costs (vs. GPT-4 at ~$0.03/1K tokens)

### Privacy & Security
- **All data stays local** (important for sensitive support tickets)
- **No external API calls** for ticket data
- **Compliance-friendly** (strict mode for traceability)

### Operational
- **Persistent storage**: Vector DB survives restarts
- **Incremental updates**: Add new tickets without full rebuild
- **Monitoring**: Track confidence scores, evaluation metrics

---

# Future Enhancements

## Already Configured (Week 06+)

- **`enable_reranking` flag**: Placeholder for semantic re-ranking
- **Metadata filtering**: Filter by type, priority, queue for targeted search
- **Hybrid search**: Combine vector similarity + keyword matching

## Potential Improvements

- **Multilingual answers**: Return responses in user's preferred language
- **API wrapper**: REST API for easy integration with existing systems
- **Feedback loop**: User ratings to improve future answers
- **Caching**: Store common query results for faster response
- **Advanced confidence**: Incorporate metadata matching, recency, user feedback

---

# Technical Skills Acquired

## What You Learned

✅ **Vector embeddings** - Convert text to semantic representations
✅ **Similarity search** - Find relevant documents using cosine distance
✅ **RAG pipeline** - Combine retrieval with generation for grounded answers
✅ **Train/test methodology** - Proper ML evaluation practices
✅ **LLM evaluation** - Use AI to judge AI outputs systematically
✅ **Batch processing** - Scale evaluation across multiple examples

---

# Design Principles Reinforced

## Software Engineering Best Practices

✅ **Configuration-first** - All parameters centralized for easy experimentation
✅ **Modular architecture** - Reusable functions for each pipeline stage
✅ **Proper evaluation** - Systematic quality assessment, not anecdotal testing
✅ **Efficiency** - Pre-translate once, reuse many times
✅ **Transparency** - Show confidence scores, similar tickets, reasoning

---

# From Concept to Production

## Journey Across Weeks

### Week 02/03: Individual Ticket Processing
- **Focus**: Proof of concept
- **Pattern**: One ticket at a time
- **Output**: Classification + translation

### Week 04: RAG Fundamentals with PDFs
- **Focus**: Core RAG technique
- **Pattern**: Document → Chunk → Embed → Search → Generate
- **Output**: Context-aware answers

### Week 05: Production-Ready Ticket RAG
- **Focus**: Real-world application
- **Pattern**: Structured data + proper evaluation
- **Output**: Evaluated, confident answers

### Week 06: Enhanced RAG (Coming Next)
- **Focus**: Optimization via re-ranking
- **Pattern**: Two-stage retrieval
- **Output**: Even better relevance

---

# Homework Experiments

## Parameter Tuning

### 1. Top-K Tuning
- **Test**: 1, 3, 5, 10 similar tickets
- **Question**: At what point does more context stop helping?
- **Expected**: Diminishing returns after 5-7 tickets (information overload)

### 2. Temperature Testing
- **Test**: 0.1, 0.5, 0.9 for LLM creativity
- **Question**: Balance factual vs. creative?
- **Expected**: ~0.2 optimal for factual support (avoid hallucinations)

### 3. Batch Evaluation
- **Test**: Run 20+ tickets
- **Question**: Do certain ticket types perform better?
- **Expected**: Some categories may have better historical coverage

---

# Resources & Files

## Project Structure

### Key Files
- **Main notebook**: `Week05/ticket_rag_system.ipynb`
- **Translation script**: `Week05/translate_tickets.py` (one-time setup)
- **Input CSV**: `dataset-tickets-multi-lang3-4k-translated-all.csv`
- **Vector DB**: `./chroma_ticket_db/` (persistent storage)

### Dependencies
```python
chromadb              # Vector database
sentence-transformers # Embedding models
pandas                # Data manipulation
requests              # LM Studio API calls
python-dotenv         # Environment management
openai                # LLM SDK integration
```

---

# Success Metrics Summary

## Overall Performance

**Quality**: 5.00/5 overall score (LLM evaluation)
**Confidence**: 87% average (high similarity to historical tickets)
**Coverage**: 3,601 tickets processed (80/20 train/test split)
**Speed**: <1s query time, ~10s with LLM generation
**Accuracy**: 100% (5/5) on batch evaluation
**Completeness**: 96% (4.8/5) - nearly comprehensive

---

# Next Steps to Week 06

## What's Coming

### Two-Stage Retrieval
- **Stage 1**: Vector search retrieves top-20 candidates (fast, broad)
- **Stage 2**: Cross-encoder re-ranks to top-5 (slow, precise)
- **Result**: Better relevance, improved answer quality

### Re-Ranking Model
- **mixedbread-ai/mxbai-rerank-base-v1** for semantic re-ordering
- **278M parameters** (~1GB memory, works on MPS)
- **Deeper semantic understanding** than pure vector similarity

### Comparison Framework
- **Same evaluation methodology** (LLM-as-judge)
- **Week05 vs Week06 performance** side-by-side
- **Quantify improvement** from re-ranking

---

# Skills to Build On

## Progression Path

### Current: Week 05
- **Pure vector similarity** (fast, good baseline)
- **Single-stage retrieval** (simple, effective)
- **Solid foundation** for enhancement

### Next: Week 06
- **Semantic re-ranking** (slower, better quality)
- **Two-stage retrieval** (recall + precision)
- **Advanced relevance** tuning

### Future: Week 07+
- **Hybrid search** (vector + keyword)
- **Metadata filtering** (type, priority, queue)
- **Multilingual support** (cross-lingual retrieval)

---

# Key Takeaways

## What Makes RAG Powerful

1. **Grounded Answers**: LLM uses real historical data, not hallucinations
2. **Scalable Knowledge**: Add new tickets without retraining LLM
3. **Traceable Sources**: Show which historical tickets influenced answer
4. **Confidence Scores**: Measure system certainty, build user trust
5. **Systematic Evaluation**: Measure quality objectively, not anecdotally

## Remember
- **RAG = Retrieval + Generation** (search precision + LLM comprehension)
- **Vector embeddings** make semantic search possible
- **Proper evaluation** (train/test split) ensures honest metrics
- **Configuration-first** enables rapid experimentation

---

# Questions & Discussion

## Topics for Exploration

- How does RAG compare to fine-tuning LLMs?
- When should you use strict vs augmented mode?
- How do you handle tickets with no similar historical examples?
- What confidence threshold should trigger human review?
- How do you keep the knowledge base current as new tickets arrive?

---

# Thank You!

## Week 05 Complete ✅

**Next**: Week 06 - Advanced RAG with Re-Ranking

**Resources**:
- Notebook: `Week05/ticket_rag_system.ipynb`
- Memory: `week05_rag_system.md`
- This presentation: `Week05_Learning_Summary.md`
