# Apps_with_AI - Session Context (2025-10-31)

## Project Overview
Educational repository demonstrating AI/ML applications with practical implementations across multiple weeks of coursework.

## Project Structure
```
Apps_with_AI/
├── Week01/          # LM Studio testing and setup
├── Week02/          # Ticket triage application (Streamlit)
├── Week03/          # Ticket triage materials
├── Week04/          # [Cleaned up - see commit 1133dad]
├── Week05/          # RAG system with ChromaDB (CSV-based)
├── Week06/          # Advanced RAG with re-ranking
├── Source_files/    # (gitignored) Data sources
├── .venv/           # Python virtual environment
└── .env             # API keys (OpenRouter, HF, Groq, Google)
```

## Current State (Git Status)
```
Branch: main
Recent Changes:
- M Week05/Week05.pptx
- M Week05/ticket_rag_system.ipynb
- ?? Week06/ (untracked)
- ?? .serena/memories/week06_rag_reranking.md (new memory)
```

## Key Applications

### 1. Ticket Triage System (Week02)
- **File**: [Week02/ticket_triage_app.py](Week02/ticket_triage_app.py)
- **Purpose**: Streamlit app for automated ticket categorization
- **Features**: Category/priority assignment, agent routing, translation
- **Data**: IT_Tickets/*.xlsx files
- **Status**: Modified in git, production-ready

### 2. RAG System v1 (Week05)
- **File**: [Week05/ticket_rag_system.ipynb](Week05/ticket_rag_system.ipynb)
- **Purpose**: Retrieval-Augmented Generation for ticket resolution
- **Data**: dataset-tickets-multi-lang3-4k.csv (72K+ tickets)
- **Vector DB**: ChromaDB with all-MiniLM-L6-v2 embeddings
- **Features**: 
  - Top-K retrieval with metadata filtering
  - LLM integration (LM Studio gpt-oss-20b)
  - Confidence scoring & evaluation
  - Train/test split (80/20)
- **Status**: Working, comprehensive evaluation system

### 3. RAG System v2 (Week06)
- **File**: [Week06/ticket_rag_system.ipynb](Week06/ticket_rag_system.ipynb)
- **Purpose**: Enhanced RAG with two-stage retrieval
- **Enhancements**:
  - Two-stage retrieval (vector + re-ranking)
  - Cross-encoder: mixedbread-ai/mxbai-rerank-base-v1
  - Dual RAG modes (strict vs augmented)
  - Enhanced LLM evaluation
- **Data**: dataset-tickets-multi-lang3-4k-translated-all.csv (3,601 tickets)
- **LM Studio**: http://192.168.7.171:1234 (configured correctly)
- **Status**: Untracked in git, production-ready with comprehensive testing

## Technical Stack

### Python Environment
- **Version**: Python 3.11
- **Environment**: .venv (virtual environment)
- **Key Dependencies**:
  - Streamlit (Week02 app)
  - ChromaDB (Week05/06 vector database)
  - sentence-transformers (embeddings & re-ranking)
  - pandas (data processing)
  - requests (LM Studio integration)
  - python-dotenv (environment management)

### External Services
- **LM Studio**: Local LLM server (gpt-oss-20b)
  - URL: http://192.168.7.171:1234
  - Used for: Answer generation, evaluation
- **OpenRouter**: API key available (.env)
- **Hugging Face**: Token available (.env)
- **Groq**: API key available (.env)
- **Google**: API key available (.env)

## Available Memories (Cross-Session)
1. **project_overview**: High-level structure and ticket triage app
2. **week05_rag_system**: Complete RAG v1 documentation
3. **week06_rag_reranking**: Advanced RAG v2 with re-ranking
4. **translation_fix_final**: Translation system fixes
5. **lm_studio_troubleshooting**: LM Studio connectivity fixes

## Recent Work Focus
- Week06 RAG system with re-ranking implementation
- Two-stage retrieval for improved relevance
- Dual RAG modes (strict vs augmented)
- LM Studio connectivity troubleshooting (resolved)
- Dataset quality analysis (near-duplicates identified)

## Session Readiness
✅ Project activated (Apps_with_AI)
✅ Memories loaded (5 available)
✅ Git status checked (main branch)
✅ File structure analyzed
✅ Dependencies identified
✅ API keys validated (.env present)
✅ Recent work context established

## Next Steps (Ready For)
- Continue Week06 development
- Commit untracked Week06 work
- Batch evaluation and comparison (v1 vs v2)
- Performance optimization
- Documentation updates
- Further enhancement to RAG pipeline
