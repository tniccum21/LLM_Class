# Week07 RAG System - Technical Architecture

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Preparation (One-Time)"
        CSV[Dataset CSV<br/>3,601 tickets]
        BUILD[build_vectordb.py]
        CSV -->|Load 100%| BUILD
        BUILD -->|Embed & Index| CHROMA
    end

    subgraph "Vector Store"
        CHROMA[(ChromaDB<br/>chroma_ticket_db/)]
    end

    subgraph "Streamlit Application"
        UI[User Interface]

        subgraph "Left Panel"
            SUBJECT[Subject Input]
            BODY[Body Input]
            OPTIONS[Options<br/>☐ Strict Mode<br/>☐ Show Refs]
            SUBMIT[Submit Button]
        end

        subgraph "Right Panel"
            ANSWER[Markdown Answer]
            METRICS[Performance Metrics<br/>⏱️ Time 📊 Confidence]
            REFS[Reference Tickets]
        end

        SUBJECT --> UI
        BODY --> UI
        OPTIONS --> UI
        SUBMIT --> PIPELINE
        PIPELINE --> ANSWER
        PIPELINE --> METRICS
        PIPELINE --> REFS
    end

    subgraph "RAG Pipeline (rag_pipeline.py)"
        PIPELINE[Pipeline Orchestrator]

        EMBED[1. Embed Query<br/>HuggingFace]
        SEARCH[2. Vector Search<br/>ChromaDB top-20]
        RERANK[3. Re-rank Results<br/>CrossEncoder top-5]
        CONTEXT[4. Build Context<br/>Format tickets]
        LLM[5. Generate Answer<br/>LangChain + LM Studio]

        PIPELINE --> EMBED
        EMBED --> SEARCH
        SEARCH --> RERANK
        RERANK --> CONTEXT
        CONTEXT --> LLM
    end

    subgraph "External Services"
        LMSTUDIO[LM Studio<br/>192.168.7.171:1234<br/>gpt-oss-20b]
    end

    SEARCH <-->|Query| CHROMA
    LLM <-->|API Call| LMSTUDIO

    style CSV fill:#e1f5ff
    style CHROMA fill:#ffe1e1
    style UI fill:#e1ffe1
    style PIPELINE fill:#fff4e1
    style LMSTUDIO fill:#f0e1ff
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Pipeline as RAG Pipeline
    participant ChromaDB as Vector Store
    participant Reranker as CrossEncoder
    participant LLM as LM Studio

    User->>UI: Enter subject & body
    User->>UI: Select options (strict/augmented)
    User->>UI: Click Submit

    UI->>Pipeline: Process ticket request

    Note over Pipeline: Step 1: Embed Query
    Pipeline->>Pipeline: Generate embedding (384-dim)

    Note over Pipeline: Step 2: Vector Search
    Pipeline->>ChromaDB: Query with embedding
    ChromaDB-->>Pipeline: Return top-20 candidates

    Note over Pipeline: Step 3: Re-rank
    Pipeline->>Reranker: Score query-document pairs
    Reranker-->>Pipeline: Return top-5 with scores

    Note over Pipeline: Step 4: Build Context
    Pipeline->>Pipeline: Format tickets for LLM

    Note over Pipeline: Step 5: Generate Answer
    Pipeline->>LLM: Send prompt + context
    LLM-->>Pipeline: Generate response

    Pipeline-->>UI: Return answer + metrics

    UI->>User: Display markdown answer
    UI->>User: Show performance metrics
    UI->>User: Show reference tickets (optional)
```

## Component Architecture

```mermaid
graph LR
    subgraph "Core Components"
        CONFIG[config.py<br/>AppConfig]
        PIPELINE[rag_pipeline.py<br/>RAG Functions]
        APP[ticket_support_app.py<br/>Streamlit UI]
        BUILD[build_vectordb.py<br/>DB Builder]
    end

    subgraph "LangChain Integration"
        LC_EMB[HuggingFaceEmbeddings]
        LC_VDB[Chroma]
        LC_LLM[ChatOpenAI]
        LC_PROMPT[ChatPromptTemplate]
        LC_CHAIN[LCEL Chain]
    end

    subgraph "External Models"
        EMB_MODEL[all-MiniLM-L6-v2<br/>Embedding Model]
        RERANK_MODEL[mxbai-rerank-base-v1<br/>Re-ranking Model]
        LLM_MODEL[gpt-oss-20b<br/>Language Model]
    end

    CONFIG --> PIPELINE
    CONFIG --> APP
    CONFIG --> BUILD

    PIPELINE --> LC_EMB
    PIPELINE --> LC_VDB
    PIPELINE --> LC_LLM
    PIPELINE --> LC_PROMPT
    PIPELINE --> LC_CHAIN

    LC_EMB --> EMB_MODEL
    LC_CHAIN --> RERANK_MODEL
    LC_LLM --> LLM_MODEL

    APP --> PIPELINE
    BUILD --> LC_VDB

    style CONFIG fill:#e1f5ff
    style PIPELINE fill:#fff4e1
    style APP fill:#e1ffe1
    style BUILD fill:#ffe1e1
```

## State Management

```mermaid
stateDiagram-v2
    [*] --> AppInit

    AppInit --> LoadingModels: First Run
    LoadingModels --> ModelsLoaded: Success
    LoadingModels --> Error: Failure

    ModelsLoaded --> Idle: Ready

    Idle --> Processing: User Submits
    Processing --> Embedding: Query
    Embedding --> VectorSearch: Embedded
    VectorSearch --> Reranking: Retrieved 20
    Reranking --> ContextBuilding: Top 5
    ContextBuilding --> LLMGeneration: Context Ready
    LLMGeneration --> DisplayAnswer: Generated
    DisplayAnswer --> Idle: Complete

    Idle --> ConfigChange: User Changes Options
    ConfigChange --> Idle: Applied

    Error --> AppInit: Retry
    DisplayAnswer --> Error: LLM Failure
```

## File Organization

```
Week07/
│
├── 📄 DESIGN.md                        # Design specification (this file)
├── 📄 ARCHITECTURE.md                  # Architecture diagrams
├── 📄 README.md                        # User documentation
│
├── 🐍 Core Application Files
│   ├── config.py                       # Configuration management
│   ├── rag_pipeline.py                 # RAG pipeline implementation
│   ├── ticket_support_app.py           # Streamlit UI application
│   └── build_vectordb.py               # Vector DB builder script
│
├── 📦 Dependencies
│   ├── requirements.txt                # Python packages
│   └── secrets.env                     # API keys (gitignored)
│
├── 💾 Data Files
│   ├── dataset-tickets-multi-lang3-4k-translated-all.csv
│   ├── dataset-tickets-multi-lang3-4k.csv
│   └── chroma_ticket_db/               # Vector database (gitignored)
│
├── 📓 Notebooks (Reference)
│   ├── ticket_rag_langchain_simple.ipynb
│   └── ticket_rag_system.ipynb
│
└── 🧪 Tests (Future)
    ├── test_config.py
    ├── test_rag_pipeline.py
    └── test_app.py
```

## Technology Stack

### Core Framework
- **Streamlit** 1.30+ - Web application framework
- **LangChain** 0.1+ - RAG orchestration framework
- **Python** 3.11+ - Runtime environment

### Vector Database
- **ChromaDB** 0.4.22+ - Vector storage and similarity search
- **SQLite** (embedded) - ChromaDB backend

### ML Models
- **HuggingFace Transformers** - Embedding model hosting
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimensions: 384
  - Size: ~80MB

- **CrossEncoder** - Re-ranking model
  - Model: `mixedbread-ai/mxbai-rerank-base-v1`
  - Parameters: 278M
  - Size: ~1GB

### LLM Integration
- **LM Studio** - Local LLM server
  - Model: `gpt-oss-20b`
  - API: OpenAI-compatible
  - Protocol: HTTP REST

### Data Processing
- **Pandas** 2.0+ - DataFrame operations
- **NumPy** (implicit) - Numerical computations

## Performance Characteristics

### Resource Requirements

| Component | Memory | Disk | CPU |
|-----------|--------|------|-----|
| Embedding Model | ~200MB | 80MB | Low |
| Re-ranker Model | ~1.2GB | 1GB | Medium |
| ChromaDB Index | ~500MB | 2GB | Low |
| Streamlit App | ~100MB | - | Low |
| **Total** | **~2GB** | **3GB** | **Medium** |

### Response Time Breakdown

```
Total Response Time: 2-5 seconds
├── Embedding:        0.05-0.1s  (2-4%)
├── Vector Search:    0.2-0.5s   (8-20%)
├── Re-ranking:       0.3-0.6s   (12-24%)
├── Context Build:    0.05-0.1s  (2-4%)
└── LLM Generation:   1.0-4.0s   (40-80%)
```

### Scalability Considerations

**Current Design** (Single User):
- ✅ Fast response times (2-5s)
- ✅ Low memory footprint (~2GB)
- ✅ No external API dependencies

**Future Scaling** (Multi-User):
- Use Redis for ChromaDB query caching
- Implement request queuing for LLM calls
- Add horizontal scaling with multiple workers
- Consider GPU acceleration for embeddings
- Implement connection pooling for LM Studio

## Security Architecture

### Data Protection
```mermaid
graph TD
    ENV[secrets.env<br/>API Keys]
    GITIGNORE[.gitignore]
    CODE[Application Code]
    CHROMA[ChromaDB]
    LMSTUDIO[LM Studio<br/>Local Network]

    ENV -.->|Excluded| GITIGNORE
    CHROMA -.->|Excluded| GITIGNORE
    ENV -->|Load at Runtime| CODE
    CODE -->|Local Connection| LMSTUDIO

    style ENV fill:#ffe1e1
    style GITIGNORE fill:#e1ffe1
    style LMSTUDIO fill:#e1f5ff
```

### Security Layers

1. **Environment Variables**
   - API keys in `.env` file
   - Gitignored to prevent commits
   - Loaded via `python-dotenv`

2. **Network Security**
   - LM Studio on local network (192.168.x.x)
   - No external API calls
   - No sensitive data transmission

3. **Input Validation**
   - Streamlit form validation
   - Query sanitization before embedding
   - Max length limits on inputs

4. **Data Privacy**
   - No user data persistence
   - Session state only (in-memory)
   - No logging of ticket content

## Error Handling Strategy

```mermaid
graph TD
    START[User Request]
    VALIDATE{Valid Input?}
    EMBED[Embed Query]
    SEARCH[Vector Search]
    RERANK[Re-rank Results]
    LLM[Generate Answer]
    DISPLAY[Display Response]

    ERROR_INPUT[Show Input Error]
    ERROR_EMBED[Use Fallback Search]
    ERROR_SEARCH[Show Search Error]
    ERROR_RERANK[Skip Re-ranking]
    ERROR_LLM[Show Generation Error]

    START --> VALIDATE
    VALIDATE -->|No| ERROR_INPUT
    VALIDATE -->|Yes| EMBED

    EMBED -->|Success| SEARCH
    EMBED -->|Failure| ERROR_EMBED

    SEARCH -->|Success| RERANK
    SEARCH -->|Failure| ERROR_SEARCH

    RERANK -->|Success| LLM
    RERANK -->|Failure| ERROR_RERANK
    ERROR_RERANK --> LLM

    LLM -->|Success| DISPLAY
    LLM -->|Failure| ERROR_LLM

    style ERROR_INPUT fill:#ffe1e1
    style ERROR_EMBED fill:#ffe1e1
    style ERROR_SEARCH fill:#ffe1e1
    style ERROR_RERANK fill:#fff4e1
    style ERROR_LLM fill:#ffe1e1
```

## Configuration System Design

```python
# config.py - Hierarchical configuration with inheritance

class BaseConfig:
    """Base configuration shared across all environments"""
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dimension: int = 384
    collection_name: str = 'support_tickets'

class DevelopmentConfig(BaseConfig):
    """Development environment overrides"""
    lm_studio_url: str = 'http://192.168.7.171:1234'
    debug: bool = True
    show_metrics: bool = True

class ProductionConfig(BaseConfig):
    """Production environment overrides"""
    lm_studio_url: str = os.getenv('LM_STUDIO_URL')
    debug: bool = False
    show_metrics: bool = False
    enable_caching: bool = True

# Auto-select based on environment
config = DevelopmentConfig() if os.getenv('ENV') == 'dev' else ProductionConfig()
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Local Development"
        DEV_APP[Streamlit App<br/>localhost:8501]
        DEV_LM[LM Studio<br/>192.168.7.171:1234]
        DEV_DB[ChromaDB<br/>./chroma_ticket_db]

        DEV_APP --> DEV_DB
        DEV_APP --> DEV_LM
    end

    subgraph "Production (Future)"
        NGINX[NGINX<br/>Reverse Proxy]
        STREAM1[Streamlit Worker 1]
        STREAM2[Streamlit Worker 2]
        REDIS[(Redis<br/>Cache)]
        PROD_LM[LM Studio Cluster<br/>Load Balanced]
        PROD_DB[(ChromaDB<br/>Persistent Volume)]

        NGINX --> STREAM1
        NGINX --> STREAM2
        STREAM1 --> REDIS
        STREAM2 --> REDIS
        STREAM1 --> PROD_DB
        STREAM2 --> PROD_DB
        STREAM1 --> PROD_LM
        STREAM2 --> PROD_LM
    end

    style DEV_APP fill:#e1ffe1
    style STREAM1 fill:#e1ffe1
    style STREAM2 fill:#e1ffe1
```

---

## Implementation Priorities

### Phase 1: Foundation (Week 1)
1. **config.py** - Configuration management
2. **build_vectordb.py** - Database setup
3. **rag_pipeline.py** - Core pipeline extraction

### Phase 2: Application (Week 1-2)
4. **ticket_support_app.py** - Streamlit UI
5. **requirements.txt** - Dependencies
6. **Testing** - End-to-end validation

### Phase 3: Polish (Week 2)
7. **Error handling** - Graceful degradation
8. **Documentation** - README and guides
9. **Performance optimization** - Caching and tuning

---

## Design Principles Applied

✅ **Separation of Concerns** - DB builder separate from app
✅ **Single Responsibility** - Each file has clear purpose
✅ **DRY (Don't Repeat Yourself)** - Shared config system
✅ **KISS (Keep It Simple)** - Straightforward architecture
✅ **Extensibility** - Easy to add options and features
✅ **Testability** - Clear interfaces for unit testing
✅ **Performance** - Model loading in session state
✅ **Security** - Proper secret management

---

## Technical Debt & Future Enhancements

### Known Limitations
- Single-user design (no concurrency)
- No persistent user sessions
- No query history/analytics
- Limited error recovery options

### Planned Enhancements
- Multi-user support with authentication
- Query history and analytics dashboard
- A/B testing between RAG modes
- Custom re-ranker model training
- Feedback collection system
- API endpoint for programmatic access
- Docker containerization
- CI/CD pipeline integration
