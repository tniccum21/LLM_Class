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
            SLIDERS[Retrieval Sliders<br/>Initial K: 5-50<br/>Reranked N: 1-20]
            RETRIEVER[Retriever Type<br/>- Standard<br/>- MultiQuery<br/>- MMR]
            OPTIONS[RAG Mode<br/>- Strict<br/>- Augmented]
            SAFETY[Safety Controls<br/>- Content Safety<br/>- Prompt Injection]
            SUBMIT[Submit Button]
        end

        subgraph "Right Panel"
            ANSWER[Markdown Answer]
            METRICS[Performance Metrics<br/>Time & Confidence]
            REFS[Reference Tickets]
            VALIDATION[Safety Validation<br/>Pass/Fail Status]
        end

        SUBJECT --> UI
        BODY --> UI
        SLIDERS --> UI
        RETRIEVER --> UI
        OPTIONS --> UI
        SAFETY --> UI
        SUBMIT --> GUARDIAN
        GUARDIAN -->|Safe| PIPELINE
        GUARDIAN -->|Unsafe| VALIDATION
        PIPELINE --> ANSWER
        PIPELINE --> METRICS
        PIPELINE --> REFS
    end

    subgraph "Input Guardian (input_guardian.py)"
        GUARDIAN[Validation Orchestrator]

        PROMPT_GUARD[Prompt Guard 2 86M<br/>HuggingFace Local<br/>50-200ms]
        LLAMA_GUARD[Llama Guard 3 8B<br/>LM Studio API<br/>500-1500ms]

        GUARDIAN -->|Layer 1| PROMPT_GUARD
        GUARDIAN -->|Layer 2| LLAMA_GUARD
        PROMPT_GUARD -->|JAILBREAK| BLOCK[Block Submission]
        LLAMA_GUARD -->|S1-S13| BLOCK
        PROMPT_GUARD -->|BENIGN| SAFE[Continue to RAG]
        LLAMA_GUARD -->|safe| SAFE
    end

    subgraph "RAG Pipeline (rag_pipeline.py)"
        PIPELINE[Pipeline Orchestrator]

        EMBED[1. Embed Query<br/>all-MiniLM-L6-v2]

        SEARCH_SWITCH{Retriever Type?}
        SEARCH_STD[2a. Standard Search<br/>Cosine Similarity<br/>100-300ms]
        SEARCH_MQ[2b. MultiQuery Search<br/>LLM Expansion + Search<br/>2-5s]
        SEARCH_MMR[2c. MMR Search<br/>Diversity-Aware<br/>200-500ms]

        RERANK[3. Re-rank Results<br/>mxbai-rerank-base-v1<br/>top-5]
        CONTEXT[4. Build Context<br/>Format tickets]
        CONFIDENCE[5. Calculate Confidence<br/>Weighted scores]
        LLM[6. Generate Answer<br/>LangChain + LM Studio]

        PIPELINE --> EMBED
        EMBED --> SEARCH_SWITCH
        SEARCH_SWITCH -->|Standard| SEARCH_STD
        SEARCH_SWITCH -->|MultiQuery| SEARCH_MQ
        SEARCH_SWITCH -->|MMR| SEARCH_MMR
        SEARCH_STD --> RERANK
        SEARCH_MQ --> RERANK
        SEARCH_MMR --> RERANK
        RERANK --> CONTEXT
        CONTEXT --> CONFIDENCE
        CONFIDENCE --> LLM
    end

    subgraph "External Services"
        LMSTUDIO[LM Studio Server<br/>192.168.7.171:1234<br/>gpt-oss-20b<br/>Llama Guard 3 8B]
    end

    SEARCH_STD <-->|Query| CHROMA
    SEARCH_MQ <-->|Query| CHROMA
    SEARCH_MMR <-->|Query| CHROMA
    LLAMA_GUARD <-->|API Call| LMSTUDIO
    LLM <-->|API Call| LMSTUDIO

    style CSV fill:#e1f5ff
    style CHROMA fill:#ffe1e1
    style UI fill:#e1ffe1
    style GUARDIAN fill:#ffe6e6
    style PIPELINE fill:#fff4e1
    style LMSTUDIO fill:#f0e1ff
    style BLOCK fill:#ff9999
    style SAFE fill:#99ff99
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Guardian as Input Guardian
    participant PromptGuard as Prompt Guard 2
    participant LlamaGuard as Llama Guard 3
    participant Pipeline as RAG Pipeline
    participant ChromaDB as Vector Store
    participant Reranker as CrossEncoder
    participant LLM as LM Studio

    User->>UI: Enter subject & body
    User->>UI: Configure sliders (K, N)
    User->>UI: Select retriever type
    User->>UI: Select RAG mode
    User->>UI: Enable safety checks
    User->>UI: Click Submit

    alt Safety Checks Enabled
        UI->>Guardian: Validate ticket content

        alt Prompt Injection Check Enabled
            Note over Guardian,PromptGuard: Layer 1: Jailbreak Detection
            Guardian->>PromptGuard: Check for prompt injection (50-200ms)
            PromptGuard-->>Guardian: BENIGN or JAILBREAK

            alt JAILBREAK Detected
                Guardian-->>UI: Validation failed (jailbreak)
                UI->>User: Display error + violation details
            end
        end

        alt Content Safety Check Enabled
            Note over Guardian,LlamaGuard: Layer 2: Content Moderation
            Guardian->>LlamaGuard: Check 13 safety categories (500-1500ms)
            LlamaGuard-->>Guardian: safe or unsafe,S#,S#

            alt Unsafe Content Detected
                Guardian-->>UI: Validation failed (categories S1-S13)
                UI->>User: Display error + violation details
            end
        end

        Guardian-->>UI: Validation passed
    end

    UI->>Pipeline: Process ticket request

    Note over Pipeline: Step 1: Embed Query
    Pipeline->>Pipeline: Generate embedding (384-dim)

    Note over Pipeline: Step 2: Vector Search
    alt Standard Retriever
        Pipeline->>ChromaDB: Cosine similarity search (100-300ms)
    else MultiQuery Retriever
        Pipeline->>Pipeline: Generate 3 query variations via LLM
        Pipeline->>ChromaDB: Search with each variation (2-5s)
        Pipeline->>Pipeline: Deduplicate and combine results
    else MMR Retriever
        Pipeline->>ChromaDB: Diversity-aware search (200-500ms)
    end
    ChromaDB-->>Pipeline: Return top-K candidates

    Note over Pipeline: Step 3: Re-rank
    Pipeline->>Reranker: Score query-document pairs (500-1500ms)
    Reranker-->>Pipeline: Return top-N with scores

    Note over Pipeline: Step 4: Build Context
    Pipeline->>Pipeline: Format tickets for LLM

    Note over Pipeline: Step 5: Calculate Confidence
    Pipeline->>Pipeline: Weighted average of re-ranking scores

    Note over Pipeline: Step 6: Generate Answer
    alt Strict Mode
        Pipeline->>LLM: Context-only prompt (2-10s)
    else Augmented Mode
        Pipeline->>LLM: Context + general knowledge prompt (2-10s)
    end
    LLM-->>Pipeline: Generate response

    Pipeline-->>UI: Return answer + confidence + metrics + references

    UI->>User: Display markdown answer
    UI->>User: Show confidence score
    UI->>User: Show performance metrics
    UI->>User: Show reference tickets
    UI->>User: Show safety validation results
```

## Component Architecture

```mermaid
graph LR
    subgraph "Core Components"
        CONFIG[config.py<br/>AppConfig]
        GUARDIAN[input_guardian.py<br/>Safety Validation]
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

    subgraph "HuggingFace Models"
        EMB_MODEL[all-MiniLM-L6-v2<br/>Embedding Model<br/>384-dim]
        RERANK_MODEL[mxbai-rerank-base-v1<br/>Re-ranking Model<br/>278M params]
        PROMPT_GUARD_MODEL[Prompt-Guard-86M<br/>Jailbreak Detection<br/>86M params]
    end

    subgraph "LM Studio Models"
        LLM_MODEL[gpt-oss-20b<br/>Language Model<br/>20B params]
        LLAMA_GUARD_MODEL[Llama-Guard-3-8B<br/>Content Safety<br/>8B params]
    end

    CONFIG --> GUARDIAN
    CONFIG --> PIPELINE
    CONFIG --> APP
    CONFIG --> BUILD

    APP --> GUARDIAN
    APP --> PIPELINE

    GUARDIAN --> PROMPT_GUARD_MODEL
    GUARDIAN --> LC_LLM

    PIPELINE --> LC_EMB
    PIPELINE --> LC_VDB
    PIPELINE --> LC_LLM
    PIPELINE --> LC_PROMPT
    PIPELINE --> LC_CHAIN

    LC_EMB --> EMB_MODEL
    LC_CHAIN --> RERANK_MODEL
    LC_LLM --> LLM_MODEL
    LC_LLM --> LLAMA_GUARD_MODEL

    BUILD --> LC_VDB

    style CONFIG fill:#e1f5ff
    style GUARDIAN fill:#ffe6e6
    style PIPELINE fill:#fff4e1
    style APP fill:#e1ffe1
    style BUILD fill:#ffe1e1
```

## State Management

```mermaid
stateDiagram-v2
    [*] --> AppInit

    AppInit --> LoadingModels: First Run
    LoadingModels --> LoadingGuardian: Pipeline Loaded
    LoadingGuardian --> ModelsLoaded: Guardian Loaded (if enabled)
    LoadingGuardian --> ModelsLoaded: Skip (if disabled)
    LoadingModels --> Error: Failure

    ModelsLoaded --> Idle: Ready

    Idle --> SafetyCheck: User Submits
    SafetyCheck --> PromptGuardCheck: Check Enabled
    SafetyCheck --> Processing: Checks Disabled

    PromptGuardCheck --> Jailbreak: JAILBREAK Detected
    PromptGuardCheck --> ContentSafetyCheck: BENIGN

    ContentSafetyCheck --> Unsafe: S1-S13 Violation
    ContentSafetyCheck --> Processing: safe

    Jailbreak --> ValidationError: Show Error
    Unsafe --> ValidationError: Show Error
    ValidationError --> Idle: User Reviews

    Processing --> RetrieverSelection: Query Validated
    RetrieverSelection --> StandardSearch: Standard
    RetrieverSelection --> MultiQuerySearch: MultiQuery
    RetrieverSelection --> MMRSearch: MMR

    StandardSearch --> Reranking: Retrieved K
    MultiQuerySearch --> Reranking: Retrieved K
    MMRSearch --> Reranking: Retrieved K

    Reranking --> ContextBuilding: Top N
    ContextBuilding --> ConfidenceCalc: Context Ready
    ConfidenceCalc --> LLMGeneration: Confidence Scored

    LLMGeneration --> DisplayAnswer: Generated
    DisplayAnswer --> Idle: Complete

    Idle --> ConfigChange: User Changes Sliders/Options
    ConfigChange --> Idle: Applied

    Error --> AppInit: Retry
    DisplayAnswer --> Error: LLM Failure
    LLMGeneration --> Error: Timeout/Connection
```

## File Organization

```
Week07/
│
├── 📄 Documentation
│   ├── ARCHITECTURE.md                 # Architecture diagrams (this file)
│   ├── README.md                       # User documentation
│   └── Week07_Presentation.md          # Classroom presentation slides
│
├── 🐍 Core Application Files (Fully Documented)
│   ├── config.py                       # Configuration management
│   ├── input_guardian.py               # Dual-layer safety validation
│   │                                   # - Llama Guard 3 8B (content safety)
│   │                                   # - Prompt Guard 2 86M (jailbreak detection)
│   ├── rag_pipeline.py                 # RAG pipeline implementation
│   │                                   # - Three retriever strategies
│   │                                   # - Two-stage retrieval
│   │                                   # - LangChain LCEL integration
│   ├── ticket_support_app.py           # Streamlit UI application
│   │                                   # - Two-column layout
│   │                                   # - Configuration sliders
│   │                                   # - Safety controls
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
├── 📓 Notebooks (Reference - Week 6)
│   ├── ticket_rag_langchain_simple.ipynb
│   └── ticket_rag_system.ipynb
│
└── 🧪 Tests (Future)
    ├── test_config.py
    ├── test_input_guardian.py
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

#### Retrieval & Ranking Models (HuggingFace)
- **Embedding Model** - `sentence-transformers/all-MiniLM-L6-v2`
  - Purpose: Query and document embeddings
  - Dimensions: 384
  - Parameters: 22M
  - Size: ~80MB
  - Device: CPU
  - Speed: ~10-50ms per query

- **Re-ranking Model** - `mixedbread-ai/mxbai-rerank-base-v1`
  - Purpose: Cross-encoder relevance scoring
  - Parameters: 278M
  - Size: ~1GB
  - Device: CPU
  - Speed: ~50-100ms per document

#### Safety Models

- **Prompt Injection Detection** - `meta-llama/Prompt-Guard-86M`
  - Purpose: Jailbreak and prompt injection detection
  - Parameters: 86M
  - Size: ~350MB
  - Device: CPU (HuggingFace transformers)
  - Speed: ~50-200ms per check
  - Classes: BENIGN (0), JAILBREAK (1)

- **Content Safety** - `meta-llama/Llama-Guard-3-8B`
  - Purpose: 13-category content moderation
  - Parameters: 8B
  - Size: ~16GB
  - Device: GPU (via LM Studio)
  - Speed: ~500-1500ms per check
  - Categories: S1-S13 (violence, hate speech, etc.)

### LLM Integration
- **LM Studio** - Local LLM server
  - Primary Model: `gpt-oss-20b` (answer generation)
  - Safety Model: `Llama-Guard-3-8B` (content moderation)
  - API: OpenAI-compatible
  - Protocol: HTTP REST
  - Network: Local (192.168.7.171:1234)

### Data Processing
- **Pandas** 2.0+ - DataFrame operations
- **NumPy** (implicit) - Numerical computations
- **PyTorch** (implicit) - Model inference backend

## Performance Characteristics

### Resource Requirements

| Component | Memory | Disk | CPU/GPU | Notes |
|-----------|--------|------|---------|-------|
| Embedding Model | ~200MB | 80MB | CPU Low | all-MiniLM-L6-v2 |
| Re-ranker Model | ~1.2GB | 1GB | CPU Medium | mxbai-rerank-base-v1 |
| Prompt Guard Model | ~400MB | 350MB | CPU Low | 86M parameters |
| ChromaDB Index | ~500MB | 2GB | CPU Low | Persistent on disk |
| Streamlit App | ~200MB | - | CPU Low | Includes UI state |
| LM Studio (LLM) | ~12GB | 20GB | GPU High | gpt-oss-20b |
| LM Studio (Guard) | ~8GB | 16GB | GPU High | Llama-Guard-3-8B |
| **Total (Local)** | **~2.5GB** | **3.5GB** | **Medium** | Without LM Studio |
| **Total (Full)** | **~22.5GB** | **39.5GB** | **GPU Required** | With LM Studio |

### Response Time Breakdown

#### Without Safety Checks (Baseline)
```
Total Response Time: 3-12 seconds
├── Embedding:        0.05-0.1s   (1-2%)
├── Vector Search:    0.1-0.5s    (2-8%)
│   ├── Standard:     0.1-0.3s
│   ├── MultiQuery:   2.0-5.0s    (includes LLM expansion)
│   └── MMR:          0.2-0.5s
├── Re-ranking:       0.5-1.5s    (10-25%)
├── Context Build:    0.05-0.1s   (1-2%)
├── Confidence Calc:  0.01-0.05s  (<1%)
└── LLM Generation:   2.0-10.0s   (40-80%)
```

#### With Safety Checks (Full Protection)
```
Total Response Time: 4-14 seconds
├── Prompt Guard:     0.05-0.2s   (1-3%)
├── Content Safety:   0.5-1.5s    (10-20%)
├── [RAG Pipeline]:   3-12s       (70-85%)
└── Overhead:         ~1-2s       (safety validation)
```

### Performance by Configuration

| Configuration | Time Range | Best For |
|---------------|------------|----------|
| Standard, No Safety | 3-5s | Development, testing |
| Standard, Full Safety | 4-7s | **Production default** |
| MultiQuery, Full Safety | 6-12s | Complex queries |
| MMR, Full Safety | 5-8s | Diverse results needed |
| Standard, Content Only | 3.5-6.5s | Trusted internal users |
| Standard, Prompt Only | 3.1-5.2s | Public-facing, speed priority |

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
