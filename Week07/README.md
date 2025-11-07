# Week07 - Intelligent Support Ticket Assistant

A production-ready Streamlit application that provides AI-powered support ticket resolution using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and two-stage retrieval.

## 🎯 Features

- **Intelligent Ticket Resolution** - AI-powered answers based on 3,600+ historical tickets
- **Two-Stage Retrieval** - Vector similarity search + cross-encoder re-ranking
- **Dual RAG Modes**
  - 🔒 **Strict Mode** - Answers only from historical ticket context
  - 🔓 **Augmented Mode** - Supplements with LLM general knowledge
- **Real-Time Performance Metrics** - Response time, confidence scores, ticket references
- **Clean Two-Column UI** - Intuitive input/output workflow
- **Extensible Configuration** - Easy to add new options and features

## 🏗️ Architecture

```
User Input → Embed Query → Vector Search (top-20)
                ↓
            Re-rank (top-5) → Build Context → LLM Generation
                ↓
         Display Answer + Metrics + References
```

**Key Components**:
- **ChromaDB** - Vector database for semantic search
- **HuggingFace Embeddings** - `all-MiniLM-L6-v2` (384-dim)
- **CrossEncoder Re-ranker** - `mxbai-rerank-base-v1` (278M params)
- **LM Studio** - Local LLM server (`gpt-oss-20b`)
- **LangChain** - RAG orchestration framework

## 📋 Prerequisites

- **Python** 3.11+
- **LM Studio** running on local network
  - Model: `gpt-oss-20b` or compatible
  - URL: `http://192.168.7.171:1234` (configurable)
- **~2GB RAM** for models (embedding + re-ranker)
- **~3GB disk space** for ChromaDB index

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Week07
pip install -r requirements.txt
```

### 2. Configure Environment

Create `secrets.env` file:

```bash
# LM Studio Configuration
LM_STUDIO_URL=http://192.168.7.171:1234
LM_STUDIO_MODEL=gpt-oss-20b

# Optional: API Keys for future features
OPENAI_API_KEY=your-key-here
HUGGINGFACE_TOKEN=your-token-here
```

### 3. Build Vector Database (One-Time Setup)

```bash
python build_vectordb.py
```

**Expected Output**:
```
Loading dataset: dataset-tickets-multi-lang3-4k-translated-all.csv
Loaded 3,601 tickets

Creating embeddings with all-MiniLM-L6-v2...
Progress: 100% [████████████████████] 3601/3601

Building ChromaDB collection...
✓ Collection 'support_tickets' created
✓ 3,601 documents indexed
✓ Database saved to ./chroma_ticket_db

Validation Report:
- Total documents: 3,601
- Embedding dimension: 384
- Disk size: 2.1 GB
- Build time: 45.3 seconds

Vector database ready for use!
```

### 4. Launch Streamlit App

```bash
streamlit run ticket_support_app.py
```

**Application URL**: http://localhost:8501

## 📖 Usage Guide

### Basic Workflow

1. **Enter Ticket Information**
   - **Subject**: Brief description (e.g., "Cannot access shared drive")
   - **Body**: Detailed explanation with context

2. **Configure Options**
   - **RAG Mode**:
     - 🔒 **Strict** - Answer only from historical tickets
     - 🔓 **Augmented** - Supplement with LLM knowledge
   - **Show References** - Display source tickets used

3. **Submit & Review**
   - Click **Submit** button
   - View AI-generated solution (markdown formatted)
   - Check performance metrics and confidence
   - Review reference tickets (optional)

### Example Tickets

**Example 1: VPN Connection Issue**
```
Subject: VPN connection keeps dropping
Body: I'm working from home and my VPN connection drops every 15-20
minutes. I'm using Cisco AnyConnect on Windows 10. Already tried
restarting my computer and router.
```

**Example 2: Email Configuration**
```
Subject: Unable to send emails from Outlook
Body: My Outlook can receive emails but cannot send. Getting error
"550 5.7.1 Relaying denied". Using Office 365 with SMTP
authentication.
```

**Example 3: Software Installation**
```
Subject: Need admin rights to install Adobe Acrobat
Body: I need to install Adobe Acrobat Pro for my project but don't
have admin permissions. How can I get this software installed?
```

## ⚙️ Configuration Options

### Current Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `rag_mode` | str | Strict or Augmented mode | `'strict'` |
| `show_references` | bool | Display reference tickets | `True` |
| `show_metrics` | bool | Display performance stats | `True` |

### Advanced Configuration

Edit `config.py` to customize:

```python
class AppConfig:
    # Vector Database
    chroma_db_path = './chroma_ticket_db'
    collection_name = 'support_tickets'

    # Embedding
    embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dimension = 384

    # Retrieval
    top_k_initial = 20      # Vector search results
    top_k_reranked = 5      # Re-ranked results
    reranker_model = 'mixedbread-ai/mxbai-rerank-base-v1'

    # LLM
    lm_studio_url = 'http://192.168.7.171:1234'
    llm_model = 'gpt-oss-20b'
    max_tokens = 6000
    temperature = 0.1

    # Future options can be added to extra_options dict
    extra_options = {}
```

## 📊 Performance

### Response Time Breakdown

```
Total: 2-5 seconds
├── Embedding:      0.05-0.1s   (2-4%)
├── Vector Search:  0.2-0.5s    (8-20%)
├── Re-ranking:     0.3-0.6s    (12-24%)
├── Context Build:  0.05-0.1s   (2-4%)
└── LLM Generation: 1.0-4.0s    (40-80%)
```

### Resource Usage

- **Memory**: ~2GB (models + ChromaDB)
- **Disk**: ~3GB (vector database)
- **CPU**: Medium (re-ranking is compute-intensive)

## 🔧 Troubleshooting

### ChromaDB Not Found

**Error**: `Error: ChromaDB not found at ./chroma_ticket_db`

**Solution**: Run `python build_vectordb.py` first

### LM Studio Connection Failed

**Error**: `ConnectionError: Failed to connect to LM Studio`

**Solutions**:
1. Verify LM Studio is running
2. Check URL in `secrets.env` matches your LM Studio server
3. Test connection: `curl http://192.168.7.171:1234/v1/models`

### Out of Memory Error

**Error**: `RuntimeError: CUDA out of memory` or similar

**Solutions**:
1. Close other applications
2. Reduce `top_k_initial` and `top_k_reranked` in config
3. Use lighter re-ranking model (or disable re-ranking)

### Slow Response Times

**Symptoms**: Response takes >10 seconds

**Solutions**:
1. Check LM Studio model is fully loaded
2. Reduce `max_tokens` in config
3. Use smaller `top_k_reranked` value (e.g., 3 instead of 5)
4. Verify network connection to LM Studio

## 📁 Project Structure

```
Week07/
├── README.md                           # This file
├── DESIGN.md                           # Design specification
├── ARCHITECTURE.md                     # Architecture diagrams
│
├── config.py                           # Configuration management
├── rag_pipeline.py                     # RAG pipeline implementation
├── ticket_support_app.py               # Main Streamlit application
├── build_vectordb.py                   # Vector DB builder
│
├── requirements.txt                    # Python dependencies
├── secrets.env                         # API keys (gitignored)
│
├── dataset-tickets-*.csv               # Training datasets
└── chroma_ticket_db/                   # Vector database (gitignored)
```

## 🧪 Testing

### Manual Testing Checklist

- [ ] **Valid Input**: Submit ticket → receives answer
- [ ] **Empty Input**: Submit empty → validation error
- [ ] **Mode Toggle**: Switch strict/augmented → different responses
- [ ] **References**: Toggle show/hide → UI updates correctly
- [ ] **Long Input**: Very long ticket → handles gracefully
- [ ] **Special Characters**: Ticket with symbols → processes correctly

### Unit Testing (Future)

```bash
pytest tests/ -v
```

## 🔒 Security

### Data Protection

- ✅ API keys stored in `.env` (gitignored)
- ✅ No external API calls (LM Studio local)
- ✅ No persistent user data storage
- ✅ Input validation and sanitization
- ⚠️ No authentication (single-user design)
- ⚠️ No rate limiting (local use only)

### Production Deployment Considerations

For production deployment, implement:
- User authentication and authorization
- HTTPS/TLS encryption
- Rate limiting per user
- Input content filtering
- Audit logging
- Data retention policies

## 🚧 Known Limitations

1. **Single User** - Not designed for concurrent users
2. **No Query History** - Sessions are stateless
3. **No Feedback Loop** - No mechanism to rate answers
4. **English Only** - LLM responses in English (dataset is multilingual)
5. **Local Only** - Requires local network access to LM Studio

## 🛣️ Roadmap

### Version 1.1 (Next Release)
- [ ] Query history and analytics
- [ ] User feedback collection (thumbs up/down)
- [ ] Export answers (PDF, Markdown)
- [ ] Multi-language support

### Version 2.0 (Future)
- [ ] Multi-user support with authentication
- [ ] API endpoint for programmatic access
- [ ] A/B testing framework (strict vs augmented)
- [ ] Custom re-ranker model training
- [ ] Docker containerization

## 📚 Additional Resources

### Documentation
- [DESIGN.md](DESIGN.md) - Complete design specification
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture diagrams
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Docs](https://docs.trychroma.com/)

### Related Notebooks
- `ticket_rag_langchain_simple.ipynb` - LangChain RAG implementation
- `ticket_rag_system.ipynb` - Advanced RAG with re-ranking

### Week Progression
- **Week05** - Basic RAG with ChromaDB
- **Week06** - Enhanced RAG with re-ranking
- **Week07** - Production Streamlit application (this week)

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings for all public functions
- Keep functions under 50 lines

### Adding New Features

1. Update `config.py` with new options
2. Implement in `rag_pipeline.py` or `ticket_support_app.py`
3. Add tests in `tests/` directory
4. Update documentation (README, DESIGN, ARCHITECTURE)
5. Test end-to-end workflow

### Example: Adding Temperature Control

```python
# 1. config.py
class AppConfig:
    temperature: float = 0.1  # Add new option

# 2. ticket_support_app.py
temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.config.temperature,
    help="Higher = more creative, Lower = more focused"
)

# 3. Update LLM with new temperature
st.session_state.llm.temperature = temperature
```

## 📝 License

This project is part of the "Apps with AI" educational repository.

## 🙏 Acknowledgments

- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **Streamlit** - Web application framework
- **HuggingFace** - Embedding models
- **Mixedbread AI** - Re-ranking model
- **LM Studio** - Local LLM server

---

## 📧 Support

For questions or issues:
1. Check [DESIGN.md](DESIGN.md) for architectural details
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system diagrams
3. Check existing notebooks for reference implementations

---

**Built with ❤️ using LangChain, ChromaDB, and Streamlit**

*Week07 - Apps with AI Course Project*
