"""
Week07 - RAG Pipeline Implementation

This module implements the core RAG (Retrieval-Augmented Generation) pipeline
for intelligent support ticket resolution. The pipeline uses historical support
tickets to provide contextually relevant solutions to new support requests.

ARCHITECTURE OVERVIEW:
---------------------
The pipeline implements a sophisticated three-stage retrieval and generation system:

1. VECTOR SIMILARITY SEARCH (Stage 1 Retrieval)
   - Technology: ChromaDB vector database
   - Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
   - Purpose: Rapidly identify potentially relevant historical tickets
   - Retrieval Strategies:
     * Standard: Traditional cosine similarity search
     * MultiQuery: Generate query variations for broader coverage
     * MMR: Maximal Marginal Relevance for diversity-aware retrieval
   - Output: top-K initial candidates (default K=20)

2. CROSS-ENCODER RE-RANKING (Stage 2 Retrieval)
   - Technology: mixedbread-ai/mxbai-rerank-base-v1
   - Purpose: Deep semantic understanding of query-document relevance
   - Method: Computes pairwise relevance scores between query and each candidate
   - Advantage: More accurate than vector search but computationally expensive
   - Output: top-N re-ranked results (default N=5)

3. LLM ANSWER GENERATION (Generation)
   - Technology: LangChain LCEL with local LLM (via LM Studio)
   - Input: User ticket + formatted context from re-ranked historical tickets
   - RAG Modes:
     * Strict Mode: Answer ONLY from historical ticket context
     * Augmented Mode: Combine context with LLM's general knowledge
   - Output: Detailed, actionable solution with confidence metrics

RETRIEVAL STRATEGIES:
--------------------
1. Standard Retrieval:
   - Simple cosine similarity between query and document embeddings
   - Fast and efficient for most use cases
   - Returns top-K most similar documents

2. MultiQuery Retrieval:
   - Generates multiple query variations using LLM
   - Searches with each variation independently
   - Deduplicates and combines results for broader coverage
   - Best for: Complex queries with multiple interpretations

3. MMR (Maximal Marginal Relevance):
   - Balances relevance with diversity
   - Prevents redundant results by penalizing similarity to already-selected docs
   - Lambda parameter controls relevance vs. diversity trade-off
   - Best for: Comprehensive coverage of a topic

RAG MODES:
----------
1. Strict Mode:
   - LLM constrained to ONLY use information from retrieved tickets
   - Prompts explicitly forbid generating new solutions
   - If no matching ticket found, requests more information
   - Best for: Ensuring consistency with historical resolutions

2. Augmented Mode:
   - LLM can supplement historical context with general knowledge
   - Prioritizes retrieved context but fills gaps with expertise
   - Provides more comprehensive solutions
   - Best for: Novel issues requiring general IT knowledge

PERFORMANCE CHARACTERISTICS:
---------------------------
- Vector Search: ~0.1-0.5s for top-20 retrieval
- Re-ranking: ~0.5-1.5s for 20 documents
- LLM Generation: ~2-10s depending on model and response length
- Total Pipeline: ~3-12s end-to-end

DATA FLOW:
----------
User Input (Subject + Body)
    |
    v
[Combine into Query Text]
    |
    v
[Generate Query Embeddings] <-- Embedding Model
    |
    v
[Vector Similarity Search] <-- ChromaDB
    |
    v
[Initial Candidates (top-20)]
    |
    v
[Cross-Encoder Re-ranking] <-- Re-ranker Model
    |
    v
[Top-N Results (top-5)]
    |
    v
[Build Context String] --> [Format with metadata]
    |
    v
[LLM Generation] <-- LangChain + LM Studio
    |
    v
[Final Answer + Metrics + References]

KEY FEATURES:
-------------
- Lazy Initialization: Models loaded on first use to save resources
- Flexible Configuration: All parameters configurable via AppConfig
- Performance Metrics: Detailed timing and confidence tracking
- Multiple Retriever Types: Standard, MultiQuery, MMR
- Dual RAG Modes: Strict vs Augmented response generation
- Context Formatting: Rich metadata in LLM prompts
- Confidence Scoring: Weighted average of re-ranking scores

INTEGRATION:
------------
This module is designed to be used by the Streamlit application
(ticket_support_app.py) and can be tested independently via the
__main__ block at the bottom of this file.

Usage Example:
    from config import get_config
    from rag_pipeline import RAGPipeline

    # Initialize pipeline
    config = get_config()
    pipeline = RAGPipeline(config)
    pipeline.initialize()

    # Process a ticket
    response = pipeline.process_ticket(
        subject="VPN connection issue",
        body="My VPN keeps disconnecting every 15 minutes."
    )

    # Access results
    print(response['answer'])
    print(f"Confidence: {response['confidence']:.1%}")
    print(f"References: {len(response['references'])} tickets")

DEPENDENCIES:
-------------
- langchain: LLM orchestration framework
- langchain-openai: OpenAI-compatible API client (works with LM Studio)
- langchain-community: HuggingFace embeddings integration
- langchain-chroma: ChromaDB vector store integration
- sentence-transformers: Embedding and re-ranking models
- chromadb: Vector database for ticket storage

CONFIGURATION:
--------------
All configuration managed via config.py AppConfig class:
- embedding_model: HuggingFace model for embeddings
- reranker_model: Cross-encoder model for re-ranking
- top_k_initial: Number of candidates from vector search
- top_k_reranked: Number of results after re-ranking
- lm_studio_url: Local LLM server endpoint
- rag_mode: 'strict' or 'augmented'
- retriever_type: 'standard', 'multiquery', or 'mmr'

CLASSROOM WALKTHROUGH:
---------------------
1. Start with RAGPipeline.__init__() to understand initialization
2. Study initialize() to see model loading sequence
3. Examine process_ticket() for complete pipeline flow
4. Deep dive into vector_search() for retrieval strategies
5. Analyze rerank_results() for cross-encoder scoring
6. Review build_context() for prompt formatting
7. Study generate_answer() for LLM integration
8. Examine PerformanceMetrics for monitoring
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import AppConfig


@dataclass
class PerformanceMetrics:
    """
    Performance metrics tracking for RAG pipeline execution.

    This dataclass captures detailed timing information and quality metrics
    for each stage of the RAG pipeline. Used for monitoring, optimization,
    and displaying performance information to users.

    Attributes:
        total_time: End-to-end pipeline execution time in seconds
                   Includes all stages from query input to final answer

        embedding_time: Time spent generating query embeddings in seconds
                       Currently included in search_time, kept for future use

        search_time: Time spent on vector similarity search in seconds
                    Includes query embedding and ChromaDB retrieval

        rerank_time: Time spent on cross-encoder re-ranking in seconds
                    Most computationally expensive stage for large candidate sets

        llm_time: Time spent on LLM answer generation in seconds
                 Varies based on model size and response length

        confidence_score: Overall confidence in the response (0.0 to 1.0)
                         Calculated from weighted average of re-ranking scores

        num_retrieved: Number of initial candidates from vector search
                      Typically 20 (config.top_k_initial)

        num_reranked: Number of results after re-ranking
                     Typically 5 (config.top_k_reranked)

        timestamp: When this pipeline execution occurred
                  Used for logging and analytics

        rag_mode: Which RAG mode was used ('strict' or 'augmented')
                 Affects how LLM generates answers

    Performance Expectations:
        - search_time: 0.1-0.5s (fast vector similarity)
        - rerank_time: 0.5-1.5s (depends on candidate count)
        - llm_time: 2-10s (depends on model and response length)
        - total_time: 3-12s (sum of all stages)

    Usage:
        metrics = PerformanceMetrics(
            total_time=5.23,
            embedding_time=0.0,
            search_time=0.35,
            rerank_time=0.89,
            llm_time=3.99,
            confidence_score=0.87,
            num_retrieved=20,
            num_reranked=5,
            timestamp=datetime.now(),
            rag_mode='strict'
        )

        # Display in UI
        display_dict = metrics.to_dict()
        for key, value in display_dict.items():
            print(f"{key}: {value}")
    """

    total_time: float
    embedding_time: float
    search_time: float
    rerank_time: float
    llm_time: float
    confidence_score: float
    num_retrieved: int
    num_reranked: int
    timestamp: datetime
    rag_mode: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary format for UI display.

        Transforms raw metric values into human-readable strings with
        appropriate formatting (time in seconds, percentage for confidence).
        Used by Streamlit UI to display performance information.

        Returns:
            Dict[str, Any]: Dictionary with user-friendly metric labels and formatted values

        Example Output:
            {
                'Total Response Time': '5.23s',
                'Vector Search': '0.35s',
                'Re-ranking': '0.89s',
                'LLM Generation': '3.99s',
                'Confidence': '87.0%',
                'Tickets Retrieved': 20,
                'Mode': 'Strict'
            }
        """
        return {
            'Total Response Time': f"{self.total_time:.2f}s",
            'Vector Search': f"{self.search_time:.2f}s",
            'Re-ranking': f"{self.rerank_time:.2f}s",
            'LLM Generation': f"{self.llm_time:.2f}s",
            'Confidence': f"{self.confidence_score:.1%}",
            'Tickets Retrieved': self.num_retrieved,
            'Mode': self.rag_mode.capitalize()
        }


class RAGPipeline:
    """
    Production RAG pipeline with two-stage retrieval and LangChain integration.

    This class orchestrates the complete RAG workflow from query input to
    answer generation. It manages model loading, vector search, re-ranking,
    and LLM-based answer generation with comprehensive performance tracking.

    Architecture Pattern:
        - Lazy Initialization: Models loaded on first use via initialize()
        - Composition Over Inheritance: Aggregates models rather than extending
        - Single Responsibility: Each method handles one stage of pipeline
        - Configuration Driven: All behavior controlled via AppConfig

    Component Lifecycle:
        1. Construction (__init__): Store config, initialize attributes to None
        2. Initialization (initialize): Load all models and build chains
        3. Processing (process_ticket): Execute full pipeline for each query
        4. Reuse: Same instance handles multiple queries efficiently

    Model Components:
        - embedder: HuggingFace sentence transformer for query/document embeddings
        - vectordb: ChromaDB vector store with embedded historical tickets
        - reranker: Cross-encoder model for semantic re-ranking
        - llm: Local LLM via LM Studio (OpenAI-compatible API)
        - chain: LangChain LCEL chain for answer generation
        - query_expansion_chain: LLM chain for MultiQuery mode (optional)

    Performance Characteristics:
        - First query: Slower due to model loading and caching
        - Subsequent queries: Fast due to warm caches
        - Memory usage: ~2-4GB for all models loaded
        - CPU/GPU: Embeddings and re-ranking on CPU, LLM on GPU if available

    Thread Safety:
        Not thread-safe. Create separate instances for concurrent processing.

    Usage Pattern:
        config = get_config()
        pipeline = RAGPipeline(config)
        pipeline.initialize()  # Load models once

        # Process multiple tickets efficiently
        for ticket in tickets:
            response = pipeline.process_ticket(
                subject=ticket.subject,
                body=ticket.body
            )
            print(response['answer'])
    """

    def __init__(self, config: AppConfig):
        """
        Initialize RAG pipeline with configuration.

        Creates pipeline instance with lazy initialization pattern. Models are NOT
        loaded during construction. Call initialize() to load models before first use.

        Why Lazy Initialization:
            - Faster startup time for testing and development
            - Allows configuration validation before expensive model loading
            - Supports dependency injection and testing with mocks
            - Enables graceful handling of missing models or network issues

        Args:
            config: Application configuration containing:
                   - Model paths and parameters
                   - Retrieval settings (top_k_initial, top_k_reranked)
                   - LLM connection details (lm_studio_url, llm_model)
                   - RAG mode ('strict' or 'augmented')
                   - Retriever type ('standard', 'multiquery', or 'mmr')

        Attributes Initialized:
            self.config: Stored configuration reference
            self.embedder: None until initialize() (HuggingFaceEmbeddings)
            self.vectordb: None until initialize() (Chroma vector store)
            self.reranker: None until initialize() (CrossEncoder)
            self.llm: None until initialize() (ChatOpenAI for LM Studio)
            self.chain: None until initialize() (LangChain LCEL chain)
            self.query_expansion_chain: None until initialize() (MultiQuery chain)

        Example:
            config = AppConfig(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                reranker_model='mixedbread-ai/mxbai-rerank-base-v1',
                lm_studio_url='http://192.168.7.171:1234',
                llm_model='gpt-oss-20b',
                rag_mode='strict',
                retriever_type='standard'
            )
            pipeline = RAGPipeline(config)  # Fast, no models loaded yet
            pipeline.initialize()  # Now load models (slow, one-time cost)
        """
        self.config = config
        # All model components set to None initially (lazy loading pattern)
        self.embedder: Optional[HuggingFaceEmbeddings] = None
        self.vectordb: Optional[Chroma] = None
        self.reranker: Optional[CrossEncoder] = None
        self.llm: Optional[ChatOpenAI] = None
        self.chain = None
        self.query_expansion_chain = None

    def initialize(self):
        """
        Initialize all pipeline components with model loading and chain construction.

        This method performs the expensive one-time setup of loading all models
        and connecting to services. Should be called once after construction
        and before processing any tickets.

        Initialization Steps:
            1. Load embedding model (HuggingFace sentence transformer)
            2. Connect to ChromaDB vector database
            3. Load cross-encoder re-ranker model
            4. Connect to LM Studio LLM server
            5. Build LangChain LCEL chain for answer generation
            6. (Optional) Build query expansion chain for MultiQuery mode

        Performance:
            - Total time: 5-15 seconds depending on models and hardware
            - Embedding model: ~2-5s (downloads on first run, cached thereafter)
            - Vector database: ~1-3s (loads existing embeddings from disk)
            - Re-ranker model: ~2-5s (downloads on first run, cached thereafter)
            - LLM connection: <1s (just establishes HTTP connection)
            - Chain building: <1s (pure Python object construction)

        Error Handling:
            - Embedding model not found: Downloads from HuggingFace
            - ChromaDB missing: Creates new empty database
            - Re-ranker not found: Downloads from HuggingFace
            - LM Studio offline: Raises connection error
            - Invalid config: Raises ValueError

        Caching Behavior:
            - HuggingFace models: Cached in ~/.cache/huggingface/
            - ChromaDB data: Persisted in config.persist_directory
            - LM Studio models: Managed by LM Studio application

        Side Effects:
            - Sets self.embedder, self.vectordb, self.reranker, self.llm
            - Sets self.chain (always) and self.query_expansion_chain (conditional)
            - Prints progress messages to console
            - Downloads models on first run (network activity)

        Thread Safety:
            Not thread-safe. Call from main thread before parallel processing.

        Example:
            pipeline = RAGPipeline(config)
            try:
                pipeline.initialize()  # May take 10-15s on first run
                print("Ready to process tickets")
            except ConnectionError:
                print("LM Studio not running")
            except Exception as e:
                print(f"Initialization failed: {e}")
        """
        print("Initializing RAG pipeline...")

        # STEP 1: Load embedding model for query and document vectorization
        print(f"  Loading embedding model: {self.config.embedding_model}")
        start = time.time()
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,  # e.g., 'all-MiniLM-L6-v2'
            model_kwargs={'device': self.config.embedding_device},  # 'cpu', 'cuda', or 'mps'
            encode_kwargs={'normalize_embeddings': True}  # L2 normalization for cosine similarity
        )
        print(f"  Embeddings loaded ({time.time() - start:.2f}s)")

        # STEP 2: Connect to ChromaDB vector database with pre-embedded tickets
        print(f"  Loading ChromaDB: {self.config.chroma_db_path}")
        start = time.time()
        self.vectordb = Chroma(
            collection_name=self.config.collection_name,  # e.g., 'support_tickets'
            embedding_function=self.embedder,  # Uses same embedder for consistency
            persist_directory=self.config.persist_directory  # On-disk storage location
        )
        print(f"  ChromaDB loaded ({time.time() - start:.2f}s)")

        # STEP 3: Load cross-encoder re-ranker for semantic relevance scoring
        print(f"  Loading re-ranker: {self.config.reranker_model}")
        start = time.time()
        self.reranker = CrossEncoder(self.config.reranker_model)  # e.g., 'mxbai-rerank-base-v1'
        print(f"  Re-ranker loaded ({time.time() - start:.2f}s)")

        # STEP 4: Connect to LM Studio local LLM server via OpenAI-compatible API
        print(f"  Connecting to LM Studio: {self.config.lm_studio_url}")
        start = time.time()
        self.llm = ChatOpenAI(
            base_url=f"{self.config.lm_studio_url}/v1",  # e.g., 'http://192.168.7.171:1234/v1'
            api_key="not-needed",  # LM Studio doesn't require authentication
            model=self.config.llm_model,  # Model name as shown in LM Studio
            temperature=self.config.temperature,  # 0.0-1.0, lower = more deterministic
            max_tokens=self.config.max_tokens,  # Maximum response length
            timeout=self.config.timeout  # HTTP request timeout in seconds
        )
        print(f"  LLM connected ({time.time() - start:.2f}s)")

        # STEP 5: Build LangChain LCEL chain for structured answer generation
        self._build_chain()

        # STEP 6: (Optional) Build query expansion chain for MultiQuery retrieval
        if self.config.retriever_type == 'multiquery':
            print(f"  Setting up query expansion...")
            start = time.time()
            self._build_query_expansion_chain()
            print(f"  Query expansion initialized ({time.time() - start:.2f}s)")

        print("RAG pipeline initialized successfully!")

    def _build_chain(self):
        """
        Build LangChain LCEL chain for structured answer generation.

        This method constructs the LangChain Expression Language (LCEL) chain
        that orchestrates the LLM-based answer generation. The chain takes
        context, subject, and body as inputs and produces the final answer.

        Chain Architecture:
            Input Dict -> Parallel Lambda Extractors -> Prompt Template -> LLM -> String Parser

        Why LCEL:
            - Declarative syntax: Clear data flow visualization
            - Automatic batching: Efficient processing of multiple requests
            - Type safety: Compile-time validation of chain structure
            - Composability: Easy to extend with additional steps
            - Observability: Built-in logging and tracing support

        Chain Components:
            1. Input Dictionary: {context: str, subject: str, body: str}
            2. Lambda Extractors: Pass-through functions for each input field
            3. Prompt Template: Combines system prompt + user inputs
            4. LLM: Generates answer via LM Studio
            5. String Parser: Extracts text from LLM response

        RAG Mode Handling:
            The system prompt varies based on config.rag_mode:
            - Strict Mode: Forces LLM to use ONLY historical context
            - Augmented Mode: Allows LLM to supplement with general knowledge

        Side Effects:
            Sets self.chain to the constructed LCEL chain

        Example Usage (internal):
            self._build_chain()  # Called during initialize()
            # Later, in generate_answer():
            answer = self.chain.invoke({
                "context": formatted_context,
                "subject": "VPN issue",
                "body": "Connection drops"
            })
        """
        # Get mode-specific system prompt from configuration
        # Strict mode: Answer ONLY from context
        # Augmented mode: Context + LLM knowledge
        system_prompt = self.config.get_prompt_template()

        # Create two-message chat template: system + user
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  # Instructions and context formatting
            ("user", "Support Ticket:\nSubject: {subject}\nBody: {body}\n\nPlease provide a detailed, actionable solution based on the similar cases above.")
        ])

        # Build LCEL chain with pipe operator (|) for sequential composition
        # Step 1: Extract fields from input dict using RunnableLambda
        # Step 2: Format into prompt template
        # Step 3: Send to LLM for generation
        # Step 4: Parse string response
        self.chain = (
            {
                "context": RunnableLambda(lambda x: x["context"]),  # Pass-through context
                "subject": RunnableLambda(lambda x: x["subject"]),  # Pass-through subject
                "body": RunnableLambda(lambda x: x["body"])  # Pass-through body
            }
            | prompt  # Inject into template
            | self.llm  # Generate with LLM
            | StrOutputParser()  # Extract text from response
        )

    def _build_query_expansion_chain(self):
        """
        Build query expansion chain for MultiQuery retrieval strategy.

        This method constructs a separate LangChain chain specifically for
        generating query variations. Used only when config.retriever_type
        is set to 'multiquery'.

        Purpose:
            Improves retrieval coverage by searching with multiple phrasings
            of the same semantic intent. Helps capture relevant documents that
            might use different terminology.

        MultiQuery Benefits:
            - Broader Coverage: Multiple searches find more relevant tickets
            - Terminology Variation: Captures different ways to phrase issues
            - Robustness: Less dependent on exact query wording
            - Redundancy Handling: Deduplication ensures no duplicate results

        MultiQuery Drawbacks:
            - Slower: Multiple LLM calls and searches increase latency
            - More Tokens: Generates additional text (3 queries vs 1)
            - Complexity: More moving parts to debug and maintain

        Chain Architecture:
            Original Query -> Prompt Template -> LLM -> String Parser -> 3 Query Variations

        Output Format:
            The LLM is prompted to generate exactly 3 query variations,
            each on a separate line. These are parsed and used for
            parallel vector searches in vector_search() method.

        Side Effects:
            Sets self.query_expansion_chain to the constructed LCEL chain

        Example Usage (internal):
            # Only called if config.retriever_type == 'multiquery'
            self._build_query_expansion_chain()
            # Later, in generate_multiple_queries():
            variations = self.query_expansion_chain.invoke({
                "question": "VPN keeps disconnecting"
            })
            # Output: "VPN connection drops repeatedly\nVPN unstable\nVPN session timeout"

        Performance:
            - Time: ~2-5s for 3 query generation (depends on LLM speed)
            - Tokens: ~100-200 tokens consumed per expansion
        """
        # Create prompt template for query expansion task
        query_expansion_prompt = PromptTemplate(
            input_variables=["question"],  # Original user query
            template="""You are an AI assistant that generates multiple search queries for IT support tickets.
Generate 3 different versions of the user's query to improve ticket retrieval.
Each version should rephrase the query while maintaining the technical intent.

Original query: {question}

Provide 3 alternative queries (one per line):"""
        )

        # Build simple chain: prompt -> LLM -> parse
        self.query_expansion_chain = (
            query_expansion_prompt  # Format original query into expansion prompt
            | self.llm  # Generate 3 variations using LLM
            | StrOutputParser()  # Extract text (will be split by newlines later)
        )

    def generate_multiple_queries(self, query: str) -> List[str]:
        """
        Generate multiple query variations for improved retrieval coverage.

        Uses the LLM to generate alternative phrasings of the same query intent.
        Only used when config.retriever_type == 'multiquery'. Helps capture
        documents that use different terminology.

        Process:
            1. Check if query expansion chain exists (MultiQuery mode enabled)
            2. If not, return just the original query
            3. If yes, invoke LLM to generate 3 variations
            4. Parse LLM output (newline-separated variations)
            5. Clean up formatting (remove numbering like "1.", "2.")
            6. Filter out short or empty variations
            7. Return original + top 2 generated variations (3 total)

        Why Limit to 3 Total:
            - Balance between coverage and performance
            - More queries = more search time + deduplication overhead
            - Diminishing returns after 3 variations
            - Token cost increases linearly with query count

        Args:
            query: Original user query text

        Returns:
            List[str]: List of 1-3 query variations
                      Always includes original as first element
                      May include 0-2 generated variations

        Example:
            original = "VPN keeps disconnecting every 15 minutes"
            variations = pipeline.generate_multiple_queries(original)
            # Returns: [
            #   "VPN keeps disconnecting every 15 minutes",  (original)
            #   "VPN connection drops repeatedly",            (variation 1)
            #   "VPN session unstable and timing out"        (variation 2)
            # ]

        Performance:
            - Time: ~2-5s for LLM generation (only in MultiQuery mode)
            - Tokens: ~100-200 consumed per expansion
            - Fallback: Returns [original] if expansion fails
        """
        # Return early if not in MultiQuery mode
        if not self.query_expansion_chain:
            return [query]

        # Generate alternative queries using LLM chain
        generated_queries_text = self.query_expansion_chain.invoke({"question": query})

        # Parse LLM output: split by newlines, remove empty strings
        generated_queries = [q.strip() for q in generated_queries_text.split('\n') if q.strip()]

        # Clean up LLM formatting (remove numbering like "1.", "2.", "3)")
        import re
        cleaned_queries = []
        for q in generated_queries:
            # Remove leading numbers and dots/parentheses
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', q).strip()
            # Only keep meaningful queries (>10 chars to filter out artifacts)
            if cleaned and len(cleaned) > 10:
                cleaned_queries.append(cleaned)

        # Return original + top 2 generated (3 total maximum)
        return [query] + cleaned_queries[:2]

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for query text.

        Converts text into a dense numerical vector using the configured
        embedding model. This vector is used for cosine similarity search
        in the ChromaDB vector database.

        Embedding Model Details:
            - Default: sentence-transformers/all-MiniLM-L6-v2
            - Dimension: 384 (configurable via config.embedding_dimension)
            - Normalization: L2-normalized for cosine similarity
            - Speed: ~10-50ms per query on CPU

        Why Embeddings:
            - Semantic similarity: Captures meaning beyond keyword matching
            - Efficient search: Fast vector operations in ChromaDB
            - Multilingual: Works across languages (model-dependent)

        Args:
            query: Text to convert into embedding vector

        Returns:
            List[float]: Dense vector representation (length = embedding_dimension)
                        Each element is a float between -1.0 and 1.0 (normalized)

        Example:
            embedding = pipeline.embed_query("VPN connection issue")
            # Returns: [0.023, -0.145, 0.089, ..., 0.112]  (384 floats)
            # Can now compute cosine similarity with document embeddings

        Thread Safety:
            Thread-safe for read operations after initialization
        """
        return self.embedder.embed_query(query)

    def vector_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform vector similarity search with configurable retrieval strategy.

        This is the first stage of the two-stage retrieval pipeline. It rapidly
        identifies potentially relevant historical tickets using vector embeddings
        and cosine similarity. The specific strategy (standard, MultiQuery, or MMR)
        is determined by config.retriever_type.

        Retrieval Strategies:
            1. Standard: Simple cosine similarity search (fastest)
            2. MultiQuery: Generate query variations and combine results (broader coverage)
            3. MMR: Maximal Marginal Relevance for diversity-aware retrieval

        Strategy Selection Guide:
            - Standard: General use, fast, good for most queries
            - MultiQuery: Complex queries with multiple interpretations
            - MMR: When diversity matters (avoid redundant similar tickets)

        Args:
            query: User's support ticket text (subject + body)
            top_k: Number of candidates to retrieve (default: config.top_k_initial = 20)
                  More candidates = better recall but slower re-ranking

        Returns:
            List[Document]: Retrieved ticket documents with metadata
                           Each Document contains:
                           - page_content: Ticket body text
                           - metadata: {subject, category, priority, ...}

        Performance:
            - Standard: ~100-300ms for top-20
            - MultiQuery: ~2-5s (includes LLM query expansion)
            - MMR: ~200-500ms (slightly slower than standard)

        Example:
            query = "VPN keeps disconnecting every 15 minutes"
            candidates = pipeline.vector_search(query, top_k=20)
            # Returns 20 potentially relevant historical tickets
            # Next stage: re-rank these 20 down to top-5

        Implementation Details:
            - Embeddings: Pre-computed and stored in ChromaDB
            - Similarity: Cosine similarity (dot product of normalized vectors)
            - Index: HNSW (Hierarchical Navigable Small World) for fast ANN search
        """
        # Use provided top_k or default from config
        k = top_k or self.config.top_k_initial

        if self.config.retriever_type == 'mmr':
            # MMR STRATEGY: Maximal Marginal Relevance
            # Balances relevance with diversity to avoid redundant results

            # MMR Algorithm:
            # 1. Fetch fetch_k candidates (3x more than needed)
            # 2. Select first document (highest similarity)
            # 3. For each remaining:
            #    score = lambda * relevance - (1-lambda) * max_similarity_to_selected
            # 4. Greedily select k documents with highest scores

            # lambda_mult parameter controls trade-off:
            # - 1.0: Pure relevance (ignores diversity)
            # - 0.0: Pure diversity (ignores relevance)
            # - 0.5: Balanced (default)

            return self.vectordb.max_marginal_relevance_search(
                query,
                k=k,  # Final number of results to return
                fetch_k=k * 3,  # Fetch 3x more candidates for diversity selection
                lambda_mult=0.5  # Balance between relevance and diversity
            )

        elif self.config.retriever_type == 'multiquery' and self.query_expansion_chain:
            # MULTIQUERY STRATEGY: Query expansion for broader coverage

            # MultiQuery Algorithm:
            # 1. Generate N query variations (typically 3)
            # 2. Search with each variation independently
            # 3. Combine results with deduplication
            # 4. Return top-k from combined set

            # Benefits:
            # - Captures different terminology
            # - Robust to query phrasing
            # - Better coverage for ambiguous queries

            # Generate 3 query variations (includes original)
            queries = self.generate_multiple_queries(query)

            # Retrieve documents for each query variation
            all_docs = []
            seen_ids = set()  # Track document IDs for deduplication

            for q in queries:
                # Search with each variation
                docs = self.vectordb.similarity_search(q, k=k)

                # Deduplicate based on document content hash
                for doc in docs:
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)

            # Return top-k from combined results
            # Note: Results are not re-sorted by similarity here
            # Cross-encoder re-ranking will handle final ordering
            return all_docs[:k]

        else:
            # STANDARD STRATEGY: Simple cosine similarity search

            # Standard Algorithm:
            # 1. Embed query into 384-dim vector
            # 2. Compute cosine similarity with all document vectors
            # 3. Return top-k most similar documents

            # Cosine Similarity Formula:
            # similarity = dot(query_embedding, doc_embedding)
            # Both vectors are L2-normalized, so this is equivalent to:
            # similarity = sum(q_i * d_i) for i in dimensions

            # Advantages:
            # - Fastest retrieval strategy
            # - Most predictable results
            # - Good for well-phrased queries

            return self.vectordb.similarity_search(query, k=k)

    def rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using cross-encoder for deep semantic relevance.

        This is the second stage of the two-stage retrieval pipeline. While
        vector search (stage 1) uses fast but approximate bi-encoder embeddings,
        re-ranking uses a slower but more accurate cross-encoder that directly
        models query-document relevance.

        Why Re-ranking:
            Vector search (bi-encoder):
            - Fast: Independent embeddings for query and documents
            - Approximate: No direct query-document interaction
            - Good for recall: Quickly narrows from 1000s to 20 candidates

            Cross-encoder re-ranking:
            - Slow: Processes each query-document pair jointly
            - Accurate: Models direct interaction between query and document
            - Good for precision: Refines top-20 down to top-5

        Cross-Encoder Process:
            1. Concatenate query + document text
            2. Feed through BERT-style model (12 transformer layers)
            3. Output single relevance score (not an embedding)
            4. Scores are NOT normalized (typically range -10 to +10)

        Args:
            query: Original user query text
            documents: Candidates from vector_search() (typically 20)
            top_k: Number of results to keep (default: config.top_k_reranked = 5)

        Returns:
            List[Tuple[Document, float]]: Documents paired with relevance scores
                                          Sorted by score descending
                                          Length = min(len(documents), top_k)

        Score Interpretation:
            - Positive scores: Relevant (higher = more relevant)
            - Negative scores: Not relevant (lower = less relevant)
            - Typical range: -10 to +10 (not bounded, not normalized)

        Performance:
            - Time: ~50-100ms per document on CPU
            - Total: ~1-2s for 20 documents
            - Scales linearly with number of documents

        Example:
            candidates = [doc1, doc2, ..., doc20]  # From vector search
            reranked = pipeline.rerank_results(query, candidates, top_k=5)
            # Returns: [(doc7, 8.2), (doc3, 7.9), (doc15, 6.1), (doc2, 5.8), (doc11, 4.9)]
            # These top-5 will be used for LLM context

        Debug Output:
            Prints raw scores and top reranked scores to console for monitoring
        """
        # Use provided top_k or default from config
        k = top_k or self.config.top_k_reranked

        # Create query-document pairs for cross-encoder input
        # Each pair is a list: [query_text, document_text]
        pairs = [[query, doc.page_content] for doc in documents]

        # Get cross-encoder relevance scores
        # The model processes each pair jointly through transformer layers
        scores = self.reranker.predict(pairs)

        # DEBUG: Monitor raw cross-encoder scores for quality assurance
        print(f"DEBUG rerank_results: Raw scores from cross-encoder: {scores[:3] if len(scores) > 3 else scores}")

        # Combine documents with their relevance scores
        doc_scores = list(zip(documents, scores))

        # Sort by relevance score descending and take top-k
        # Higher scores = more relevant documents
        reranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:k]

        # DEBUG: Monitor top reranked scores for quality assurance
        print(f"DEBUG rerank_results: Top reranked scores: {[float(s) for _, s in reranked[:3]]}")

        return reranked

    def build_context(
        self,
        doc_scores: List[Tuple[Document, float]]
    ) -> str:
        """
        Build formatted context string from re-ranked documents for LLM prompt.

        Takes the top-N re-ranked tickets and formats them into a structured
        context string that will be included in the LLM prompt. The formatting
        includes metadata (subject, category, priority) and relevance scores
        to help the LLM understand the quality of retrieved context.

        Context Format:
            Each ticket is formatted as:
            Ticket #N (Relevance: XX.X%)
            Subject: [subject]
            Category: [category]
            Priority: [priority]
            Content: [body]

            Tickets are separated by "---" dividers

        Why Include Relevance Scores:
            - Helps LLM weight more relevant tickets higher
            - Provides transparency for debugging
            - Allows LLM to express uncertainty for low-relevance context

        Args:
            doc_scores: Re-ranked documents with scores from rerank_results()
                       Typically 5 documents with scores ranging -10 to +10

        Returns:
            str: Formatted context string ready for LLM prompt
                Empty string if no documents provided

        Example Output:
            Ticket #1 (Relevance: 82.0%)
            Subject: VPN connection drops
            Category: Network
            Priority: High
            Content: User reports VPN disconnecting every 15 minutes...

            ---

            Ticket #2 (Relevance: 79.0%)
            Subject: Cisco AnyConnect timeout
            Category: Network
            Priority: Medium
            Content: VPN client times out during file transfers...

        Token Cost:
            Approximately 100-200 tokens per ticket (varies with content length)
            Total: 500-1000 tokens for 5 tickets
        """
        context_parts = []

        # Format each ticket with numbered index and metadata
        for i, (doc, score) in enumerate(doc_scores, 1):
            metadata = doc.metadata
            subject = metadata.get('subject', 'N/A')
            body = doc.page_content
            category = metadata.get('category', 'N/A')
            priority = metadata.get('priority', 'N/A')

            # Format single ticket block
            context_part = f"""
Ticket #{i} (Relevance: {score:.2%})
Subject: {subject}
Category: {category}
Priority: {priority}
Content: {body}
"""
            context_parts.append(context_part.strip())

        # Join tickets with visual separator
        return "\n\n---\n\n".join(context_parts)

    def calculate_confidence(
        self,
        doc_scores: List[Tuple[Document, float]]
    ) -> float:
        """
        Calculate overall confidence score from re-ranking scores.

        Computes a single confidence metric (0-1) that indicates how confident
        the system is in the retrieved context. This helps users understand
        the reliability of the generated answer.

        Calculation Process:
            1. Compute weighted average of re-ranking scores
               - Top result gets weight 1.0
               - Second result gets weight 0.5
               - Third result gets weight 0.33
               - And so on (weight = 1 / position)
            2. Normalize to 0-1 range using sigmoid function
               - Maps unbounded scores to bounded confidence
               - Centered around score of 0

        Why Weighted Average:
            - Top results matter more than lower results
            - Reflects diminishing importance of lower-ranked tickets
            - Single high-relevance ticket can drive high confidence

        Normalization Formula:
            confidence = 1 / (1 + e^(-weighted_score))
            - Score -5 -> Confidence ~0.7%
            - Score  0 -> Confidence  50%
            - Score +5 -> Confidence ~99.3%

        Args:
            doc_scores: Re-ranked documents with cross-encoder scores
                       Typically 5 tuples of (Document, float)

        Returns:
            float: Confidence score between 0.0 and 1.0
                  0.0 = No confidence (no relevant context)
                  0.5 = Moderate confidence (neutral scores)
                  1.0 = High confidence (strong relevance)

        Example:
            # Strong match: scores [8.2, 7.9, 6.1, 5.8, 4.9]
            confidence = 0.95  # Very confident

            # Weak match: scores [0.5, 0.2, -0.1, -0.3, -0.8]
            confidence = 0.53  # Low confidence

            # No results: scores []
            confidence = 0.0  # No confidence
        """
        if not doc_scores:
            return 0.0

        # Calculate position-based weights (1/1, 1/2, 1/3, ...)
        weights = [1.0 / (i + 1) for i in range(len(doc_scores))]

        # Extract just the scores from (doc, score) tuples
        scores = [score for _, score in doc_scores]

        # Compute weighted average: sum(w_i * s_i) / sum(w_i)
        weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)

        # Normalize to 0-1 using sigmoid function centered at 0
        # e = 2.71828 (Euler's number)
        normalized = 1 / (1 + pow(2.71828, -weighted_score))

        return normalized

    def generate_answer(
        self,
        subject: str,
        body: str,
        context: str
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        This is the final stage of the RAG pipeline. Takes the formatted
        context from retrieved tickets and the user's query, then invokes
        the LLM to generate a detailed, actionable solution.

        LLM Behavior:
            - Strict Mode: Answer ONLY from context (no hallucination)
            - Augmented Mode: Context + LLM's general IT knowledge

        Process:
            1. Combine inputs into chain input dict
            2. Invoke LangChain LCEL chain
               - Lambda extractors pass through inputs
               - Prompt template formats system + user messages
               - LLM generates response
               - String parser extracts text
            3. Return generated answer text

        Args:
            subject: User's ticket subject line
            body: User's ticket description/body
            context: Formatted historical tickets from build_context()

        Returns:
            str: Generated answer from LLM
                Typically 100-500 words of technical guidance

        Performance:
            - Time: 2-10s depending on LLM model and response length
            - Tokens: Input (query + context) + Output (answer)
            - Local LLM: Depends on LM Studio server performance

        Example:
            answer = pipeline.generate_answer(
                subject="VPN keeps disconnecting",
                body="My VPN drops every 15 minutes on Windows 10",
                context="Ticket #1...\n\n---\n\nTicket #2..."
            )
            # Returns: "Based on similar cases, this appears to be a timeout issue..."

        Error Handling:
            - LM Studio offline: Raises ConnectionError
            - Timeout: Raises TimeoutError after config.timeout seconds
            - Invalid model: Raises ModelNotFoundError
        """
        # Invoke the LangChain LCEL chain with all inputs
        response = self.chain.invoke({
            "subject": subject,
            "body": body,
            "context": context
        })

        return response

    def process_ticket(
        self,
        subject: str,
        body: str,
        top_k_initial: Optional[int] = None,
        top_k_reranked: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a support ticket through the complete RAG pipeline.

        This is the main entry point for ticket processing. It orchestrates
        all pipeline stages and returns a comprehensive response with answer,
        confidence, references, and performance metrics.

        Pipeline Stages:
            1. Query Formation: Combine subject + body into single query text
            2. Query Expansion: (Optional) Generate variations for MultiQuery mode
            3. Vector Search: Retrieve top-K candidates from ChromaDB (typically 20)
            4. Cross-Encoder Re-ranking: Re-rank to top-N (typically 5)
            5. Context Building: Format top-N tickets for LLM prompt
            6. Confidence Calculation: Compute confidence score from re-ranking scores
            7. Answer Generation: Invoke LLM with context to generate solution
            8. Response Assembly: Package answer, metrics, and references

        Timing Breakdown (typical):
            - Vector Search: 0.1-0.5s
            - Re-ranking: 0.5-1.5s
            - LLM Generation: 2-10s
            - Total: 3-12s

        Args:
            subject: User's ticket subject line
            body: User's ticket description/body text
            top_k_initial: Override for initial retrieval count (default: config.top_k_initial)
            top_k_reranked: Override for re-ranked count (default: config.top_k_reranked)

        Returns:
            Dict[str, Any]: Comprehensive response dictionary containing:
                answer: str - Generated solution from LLM
                confidence: float - Overall confidence score (0-1)
                references: List[Dict] - Referenced tickets with metadata and scores
                metrics: PerformanceMetrics - Detailed timing and performance data
                generated_queries: List[str] - Query variations (MultiQuery mode only)

        Response Structure:
            {
                'answer': "Based on similar cases, try these steps...",
                'confidence': 0.87,
                'references': [
                    {
                        'subject': "VPN connection drops",
                        'body': "User reports...",
                        'category': "Network",
                        'priority': "High",
                        'score': 8.2  # Cross-encoder relevance score
                    },
                    ...  # 4 more tickets
                ],
                'metrics': PerformanceMetrics(
                    total_time=5.23,
                    search_time=0.35,
                    rerank_time=0.89,
                    llm_time=3.99,
                    ...
                ),
                'generated_queries': [  # Only if MultiQuery mode
                    "VPN keeps disconnecting",
                    "VPN connection unstable",
                    "VPN drops repeatedly"
                ]
            }

        Example Usage:
            pipeline = RAGPipeline(config)
            pipeline.initialize()

            response = pipeline.process_ticket(
                subject="VPN connection issue",
                body="My VPN keeps disconnecting every 15 minutes on Windows 10"
            )

            print(response['answer'])
            print(f"Confidence: {response['confidence']:.1%}")
            print(f"Based on {len(response['references'])} similar tickets")

        Error Handling:
            - LM Studio offline: Raises ConnectionError during LLM generation
            - Empty results: Returns low confidence with generic message
            - Timeout: Raises TimeoutError after config.timeout seconds

        Thread Safety:
            Not thread-safe. Create separate pipeline instances for concurrent use.

        Performance Optimization:
            - Warm cache: First query slower due to model loading
            - Subsequent queries: Fast due to warm caches
            - Adjust top_k_initial/top_k_reranked to balance speed vs quality
        """
        # Start total time tracking
        start_time = time.time()

        # STAGE 1: Query Formation
        # Combine subject and body into single query string
        query = f"{subject}\n\n{body}"

        # STAGE 2: Query Expansion (Optional - MultiQuery mode only)
        generated_queries = None
        if self.config.retriever_type == 'multiquery' and self.query_expansion_chain:
            generated_queries = self.generate_multiple_queries(query)

        # STAGE 3: Vector Search - Retrieve initial candidates
        search_start = time.time()
        initial_results = self.vector_search(query, top_k=top_k_initial)
        search_time = time.time() - search_start

        # STAGE 4: Cross-Encoder Re-ranking - Refine to top-N
        rerank_start = time.time()
        reranked_results = self.rerank_results(query, initial_results, top_k=top_k_reranked)
        rerank_time = time.time() - rerank_start

        # STAGE 5: Context Building - Format tickets for LLM prompt
        context = self.build_context(reranked_results)

        # STAGE 6: Confidence Calculation - Compute overall confidence
        confidence = self.calculate_confidence(reranked_results)

        # STAGE 7: Answer Generation - Invoke LLM with context
        llm_start = time.time()
        answer = self.generate_answer(subject, body, context)
        llm_time = time.time() - llm_start

        # Calculate total pipeline time
        total_time = time.time() - start_time

        # DEBUG: Monitor re-ranking scores for quality assurance
        print(f"DEBUG process_ticket: Reranked results scores: {[float(s) for _, s in reranked_results]}")

        # STAGE 8: Response Assembly - Build comprehensive response dict
        references = []
        for doc, score in reranked_results:
            # Convert score to Python float (from numpy.float32 or similar)
            ref_score = float(score)
            print(f"DEBUG process_ticket: Converting score {score} (type: {type(score)}) -> {ref_score}")

            # Package ticket metadata and score for UI display
            references.append({
                'subject': doc.metadata.get('subject', 'N/A'),
                'body': doc.page_content,
                'category': doc.metadata.get('category', 'N/A'),
                'priority': doc.metadata.get('priority', 'N/A'),
                'score': ref_score,  # Raw cross-encoder relevance score
            })

        # Assemble final response dictionary
        response = {
            'answer': answer,
            'confidence': confidence,
            'references': references,
            'metrics': PerformanceMetrics(
                total_time=total_time,
                embedding_time=0.0,  # Embedded in search_time
                search_time=search_time,
                rerank_time=rerank_time,
                llm_time=llm_time,
                confidence_score=confidence,
                num_retrieved=len(initial_results),
                num_reranked=len(reranked_results),
                timestamp=datetime.now(),
                rag_mode=self.config.rag_mode
            ),
            'generated_queries': generated_queries  # None if not MultiQuery mode
        }

        return response


if __name__ == '__main__':
    # Test pipeline initialization
    from config import get_config

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    config = get_config()
    pipeline = RAGPipeline(config)

    try:
        pipeline.initialize()
        print("\n✅ Pipeline initialization successful!")

        # Test query
        print("\n🧪 Testing with sample ticket...")
        response = pipeline.process_ticket(
            subject="VPN connection issue",
            body="My VPN keeps disconnecting every 15 minutes. Using Cisco AnyConnect on Windows 10."
        )

        print("\n📊 Response:")
        print(f"Answer length: {len(response['answer'])} characters")
        print(f"Confidence: {response['confidence']:.1%}")
        print(f"References: {len(response['references'])} tickets")
        print(f"\nMetrics:")
        for key, value in response['metrics'].to_dict().items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
