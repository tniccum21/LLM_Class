"""
Week07 - Intelligent Support Ticket Assistant (Streamlit App)

This module implements the production Streamlit web application that provides
a user-friendly interface for AI-powered support ticket resolution using RAG
(Retrieval-Augmented Generation) with dual-layer safety guardrails.

ARCHITECTURE OVERVIEW:
---------------------
This application integrates three major components:

1. RAG Pipeline (rag_pipeline.py)
   - Two-stage retrieval: Vector search + cross-encoder re-ranking
   - LangChain LCEL for LLM integration
   - Multiple retriever strategies: standard, multiquery, MMR
   - ChromaDB vector database for ticket storage

2. Input Guardian (input_guardian.py)
   - Llama Guard 3 8B for content safety (via LM Studio)
   - Llama Prompt Guard 2 86M for prompt injection detection
   - Independent checkbox control for each layer
   - Configurable safety thresholds

3. Streamlit UI
   - Two-column layout: input panel (left) + response panel (right)
   - Real-time configuration via sliders and radio buttons
   - Performance metrics display
   - Reference ticket visualization

APPLICATION FLOW:
----------------
User Input -> Safety Validation -> RAG Pipeline -> Response Display

1. User enters ticket (subject + body) and configures options
2. Click Submit triggers validation and processing
3. Input Guardian validates for safety (if enabled)
4. If safe (or guardrails disabled), RAG pipeline processes ticket
5. Response displayed with metrics and reference tickets

KEY FEATURES:
------------
- Dual RAG Modes:
  * Strict: Answer only from historical ticket context
  * Augmented: Supplement context with LLM general knowledge

- Three Retriever Types:
  * Standard: Single query vector search
  * MultiQuery: Generate multiple query variations
  * MMR: Maximal Marginal Relevance (diversity-focused)

- Configurable Retrieval Parameters:
  * top_k_initial: Tickets retrieved from vector DB (5-50, default 20)
  * top_k_reranked: Final tickets after re-ranking (1-15, default 5)

- Independent Safety Controls:
  * Content Safety Check (Llama Guard): On/Off
  * Prompt Injection Detection (Prompt Guard): On/Off
  * Safety Threshold: 0.5-0.95 (default 0.7)

- Performance Optimization:
  * @st.cache_resource for model caching (pipeline, guardian)
  * Models loaded once and reused across sessions
  * Fast response times after initial load

CACHING STRATEGY:
----------------
Streamlit's @st.cache_resource decorator caches expensive model initialization:

1. load_pipeline(rag_mode, retriever_type):
   - Caches per unique (rag_mode, retriever_type) combination
   - Loads embedding model, vector DB, re-ranker, LLM, LangChain chain
   - Cache persists across all user sessions

2. load_guardian(enable_guardrails, enable_llama_guard, enable_prompt_guard):
   - Caches per unique guardrail configuration
   - Loads Llama Guard (via LM Studio) and Prompt Guard (local)
   - Graceful failure if LM Studio unavailable

Cache invalidation: Restart Streamlit app to clear cache and reload models

UI LAYOUT:
---------
+------------------------------------------------------------------+
|                  Intelligent Support Ticket Assistant            |
+------------------------------------------------------------------+
|  LEFT PANEL (Input)          |   RIGHT PANEL (Response)          |
|  - Ticket Input              |   - AI-Generated Answer           |
|    * Subject                 |   - Generated Query Variations    |
|    * Body                    |   - Performance Metrics           |
|  - Options                   |   - Reference Tickets (expandable)|
|    * RAG Mode (radio)        |                                   |
|    * Retriever Type (radio)  |   OR (before submit):             |
|  - Retrieval Parameters      |   - Empty State                   |
|    * Top-K sliders           |   - Usage Instructions            |
|  - Safety Options            |   - Example Tickets               |
|    * Checkboxes              |                                   |
|    * Safety Threshold        |                                   |
|  - Submit / Clear buttons    |                                   |
+------------------------------------------------------------------+

ERROR HANDLING:
--------------
The application handles several error scenarios gracefully:

1. Vector Database Not Found:
   - Shows error with instructions to run build_vectordb.py
   - Stops execution with st.stop()

2. Pipeline Initialization Failed:
   - Displays error message with details
   - Common causes: LM Studio not running, model not loaded

3. Guardian Initialization Failed:
   - Shows warning (not error) and continues without guardrails
   - Application remains functional for ticket processing

4. Validation Failure (during processing):
   - Catches exceptions and continues without safety validation
   - Shows warning to user

5. Ticket Processing Error:
   - Displays error with full stack trace for debugging
   - Does not crash the application

CONFIGURATION:
-------------
All configuration is managed through config.py:
- Vector database paths
- Embedding model settings
- LLM connection details
- Safety thresholds
- Default UI values

USAGE EXAMPLE:
-------------
To run the application:
    streamlit run ticket_support_app.py

To use programmatically (not typical):
    if __name__ == '__main__':
        main()

DEPLOYMENT NOTES:
----------------
- Requires LM Studio running with appropriate models loaded
- Vector database must be built before first use (run build_vectordb.py)
- HuggingFace transformers required for Prompt Guard
- Streamlit automatically handles port selection (default 8501)

CLASSROOM WALKTHROUGH GUIDE:
---------------------------
1. Start with module docstring and architecture overview
2. Explain caching strategy (@st.cache_resource decorators)
3. Walk through main() function flow
4. Show render_input_panel() for UI creation
5. Demonstrate validation logic in submit handler
6. Explain render_response_panel() for result display
7. Discuss error handling at each stage
"""

import streamlit as st
from pathlib import Path

from config import get_config
from rag_pipeline import RAGPipeline
from input_guardian import InputGuardian, ValidationResult


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
# This must be the first Streamlit command and can only be called once
st.set_page_config(
    page_title="Intelligent Support Ticket Assistant",
    page_icon="🎫",
    layout="wide",              # Use full screen width
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)


# ============================================================================
# MODEL LOADING FUNCTIONS (CACHED)
# ============================================================================

@st.cache_resource
def load_pipeline(rag_mode='strict', retriever_type='standard'):
    """
    Load and initialize RAG pipeline with caching for performance.

    This function is cached by Streamlit's @st.cache_resource decorator,
    meaning it will only execute once per unique (rag_mode, retriever_type)
    combination. Subsequent calls with the same parameters return the cached
    pipeline instance instantly.

    Caching Strategy:
        - Cache key: (rag_mode, retriever_type)
        - Cache lifetime: Until Streamlit app restart
        - Shared across: All user sessions
        - Memory impact: ~2-3GB per cached pipeline

    What Gets Cached:
        - Embedding model (sentence-transformers, ~400MB)
        - Vector database connection (ChromaDB)
        - Cross-encoder re-ranker (~1GB)
        - LLM connection (LM Studio API client)
        - LangChain LCEL chain
        - Query expansion chain (if multiquery mode)

    Args:
        rag_mode: RAG behavior mode
            - 'strict': Answer only from retrieved context (no hallucination)
            - 'augmented': Supplement context with LLM general knowledge
        retriever_type: Retrieval strategy
            - 'standard': Single query vector search
            - 'multiquery': Generate and search multiple query variations
            - 'mmr': Maximal Marginal Relevance (balances relevance + diversity)

    Returns:
        tuple: (pipeline, config)
            - pipeline: Initialized RAGPipeline instance
            - config: AppConfig instance with applied settings

    Raises:
        FileNotFoundError: If vector database doesn't exist
        ConnectionError: If cannot connect to LM Studio
        Exception: Other initialization failures

    Side Effects:
        - Loads models into memory (~2-3GB total)
        - Connects to LM Studio API
        - Prints initialization progress to console

    Performance:
        - First call: 5-15 seconds (model loading)
        - Cached calls: <0.01 seconds (instant)
        - Cache persists until app restart

    Example:
        # First call loads models (slow)
        pipeline1, config1 = load_pipeline('strict', 'standard')

        # Second call with same params returns cached instance (instant)
        pipeline2, config2 = load_pipeline('strict', 'standard')

        # Different params triggers new load
        pipeline3, config3 = load_pipeline('augmented', 'multiquery')
    """
    # Get base configuration
    config = get_config()

    # Apply user-selected RAG mode and retriever type
    # These override the defaults from config.py
    config.rag_mode = rag_mode
    config.retriever_type = retriever_type

    # Initialize pipeline (loads all models)
    pipeline = RAGPipeline(config)
    pipeline.initialize()

    return pipeline, config


@st.cache_resource
def load_guardian(enable_guardrails=True, enable_llama_guard=True, enable_prompt_guard=True):
    """
    Load and initialize input guardian with caching for performance.

    This function is cached by Streamlit's @st.cache_resource decorator,
    meaning it will only execute once per unique guardrail configuration.
    The guardian is shared across all user sessions for efficiency.

    Caching Strategy:
        - Cache key: (enable_guardrails, enable_llama_guard, enable_prompt_guard)
        - Cache lifetime: Until Streamlit app restart
        - Shared across: All user sessions
        - Memory impact: ~350MB for Prompt Guard model

    What Gets Cached:
        - Prompt Guard model (86M params, ~350MB if enabled)
        - Prompt Guard tokenizer
        - LM Studio API client (for Llama Guard)
        - Configuration settings

    Args:
        enable_guardrails: Master switch for guardrails system
            - If False, returns None immediately (no guardian)
            - If True, proceeds with initialization
        enable_llama_guard: Enable Llama Guard 3 8B content safety check
            - Requires LM Studio running with llama-guard-3-8b loaded
            - Checks for harmful content across 13 categories
        enable_prompt_guard: Enable Llama Prompt Guard 2 86M injection detection
            - Loads local transformer model from HuggingFace
            - Detects jailbreak and prompt injection attempts

    Returns:
        InputGuardian or None:
            - InputGuardian: Initialized guardian instance (if successful)
            - None: If guardrails disabled or initialization failed

    Error Handling:
        The function uses graceful degradation - it never raises exceptions:

        1. If enable_guardrails=False:
           - Returns None immediately without attempting initialization

        2. If LM Studio connection fails (ConnectionError):
           - Shows st.warning with connection details
           - Returns None (allows app to run without guardrails)
           - Common cause: LM Studio not running or wrong URL

        3. If Prompt Guard loading fails:
           - Handled internally in guardian.initialize()
           - Guardian still returns but with Prompt Guard disabled
           - Shows warning in console

        4. Any other initialization error:
           - Shows st.error with error details
           - Returns None (graceful degradation)

    Side Effects:
        - Connects to LM Studio API (test request)
        - Downloads Prompt Guard model from HuggingFace (first time only)
        - Loads Prompt Guard into memory (~350MB)
        - Displays warnings/errors in Streamlit UI

    Performance:
        - First call (cold): 2-5 seconds (model loading)
        - First call (Prompt Guard download): 30-60 seconds (one-time)
        - Cached calls: <0.01 seconds (instant)

    Example Output on Success:
        (Console output from guardian.initialize())
        Initializing Input Guardian...
          LM Studio URL: http://192.168.7.171:1234
          Model: llama-guard-3-8b-imat
          Connected to LM Studio
          Safety threshold: 0.7
          Loading Prompt Guard model...
          Prompt Guard loaded on mps
        Input Guardian initialized successfully!

    Example Output on LM Studio Failure:
        (Streamlit warning displayed)
        Guardrails disabled: Cannot connect to LM Studio at http://...
        Ensure LM Studio is running and accessible

    Design Rationale:
        - Never crashes the app due to guardrail failures
        - Allows users to process tickets even if safety checks unavailable
        - Makes LM Studio dependency optional (convenient for development)
        - Provides clear feedback about what went wrong
    """
    # If master switch is off, don't initialize anything
    if not enable_guardrails:
        return None

    # Get configuration
    config = get_config()

    # Apply checkbox settings to config before initialization
    # This allows independent control of each guardrail layer
    config.enable_llama_guard = enable_llama_guard
    config.enable_prompt_guard = enable_prompt_guard

    # Create guardian instance
    guardian = InputGuardian(config)

    try:
        # Attempt initialization (connects to LM Studio, loads Prompt Guard)
        guardian.initialize()
        return guardian

    except ConnectionError as e:
        # LM Studio connection failed (not running or wrong URL)
        # Show warning but don't crash - app can run without guardrails
        st.warning(f"Guardrails disabled: {str(e)}")
        return None

    except Exception as e:
        # Unexpected error during initialization
        # Show error and return None for graceful degradation
        st.error(f"Guardrails initialization failed: {str(e)}")
        return None


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_header():
    """
    Render the application header section.

    This function creates the top section of the application with the title
    and subtitle. Uses Streamlit's markdown rendering for formatted text.

    Components:
        - st.title(): Large heading with emoji
        - st.markdown(): Subtitle text
        - st.divider(): Horizontal line separator

    Side Effects:
        - Displays header in Streamlit UI
        - Adds spacing with divider

    Design Note:
        Kept simple and clean for professional appearance. The emoji adds
        visual interest without being distracting.
    """
    st.title("🎫 Intelligent Support Ticket Assistant")
    st.markdown("AI-powered ticket resolution using Retrieval-Augmented Generation")
    st.divider()


def render_input_panel():
    """
    Render the left panel containing ticket input and configuration options.

    This function creates the complete input interface for the application,
    including ticket input fields, RAG configuration options, retrieval
    parameters, and safety controls. It returns all user inputs as a dictionary
    for processing in the main application flow.

    Layout Structure:
        1. Ticket Input Section
           - Subject: Single-line text input
           - Body: Multi-line text area (200px height)

        2. Options Section (two columns)
           - Column 1: RAG Mode (radio buttons)
             * Strict: Context-only answers
             * Augmented: Context + LLM knowledge
           - Column 2: Retriever Type (radio buttons)
             * Standard: Single query
             * MultiQuery: Query variations
             * MMR: Diversity-focused

        3. Reference Display Toggle
           - Checkbox: Show/hide reference tickets in response

        4. Retrieval Parameters Section
           - Top-K Initial slider (5-50, step 5)
           - Top-K Reranked slider (1-15, step 1)
           - Dynamic validation: reranked <= initial

        5. Safety Options Section
           - Content Safety Check checkbox (Llama Guard)
           - Prompt Injection Detection checkbox (Prompt Guard)
           - Safety Threshold slider (0.5-0.95, step 0.05)
           - Only visible when either safety option enabled

        6. Action Buttons
           - Submit: Process ticket (primary button)
           - Clear: Reset form and rerun app

    Returns:
        dict: Dictionary containing all user inputs with keys:
            - subject (str): Ticket subject line
            - body (str): Ticket description
            - rag_mode (str): 'strict' or 'augmented'
            - retriever_type (str): 'standard', 'multiquery', or 'mmr'
            - show_references (bool): Whether to display reference tickets
            - top_k_initial (int): Number of tickets from vector search
            - top_k_reranked (int): Number of tickets after re-ranking
            - enable_guardrails (bool): Content safety check enabled
            - enable_prompt_guard (bool): Prompt injection detection enabled
            - safety_threshold (float): Safety score threshold (0.0-1.0)
            - submit (bool): True if Submit button clicked
            - clear (bool): True if Clear button clicked

    UI/UX Design Decisions:
        1. Radio Buttons for Mutually Exclusive Options:
           - RAG mode and retriever type use radio buttons
           - Clear visual separation of options
           - Format_func adds icons and descriptive text

        2. Sliders for Numerical Parameters:
           - Intuitive for range-based values
           - Real-time feedback on selected values
           - Step size chosen for practical granularity

        3. Checkboxes for Independent Boolean Options:
           - Each guardrail layer can be toggled independently
           - Clear on/off state

        4. Conditional Display:
           - Safety threshold only shown when guardrails enabled
           - Reduces UI clutter when options not relevant

        5. Help Text:
           - Every input has help text via help parameter
           - Explains purpose and constraints
           - Users can hover for details

    Validation Logic:
        - top_k_reranked max value: min(15, top_k_initial)
        - top_k_reranked default value: min(5, top_k_initial)
        - Ensures reranked <= initial (can't select more than retrieved)

    Example Return Value:
        {
            'subject': 'VPN disconnecting',
            'body': 'My VPN drops every 15 minutes on Windows 10',
            'rag_mode': 'strict',
            'retriever_type': 'standard',
            'show_references': True,
            'top_k_initial': 20,
            'top_k_reranked': 5,
            'enable_guardrails': True,
            'enable_prompt_guard': True,
            'safety_threshold': 0.7,
            'submit': False,
            'clear': False
        }
    """
    st.header("📝 Ticket Input")

    # Ticket subject input
    subject = st.text_input(
        "Subject",
        placeholder="Brief description of the issue",
        help="Enter a concise subject line for your support request"
    )

    # Ticket body input (multi-line)
    body = st.text_area(
        "Body",
        height=200,
        placeholder="Detailed description of your issue...",
        help="Provide as much detail as possible to get accurate assistance"
    )

    st.subheader("⚙️ Options")

    # Two-column layout for RAG mode and retriever type
    col1, col2 = st.columns(2)

    with col1:
        # RAG Mode selection
        rag_mode = st.radio(
            "RAG Mode",
            options=['strict', 'augmented'],
            format_func=lambda x: {
                'strict': '🔒 Strict (Context Only)',
                'augmented': '🔓 Augmented (+ LLM Knowledge)'
            }[x],
            help="Strict: Answer only from historical tickets | Augmented: Supplement with LLM knowledge"
        )

    with col2:
        # Retriever type selection
        retriever_type = st.radio(
            "Retriever Type",
            options=['standard', 'multiquery', 'mmr'],
            format_func=lambda x: {
                'standard': '📍 Standard',
                'multiquery': '🔄 MultiQuery',
                'mmr': '🎯 MMR (Diverse)'
            }[x],
            help="Standard: Single query | MultiQuery: Multiple query variations | MMR: Balances relevance with diversity"
        )

    # Reference ticket display toggle
    show_references = st.checkbox(
        "Show Reference Tickets",
        value=True,
        help="Display the historical tickets used to generate the answer"
    )

    st.subheader("🎛️ Retrieval Parameters")

    # Top-K initial retrieval slider
    top_k_initial = st.slider(
        "Tickets Retrieved (Vector Search)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of tickets to retrieve from vector database before re-ranking"
    )

    # Top-K reranked slider (dynamically constrained by top_k_initial)
    top_k_reranked = st.slider(
        "Tickets Selected (After Re-ranking)",
        min_value=1,
        max_value=min(15, top_k_initial),  # Can't rerank more than retrieved
        value=min(5, top_k_initial),       # Default 5 or less if initial < 5
        step=1,
        help="Number of top tickets to use after cross-encoder re-ranking (must be ≤ retrieved tickets)"
    )

    st.subheader("🛡️ Safety Options")

    # Content safety check toggle (Llama Guard)
    enable_guardrails = st.checkbox(
        "Enable Content Safety Check",
        value=True,
        help="Screen input for malicious intent and abusive language using Llama Guard 3 8B via LM Studio"
    )

    # Prompt injection detection toggle (Prompt Guard)
    enable_prompt_guard = st.checkbox(
        "Enable Prompt Injection Detection",
        value=True,
        help="Detect prompt injection attacks using Llama Prompt Guard 2 86M (requires transformers)"
    )

    # Safety threshold (only shown when at least one guardrail enabled)
    if enable_guardrails:
        safety_threshold = st.slider(
            "Safety Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Higher = stricter validation (0.7 recommended)"
        )
    else:
        # Default value when not displayed
        safety_threshold = 0.7

    st.divider()

    # Action buttons (two columns for side-by-side layout)
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Submit", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

    # Return all inputs as dictionary
    return {
        'subject': subject,
        'body': body,
        'rag_mode': rag_mode,
        'retriever_type': retriever_type,
        'show_references': show_references,
        'top_k_initial': top_k_initial,
        'top_k_reranked': top_k_reranked,
        'enable_guardrails': enable_guardrails,
        'enable_prompt_guard': enable_prompt_guard,
        'safety_threshold': safety_threshold,
        'submit': submit,
        'clear': clear
    }


def render_response_panel(response, show_references):
    """
    Render the right panel displaying AI-generated response and metadata.

    This function displays the complete RAG pipeline output including the
    generated answer, performance metrics, and optionally the reference tickets
    used to generate the answer.

    Args:
        response: Dictionary returned from pipeline.process_ticket() containing:
            - answer (str): AI-generated solution text
            - generated_queries (List[str] or None): Query variations if multiquery mode
            - metrics (PerformanceMetrics): Timing and confidence data
            - references (List[dict]): Retrieved tickets with metadata
                Each reference contains:
                - subject (str): Ticket subject
                - body (str): Ticket content
                - category (str): Ticket category
                - priority (str): Ticket priority
                - score (float): Cross-encoder relevance score
        show_references: Whether to display reference tickets section

    Layout Structure:
        1. Header: "AI-Generated Response"

        2. Generated Query Variations (if multiquery mode)
           - Expandable section (collapsed by default)
           - Shows original query + variations
           - Helps users understand multiquery strategy

        3. Answer Text
           - Markdown-rendered for formatting
           - Main AI-generated solution

        4. Performance Metrics (3 columns)
           - Column 1: Total Response Time
           - Column 2: Confidence Score
           - Column 3: Number of Tickets Retrieved

        5. Detailed Timing (expandable, collapsed by default)
           - Vector search time
           - Re-ranking time
           - LLM generation time
           - RAG mode used

        6. Reference Tickets (expandable, expanded by default, if enabled)
           - Only shown if show_references=True and references exist
           - For each ticket:
             * Ticket number and relevance score
             * Subject, category, priority (2 columns)
             * Content (read-only text area)
             * Divider between tickets

    Display Format Examples:
        Generated Queries (if multiquery):
            MultiQuery mode generated these search variations:
            1. Original Query:
               VPN keeps disconnecting every 15 minutes
            2. Variation 1:
               VPN connection drops frequently on corporate network
            3. Variation 2:
               Intermittent VPN disconnection issues Windows 10

        Metrics:
            Response Time: 1.23s
            Confidence: 87%
            Retrieved: 20 tickets

        Detailed Timing:
            Vector Search: 0.45s
            Re-ranking: 0.32s
            LLM Generation: 0.46s
            Mode: Strict

        Reference Ticket:
            Ticket #1 - Relevance: 8.45
            Subject: VPN disconnection troubleshooting
            Category: Network    Priority: High
            [Content in read-only text area]

    Side Effects:
        - Displays response content in Streamlit UI
        - Creates expandable sections
        - Renders markdown content

    Design Decisions:
        1. Metrics in Columns:
           - Visual separation of key metrics
           - Easy to scan and compare

        2. Expandable Sections:
           - Detailed timing collapsed by default (advanced users)
           - Query variations collapsed (supplementary info)
           - References expanded (important context)
           - Reduces initial visual clutter

        3. Read-Only Text Areas for Content:
           - Preserves formatting and line breaks
           - Scroll bars for long content
           - Disabled state prevents editing
           - Unique keys prevent Streamlit warnings

        4. Score Display Format:
           - Relevance shown as float with 2 decimals
           - Raw cross-encoder scores (typically -10 to +10)
           - Higher = more relevant

    Performance Notes:
        - Markdown rendering is fast (<1ms)
        - Text areas render instantly
        - No heavy computations in display logic
    """
    st.header("💡 AI-Generated Response")

    # Display generated query variations if MultiQuery mode was used
    if response.get('generated_queries') and len(response['generated_queries']) > 1:
        with st.expander("🔄 Generated Query Variations", expanded=False):
            st.markdown("**MultiQuery mode generated these search variations:**")
            for i, query in enumerate(response['generated_queries'], 1):
                if i == 1:
                    st.markdown(f"**{i}. Original Query:**")
                else:
                    st.markdown(f"**{i}. Variation {i-1}:**")
                st.text(query)
                # Add divider between queries (except after last one)
                if i < len(response['generated_queries']):
                    st.divider()

    # Display the AI-generated answer
    st.markdown(response['answer'])

    # Performance metrics section
    st.divider()

    # Convert metrics to dictionary for display
    metrics = response['metrics'].to_dict()

    # Display key metrics in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⏱️ Response Time", metrics['Total Response Time'])
    with col2:
        st.metric("📊 Confidence", metrics['Confidence'])
    with col3:
        st.metric("🔍 Retrieved", metrics['Tickets Retrieved'])

    # Detailed timing breakdown (collapsed by default)
    with st.expander("⏰ Detailed Timing", expanded=False):
        timing_col1, timing_col2 = st.columns(2)
        with timing_col1:
            st.text(f"Vector Search: {metrics['Vector Search']}")
            st.text(f"Re-ranking: {metrics['Re-ranking']}")
        with timing_col2:
            st.text(f"LLM Generation: {metrics['LLM Generation']}")
            st.text(f"Mode: {metrics['Mode']}")

    # Reference tickets section (only if enabled and references exist)
    if show_references and response.get('references'):
        with st.expander("📚 Reference Tickets", expanded=True):
            for i, ref in enumerate(response['references'], 1):
                st.markdown(f"**Ticket #{i}** - Relevance: {ref['score']:.2f}")

                # Display metadata in two columns
                ref_col1, ref_col2 = st.columns([1, 1])
                with ref_col1:
                    st.text(f"Subject: {ref['subject']}")
                    st.text(f"Category: {ref['category']}")
                with ref_col2:
                    st.text(f"Priority: {ref['priority']}")

                # Display ticket content in read-only text area
                with st.container():
                    st.text_area(
                        f"Content",
                        value=ref['body'],
                        height=100,
                        disabled=True,  # Read-only
                        key=f"ref_{i}",  # Unique key to avoid Streamlit warnings
                        label_visibility="collapsed"  # Hide redundant label
                    )

                # Add divider between tickets (except after last one)
                if i < len(response['references']):
                    st.divider()


def render_validation_panel(validation: ValidationResult):
    """
    Render input validation results from the guardrail system.

    This function displays the safety validation outcome in a clear, color-coded
    format with detailed information about any detected violations.

    Args:
        validation: ValidationResult from guardian.validate_ticket() containing:
            - is_safe (bool): Whether input passed validation
            - safety_score (float): Overall safety score (0.0-1.0)
            - violations (List[SafetyViolation]): Detected violations
            - processing_time (float): Validation duration in seconds
            - timestamp (datetime): When validation was performed

    Display Logic:
        If Safe:
            - Green success box with checkmark
            - Shows safety score as percentage
            - Format: "Input Validated - Safety Score: 95%"

        If Unsafe:
            - Red error box with alert icon
            - Shows safety score as percentage
            - Lists all violations with details
            - Format: "Safety Violations Detected - Score: 45%"

    Violation Display Format:
        For each violation:
        - Icon based on severity (see severity_icons mapping)
        - Category name (formatted: underscores to spaces, title case)
        - Severity level (uppercase)
        - Confidence score (percentage)
        - Explanation (what was detected and why)

        Example:
            🔴 Prompt Injection
            - Severity: CRITICAL
            - Confidence: 98.5%
            - Issue: Prompt injection detected by Llama Prompt Guard (confidence: 98.5%)

    Severity Icon Mapping:
        - critical: 🔴 (red circle)
        - high: 🟠 (orange circle)
        - medium: 🟡 (yellow circle)
        - low: 🟢 (green circle)
        - unknown: ⚪ (white circle)

    Validation Metrics Display:
        - Expandable section (collapsed by default)
        - Processing time in seconds (2 decimal places)
        - Timestamp in YYYY-MM-DD HH:MM:SS format

    Side Effects:
        - Displays validation results in Streamlit UI
        - Uses st.success() for safe content (green box)
        - Uses st.error() for unsafe content (red box)
        - Uses st.warning() for individual violations (yellow boxes)
        - Creates expandable metrics section

    Color Coding Purpose:
        - Green (success): User can proceed immediately
        - Red (error): Clear visual indication of blocking
        - Yellow (warning): Detailed violation info without additional alarm
        - Helps users quickly understand validation outcome

    Example Output (Safe):
        [Green box]
        ✅ Input Validated - Safety Score: 95%

        ⏱️ Validation Metrics (collapsed)
        Processing Time: 0.23s
        Timestamp: 2025-01-15 14:32:10

    Example Output (Unsafe):
        [Red box]
        🚨 Safety Violations Detected - Score: 45%

        [Yellow box]
        🔴 Prompt Injection
        - Severity: CRITICAL
        - Confidence: 98.5%
        - Issue: Prompt injection detected by Llama Prompt Guard (confidence: 98.5%)

        [Yellow box]
        🟠 Llama Guard Violation
        - Severity: HIGH
        - Confidence: 85%
        - Issue: Llama Guard detected unsafe content: S1

        ⏱️ Validation Metrics (collapsed)
        Processing Time: 0.45s
        Timestamp: 2025-01-15 14:32:10

    Design Rationale:
        - Clear visual distinction between safe and unsafe
        - Detailed information for unsafe content (helps users understand why)
        - Severity icons provide quick visual scanning
        - Collapsed metrics reduce clutter for successful validations
        - Consistent formatting across different violation types
    """
    # Display overall validation result with appropriate styling
    if validation.is_safe:
        # Safe content: green success box
        st.success(f"✅ Input Validated - Safety Score: {validation.safety_score:.1%}")
    else:
        # Unsafe content: red error box
        st.error(f"🚨 Safety Violations Detected - Score: {validation.safety_score:.1%}")

        # Display details for each violation
        for violation in validation.violations:
            # Map severity to emoji icons for visual distinction
            severity_icons = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }

            # Get appropriate icon (default to white circle if unknown severity)
            icon = severity_icons.get(violation.severity, '⚪')

            # Format category name: underscores to spaces, title case
            # Example: "prompt_injection" -> "Prompt Injection"
            category_display = violation.category.replace('_', ' ').title()

            # Display violation in yellow warning box
            st.warning(f"""
**{icon} {category_display}**
- Severity: {violation.severity.upper()}
- Confidence: {violation.confidence:.1%}
- Issue: {violation.explanation}
            """)

    # Validation metrics (collapsed by default to reduce clutter)
    with st.expander("⏱️ Validation Metrics", expanded=False):
        st.text(f"Processing Time: {validation.processing_time:.2f}s")
        st.text(f"Timestamp: {validation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def render_empty_state():
    """
    Render empty state when no response has been generated yet.

    This function displays helpful information and examples to guide users
    when they first load the application or after clicking Clear.

    Display Components:
        1. Header: "AI-Generated Response"

        2. Info Box: Brief instruction to enter ticket and submit

        3. How It Works Section:
           - Numbered steps explaining the workflow
           - Covers input, configuration, and processing

        4. Example Tickets Section:
           - Three realistic examples with subjects and bodies
           - Helps users understand what input format works best
           - Examples cover common IT support scenarios

    Example Tickets Provided:
        1. VPN Connection Issue:
           - Subject: VPN keeps disconnecting
           - Body: My VPN connection drops every 15 minutes using Cisco AnyConnect on Windows 10
           - Category: Network connectivity

        2. Email Problem:
           - Subject: Cannot send emails from Outlook
           - Body: Outlook receives emails but cannot send. Getting error 550 5.7.1 Relaying denied
           - Category: Email server

        3. Software Installation:
           - Subject: Need admin rights for software
           - Body: Need to install Adobe Acrobat Pro but don't have admin permissions
           - Category: Permissions/access

    Side Effects:
        - Displays empty state content in Streamlit UI
        - Uses st.info() for gentle guidance (blue info box)
        - Uses st.markdown() for formatted content

    Design Purpose:
        - Reduces confusion for first-time users
        - Provides concrete examples to help users get started
        - Explains the application workflow clearly
        - Fills empty space with useful information
        - Encourages engagement by showing example use cases

    When This Is Displayed:
        - On initial page load (before any submission)
        - After clicking Clear button (form reset)
        - When submit=False in render_input_panel return value
    """
    st.header("💡 AI-Generated Response")
    st.info("👈 Enter a support ticket and click Submit to get AI-powered assistance")

    st.markdown("""
    ### How it works:

    1. **Enter your ticket** - Provide a subject and detailed description
    2. **Choose RAG mode** - Select strict (context-only) or augmented (+ LLM knowledge)
    3. **Choose retriever** - Standard, MultiQuery (multiple variations), or MMR (diverse results)
    4. **Get AI assistance** - Receive intelligent solutions based on historical tickets

    ### Example tickets to try:

    **VPN Connection Issue**
    - Subject: VPN keeps disconnecting
    - Body: My VPN connection drops every 15 minutes using Cisco AnyConnect on Windows 10

    **Email Problem**
    - Subject: Cannot send emails from Outlook
    - Body: Outlook receives emails but cannot send. Getting error 550 5.7.1 Relaying denied

    **Software Installation**
    - Subject: Need admin rights for software
    - Body: Need to install Adobe Acrobat Pro but don't have admin permissions
    """)


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """
    Main application entry point and control flow.

    This function orchestrates the entire application, coordinating all
    components and handling the complete user interaction workflow.

    Execution Flow:
        1. Render header
        2. Create two-column layout (input | response)
        3. Render input panel in left column (collect user inputs)
        4. Initialize RAG pipeline with user-selected configuration
        5. Initialize input guardian (if any guardrail enabled)
        6. In right column, handle user actions:
           - Clear button: Rerun app (reset form)
           - Submit button:
             a. Validate inputs (subject and body required)
             b. Run safety validation (if guardian exists)
             c. Display validation results
             d. Block if unsafe (st.stop())
             e. Process ticket through RAG pipeline
             f. Display response with metrics and references
           - No action: Show empty state with instructions

    Error Handling Strategy:
        The application uses defensive error handling at each stage:

        1. Pipeline Initialization:
           - FileNotFoundError: Vector database not found
             * Shows detailed error with build instructions
             * Stops execution (cannot proceed without DB)

           - Generic Exception: Initialization failed
             * Shows error message with details
             * Stops execution (cannot proceed without pipeline)

        2. Guardian Initialization:
           - Handled in load_guardian() function
           - Never raises exceptions (graceful degradation)
           - Returns None if initialization fails
           - App continues without guardrails

        3. Validation:
           - Wrapped in try-except block
           - Catches all exceptions during validation
           - Shows warning and proceeds without safety check
           - Never blocks due to validation errors

        4. Ticket Processing:
           - Wrapped in try-except block
           - Catches all processing exceptions
           - Displays error message with full stack trace
           - Does not crash the application

    Layout Structure:
        +---------------------------------------------------------------+
        |                         Header                                |
        +---------------------------------------------------------------+
        |  Left Column (1)       |   Right Column (1.5)                |
        |  render_input_panel()  |   If submit:                        |
        |                        |     - Validation results            |
        |                        |     - Response or error             |
        |                        |   Else:                             |
        |                        |     - Empty state                   |
        +---------------------------------------------------------------+

    Column Width Ratio:
        - Left (input): 1 unit
        - Right (response): 1.5 units
        - Total: 2.5 units
        - Gives more space to response for better readability

    Streamlit Execution Model:
        Streamlit reruns the entire script on every user interaction:
        - Button clicks trigger full rerun
        - Slider changes trigger full rerun
        - Text input triggers rerun on Enter or blur
        - Cached functions (@st.cache_resource) skip reexecution

    State Management:
        - No explicit session state needed for basic operation
        - Cached resources persist across reruns
        - Form inputs reset when st.rerun() called
        - User interactions captured in inputs dictionary

    Performance Characteristics:
        - First run: 5-15 seconds (load all models)
        - Subsequent runs: <1 second (cached models)
        - Validation: 0.1-0.5 seconds (if enabled)
        - Ticket processing: 1-3 seconds (depends on complexity)

    Side Effects:
        - Renders entire UI
        - Loads models into memory (via cached functions)
        - Connects to LM Studio API
        - Processes user inputs
        - Displays results in Streamlit interface

    Example User Flow:
        1. User loads page -> Header + Input panel + Empty state displayed
        2. User enters ticket and clicks Submit
        3. Pipeline initialized (or retrieved from cache)
        4. Guardian initialized (or retrieved from cache)
        5. Validation runs (if enabled)
        6. If safe, ticket processed
        7. Response displayed with metrics and references
        8. User clicks Clear -> Page reloads with empty form
    """
    # Render application header
    render_header()

    # Create two-column layout for input (left) and response (right)
    # Ratio 1:1.5 gives more space to response panel
    left_col, right_col = st.columns([1, 1.5])

    # LEFT COLUMN: Render input panel and collect user configuration
    with left_col:
        inputs = render_input_panel()

    # STEP 1: Initialize RAG Pipeline
    # This is where vector DB, embeddings, re-ranker, and LLM are loaded
    try:
        with st.spinner("🔧 Initializing RAG pipeline..."):
            pipeline, config = load_pipeline(
                rag_mode=inputs['rag_mode'],
                retriever_type=inputs['retriever_type']
            )

    except FileNotFoundError:
        # Vector database doesn't exist - show build instructions
        st.error("""
        ❌ **Vector database not found!**

        Please build the vector database first:

        ```bash
        python build_vectordb.py
        ```

        This will create the ChromaDB index from your ticket dataset.
        """)
        st.stop()  # Halt execution - cannot proceed without DB

    except Exception as e:
        # Generic initialization error
        st.error(f"❌ **Initialization failed:** {str(e)}")
        st.stop()  # Halt execution

    # STEP 2: Initialize Input Guardian (if any guardrail enabled)
    # Guardian initialization never raises exceptions (graceful degradation)
    guardian = None
    if inputs['enable_guardrails'] or inputs['enable_prompt_guard']:
        with st.spinner("🛡️ Initializing input guardian..."):
            guardian = load_guardian(
                enable_guardrails=True,  # Keep True to avoid early return in load_guardian
                enable_llama_guard=inputs['enable_guardrails'],
                enable_prompt_guard=inputs['enable_prompt_guard']
            )

    # RIGHT COLUMN: Handle user actions and display results
    with right_col:
        # Handle Clear button: reset form by rerunning app
        if inputs['clear']:
            st.rerun()

        # Handle Submit button: validate and process ticket
        if inputs['submit']:
            # Validate that required fields are filled
            if not inputs['subject'].strip() or not inputs['body'].strip():
                st.error("⚠️ Please provide both subject and body")
                render_empty_state()
            else:
                # STEP 3: Safety Validation (if guardian exists)
                validation = None
                if guardian:
                    with st.spinner("🛡️ Validating input for safety..."):
                        try:
                            # Update safety threshold from user input
                            config.safety_threshold = inputs['safety_threshold']

                            # Run validation
                            validation = guardian.validate_ticket(
                                subject=inputs['subject'],
                                body=inputs['body']
                            )

                            # Display validation results
                            render_validation_panel(validation)

                            # Block if unsafe
                            if not validation.is_safe:
                                st.stop()  # Halt execution - unsafe content

                        except Exception as e:
                            # Validation failed due to error - show warning but continue
                            st.warning(f"⚠️ Guardrail validation failed: {str(e)}")
                            st.warning("Proceeding without safety validation")

                # STEP 4: Process Ticket (only if safe or guardrails disabled)
                with st.spinner("🤔 Analyzing your ticket and generating solution..."):
                    try:
                        # Run RAG pipeline
                        response = pipeline.process_ticket(
                            subject=inputs['subject'],
                            body=inputs['body'],
                            top_k_initial=inputs['top_k_initial'],
                            top_k_reranked=inputs['top_k_reranked']
                        )

                        # Display response with metrics and references
                        render_response_panel(response, inputs['show_references'])

                    except Exception as e:
                        # Processing failed - show error with stack trace
                        st.error(f"❌ **Error processing ticket:** {str(e)}")
                        st.exception(e)  # Full stack trace for debugging
        else:
            # No action taken - show empty state with instructions
            render_empty_state()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
