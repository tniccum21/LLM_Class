"""
Week07 - Intelligent Support Ticket Assistant (Streamlit App)

Production Streamlit application for AI-powered support ticket resolution
using RAG with two-stage retrieval and LangChain integration.
"""

import streamlit as st
from pathlib import Path

from config import get_config
from rag_pipeline import RAGPipeline
from input_guardian import InputGuardian, ValidationResult


# Page configuration
st.set_page_config(
    page_title="Intelligent Support Ticket Assistant",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_pipeline(rag_mode='strict', retriever_type='standard'):
    """Load and initialize RAG pipeline (cached for performance)"""
    config = get_config()
    config.rag_mode = rag_mode
    config.retriever_type = retriever_type
    pipeline = RAGPipeline(config)
    pipeline.initialize()
    return pipeline, config


@st.cache_resource
def load_guardian(enable_guardrails=True, enable_llama_guard=True, enable_prompt_guard=True):
    """Load and initialize input guardian (cached for performance)"""
    if not enable_guardrails:
        return None

    config = get_config()
    # Apply checkbox settings before initialization
    config.enable_llama_guard = enable_llama_guard
    config.enable_prompt_guard = enable_prompt_guard
    guardian = InputGuardian(config)

    try:
        guardian.initialize()
        return guardian
    except ConnectionError as e:
        st.warning(f"⚠️ Guardrails disabled: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Guardrails initialization failed: {str(e)}")
        return None


def render_header():
    """Render application header"""
    st.title("🎫 Intelligent Support Ticket Assistant")
    st.markdown("AI-powered ticket resolution using Retrieval-Augmented Generation")
    st.divider()


def render_input_panel():
    """Render left panel with ticket input and options"""
    st.header("📝 Ticket Input")

    subject = st.text_input(
        "Subject",
        placeholder="Brief description of the issue",
        help="Enter a concise subject line for your support request"
    )

    body = st.text_area(
        "Body",
        height=200,
        placeholder="Detailed description of your issue...",
        help="Provide as much detail as possible to get accurate assistance"
    )

    st.subheader("⚙️ Options")

    col1, col2 = st.columns(2)

    with col1:
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

    show_references = st.checkbox(
        "Show Reference Tickets",
        value=True,
        help="Display the historical tickets used to generate the answer"
    )

    st.subheader("🎛️ Retrieval Parameters")

    top_k_initial = st.slider(
        "Tickets Retrieved (Vector Search)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of tickets to retrieve from vector database before re-ranking"
    )

    top_k_reranked = st.slider(
        "Tickets Selected (After Re-ranking)",
        min_value=1,
        max_value=min(15, top_k_initial),
        value=min(5, top_k_initial),
        step=1,
        help="Number of top tickets to use after cross-encoder re-ranking (must be ≤ retrieved tickets)"
    )

    st.subheader("🛡️ Safety Options")

    enable_guardrails = st.checkbox(
        "Enable Content Safety Check",
        value=True,
        help="Screen input for malicious intent and abusive language using Llama Guard 3 8B via LM Studio"
    )

    enable_prompt_guard = st.checkbox(
        "Enable Prompt Injection Detection",
        value=True,
        help="Detect prompt injection attacks using Llama Prompt Guard 2 86M (requires transformers)"
    )

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
        safety_threshold = 0.7

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Submit", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

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
    """Render right panel with AI-generated response"""
    st.header("💡 AI-Generated Response")

    # Display generated queries if MultiQuery mode was used
    if response.get('generated_queries') and len(response['generated_queries']) > 1:
        with st.expander("🔄 Generated Query Variations", expanded=False):
            st.markdown("**MultiQuery mode generated these search variations:**")
            for i, query in enumerate(response['generated_queries'], 1):
                if i == 1:
                    st.markdown(f"**{i}. Original Query:**")
                else:
                    st.markdown(f"**{i}. Variation {i-1}:**")
                st.text(query)
                if i < len(response['generated_queries']):
                    st.divider()

    # Display answer
    st.markdown(response['answer'])

    # Performance metrics
    st.divider()

    metrics = response['metrics'].to_dict()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⏱️ Response Time", metrics['Total Response Time'])
    with col2:
        st.metric("📊 Confidence", metrics['Confidence'])
    with col3:
        st.metric("🔍 Retrieved", metrics['Tickets Retrieved'])

    # Detailed timing
    with st.expander("⏰ Detailed Timing", expanded=False):
        timing_col1, timing_col2 = st.columns(2)
        with timing_col1:
            st.text(f"Vector Search: {metrics['Vector Search']}")
            st.text(f"Re-ranking: {metrics['Re-ranking']}")
        with timing_col2:
            st.text(f"LLM Generation: {metrics['LLM Generation']}")
            st.text(f"Mode: {metrics['Mode']}")

    # Reference tickets
    if show_references and response.get('references'):
        with st.expander("📚 Reference Tickets", expanded=True):
            for i, ref in enumerate(response['references'], 1):
                st.markdown(f"**Ticket #{i}** - Relevance: {ref['score']:.2f}")

                ref_col1, ref_col2 = st.columns([1, 1])
                with ref_col1:
                    st.text(f"Subject: {ref['subject']}")
                    st.text(f"Category: {ref['category']}")
                with ref_col2:
                    st.text(f"Priority: {ref['priority']}")

                with st.container():
                    st.text_area(
                        f"Content",
                        value=ref['body'],
                        height=100,
                        disabled=True,
                        key=f"ref_{i}",
                        label_visibility="collapsed"
                    )

                if i < len(response['references']):
                    st.divider()


def render_validation_panel(validation: ValidationResult):
    """Render input validation results"""

    if validation.is_safe:
        st.success(f"✅ Input Validated - Safety Score: {validation.safety_score:.1%}")
    else:
        st.error(f"🚨 Safety Violations Detected - Score: {validation.safety_score:.1%}")

        for violation in validation.violations:
            severity_icons = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }

            icon = severity_icons.get(violation.severity, '⚪')

            st.warning(f"""
**{icon} {violation.category.replace('_', ' ').title()}**
- Severity: {violation.severity.upper()}
- Confidence: {violation.confidence:.1%}
- Issue: {violation.explanation}
            """)

    with st.expander("⏱️ Validation Metrics", expanded=False):
        st.text(f"Processing Time: {validation.processing_time:.2f}s")
        st.text(f"Timestamp: {validation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def render_empty_state():
    """Render empty state when no response yet"""
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


def main():
    """Main application logic"""
    # Render header first
    render_header()

    # Two-column layout for input
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        inputs = render_input_panel()

    # Initialize pipeline with selected options
    try:
        with st.spinner("🔧 Initializing RAG pipeline..."):
            pipeline, config = load_pipeline(
                rag_mode=inputs['rag_mode'],
                retriever_type=inputs['retriever_type']
            )

    except FileNotFoundError:
        st.error("""
        ❌ **Vector database not found!**

        Please build the vector database first:

        ```bash
        python build_vectordb.py
        ```

        This will create the ChromaDB index from your ticket dataset.
        """)
        st.stop()

    except Exception as e:
        st.error(f"❌ **Initialization failed:** {str(e)}")
        st.stop()

    # Initialize guardian if ANY guardrail is enabled
    guardian = None
    if inputs['enable_guardrails'] or inputs['enable_prompt_guard']:
        with st.spinner("🛡️ Initializing input guardian..."):
            guardian = load_guardian(
                enable_guardrails=True,  # Keep True to avoid early return in load_guardian
                enable_llama_guard=inputs['enable_guardrails'],
                enable_prompt_guard=inputs['enable_prompt_guard']
            )

    with right_col:
        # Handle clear button
        if inputs['clear']:
            st.rerun()

        # Handle submit button
        if inputs['submit']:
            # Validation
            if not inputs['subject'].strip() or not inputs['body'].strip():
                st.error("⚠️ Please provide both subject and body")
                render_empty_state()
            else:
                # Step 1: Guardrails validation (if any guardrail enabled)
                validation = None
                if guardian:
                    with st.spinner("🛡️ Validating input for safety..."):
                        try:
                            # Update safety threshold in config
                            config.safety_threshold = inputs['safety_threshold']

                            validation = guardian.validate_ticket(
                                subject=inputs['subject'],
                                body=inputs['body']
                            )

                            # Display validation results
                            render_validation_panel(validation)

                            # Block if unsafe
                            if not validation.is_safe:
                                st.stop()

                        except Exception as e:
                            st.warning(f"⚠️ Guardrail validation failed: {str(e)}")
                            st.warning("Proceeding without safety validation")

                # Step 2: Process ticket (only if safe or guardrails disabled)
                with st.spinner("🤔 Analyzing your ticket and generating solution..."):
                    try:
                        response = pipeline.process_ticket(
                            subject=inputs['subject'],
                            body=inputs['body'],
                            top_k_initial=inputs['top_k_initial'],
                            top_k_reranked=inputs['top_k_reranked']
                        )

                        # Display response
                        render_response_panel(response, inputs['show_references'])

                    except Exception as e:
                        st.error(f"❌ **Error processing ticket:** {str(e)}")
                        st.exception(e)
        else:
            # Empty state
            render_empty_state()


if __name__ == '__main__':
    main()
