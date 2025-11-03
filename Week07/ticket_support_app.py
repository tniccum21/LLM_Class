"""
Week07 - Intelligent Support Ticket Assistant (Streamlit App)

Production Streamlit application for AI-powered support ticket resolution
using RAG with two-stage retrieval and LangChain integration.
"""

import streamlit as st
from pathlib import Path

from config import get_config
from rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Intelligent Support Ticket Assistant",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_pipeline():
    """Load and initialize RAG pipeline (cached for performance)"""
    config = get_config()
    pipeline = RAGPipeline(config)
    pipeline.initialize()
    return pipeline, config


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

    rag_mode = st.radio(
        "RAG Mode",
        options=['strict', 'augmented'],
        format_func=lambda x: {
            'strict': '🔒 Strict (Context Only)',
            'augmented': '🔓 Augmented (+ LLM Knowledge)'
        }[x],
        help="Strict: Answer only from historical tickets | Augmented: Supplement with LLM knowledge"
    )

    show_references = st.checkbox(
        "Show Reference Tickets",
        value=True,
        help="Display the historical tickets used to generate the answer"
    )

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
        'show_references': show_references,
        'submit': submit,
        'clear': clear
    }


def render_response_panel(response, show_references):
    """Render right panel with AI-generated response"""
    st.header("💡 AI-Generated Response")

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
                st.markdown(f"**Ticket #{i}** - Match: {ref['score']:.1%}")

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


def render_empty_state():
    """Render empty state when no response yet"""
    st.header("💡 AI-Generated Response")
    st.info("👈 Enter a support ticket and click Submit to get AI-powered assistance")

    st.markdown("""
    ### How it works:

    1. **Enter your ticket** - Provide a subject and detailed description
    2. **Choose RAG mode** - Select strict (context-only) or augmented (+ LLM knowledge)
    3. **Get AI assistance** - Receive intelligent solutions based on historical tickets

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
    # Initialize pipeline
    try:
        with st.spinner("🔧 Initializing RAG pipeline..."):
            pipeline, config = load_pipeline()

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

    # Render header
    render_header()

    # Two-column layout
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        inputs = render_input_panel()

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
                # Update config if RAG mode changed
                if inputs['rag_mode'] != config.rag_mode:
                    config.rag_mode = inputs['rag_mode']
                    pipeline._build_chain()  # Rebuild chain with new mode

                # Process ticket
                with st.spinner("🤔 Analyzing your ticket and generating solution..."):
                    try:
                        response = pipeline.process_ticket(
                            subject=inputs['subject'],
                            body=inputs['body']
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
