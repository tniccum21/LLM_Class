import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Ticket Triage System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .ticket-header {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .ticket-body {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        min-height: 300px;
    }
    .nav-button {
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Load the CSV file
@st.cache_data
def load_data():
    csv_path = os.path.join('IT_Tickets', 'dataset-tickets-multi-lang3-4k.csv')
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error(f"CSV file not found at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# Main app
def main():
    st.title("🎫 Ticket Triage System")
    st.markdown("Review and manage support tickets efficiently")
    
    # Load data
    df = load_data()
    
    if df is None:
        return
    
    # Sidebar with statistics
    with st.sidebar:
        st.header("📊 Ticket Statistics")
        st.metric("Total Tickets", len(df))
        st.metric("Current Ticket", st.session_state.current_index + 1)
        
        # Quick stats
        if 'priority' in df.columns:
            st.subheader("Priority Distribution")
            priority_counts = df['priority'].value_counts()
            for priority, count in priority_counts.items():
                st.write(f"**{priority}**: {count}")
        
        if 'language' in df.columns:
            st.subheader("Language Distribution")
            lang_counts = df['language'].value_counts().head(5)
            for lang, count in lang_counts.items():
                st.write(f"**{lang}**: {count}")
        
        # Jump to ticket
        st.subheader("Jump to Ticket")
        jump_to = st.number_input(
            "Ticket Number", 
            min_value=1, 
            max_value=len(df), 
            value=st.session_state.current_index + 1
        )
        if st.button("Go"):
            st.session_state.current_index = jump_to - 1
            st.rerun()
    
    # Navigation buttons at the top
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True):
            st.session_state.current_index = 0
            st.rerun()
    
    with col2:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
    
    with col3:
        st.markdown(f"<center><h3>Ticket {st.session_state.current_index + 1} of {len(df)}</h3></center>", unsafe_allow_html=True)
    
    with col4:
        if st.button("Next ➡️", use_container_width=True):
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    with col5:
        if st.button("Last ⏭️", use_container_width=True):
            st.session_state.current_index = len(df) - 1
            st.rerun()
    
    # Display current ticket
    st.markdown("---")
    
    current_ticket = df.iloc[st.session_state.current_index]
    
    # Ticket metadata in columns
    meta_cols = st.columns(4)
    
    with meta_cols[0]:
        if 'type' in df.columns:
            st.metric("Type", current_ticket.get('type', 'N/A'))
    
    with meta_cols[1]:
        if 'priority' in df.columns:
            priority = current_ticket.get('priority', 'N/A')
            color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(str(priority).lower(), '⚪')
            st.metric("Priority", f"{color} {priority}")
    
    with meta_cols[2]:
        if 'language' in df.columns:
            st.metric("Language", current_ticket.get('language', 'N/A'))
    
    with meta_cols[3]:
        if 'queue' in df.columns:
            st.metric("Queue", current_ticket.get('queue', 'N/A'))
    
    # Subject
    st.markdown("### 📋 Subject")
    subject_container = st.container()
    with subject_container:
        st.info(current_ticket.get('subject', 'No subject available'))
    
    # Body
    st.markdown("### 📝 Body")
    body_container = st.container()
    with body_container:
        body_text = current_ticket.get('body', 'No body content available')
        st.text_area("", value=body_text, height=300, disabled=True, key=f"body_{st.session_state.current_index}")
    
    # Additional ticket information (collapsible)
    with st.expander("📎 Additional Information"):
        # Display other columns
        other_cols = [col for col in df.columns if col not in ['subject', 'body']]
        
        for col in other_cols:
            value = current_ticket.get(col, 'N/A')
            if pd.notna(value) and str(value).strip():
                st.write(f"**{col.replace('_', ' ').title()}:** {value}")
    
    # Keyboard shortcuts hint
    st.markdown("---")
    st.caption("💡 Tip: Use the navigation buttons above to browse through tickets. You can also jump to a specific ticket using the sidebar.")

if __name__ == "__main__":
    main()