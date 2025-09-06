import streamlit as st
import pandas as pd
import os
import requests
import json

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
if 'ai_predictions' not in st.session_state:
    st.session_state.ai_predictions = {}

# Load the CSV file
@st.cache_data
def load_data():
    csv_path = './IT_Tickets/dataset-tickets-multi-lang3-4k.csv'
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error(f"CSV file not found at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# Function to call LM Studio LLM
def call_lm_studio(system_content, user_content):
    """
    Basic function to call LM Studio API
    """
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 1200,
        "temperature": 0.3
    }
    
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

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
    
    # Display current ticket in two columns
    st.markdown("---")
    
    current_ticket = df.iloc[st.session_state.current_index]
    
    # Create two main columns
    left_col, right_col = st.columns(2, gap="large")
    
    # LEFT COLUMN - Original Data
    with left_col:
        st.markdown("### 📄 Original Ticket Data")
        
        # Subject at the top
        st.markdown("#### 📋 Subject")
        st.info(current_ticket.get('subject', 'No subject available'))
        
        # Body second
        st.markdown("#### 📝 Body")
        body_text = current_ticket.get('body', 'No body content available')
        st.text_area("", value=body_text, height=250, disabled=True, key=f"orig_body_{st.session_state.current_index}")
        
        # Other fields below
        st.markdown("#### 📎 Additional Fields")
        
        # Get all other columns (excluding subject and body)
        other_cols = [col for col in df.columns if col not in ['subject', 'body']]
        
        for col in other_cols:
            value = current_ticket.get(col, 'N/A')
            if pd.notna(value):
                # Special formatting for certain fields
                if col == 'priority':
                    color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(str(value).lower(), '⚪')
                    st.write(f"**{col.replace('_', ' ').title()}:** {color} {value}")
                else:
                    st.write(f"**{col.replace('_', ' ').title()}:** {value}")
    
    # RIGHT COLUMN - AI Predictions
    with right_col:
        st.markdown("### 🤖 AI Predictions")
        
        # Add AI Analysis button
        if st.button("🔮 Analyze with AI", type="primary", use_container_width=True):
            with st.spinner("Analyzing ticket..."):
                # Call the LM Studio function
                system_prompt = "You are a helpful assistant analyzing support tickets."
                user_prompt = f"Subject: {current_ticket.get('subject', '')}\nBody: {current_ticket.get('body', '')}"
                
                ai_response = call_lm_studio(system_prompt, user_prompt)
                
                # Store the response in session state
                ticket_key = f"ticket_{st.session_state.current_index}"
                st.session_state.ai_predictions[ticket_key] = ai_response
                st.success("AI analysis complete!")
        
        # Check if we have predictions for this ticket
        ticket_key = f"ticket_{st.session_state.current_index}"
        ai_response = st.session_state.ai_predictions.get(ticket_key, "")
        
        # Subject prediction placeholder
        st.markdown("#### 📋 Subject (AI)")
        st.text_input("", value="", placeholder="AI will predict subject...", disabled=True, key=f"ai_subject_{st.session_state.current_index}")
        
        # Body prediction - show AI response if available
        st.markdown("#### 📝 Body (AI)")
        st.text_area("", value=ai_response, placeholder="AI will generate response...", height=250, disabled=True, key=f"ai_body_{st.session_state.current_index}")
        
        # Other fields predictions
        st.markdown("#### 📎 Additional Fields (AI)")
        
        # Create empty placeholders for all other fields
        for col in other_cols:
            field_name = col.replace('_', ' ').title()
            
            # Use appropriate input type based on field
            if col == 'priority':
                st.selectbox(f"**{field_name}:**", ["", "high", "medium", "low"], disabled=True, key=f"ai_{col}_{st.session_state.current_index}")
            elif col == 'language':
                st.selectbox(f"**{field_name}:**", ["", "en", "es", "fr", "de", "ja", "zh", "pt", "it", "ru", "ko"], disabled=True, key=f"ai_{col}_{st.session_state.current_index}")
            else:
                st.text_input(f"**{field_name}:**", value="", placeholder=f"AI prediction for {field_name.lower()}...", disabled=True, key=f"ai_{col}_{st.session_state.current_index}")
    
    # Keyboard shortcuts hint
    st.markdown("---")
    st.caption("💡 Tip: Use the navigation buttons above to browse through tickets. You can also jump to a specific ticket using the sidebar.")

if __name__ == "__main__":
    main()