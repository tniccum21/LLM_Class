# Ticket Triage Application

## Structure
- **Main Application**: Week02/ticket_triage_app.py - Streamlit app with two-column layout
- **Core Functions**: 
  - `load_data()`: Loads tickets from Excel files
  - `call_lm_studio()`: AI integration for ticket analysis
  - `main()`: Streamlit app with category/priority/agent assignment
- **Data Source**: Week02/IT_Tickets/*.xlsx files
- **Dependencies**: Streamlit, pandas, OpenAI API, googletrans

## Current State
- Modified file tracked in git (Week02/ticket_triage_app.py)
- Recent commits show AI integration and translation capabilities
- App provides ticket categorization, priority assignment, and agent routing