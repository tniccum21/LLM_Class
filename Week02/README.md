# Ticket Triage System

A Streamlit-based interface for reviewing and managing support tickets from a CSV dataset.

## Features

- 📋 Two-column layout: Original data vs AI predictions
- 📄 Left column displays original ticket data with subject and body at top
- 🤖 Right column prepared for AI-generated predictions (currently blank placeholders)
- ⬅️➡️ Navigate through tickets with Previous/Next buttons
- ⏮️⏭️ Jump to first or last ticket quickly
- 📊 Sidebar statistics showing ticket distribution
- 🔢 Direct navigation to any ticket number
- 🌍 Multi-language ticket support

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the Week02 directory:
```bash
cd Week02
```

2. Run the Streamlit application:
```bash
streamlit run ticket_triage_app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Interface Overview

- **Two-Column Layout**: 
  - **Left Column**: Original ticket data with subject and body prominently displayed, followed by all other fields
  - **Right Column**: AI prediction placeholders (ready for future AI integration)
- **Navigation Bar**: Contains First, Previous, Next, and Last buttons for easy navigation
- **Sidebar**: Shows total tickets, current position, and ticket statistics
- **Jump to Ticket**: Enter a specific ticket number to navigate directly

## Data Structure

The application expects a CSV file located at `IT_Tickets/dataset-tickets-multi-lang3-4k.csv` with at minimum:
- `subject`: Ticket subject/title
- `body`: Ticket description/content

Optional fields that will be displayed if present:
- `type`: Ticket type
- `priority`: Ticket priority level
- `language`: Language of the ticket
- `queue`: Assignment queue
- Additional metadata fields