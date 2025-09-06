# Ticket Triage System

A Streamlit-based interface for reviewing and managing support tickets from a CSV dataset.

## Features

- 📋 View ticket subjects and bodies in a clean interface
- ⬅️➡️ Navigate through tickets with Previous/Next buttons
- ⏮️⏭️ Jump to first or last ticket quickly
- 📊 Sidebar statistics showing ticket distribution
- 🔢 Direct navigation to any ticket number
- 📎 Expandable section for additional ticket information
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

- **Main View**: Displays the current ticket's subject and body
- **Navigation Bar**: Contains First, Previous, Next, and Last buttons for easy navigation
- **Sidebar**: Shows total tickets, current position, and ticket statistics
- **Jump to Ticket**: Enter a specific ticket number to navigate directly
- **Additional Information**: Expandable section showing all other ticket fields

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