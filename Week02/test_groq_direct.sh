#!/bin/bash

# Test Groq API directly with curl
# Replace with your actual API key
GROQ_API_KEY="your_key_here"

curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'