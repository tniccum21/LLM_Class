#!/usr/bin/env python3
"""
Ticket Translation Script - Week 05
Converts notebook to command-line script for faster execution

Usage:
    python translate_tickets.py [--limit N] [--provider groq|lm_studio]

Examples:
    python translate_tickets.py --limit 100               # Test with 100 tickets
    python translate_tickets.py                           # Process all tickets
    python translate_tickets.py --provider lm_studio      # Use LM Studio instead
"""

import os
import sys
import argparse
import pandas as pd
import json
import re
from typing import List, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from secrets.env
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(script_dir, 'secrets.env')
load_dotenv(secrets_path)

# =====================================================
# CONFIGURATION
# =====================================================

CONFIG = {
    'input_csv': '../Source_files/dataset-tickets-multi-lang3-4k.csv',
    'output_csv': '../Source_files/dataset-tickets-multi-lang3-4k-translated-all.csv',

    # Fields to translate
    'fields_to_translate': ['subject', 'body', 'answer'],

    # LLM configuration
    'api_provider': 'groq',  # 'groq' or 'lm_studio'
    'groq_url': 'https://api.groq.com/openai/v1',
    'lm_studio_url': 'http://169.254.233.17:1234',
    'llm_model': 'openai/gpt-oss-20b',  # Groq model
    'use_consolidation': True,  # Week03: Use 1 LLM call instead of 4
    'temperature': 0.1,

    # Processing
    'progress_interval': 10,  # Show progress every N tickets
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def show_error(err_string: str):
    """Print an error message and stop execution."""
    print(f"❌ Error: {err_string}")
    sys.exit(1)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load ticket data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} tickets from CSV")
        print(f"📋 Columns: {', '.join(df.columns)}")
        return df
    except FileNotFoundError:
        show_error(f"CSV file not found at {csv_path}")
    except Exception as e:
        show_error(f"Error loading CSV: {str(e)}")


# =====================================================
# INITIALIZE LLM CLIENT
# =====================================================

def init_client(provider: str = None):
    global client
    """Initialize OpenAI client based on provider."""
    global client

    if provider:
        CONFIG['api_provider'] = provider

    if CONFIG['api_provider'] == 'groq':
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            show_error("GROQ_API_KEY not found in secrets.env")

        client = OpenAI(
            base_url=CONFIG['groq_url'],
            api_key=groq_api_key
        )
        print(f"✅ Using Groq API with model: {CONFIG['llm_model']}")
    else:
        client = OpenAI(
            base_url=f"{CONFIG['lm_studio_url']}/v1",
            api_key="lm-studio"
        )
        print(f"✅ Using LM Studio with model: {CONFIG['llm_model']}")

    # Test connection
    try:
        response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[{"role": "user", "content": "Say 'ready'"}],
            temperature=0.1,
            max_tokens=10,
            timeout=30
        )
        print(f"✅ {CONFIG['api_provider'].upper()} is connected and ready\n")
    except Exception as e:
        show_error(f"{CONFIG['api_provider'].upper()} connection failed: {e}")


# =====================================================
# TRANSLATION FUNCTIONS
# =====================================================

def detect_and_translate_llm_consolidated(texts: List[str]) -> Tuple[str, List[str]]:
    """
    Week03 consolidation: 1 LLM call instead of 4.
    Combines language detection + all translations into single call with JSON output.
    """
    try:
        if not texts or all(not t or len(str(t).strip()) < 3 for t in texts):
            return 'en', texts

        subject = str(texts[0]) if len(texts) > 0 else ""
        body = str(texts[1]) if len(texts) > 1 else ""
        answer = str(texts[2]) if len(texts) > 2 else ""

        system_prompt = """You are a professional translation assistant specializing in support ticket translation.
You must return your response as a valid JSON object with no additional text or commentary."""

        user_prompt = f"""Analyze and translate this support ticket. Return a JSON object with the following structure:

###
Subject: {subject}
Body: {body}
Answer: {answer}
###

Please analyze this ticket and return a JSON object with this exact structure:
{{
  "detected_language": "one of [German, English, French, Portuguese, Spanish, Unknown]",
  "subject_english": "English translation of subject with American idioms",
  "body_english": "English translation of body with American idioms",
  "answer_english": "English translation of answer with American idioms"
}}

Instructions:
1. Detect the predominant language (if multiple languages, choose the most common one)
2. For each field, provide a faithful English translation adapted to American idioms and phrasing
3. Preserve technical terminology and proper nouns
4. If a field is empty, return an empty string for that translation
5. Return ONLY the JSON object, no other text"""

        # Calculate dynamic max_tokens
        total_chars = len(subject) + len(body) + len(answer)
        estimated_tokens = int((total_chars / 4) * 1.5) + 500
        max_tokens = max(estimated_tokens, 2000)

        # Single consolidated call
        response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=90
        )

        # Parse JSON response
        response_text = response.choices[0].message.content.strip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not parse JSON from LLM response")

        # Map language name to code
        lang_map = {
            'german': 'de',
            'english': 'en',
            'french': 'fr',
            'portuguese': 'pt',
            'spanish': 'es',
            'unknown': 'en'
        }

        detected_lang_name = data.get('detected_language', 'Unknown').lower()
        detected_lang = lang_map.get(detected_lang_name, 'en')

        # Extract translations
        translations = [
            data.get('subject_english', ''),
            data.get('body_english', ''),
            data.get('answer_english', '')
        ]

        return detected_lang, translations

    except Exception as e:
        print(f"   ⚠️ Consolidated translation failed: {e}")
        print(f"   ⚠️ Falling back to original 4-call method...")
        return detect_and_translate_llm(texts)


def detect_and_translate_llm(texts: List[str]) -> Tuple[str, List[str]]:
    """
    Week02 original: 4 separate LLM calls.
    Used as fallback if consolidated approach fails.
    """
    try:
        if not texts or all(not t or len(str(t).strip()) < 3 for t in texts):
            return 'en', texts

        subject = str(texts[0]) if len(texts) > 0 else ""
        body = str(texts[1]) if len(texts) > 1 else ""
        answer = str(texts[2]) if len(texts) > 2 else ""

        # Language detection
        lang_detect_prompt = f"""Analyze the following support request email and return ONLY the name of the language it is written in,
return one of the following languages:
German; English; French; Portuguese; Spanish; Unknown;

###
Subject: {subject}
Body: {body}
###

Your response should be simply one of [German, English, French, Portuguese, Spanish or Unknown], with no additional commentary or characters;
If there is more than one language present, choose the predominent one."""

        lang_response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing support tickets."},
                {"role": "user", "content": lang_detect_prompt}
            ],
            temperature=0.1,
            max_tokens=1200,
            timeout=30
        )

        detected_lang_name = lang_response.choices[0].message.content.strip().lower()
        if not detected_lang_name:
            detected_lang_name = 'unknown'

        lang_map = {'german': 'de', 'english': 'en', 'french': 'fr', 'portuguese': 'pt', 'spanish': 'es', 'unknown': 'en'}
        detected_lang = lang_map.get(detected_lang_name, 'en')

        translations = []

        # Translate subject
        subj_prompt = f"""Translate the following email subject line from {detected_lang_name} to English, while adapting it to American idioms and phrasing -
your translation should faithfully match the meaning of the original:

###
{subject}
###

Your response should simply be the English translation with no other information."""

        subj_response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant whose assigned the job of translating support tickets from the original language to English."},
                {"role": "user", "content": subj_prompt}
            ],
            temperature=0.1,
            max_tokens=400,
            timeout=30
        )
        translations.append(subj_response.choices[0].message.content.strip())

        # Translate body
        body_prompt = f"""Translate the following email body from {detected_lang_name} to English, while adapting it to American idioms and phrasing -
your translation should faithfully match the meaning of the original:

###
{body}
###

Your response should simply be the English translation with no other information."""

        body_char_count = len(body)
        body_max_tokens = max(int((body_char_count / 4) * 1.5) + 200, 1200)

        body_response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant whose assigned the job of translating support tickets from the original language to English."},
                {"role": "user", "content": body_prompt}
            ],
            temperature=0.1,
            max_tokens=body_max_tokens,
            timeout=60
        )
        translations.append(body_response.choices[0].message.content.strip())

        # Translate answer
        answer_prompt = f"""Translate the following email answer from {detected_lang_name} to English, while adapting it to American idioms and phrasing -
your translation should faithfully match the meaning of the original:

###
{answer}
###

Your response should simply be the English translation with no other information."""

        answer_char_count = len(answer)
        answer_max_tokens = max(int((answer_char_count / 4) * 1.5) + 200, 1200)

        answer_response = client.chat.completions.create(
            model=CONFIG['llm_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant whose assigned the job of translating support tickets from the original language to English."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.1,
            max_tokens=answer_max_tokens,
            timeout=60
        )
        translations.append(answer_response.choices[0].message.content.strip())

        return detected_lang, translations

    except Exception as e:
        print(f"   ⚠️ LLM detection/translation failed: {e}, using original texts")
        return 'en', texts


def translate_ticket(row: pd.Series, fields: List[str]) -> Tuple[pd.Series, str]:
    """Translate a single ticket row."""
    texts_to_translate = []
    for field in fields:
        text = str(row.get(field, '')) if pd.notna(row.get(field)) else ''
        texts_to_translate.append(text)

    # Use consolidation or original based on config
    if CONFIG.get('use_consolidation', True):
        detected_lang, translations = detect_and_translate_llm_consolidated(texts_to_translate)
    else:
        detected_lang, translations = detect_and_translate_llm(texts_to_translate)

    updated_row = row.copy()
    for i, field in enumerate(fields):
        if i < len(translations):
            updated_row[field] = translations[i]

    return updated_row, detected_lang


# =====================================================
# MAIN PROCESSING
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='Translate support tickets using LLM')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickets to process')
    parser.add_argument('--provider', type=str, choices=['groq', 'lm_studio'], default='groq', help='LLM provider to use')
    args = parser.parse_args()

    print("="*80)
    print("🌐 TICKET TRANSLATION - Week 05")
    print("="*80)

    # Initialize client
    init_client(args.provider)

    # Load data
    print(f"\n📂 Loading CSV from: {CONFIG['input_csv']}")
    df_original = load_data(CONFIG['input_csv'])

    # Determine limit
    test_limit = args.limit if args.limit else len(df_original)

    print(f"\n🌐 Starting translation...")
    print(f"   Fields to translate: {', '.join(CONFIG['fields_to_translate'])}")
    print(f"   Tickets to process: {test_limit:,} of {len(df_original):,}")
    print(f"   Using: {'Week03 consolidation (1 call)' if CONFIG['use_consolidation'] else 'Week02 original (4 calls)'}")

    # Create copy for translated data
    df_translated = df_original.copy()

    # Add columns for translations
    for field in CONFIG['fields_to_translate']:
        df_translated[f'{field}_english'] = ''

    detected_languages = []
    start_time = datetime.now()

    # Process each ticket
    for idx, row in df_original.iterrows():
        translated_row, detected_lang = translate_ticket(row, CONFIG['fields_to_translate'])

        # Update dataframe
        for field in CONFIG['fields_to_translate']:
            df_translated.at[idx, f'{field}_english'] = translated_row[field]

        detected_languages.append(detected_lang)

        # Progress indicator
        if (idx + 1) % CONFIG['progress_interval'] == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (test_limit - idx - 1) / rate if rate > 0 else 0
            avg_time = elapsed / (idx + 1)
            print(f"   [{idx + 1:,}/{test_limit:,}] {(idx+1)/test_limit*100:.1f}% | Avg: {avg_time:.2f}s/ticket | ETA: {remaining/60:.1f} min")

        # Stop at limit
        if idx + 1 >= test_limit:
            break

    # Add detected language column
    detected_languages.extend(['unknown'] * (len(df_original) - len(detected_languages)))
    df_translated['original_language'] = detected_languages

    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ Translation complete!")
    print(f"   Total time: {elapsed_total/60:.2f} minutes")
    print(f"   Tickets processed: {min(test_limit, len(df_original)):,}")
    print(f"   Average: {elapsed_total/min(test_limit, len(df_original)):.2f} seconds per ticket")

    # Save results
    print(f"\n💾 Saving translated CSV to: {CONFIG['output_csv']}")
    df_translated.to_csv(CONFIG['output_csv'], index=False)

    # Show language distribution
    print(f"\n📊 Language Distribution:")
    lang_counts = df_translated['original_language'].value_counts()
    for lang, count in lang_counts.items():
        if lang != 'unknown':
            print(f"   {lang}: {count:,} tickets ({count/len(df_translated)*100:.1f}%)")

    print(f"\n🎯 Next step: Use '{CONFIG['output_csv']}' in the main RAG notebook")
    print("="*80)


if __name__ == "__main__":
    main()
