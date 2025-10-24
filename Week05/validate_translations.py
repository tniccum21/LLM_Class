#!/usr/bin/env python3
"""
Validation Script for Translated Tickets CSV

Checks:
1. Column structure (original + _english columns exist)
2. Translation completeness (no empty/truncated translations)
3. Language distribution
4. Sample translations quality
"""

import pandas as pd
import sys
from pathlib import Path

def validate_csv(csv_path: str):
    """Validate the translated CSV file structure and content"""

    print("="*80)
    print("TRANSLATION VALIDATION REPORT")
    print("="*80)

    # Check file exists
    if not Path(csv_path).exists():
        print(f"❌ ERROR: File not found: {csv_path}")
        return False

    # Load CSV
    print(f"\n📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total rows: {len(df):,}")

    # 1. Check column structure
    print("\n1️⃣  COLUMN STRUCTURE CHECK")
    print("-"*80)

    required_original = ['subject', 'body', 'answer']
    required_english = ['subject_english', 'body_english', 'answer_english']
    required_meta = ['original_language']

    all_required = required_original + required_english + required_meta
    missing = [col for col in all_required if col not in df.columns]

    if missing:
        print(f"   ❌ Missing columns: {', '.join(missing)}")
        return False
    else:
        print("   ✅ All required columns present")
        print(f"      - Original: {', '.join(required_original)}")
        print(f"      - English: {', '.join(required_english)}")
        print(f"      - Metadata: {', '.join(required_meta)}")

    # 2. Check translation completeness
    print("\n2️⃣  TRANSLATION COMPLETENESS CHECK")
    print("-"*80)

    issues = []
    for col in required_english:
        # Check for empty/null values
        empty = df[col].isna().sum() + (df[col] == '').sum()

        # Check for 'unknown' placeholder
        unknown = (df[col] == 'unknown').sum()

        # Check for suspiciously short translations (likely truncated)
        # Subject should be > 10 chars, body/answer > 50 chars
        min_len = 10 if 'subject' in col else 50
        short = (df[col].str.len() < min_len).sum()

        print(f"\n   {col}:")
        print(f"      Empty/null: {empty:,} ({empty/len(df)*100:.1f}%)")
        print(f"      'unknown': {unknown:,} ({unknown/len(df)*100:.1f}%)")
        print(f"      Suspiciously short (<{min_len} chars): {short:,} ({short/len(df)*100:.1f}%)")

        # Only flag issues if they indicate translation problems
        # Note: Empty subjects can be normal if original data had empty subjects
        if col == 'subject_english' and empty > len(df) * 0.15:  # More than 15% empty subjects
            issues.append(f"{col} has {empty:,} empty values (may indicate translation issues)")
        elif col != 'subject_english' and empty > len(df) * 0.05:  # More than 5% empty for body/answer
            issues.append(f"{col} has {empty:,} empty values")
        if unknown > len(df) * 0.01:  # More than 1% unknown
            issues.append(f"{col} has {unknown:,} 'unknown' values")
        if short > len(df) * 0.10:  # More than 10% suspiciously short
            issues.append(f"{col} has {short:,} suspiciously short translations")

    if issues:
        print("\n   ⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("\n   ✅ All translations appear complete")

    # 3. Language distribution
    print("\n3️⃣  LANGUAGE DISTRIBUTION")
    print("-"*80)

    lang_counts = df['original_language'].value_counts()
    for lang, count in lang_counts.items():
        print(f"   {lang}: {count:,} tickets ({count/len(df)*100:.1f}%)")

    # 4. Sample translations
    print("\n4️⃣  SAMPLE TRANSLATIONS (First 3 Non-English Tickets)")
    print("-"*80)

    non_english = df[df['original_language'] != 'en'].head(3)

    for idx, row in non_english.iterrows():
        print(f"\n   Ticket #{idx} - Language: {row['original_language'].upper()}")
        print(f"   {'─'*76}")

        # Subject
        print(f"\n   📌 Original Subject ({row['original_language']}):")
        print(f"      {row['subject'][:100]}")
        print(f"\n   ✅ English Translation:")
        print(f"      {row['subject_english'][:100]}")

        # Body (first 150 chars)
        print(f"\n   📌 Original Body ({row['original_language']}, first 150 chars):")
        print(f"      {row['body'][:150]}...")
        print(f"\n   ✅ English Translation (first 150 chars):")
        print(f"      {row['body_english'][:150]}...")

        # Check if translation differs from original
        if row['subject'] == row['subject_english']:
            print(f"\n   ⚠️  WARNING: Subject translation identical to original!")

        if row['body'][:100] == row['body_english'][:100]:
            print(f"\n   ⚠️  WARNING: Body translation identical to original!")

    # 5. Final verdict
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if not missing and not issues:
        print("✅ CSV VALIDATION PASSED")
        print("   - All required columns present")
        print("   - Translations appear complete and of good quality")
        print("   - Ready to use with ticket_rag_system.ipynb")
        return True
    else:
        print("⚠️  CSV VALIDATION FAILED")
        if missing:
            print(f"   - Missing columns: {', '.join(missing)}")
        if issues:
            print("   - Translation issues detected (see details above)")
        print("   - Recommend re-running translate_tickets.ipynb")
        return False


if __name__ == "__main__":
    # Default to the fully translated CSV in Week05 directory
    csv_path = "dataset-tickets-multi-lang3-4k-translated-all.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    success = validate_csv(csv_path)
    sys.exit(0 if success else 1)
