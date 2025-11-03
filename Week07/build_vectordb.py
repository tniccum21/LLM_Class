"""
Week07 - Vector Database Builder

One-time setup script to build ChromaDB vector database from complete dataset.
Uses 100% of available tickets (no train/test split) for production retrieval.
"""

import time
from pathlib import Path
from typing import List

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config import get_config, AppConfig


def load_tickets(csv_path: str) -> pd.DataFrame:
    """
    Load tickets from CSV file

    Args:
        csv_path: Path to CSV file

    Returns:
        pd.DataFrame: Loaded tickets
    """
    print(f"📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} tickets")

    # Display dataset info
    print(f"\n📊 Dataset Information:")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    print(f"  Shape: {df.shape}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")

    return df


def create_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert DataFrame to LangChain Document objects with metadata

    Args:
        df: DataFrame with ticket data

    Returns:
        List[Document]: LangChain documents
    """
    print(f"\n📝 Creating document objects...")

    documents = []

    for idx, row in df.iterrows():
        # Combine translated columns if available, fallback to original
        subject = row.get('subject_translated', row.get('subject', ''))
        body = row.get('body_translated', row.get('body', ''))

        # Create document content
        content = f"{subject}\n\n{body}"

        # Build metadata
        metadata = {
            'ticket_id': idx,
            'subject': str(subject),
            'category': str(row.get('category', 'Unknown')),
            'priority': str(row.get('priority', 'Unknown')),
            'source_lang': str(row.get('language', 'en')),
        }

        # Add optional fields if present
        if 'agent' in row:
            metadata['agent'] = str(row['agent'])
        if 'resolution' in row:
            metadata['resolution'] = str(row['resolution'])

        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1:,}/{len(df):,} ({(idx + 1)/len(df)*100:.1f}%)")

    print(f"✓ Created {len(documents):,} document objects")
    return documents


def build_vectordb(
    documents: List[Document],
    config: AppConfig
) -> Chroma:
    """
    Build ChromaDB vector database with embeddings

    Args:
        documents: List of documents to index
        config: Application configuration

    Returns:
        Chroma: Vector database instance
    """
    print(f"\n🔧 Building ChromaDB...")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  Collection: {config.collection_name}")
    print(f"  Persist directory: {config.persist_directory}")

    # Initialize embeddings
    print(f"\n📊 Initializing embedding model...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={'device': config.embedding_device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"✓ Embeddings initialized ({time.time() - start:.2f}s)")

    # Create vector database
    print(f"\n💾 Creating vector database...")
    start = time.time()

    # Process in batches to avoid memory issues
    batch_size = config.batch_size
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"  Processing {len(documents):,} documents in {total_batches} batches of {batch_size}")

    # Create initial database with first batch
    first_batch = documents[:batch_size]
    vectordb = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=config.collection_name,
        persist_directory=config.persist_directory
    )
    print(f"  ✓ Batch 1/{total_batches} complete")

    # Add remaining batches
    for i in range(1, total_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(documents))
        batch = documents[batch_start:batch_end]

        vectordb.add_documents(batch)
        print(f"  ✓ Batch {i+1}/{total_batches} complete ({batch_end:,}/{len(documents):,} documents)")

    build_time = time.time() - start
    print(f"\n✓ Vector database created ({build_time:.2f}s)")

    return vectordb


def validate_vectordb(vectordb: Chroma, config: AppConfig) -> dict:
    """
    Validate vector database and generate statistics

    Args:
        vectordb: Vector database to validate
        config: Application configuration

    Returns:
        dict: Validation statistics
    """
    print(f"\n🔍 Validating vector database...")

    # Get collection info
    collection = vectordb._collection
    count = collection.count()

    print(f"  Documents indexed: {count:,}")

    # Test query
    print(f"\n🧪 Testing similarity search...")
    test_query = "VPN connection issue"
    results = vectordb.similarity_search(test_query, k=5)

    print(f"  Query: '{test_query}'")
    print(f"  Results: {len(results)} documents retrieved")

    if results:
        print(f"\n  Top result:")
        top_result = results[0]
        print(f"    Subject: {top_result.metadata.get('subject', 'N/A')}")
        print(f"    Category: {top_result.metadata.get('category', 'N/A')}")
        print(f"    Preview: {top_result.page_content[:100]}...")

    # Calculate database size
    db_path = Path(config.persist_directory)
    if db_path.exists():
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        print(f"\n  Database size: {size_mb:.1f} MB ({size_gb:.2f} GB)")

    stats = {
        'total_documents': count,
        'collection_name': config.collection_name,
        'embedding_model': config.embedding_model,
        'embedding_dimension': config.embedding_dimension,
        'database_size_mb': size_mb if 'size_mb' in locals() else 0,
    }

    return stats


def main():
    """Main execution function"""
    print("=" * 70)
    print("Week07 - Vector Database Builder")
    print("=" * 70)

    start_time = time.time()

    # Load configuration
    config = get_config()

    # Step 1: Load tickets
    df = load_tickets(config.csv_path)

    # Step 2: Create documents
    documents = create_documents(df)

    # Step 3: Build vector database
    vectordb = build_vectordb(documents, config)

    # Step 4: Validate database
    stats = validate_vectordb(vectordb, config)

    # Summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ Vector Database Build Complete!")
    print("=" * 70)

    print(f"\n📊 Build Summary:")
    print(f"  Total Documents: {stats['total_documents']:,}")
    print(f"  Collection Name: {stats['collection_name']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print(f"  Embedding Dimension: {stats['embedding_dimension']}")
    print(f"  Database Size: {stats['database_size_mb']:.1f} MB")
    print(f"  Build Time: {total_time:.1f} seconds")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Vector database is ready at: {config.persist_directory}")
    print(f"  2. Run the Streamlit app: streamlit run ticket_support_app.py")
    print(f"  3. Test with sample support tickets")

    print("\n✨ Database is ready for production use!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Build failed: {str(e)}")
        raise
