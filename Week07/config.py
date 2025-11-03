"""
Week07 - Configuration Management System

Centralized configuration for the RAG Support Ticket System with
flexible extensibility for future options and environment-based overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


# Load environment variables from secrets.env
ENV_FILE = Path(__file__).parent / 'secrets.env'
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=True)


@dataclass
class AppConfig:
    """
    Application configuration with sensible defaults and extensibility.

    All configuration options are defined here with type hints for IDE support
    and validation. New options can be added to extra_options dict for
    experimental features without modifying the class structure.
    """

    # ===== Vector Database Configuration =====
    chroma_db_path: str = './chroma_ticket_db'
    collection_name: str = 'support_tickets'
    persist_directory: Optional[str] = None  # Auto-set from chroma_db_path

    # ===== Embedding Configuration =====
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dimension: int = 384
    embedding_device: str = 'cpu'  # 'cpu', 'cuda', or 'mps'

    # ===== Retrieval Configuration =====
    top_k_initial: int = 20  # Initial vector search results
    top_k_reranked: int = 5  # After cross-encoder re-ranking
    reranker_model: str = 'mixedbread-ai/mxbai-rerank-base-v1'
    similarity_threshold: float = 0.5  # Minimum similarity score

    # ===== LLM Configuration =====
    lm_studio_url: str = field(
        default_factory=lambda: os.getenv('LM_STUDIO_URL', 'http://192.168.7.171:1234')
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv('LM_STUDIO_MODEL', 'gpt-oss-20b')
    )
    max_tokens: int = 6000
    temperature: float = 0.1
    timeout: int = 60  # Seconds

    # ===== RAG Mode Configuration =====
    rag_mode: str = 'strict'  # 'strict' or 'augmented'

    # ===== UI Configuration =====
    show_references: bool = True
    show_metrics: bool = True
    show_debug_info: bool = False
    enable_caching: bool = True

    # ===== Performance Configuration =====
    batch_size: int = 100  # For batch embedding generation
    cache_embeddings: bool = True

    # ===== Data Configuration =====
    csv_path: str = 'dataset-tickets-multi-lang3-4k-translated-all.csv'

    # ===== Future Extensibility =====
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing"""
        # Set persist_directory from chroma_db_path if not explicitly set
        if self.persist_directory is None:
            self.persist_directory = self.chroma_db_path

        # Validate RAG mode
        if self.rag_mode not in ['strict', 'augmented']:
            raise ValueError(f"Invalid rag_mode: {self.rag_mode}. Must be 'strict' or 'augmented'")

        # Validate paths exist for required files
        csv_file = Path(self.csv_path)
        if not csv_file.is_absolute():
            csv_file = Path(__file__).parent / self.csv_path

        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_file}")

    def get_prompt_template(self) -> str:
        """
        Get system prompt template based on RAG mode

        Returns:
            str: System prompt for LLM
        """
        if self.rag_mode == 'strict':
            return """You are a technical support assistant helping resolve IT support tickets.

IMPORTANT: You must ONLY use information from the provided historical ticket context below.
Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

If the historical tickets do not contain sufficient information to answer the question,
clearly state that you cannot provide an answer based on the available context.

Historical Ticket Context:
{context}

Guidelines:
- Base your answer entirely on the historical tickets provided
- Reference specific ticket numbers when applicable
- If multiple solutions exist, present all relevant options
- If context is insufficient, explicitly state what information is missing
"""
        else:  # augmented
            return """You are a technical support assistant helping resolve IT support tickets.

You have access to historical ticket context below, which should be your PRIMARY reference.
However, you may supplement this information with your general IT knowledge when it provides
helpful additional context or clarification.

Historical Ticket Context:
{context}

Guidelines:
- Prioritize information from the historical tickets
- Supplement with general IT knowledge when helpful
- Clearly distinguish between context-based and knowledge-based information
- Reference specific ticket numbers when using historical context
- Provide comprehensive, actionable solutions
"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns:
            dict: Configuration as dictionary
        """
        return {
            'vector_db': {
                'path': self.chroma_db_path,
                'collection': self.collection_name,
            },
            'embedding': {
                'model': self.embedding_model,
                'dimension': self.embedding_dimension,
                'device': self.embedding_device,
            },
            'retrieval': {
                'initial_k': self.top_k_initial,
                'reranked_k': self.top_k_reranked,
                'reranker': self.reranker_model,
                'threshold': self.similarity_threshold,
            },
            'llm': {
                'url': self.lm_studio_url,
                'model': self.llm_model,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'timeout': self.timeout,
            },
            'rag_mode': self.rag_mode,
            'ui_options': {
                'show_references': self.show_references,
                'show_metrics': self.show_metrics,
                'show_debug': self.show_debug_info,
            },
            'extra_options': self.extra_options,
        }


@dataclass
class DevelopmentConfig(AppConfig):
    """Development environment configuration with debugging enabled"""

    show_debug_info: bool = True
    enable_caching: bool = False  # Disable for development

    def __post_init__(self):
        super().__post_init__()
        print("🔧 Development mode enabled")


@dataclass
class ProductionConfig(AppConfig):
    """Production environment configuration with optimizations"""

    show_debug_info: bool = False
    enable_caching: bool = True
    timeout: int = 120  # Longer timeout for production

    def __post_init__(self):
        super().__post_init__()
        # Validate production requirements
        if not os.getenv('LM_STUDIO_URL'):
            raise EnvironmentError("LM_STUDIO_URL must be set for production")


def get_config(env: Optional[str] = None) -> AppConfig:
    """
    Get configuration based on environment

    Args:
        env: Environment name ('dev', 'prod', or None for auto-detect)

    Returns:
        AppConfig: Configuration instance for the environment
    """
    if env is None:
        env = os.getenv('ENV', 'dev').lower()

    if env == 'prod':
        return ProductionConfig()
    elif env == 'dev':
        return DevelopmentConfig()
    else:
        return AppConfig()


# Default configuration instance
config = get_config()


if __name__ == '__main__':
    # Configuration validation and display
    print("=" * 60)
    print("Week07 RAG System Configuration")
    print("=" * 60)

    cfg = get_config()

    print("\n📋 Configuration Summary:")
    print(f"  Environment: {os.getenv('ENV', 'development')}")
    print(f"  RAG Mode: {cfg.rag_mode}")
    print(f"  Vector DB: {cfg.chroma_db_path}")
    print(f"  Collection: {cfg.collection_name}")
    print(f"  Embedding Model: {cfg.embedding_model}")
    print(f"  Re-ranker: {cfg.reranker_model}")
    print(f"  LM Studio: {cfg.lm_studio_url}")
    print(f"  LLM Model: {cfg.llm_model}")
    print(f"  Initial Retrieval: top-{cfg.top_k_initial}")
    print(f"  After Re-ranking: top-{cfg.top_k_reranked}")

    print("\n🎯 UI Options:")
    print(f"  Show References: {cfg.show_references}")
    print(f"  Show Metrics: {cfg.show_metrics}")
    print(f"  Debug Mode: {cfg.show_debug_info}")

    print("\n✅ Configuration validated successfully!")

    # Display as dictionary
    print("\n📊 Full Configuration:")
    import json
    print(json.dumps(cfg.to_dict(), indent=2))
