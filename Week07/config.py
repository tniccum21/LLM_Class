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
    retriever_type: str = 'standard'  # 'standard', 'multiquery', or 'mmr'

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

    # ===== Input Guardrails Configuration =====
    enable_guardrails: bool = True
    guardrail_model_path: str = field(
        default_factory=lambda: os.getenv('GUARDRAIL_MODEL', 'llama-guard-3-8b')
    )  # Model name in LM Studio

    # Safety thresholds (0.0-1.0, higher = stricter)
    safety_threshold: float = 0.7  # Block if safety_score < threshold

    # Violation severity blocking
    block_critical: bool = True    # Always block critical violations
    block_high: bool = True        # Block high severity violations
    block_medium: bool = False     # Warn but allow medium violations
    block_low: bool = False        # Allow low severity violations

    # Performance
    guardrail_timeout: int = 10    # Seconds
    guardrail_max_tokens: int = 512

    # ===== Future Extensibility =====
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing"""
        # Resolve chroma_db_path to absolute path
        chroma_path = Path(self.chroma_db_path)
        if not chroma_path.is_absolute():
            chroma_path = Path(__file__).parent / self.chroma_db_path
        self.chroma_db_path = str(chroma_path)

        # Set persist_directory from chroma_db_path if not explicitly set
        if self.persist_directory is None:
            self.persist_directory = self.chroma_db_path

        # Validate RAG mode
        if self.rag_mode not in ['strict', 'augmented']:
            raise ValueError(f"Invalid rag_mode: {self.rag_mode}. Must be 'strict' or 'augmented'")

        # Validate retriever type
        if self.retriever_type not in ['standard', 'multiquery', 'mmr']:
            raise ValueError(f"Invalid retriever_type: {self.retriever_type}. Must be 'standard', 'multiquery', or 'mmr'")

        # Validate and resolve CSV path
        csv_file = Path(self.csv_path)
        if not csv_file.is_absolute():
            csv_file = Path(__file__).parent / self.csv_path

        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_file}")

        # Update csv_path to absolute path
        self.csv_path = str(csv_file)

    def get_prompt_template(self) -> str:
        """
        Get system prompt template based on RAG mode

        Returns:
            str: System prompt for LLM
        """
        if self.rag_mode == 'strict':
            return """You are a technical support assistant helping resolve IT support tickets.

CRITICAL INSTRUCTION: You must respond using ONLY the exact language and approaches from the Resolution sections in the similar cases below. Do NOT generate new text, paraphrase, or use your own knowledge.

Your response must be based on copying and adapting the Resolution text from one or more similar cases above. If you cannot find a matching resolution in the similar cases, say: "I need more information to assist you. Please provide [specific details]."

Similar Support Cases:
{context}

Guidelines:
- Copy the Resolution approach from the most similar case above
- Use the exact phrasing and structure from those Resolutions
- You may combine elements from multiple Resolutions if they all apply
- Do NOT create new language - only use what appears in the Resolution sections
- Respond as if you are directly helping the customer (don't mention "similar cases" or "context")
"""
        else:  # augmented
            return """You are a technical support assistant helping resolve IT support tickets.

You have access to similar support cases below, which should be your PRIMARY reference.
However, you may supplement this information with your general IT knowledge when it provides
helpful additional context or clarification.

Similar Support Cases:
{context}

Guidelines:
- Prioritize solutions from the similar support cases above
- Supplement with general IT knowledge when helpful
- Provide comprehensive, actionable solutions
- Respond as if you are directly helping the customer (don't mention "historical tickets" or "context")
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
            'retriever_type': self.retriever_type,
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
