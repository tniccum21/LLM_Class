"""
Week07 - RAG Pipeline Implementation

Core RAG (Retrieval-Augmented Generation) pipeline with two-stage retrieval:
1. Vector similarity search (ChromaDB)
2. Cross-encoder re-ranking
3. LLM answer generation with LangChain

Extracted from notebook implementation and optimized for production use.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import AppConfig


@dataclass
class PerformanceMetrics:
    """Performance metrics for RAG pipeline execution"""

    total_time: float
    embedding_time: float
    search_time: float
    rerank_time: float
    llm_time: float
    confidence_score: float
    num_retrieved: int
    num_reranked: int
    timestamp: datetime
    rag_mode: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for display"""
        return {
            'Total Response Time': f"{self.total_time:.2f}s",
            'Vector Search': f"{self.search_time:.2f}s",
            'Re-ranking': f"{self.rerank_time:.2f}s",
            'LLM Generation': f"{self.llm_time:.2f}s",
            'Confidence': f"{self.confidence_score:.1%}",
            'Tickets Retrieved': self.num_retrieved,
            'Mode': self.rag_mode.capitalize()
        }


class RAGPipeline:
    """
    Production RAG pipeline with two-stage retrieval and LangChain integration
    """

    def __init__(self, config: AppConfig):
        """
        Initialize RAG pipeline components

        Args:
            config: Application configuration
        """
        self.config = config
        self.embedder: Optional[HuggingFaceEmbeddings] = None
        self.vectordb: Optional[Chroma] = None
        self.reranker: Optional[CrossEncoder] = None
        self.llm: Optional[ChatOpenAI] = None
        self.chain = None
        self.query_expansion_chain = None

    def initialize(self):
        """Initialize all pipeline components (models, database, chain)"""
        print("🔧 Initializing RAG pipeline...")

        # Initialize embeddings
        print(f"  Loading embedding model: {self.config.embedding_model}")
        start = time.time()
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': self.config.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"  ✓ Embeddings loaded ({time.time() - start:.2f}s)")

        # Initialize vector database
        print(f"  Loading ChromaDB: {self.config.chroma_db_path}")
        start = time.time()
        self.vectordb = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.config.persist_directory
        )
        print(f"  ✓ ChromaDB loaded ({time.time() - start:.2f}s)")

        # Initialize re-ranker
        print(f"  Loading re-ranker: {self.config.reranker_model}")
        start = time.time()
        self.reranker = CrossEncoder(self.config.reranker_model)
        print(f"  ✓ Re-ranker loaded ({time.time() - start:.2f}s)")

        # Initialize LLM
        print(f"  Connecting to LM Studio: {self.config.lm_studio_url}")
        start = time.time()
        self.llm = ChatOpenAI(
            base_url=f"{self.config.lm_studio_url}/v1",
            api_key="not-needed",  # LM Studio doesn't require API key
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        print(f"  ✓ LLM connected ({time.time() - start:.2f}s)")

        # Build LangChain chain
        self._build_chain()

        # Initialize query expansion chain if configured
        if self.config.retriever_type == 'multiquery':
            print(f"  Setting up query expansion...")
            start = time.time()
            self._build_query_expansion_chain()
            print(f"  ✓ Query expansion initialized ({time.time() - start:.2f}s)")

        print("✅ RAG pipeline initialized successfully!")

    def _build_chain(self):
        """Build LangChain LCEL chain for answer generation"""
        # Create prompt template based on RAG mode
        system_prompt = self.config.get_prompt_template()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Support Ticket:\nSubject: {subject}\nBody: {body}\n\nPlease provide a detailed, actionable solution based on the similar cases above.")
        ])

        # Build chain with LCEL
        self.chain = (
            {
                "context": RunnableLambda(lambda x: x["context"]),
                "subject": RunnableLambda(lambda x: x["subject"]),
                "body": RunnableLambda(lambda x: x["body"])
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_query_expansion_chain(self):
        """Build query expansion chain for MultiQuery retrieval"""
        query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant that generates multiple search queries for IT support tickets.
Generate 3 different versions of the user's query to improve ticket retrieval.
Each version should rephrase the query while maintaining the technical intent.

Original query: {question}

Provide 3 alternative queries (one per line):"""
        )

        self.query_expansion_chain = (
            query_expansion_prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_multiple_queries(self, query: str) -> List[str]:
        """
        Generate multiple query variations for improved retrieval

        Args:
            query: Original query text

        Returns:
            List[str]: List of query variations including original
        """
        if not self.query_expansion_chain:
            return [query]

        # Generate alternative queries
        generated_queries_text = self.query_expansion_chain.invoke({"question": query})

        # Parse generated queries (one per line)
        generated_queries = [q.strip() for q in generated_queries_text.split('\n') if q.strip()]

        # Filter out empty strings and numbered prefixes like "1.", "2.", etc.
        import re
        cleaned_queries = []
        for q in generated_queries:
            # Remove leading numbers and dots
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', q).strip()
            if cleaned and len(cleaned) > 10:  # Ensure meaningful queries
                cleaned_queries.append(cleaned)

        # Return original + generated (limit to 3 total variations)
        return [query] + cleaned_queries[:2]

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query text

        Args:
            query: Query text

        Returns:
            List[float]: Query embedding vector
        """
        return self.embedder.embed_query(query)

    def vector_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform vector similarity search (standard, multiquery, or MMR)

        Args:
            query: Query text
            top_k: Number of results to return (default: config.top_k_initial)

        Returns:
            List[Document]: Retrieved documents
        """
        k = top_k or self.config.top_k_initial

        if self.config.retriever_type == 'mmr':
            # MMR (Maximal Marginal Relevance) search
            # Balances relevance with diversity using lambda_mult parameter
            # lambda_mult=1.0 is pure relevance, 0.0 is pure diversity
            # Default 0.5 balances both
            return self.vectordb.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k * 3,  # Fetch 3x more candidates for diversity selection
                lambda_mult=0.5  # Balance between relevance (1.0) and diversity (0.0)
            )
        elif self.config.retriever_type == 'multiquery' and self.query_expansion_chain:
            # Generate multiple query variations
            queries = self.generate_multiple_queries(query)

            # Retrieve documents for each query variation
            all_docs = []
            seen_ids = set()

            for q in queries:
                docs = self.vectordb.similarity_search(q, k=k)
                # Deduplicate based on document content
                for doc in docs:
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)

            # Return top-k from combined results
            return all_docs[:k]
        else:
            # Standard vector similarity search
            return self.vectordb.similarity_search(query, k=k)

    def rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using cross-encoder

        Args:
            query: Query text
            documents: Documents to re-rank
            top_k: Number of top results to return (default: config.top_k_reranked)

        Returns:
            List[Tuple[Document, float]]: Re-ranked documents with scores
        """
        k = top_k or self.config.top_k_reranked

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get cross-encoder scores
        scores = self.reranker.predict(pairs)

        # DEBUG: Print raw scores
        print(f"DEBUG rerank_results: Raw scores from cross-encoder: {scores[:3] if len(scores) > 3 else scores}")

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Sort by score (descending) and take top-k
        reranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:k]

        # DEBUG: Print reranked scores
        print(f"DEBUG rerank_results: Top reranked scores: {[float(s) for _, s in reranked[:3]]}")

        return reranked

    def build_context(
        self,
        doc_scores: List[Tuple[Document, float]]
    ) -> str:
        """
        Build context string from re-ranked documents

        Args:
            doc_scores: List of (document, score) tuples

        Returns:
            str: Formatted context for LLM
        """
        context_parts = []

        for i, (doc, score) in enumerate(doc_scores, 1):
            metadata = doc.metadata
            subject = metadata.get('subject', 'N/A')
            body = doc.page_content
            category = metadata.get('category', 'N/A')
            priority = metadata.get('priority', 'N/A')

            context_part = f"""
Ticket #{i} (Relevance: {score:.2%})
Subject: {subject}
Category: {category}
Priority: {priority}
Content: {body}
"""
            context_parts.append(context_part.strip())

        return "\n\n---\n\n".join(context_parts)

    def calculate_confidence(
        self,
        doc_scores: List[Tuple[Document, float]]
    ) -> float:
        """
        Calculate confidence score from re-ranking scores

        Args:
            doc_scores: List of (document, score) tuples

        Returns:
            float: Confidence score (0-1)
        """
        if not doc_scores:
            return 0.0

        # Weighted average with higher weight on top results
        weights = [1.0 / (i + 1) for i in range(len(doc_scores))]
        scores = [score for _, score in doc_scores]

        weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)

        # Normalize to 0-1 range (cross-encoder scores are typically -10 to 10)
        # Map to sigmoid-like curve centered around 0
        normalized = 1 / (1 + pow(2.71828, -weighted_score))

        return normalized

    def generate_answer(
        self,
        subject: str,
        body: str,
        context: str
    ) -> str:
        """
        Generate answer using LLM with context

        Args:
            subject: Ticket subject
            body: Ticket body
            context: Historical ticket context

        Returns:
            str: Generated answer
        """
        response = self.chain.invoke({
            "subject": subject,
            "body": body,
            "context": context
        })

        return response

    def process_ticket(
        self,
        subject: str,
        body: str,
        top_k_initial: Optional[int] = None,
        top_k_reranked: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a support ticket through the complete RAG pipeline

        Args:
            subject: Ticket subject
            body: Ticket description
            top_k_initial: Number of tickets to retrieve (default: config.top_k_initial)
            top_k_reranked: Number of tickets after re-ranking (default: config.top_k_reranked)

        Returns:
            dict: Response with answer, metrics, and references
        """
        start_time = time.time()

        # Combine subject and body for query
        query = f"{subject}\n\n{body}"

        # Generate query variations if using multiquery
        generated_queries = None
        if self.config.retriever_type == 'multiquery' and self.query_expansion_chain:
            generated_queries = self.generate_multiple_queries(query)

        # Step 1: Vector search
        search_start = time.time()
        initial_results = self.vector_search(query, top_k=top_k_initial)
        search_time = time.time() - search_start

        # Step 2: Re-rank results
        rerank_start = time.time()
        reranked_results = self.rerank_results(query, initial_results, top_k=top_k_reranked)
        rerank_time = time.time() - rerank_start

        # Step 3: Build context
        context = self.build_context(reranked_results)

        # Step 4: Calculate confidence
        confidence = self.calculate_confidence(reranked_results)

        # Step 5: Generate answer
        llm_start = time.time()
        answer = self.generate_answer(subject, body, context)
        llm_time = time.time() - llm_start

        total_time = time.time() - start_time

        # DEBUG: Print scores before building response
        print(f"DEBUG process_ticket: Reranked results scores: {[float(s) for _, s in reranked_results]}")

        # Build response
        references = []
        for doc, score in reranked_results:
            ref_score = float(score)
            print(f"DEBUG process_ticket: Converting score {score} (type: {type(score)}) -> {ref_score}")
            references.append({
                'subject': doc.metadata.get('subject', 'N/A'),
                'body': doc.page_content,
                'category': doc.metadata.get('category', 'N/A'),
                'priority': doc.metadata.get('priority', 'N/A'),
                'score': ref_score,  # Raw cross-encoder relevance score
            })

        response = {
            'answer': answer,
            'confidence': confidence,
            'references': references,
            'metrics': PerformanceMetrics(
                total_time=total_time,
                embedding_time=0.0,  # Included in search_time
                search_time=search_time,
                rerank_time=rerank_time,
                llm_time=llm_time,
                confidence_score=confidence,
                num_retrieved=len(initial_results),
                num_reranked=len(reranked_results),
                timestamp=datetime.now(),
                rag_mode=self.config.rag_mode
            ),
            'generated_queries': generated_queries  # Include generated queries if multiquery mode
        }

        return response


if __name__ == '__main__':
    # Test pipeline initialization
    from config import get_config

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    config = get_config()
    pipeline = RAGPipeline(config)

    try:
        pipeline.initialize()
        print("\n✅ Pipeline initialization successful!")

        # Test query
        print("\n🧪 Testing with sample ticket...")
        response = pipeline.process_ticket(
            subject="VPN connection issue",
            body="My VPN keeps disconnecting every 15 minutes. Using Cisco AnyConnect on Windows 10."
        )

        print("\n📊 Response:")
        print(f"Answer length: {len(response['answer'])} characters")
        print(f"Confidence: {response['confidence']:.1%}")
        print(f"References: {len(response['references'])} tickets")
        print(f"\nMetrics:")
        for key, value in response['metrics'].to_dict().items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
