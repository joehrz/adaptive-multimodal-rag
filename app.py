"""
Adaptive Multimodal RAG - Streamlit Web UI
Production-ready web interface with all SOTA techniques
"""

import streamlit as st
import sys
import time
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import json

# Ensure project root is in path for imports
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.schema import Document

# Import components
from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter, RAGStrategy
from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer
from src.experiments.streaming.ollama_streaming_rag import OllamaStreamingRAG
from src.core.ollama_rag import OllamaRAG
from src.core.caching_system import RAGCacheManager
from src.core.config import get_config, reload_config
from src.core.debug_logger import DebugLogger, get_debug_logger, init_debug_logger

# Import Self-RAG and GraphRAG
try:
    from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG, ReflectionResult
    SELF_RAG_AVAILABLE = True
except ImportError:
    SELF_RAG_AVAILABLE = False

try:
    from src.experiments.graph_reasoning.ollama_graph_rag import OllamaGraphRAG
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False

try:
    from src.experiments.hyde.ollama_hyde import OllamaHyDE
    HYDE_AVAILABLE = True
except ImportError:
    HYDE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .strategy-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .baseline { background-color: #90EE90; color: #000; }
    .hyde { background-color: #FFD700; color: #000; }
    .hyde_self_rag { background-color: #FF6B6B; color: #fff; }
    .self_rag { background-color: #87CEEB; color: #000; }
    .graphrag { background-color: #9370DB; color: #fff; }
    .multimodal { background-color: #FFA500; color: #000; }
    .reflection-token {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    .relevant { background-color: #90EE90; }
    .partially_relevant { background-color: #FFD700; }
    .irrelevant { background-color: #FF6B6B; color: #fff; }
    .fully_supported { background-color: #90EE90; }
    .partially_supported { background-color: #FFD700; }
    .no_support { background-color: #FF6B6B; color: #fff; }
    .useful { background-color: #90EE90; }
    .somewhat_useful { background-color: #FFD700; }
    .not_useful { background-color: #FF6B6B; color: #fff; }
</style>
""", unsafe_allow_html=True)

# Load config
config = get_config()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'router' not in st.session_state:
    st.session_state.router = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = config.llm.model
if 'streaming_rag' not in st.session_state:
    st.session_state.streaming_rag = None
if 'base_rag' not in st.session_state:
    st.session_state.base_rag = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0.0
# New components
if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = None
if 'self_rag' not in st.session_state:
    st.session_state.self_rag = None
if 'graph_rag' not in st.session_state:
    st.session_state.graph_rag = None
if 'last_reflection' not in st.session_state:
    st.session_state.last_reflection = None
if 'last_reasoning_path' not in st.session_state:
    st.session_state.last_reasoning_path = None
if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0
if 'hyde_rag' not in st.session_state:
    st.session_state.hyde_rag = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'debug_logger' not in st.session_state:
    st.session_state.debug_logger = None
if 'debug_enabled' not in st.session_state:
    st.session_state.debug_enabled = True
if 'show_groundedness' not in st.session_state:
    st.session_state.show_groundedness = True
if 'show_retrieved_chunks' not in st.session_state:
    st.session_state.show_retrieved_chunks = True


def get_available_models() -> list:
    """Get list of available Ollama models"""
    try:
        import ollama
        models = ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return [config.llm.model]  # Fallback to config default


def initialize_system(model: str = None):
    """Initialize RAG components with selected model"""
    # Use provided model or session state or config default
    selected_model = model or st.session_state.selected_model or config.llm.model

    # Initialize debug logger if not already done
    if st.session_state.debug_logger is None and st.session_state.debug_enabled:
        st.session_state.debug_logger = init_debug_logger(
            output_dir="./debug_logs",
            enabled=True,
            save_format="both"
        )

    if st.session_state.router is None:
        with st.spinner(f"Initializing Adaptive RAG System with {selected_model}..."):
            try:
                # Initialize cache manager first
                st.session_state.cache_manager = RAGCacheManager(enable_auto_cleanup=True)

                # Initialize components with selected model
                analyzer = OllamaQueryAnalyzer(model=selected_model, verbose=False)
                st.session_state.router = OllamaAdaptiveRouter(
                    query_analyzer=analyzer, verbose=False)
                st.session_state.streaming_rag = OllamaStreamingRAG(
                    model=selected_model, verbose=False)
                st.session_state.base_rag = OllamaRAG(
                    model=selected_model,
                    verbose=False,
                    cache_manager=st.session_state.cache_manager
                )

                # Initialize Self-RAG if available
                if SELF_RAG_AVAILABLE:
                    st.session_state.self_rag = OllamaSelfRAG(
                        model=selected_model, verbose=False)

                # Initialize GraphRAG if available
                if GRAPHRAG_AVAILABLE:
                    st.session_state.graph_rag = OllamaGraphRAG(
                        model=selected_model, verbose=False)

                # Initialize HyDE if available
                if HYDE_AVAILABLE:
                    st.session_state.hyde_rag = OllamaHyDE(
                        model=selected_model, verbose=False)

                # Initialize empty document list
                if not st.session_state.documents:
                    st.session_state.documents = []

                return True
            except Exception as e:
                st.error(f"Failed to initialize system: {str(e)}")
                return False
    return True


def reset_vector_database():
    """Clear vector database for fresh session - properly clears persisted ChromaDB data"""
    try:
        # Use the new clear_vector_store method if available
        if hasattr(st.session_state, 'base_rag') and st.session_state.base_rag:
            if hasattr(st.session_state.base_rag, 'clear_vector_store'):
                st.session_state.base_rag.clear_vector_store()
            else:
                # Fallback for older implementation
                if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store is not None:
                    try:
                        st.session_state.base_rag.vector_store.delete_collection()
                    except Exception:
                        pass
                    st.session_state.base_rag.vector_store = None
                st.session_state.base_rag.documents = []

        # Also manually ensure persist directory is cleared (belt and suspenders)
        persist_dir = "./data/chroma_db_ollama"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        st.session_state.documents = []

        if 'query_history' in st.session_state:
            st.session_state.query_history = []

        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_time = 0.0

        # Clear cache
        if st.session_state.cache_manager:
            st.session_state.cache_manager.clear_all()

        # Clear GraphRAG
        if st.session_state.graph_rag:
            st.session_state.graph_rag.clear_graph()

        st.session_state.cache_hits = 0
        st.session_state.last_reflection = None
        st.session_state.last_reasoning_path = None

        return True
    except Exception as e:
        st.error(f"Error resetting database: {str(e)}")
        return False


def get_strategy_badge(strategy: str) -> str:
    """Get HTML badge for strategy"""
    badge_class = strategy.lower().replace(' ', '_').replace('+', '_')
    return f'<span class="strategy-badge {badge_class}">{strategy.upper()}</span>'


def _retrieve_context_docs(query: str) -> List[Document]:
    """Retrieve documents used as context for the query"""
    try:
        if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
            return st.session_state.base_rag.retrieve_documents(query, k=10)
    except Exception:
        pass
    return []


def _display_retrieved_chunks(retrieved_docs: List[Document]):
    """Display retrieved document chunks in an expander"""
    if not retrieved_docs:
        return

    with st.expander(f"Retrieved Context ({len(retrieved_docs)} chunks)", expanded=False):
        st.caption("These are the actual document chunks retrieved from your uploaded files and fed to the LLM as context.")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            page_str = f" | Page {page}" if page else ""
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            st.markdown(f"**Chunk {i+1}** — `{source}{page_str}`")
            st.text_area(
                f"chunk_{i+1}",
                content_preview,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )


def _run_groundedness_check(query: str, response: str, retrieved_docs: List[Document]):
    """Run a lightweight groundedness check comparing RAG vs no-retrieval answers"""
    if not retrieved_docs:
        return

    with st.expander("Groundedness Check", expanded=False):
        st.caption("Compares the RAG answer against an answer generated without document context to verify retrieval is being used.")

        with st.spinner("Generating LLM-only answer for comparison..."):
            llm_only_answer = st.session_state.base_rag._generate_response(query, context="", require_citations=False)

        if llm_only_answer.startswith("Error generating"):
            st.error(f"Could not generate LLM-only answer: {llm_only_answer}")
            return

        # Calculate word overlap
        words_rag = set(response.lower().split())
        words_llm = set(llm_only_answer.lower().split())
        overlap = len(words_rag & words_llm) / max(len(words_rag | words_llm), 1)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown("**RAG Answer** (with documents)")
            st.text_area("rag_answer", response[:500] + "..." if len(response) > 500 else response, height=150, disabled=True, label_visibility="collapsed")
        with col2:
            st.markdown("**LLM-Only Answer** (no documents)")
            st.text_area("llm_answer", llm_only_answer[:500] + "..." if len(llm_only_answer) > 500 else llm_only_answer, height=150, disabled=True, label_visibility="collapsed")
        with col3:
            st.markdown("**Overlap**")
            st.metric("Word Overlap", f"{overlap:.0%}")
            if overlap > 0.8:
                st.error("HIGH — LLM may be using training data")
            elif overlap > 0.5:
                st.warning("MEDIUM — Partial reliance on training data")
            else:
                st.success("LOW — RAG is providing unique info")

        # Check for document citations
        has_citations = "[Document" in response or "[Doc" in response
        if has_citations:
            st.success("Answer contains document citations — good sign RAG is grounding the response")
        else:
            st.info("No explicit citations found. Consider if the answer references specific details only found in your documents.")


def stream_response(query: str, strategy: RAGStrategy, documents: List[Document], conversation_history: list = None):
    """Stream response with progress updates"""
    response_placeholder = st.empty()
    progress_placeholder = st.empty()
    debug_placeholder = st.empty()
    reflection_placeholder = st.empty()
    reasoning_placeholder = st.empty()
    full_response = ""

    stages = {
        "analyzing": "Analyzing query complexity...",
        "retrieving": "Retrieving relevant documents...",
        "generating": "Generating response...",
        "reflecting": "Running reflection tokens...",
        "graph_traversing": "Traversing knowledge graph..."
    }

    try:
        # Handle different strategies
        if strategy == RAGStrategy.BASELINE:
            progress_placeholder.info(stages["generating"])
            response = st.session_state.base_rag.query(query, conversation_history=conversation_history)
            if isinstance(response, dict):
                full_response = response.get('answer', str(response))
            else:
                full_response = str(response)

        elif strategy == RAGStrategy.SELF_RAG and SELF_RAG_AVAILABLE and st.session_state.self_rag:
            progress_placeholder.info(stages["generating"])

            # Get retrieved documents
            retrieved_docs = []
            if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                retrieved_docs = st.session_state.base_rag.retrieve_documents(query, k=5)

            if not retrieved_docs:
                retrieved_docs = documents[:5] if documents else []

            # Run Self-RAG with reflection
            progress_placeholder.info(stages["reflecting"])
            result = st.session_state.self_rag.query_with_reflection(query, retrieved_docs)
            full_response = result.answer

            # Store reflection for display
            st.session_state.last_reflection = result.reflection

            # Display reflection tokens
            with reflection_placeholder.expander("Reflection Tokens", expanded=True):
                reflection = result.reflection
                col1, col2, col3 = st.columns(3)
                with col1:
                    token_class = reflection.relevance.value.lower()
                    st.markdown(f"**Relevance:** <span class='reflection-token {token_class}'>{reflection.relevance.value}</span>", unsafe_allow_html=True)
                with col2:
                    token_class = reflection.support.value.lower()
                    st.markdown(f"**Support:** <span class='reflection-token {token_class}'>{reflection.support.value}</span>", unsafe_allow_html=True)
                with col3:
                    token_class = reflection.utility.value.lower()
                    st.markdown(f"**Utility:** <span class='reflection-token {token_class}'>{reflection.utility.value}</span>", unsafe_allow_html=True)

                st.markdown(f"**Overall Score:** {reflection.overall_score:.2f}")
                if result.was_regenerated:
                    st.info(f"Answer was regenerated {result.regeneration_count} time(s) for quality improvement")

        elif strategy == RAGStrategy.GRAPHRAG and GRAPHRAG_AVAILABLE and st.session_state.graph_rag:
            progress_placeholder.info("Building/using knowledge graph...")

            # Build graph from documents if not already built
            if st.session_state.graph_rag.graph.number_of_nodes() == 0 and documents:
                progress_placeholder.info("Building knowledge graph from documents...")
                st.session_state.graph_rag.build_graph_from_documents(documents)

                # Auto-save graph to cache for future reuse
                try:
                    import re as _re
                    source_names = set(doc.metadata.get("source", "unknown") for doc in documents)
                    sanitized = _re.sub(r'[^a-zA-Z0-9]', '_', "_".join(sorted(source_names)))
                    graph_save_path = os.path.join("data", "graph_cache", f"{sanitized}_graph.json")
                    os.makedirs(os.path.dirname(graph_save_path), exist_ok=True)
                    st.session_state.graph_rag.save_graph(graph_save_path)
                except Exception:
                    pass  # Non-critical, graph is still in memory

            # Get retrieved documents from vector store to provide as additional context
            retrieved_docs = []
            if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                retrieved_docs = st.session_state.base_rag.retrieve_documents(query, k=5)

            # Query the graph with retrieved docs as context
            progress_placeholder.info(stages["graph_traversing"])
            result = st.session_state.graph_rag.query(query, retrieved_docs=retrieved_docs)
            full_response = result.answer

            # Store reasoning path for display
            st.session_state.last_reasoning_path = result.reasoning_path

            # Display reasoning path
            with reasoning_placeholder.expander("GraphRAG Reasoning Path", expanded=True):
                if result.reasoning_path:
                    st.markdown("**Knowledge Graph Traversal:**")
                    for step in result.reasoning_path[:10]:
                        st.markdown(f"- {step['from']} --[{step['relation']}]--> {step['to']}")
                    st.markdown(f"\n**Entities Used:** {len(result.entities_used)}")
                    st.markdown(f"**Hops:** {result.num_hops}")
                else:
                    st.info("No reasoning path generated - graph may need more documents")

        elif strategy == RAGStrategy.HYDE and HYDE_AVAILABLE and st.session_state.hyde_rag:
            # Use real HyDE pipeline
            progress_placeholder.info("Generating hypothetical document...")

            # Add documents to HyDE if not already added
            if st.session_state.hyde_rag.vector_store is None and documents:
                st.session_state.hyde_rag.add_documents(documents)

            progress_placeholder.info("Retrieving with HyDE embeddings...")
            result = st.session_state.hyde_rag.query(query)
            full_response = result.answer

            # Log HyDE to debug logger
            debug_log = st.session_state.debug_logger
            if debug_log and st.session_state.debug_enabled:
                debug_log.log_hyde(
                    hypothetical_document=result.hypothetical_document,
                    generation_time=result.hyde_generation_time
                )

            # Show HyDE details
            with debug_placeholder.expander("HyDE Details", expanded=True):
                st.markdown("**Hypothetical Document Generated:**")
                st.text_area("Hypothetical", result.hypothetical_document[:500], height=100, disabled=True)
                st.markdown(f"**HyDE Retrieved:** {result.hyde_retrieval_count} docs")
                st.markdown(f"**Standard Retrieved:** {result.standard_retrieval_count} docs")
                st.markdown(f"**Generation Time:** {result.hyde_generation_time:.1f}s")

        else:
            # HyDE+Self-RAG or fallback streaming strategies
            # Get vector store for real HyDE embedding retrieval
            vector_store = None
            if hasattr(st.session_state.base_rag, 'vector_store'):
                vector_store = st.session_state.base_rag.vector_store

            for chunk in st.session_state.streaming_rag.stream_multi_stage_rag(
                query=query,
                documents=documents,
                use_hyde=(strategy in [RAGStrategy.HYDE, RAGStrategy.HYDE_SELF_RAG]),
                vector_store=vector_store,
                top_k=5,
                on_token=None,
                on_progress=None,
                conversation_history=conversation_history
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "|")

            # If HyDE+Self-RAG, run reflection on the final answer
            if strategy == RAGStrategy.HYDE_SELF_RAG and SELF_RAG_AVAILABLE and st.session_state.self_rag:
                progress_placeholder.info(stages["reflecting"])
                retrieved_docs = []
                if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                    retrieved_docs = st.session_state.base_rag.retrieve_documents(query, k=5)

                if not retrieved_docs:
                    retrieved_docs = documents[:5] if documents else []

                reflection = st.session_state.self_rag.reflect_on_answer(query, full_response, retrieved_docs)
                st.session_state.last_reflection = reflection

                with reflection_placeholder.expander("Reflection Tokens", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        token_class = reflection.relevance.value.lower()
                        st.markdown(f"**Relevance:** <span class='reflection-token {token_class}'>{reflection.relevance.value}</span>", unsafe_allow_html=True)
                    with col2:
                        token_class = reflection.support.value.lower()
                        st.markdown(f"**Support:** <span class='reflection-token {token_class}'>{reflection.support.value}</span>", unsafe_allow_html=True)
                    with col3:
                        token_class = reflection.utility.value.lower()
                        st.markdown(f"**Utility:** <span class='reflection-token {token_class}'>{reflection.utility.value}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Overall Score:** {reflection.overall_score:.2f}")

        progress_placeholder.empty()
        response_placeholder.markdown(full_response)

        # Show retrieved chunks and groundedness check for transparency
        retrieved_docs = _retrieve_context_docs(query)
        if st.session_state.get('show_retrieved_chunks', True):
            _display_retrieved_chunks(retrieved_docs)
        if st.session_state.get('show_groundedness', True):
            _run_groundedness_check(query, full_response, retrieved_docs)

        return full_response

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"


def main():
    # Header
    st.markdown('<div class="main-header">Adaptive Multimodal RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adaptive Query Routing for Document Question Answering</div>', unsafe_allow_html=True)

    # Initialize system
    if not initialize_system():
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Model Selection
        if not st.session_state.available_models:
            st.session_state.available_models = get_available_models()

        available_models = st.session_state.available_models

        # Find index of current model
        current_model = st.session_state.selected_model
        model_index = 0
        if current_model in available_models:
            model_index = available_models.index(current_model)

        selected_model = st.selectbox(
            "LLM Model",
            options=available_models,
            index=model_index,
            help="Select the Ollama model to use. Larger models are more capable but slower."
        )

        # Check if model changed and reinitialize if needed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            # Reset router to force reinitialization with new model
            st.session_state.router = None
            st.session_state.base_rag = None
            st.session_state.streaming_rag = None
            st.session_state.self_rag = None
            st.session_state.graph_rag = None
            st.session_state.hyde_rag = None
            st.rerun()

        st.markdown("---")

        strategy_mode = st.radio(
            "Strategy Selection",
            ["Adaptive (Automatic)", "Manual", "Verification Mode"],
            help="Adaptive: auto-select strategy | Manual: choose strategy | Verification: compare RAG vs LLM-only answers"
        )

        manual_strategy = None
        if strategy_mode == "Verification Mode":
            st.info("**Verification Mode**: Compares answers WITH and WITHOUT document retrieval to verify RAG is working correctly.")

        st.markdown("---")

        st.session_state.show_groundedness = st.checkbox(
            "Show Groundedness Check",
            value=st.session_state.get('show_groundedness', True),
            help="Compare RAG answer vs LLM-only answer after each query to verify documents are being used. Adds ~5-10s per query."
        )
        st.session_state.show_retrieved_chunks = st.checkbox(
            "Show Retrieved Chunks",
            value=st.session_state.get('show_retrieved_chunks', True),
            help="Display the actual document chunks retrieved and fed to the LLM as context."
        )

        if strategy_mode == "Manual":
            strategy_options = ["Baseline RAG", "HyDE RAG", "HyDE + Self-RAG"]
            if SELF_RAG_AVAILABLE:
                strategy_options.append("Self-RAG Only")
            if GRAPHRAG_AVAILABLE:
                strategy_options.append("GraphRAG")

            manual_strategy = st.selectbox(
                "Select Strategy",
                strategy_options,
                help="Baseline: Fast (3s), HyDE: Accurate (35s), Self-RAG: Quality-checked (20s), GraphRAG: Multi-hop reasoning (25s)"
            )

        st.markdown("---")

        # System stats
        st.header("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.total_queries)
        with col2:
            avg_time = st.session_state.total_time / max(st.session_state.total_queries, 1)
            st.metric("Avg Time", f"{avg_time:.1f}s")

        st.metric("Documents", len(st.session_state.documents))
        st.metric("API Cost", "$0.00", help="All processing is local with Ollama")

        # Cache statistics
        if st.session_state.cache_manager:
            with st.expander("Cache Statistics"):
                cache_stats = st.session_state.cache_manager.get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cache Hits", cache_stats["summary"]["total_hits"])
                with col2:
                    st.metric("Cache Misses", cache_stats["summary"]["total_misses"])

                hit_rate = st.session_state.cache_manager.get_hit_rate()
                st.progress(hit_rate, text=f"Hit Rate: {hit_rate:.1%}")

                if st.button("Clear Cache"):
                    st.session_state.cache_manager.clear_all()
                    st.success("Cache cleared!")
                    st.rerun()

        # GraphRAG statistics
        if GRAPHRAG_AVAILABLE and st.session_state.graph_rag:
            graph_stats = st.session_state.graph_rag.get_graph_stats()
            if graph_stats["nodes"] > 0:
                with st.expander("Knowledge Graph"):
                    st.metric("Entities", graph_stats["entities"])
                    st.metric("Relationships", graph_stats["relationships"])
                    st.metric("Communities", graph_stats["communities"])

                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        if st.button("Clear Graph"):
                            st.session_state.graph_rag.clear_graph()
                            st.success("Graph cleared!")
                            st.rerun()
                    with col_g2:
                        if st.button("Clear Graph Cache"):
                            cache_dir = os.path.join("data", "graph_cache")
                            if os.path.exists(cache_dir):
                                shutil.rmtree(cache_dir)
                                st.success("Graph cache cleared")
                            else:
                                st.info("No graph cache to clear")
                            st.rerun()

        # Sample documents for testing
        if len(st.session_state.documents) == 0:
            if st.button("Load Sample Documents", help="Load sample documents for testing the system"):
                sample_docs = [
                    Document(
                        page_content="Machine learning is a branch of artificial intelligence that enables computers to learn from data without explicit programming.",
                        metadata={"source": "SAMPLE: ml_intro", "type": "sample"}
                    ),
                    Document(
                        page_content="Deep learning uses multi-layer neural networks to automatically learn hierarchical representations from data.",
                        metadata={"source": "SAMPLE: dl_intro", "type": "sample"}
                    ),
                    Document(
                        page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with language generation for more accurate responses.",
                        metadata={"source": "SAMPLE: rag_basics", "type": "sample"}
                    )
                ]
                st.session_state.documents.extend(sample_docs)
                st.session_state.base_rag.add_documents(sample_docs)
                st.success("Sample documents loaded for testing")
                st.rerun()

        st.markdown("---")

        # Document upload
        st.header("Documents")

        pdf_mode = st.radio(
            "PDF Processing Mode",
            ["Text Only (Fast)", "Multimodal (Slow but Complete)"],
            help="Text Only: Extract text content only (10x faster). Multimodal: Process images/tables with LLaVA (slower but sees everything)"
        )

        use_multimodal = (pdf_mode == "Multimodal (Slow but Complete)")
        use_text_only = (pdf_mode == "Text Only (Fast)")

        file_types = ['txt', 'md', 'pdf'] if (use_multimodal or use_text_only) else ['txt', 'md']
        if use_multimodal:
            help_text = "Upload documents including PDFs with full multimodal processing (images/tables)"
        elif use_text_only:
            help_text = "Upload documents including PDFs (text extraction only - faster)"
        else:
            help_text = "Upload text files to add to the knowledge base"

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=file_types,
            help=help_text,
            accept_multiple_files=True
        )

        if uploaded_files:
            multimodal_rag = None
            if use_multimodal:
                try:
                    from src.experiments.multimodal.llava_multimodal_rag import LLaVAMultimodalRAG
                    multimodal_rag = LLaVAMultimodalRAG(verbose=False)
                except ImportError:
                    st.error("Multimodal processing not available. Install dependencies: pip install PyMuPDF camelot-py[cv]")
                    multimodal_rag = None

            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")

                    if uploaded_file.type == "application/pdf" and use_text_only:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_pdf_path = temp_file.name

                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(temp_pdf_path)

                            # Create separate documents per page for better retrieval
                            page_docs = []
                            for page_num, page in enumerate(reader.pages, 1):
                                page_text = page.extract_text()
                                if page_text.strip():
                                    # Include page number in content for searchability
                                    page_content = f"[Page {page_num}] {page_text}"
                                    doc = Document(
                                        page_content=page_content,
                                        metadata={
                                            "source": uploaded_file.name,
                                            "type": "pdf_text",
                                            "page": page_num,
                                            "total_pages": len(reader.pages)
                                        }
                                    )
                                    page_docs.append(doc)

                            if page_docs:
                                st.session_state.documents.extend(page_docs)
                                st.session_state.base_rag.add_documents(page_docs)

                            os.unlink(temp_pdf_path)
                            st.success(f"Processed {uploaded_file.name} - Extracted {len(page_docs)} pages (page-level indexing)")

                        except Exception as e:
                            st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
                            try:
                                os.unlink(temp_pdf_path)
                            except OSError:
                                pass

                    elif uploaded_file.type == "application/pdf" and use_multimodal and multimodal_rag:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_pdf_path = temp_file.name

                        try:
                            processed_docs = multimodal_rag.process_pdf(temp_pdf_path)

                            # Initialize counters with default values
                            images_found = 0
                            tables_found = 0
                            text_pieces = 0

                            if isinstance(processed_docs, list) and processed_docs:
                                for doc in processed_docs:
                                    doc.metadata["source"] = uploaded_file.name
                                    st.session_state.documents.append(doc)

                                st.session_state.base_rag.add_documents(processed_docs)

                                images_found = sum(1 for doc in processed_docs if doc.metadata.get('type') == 'image')
                                tables_found = sum(1 for doc in processed_docs if doc.metadata.get('type') == 'table')
                                text_pieces = sum(1 for doc in processed_docs if doc.metadata.get('type') == 'text')

                            os.unlink(temp_pdf_path)
                            st.success(f"Processed {uploaded_file.name} - Found: {images_found} images, {tables_found} tables, {text_pieces} text sections")

                        except Exception as e:
                            st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
                            try:
                                os.unlink(temp_pdf_path)
                            except OSError:
                                pass

                    elif uploaded_file.type == "application/pdf":
                        st.warning(f"PDF {uploaded_file.name} detected. Enable 'Multimodal' to process images and tables.")
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        new_doc = Document(
                            page_content=content,
                            metadata={"source": uploaded_file.name, "type": "uploaded"}
                        )
                        st.session_state.documents.append(new_doc)
                        st.session_state.base_rag.add_documents([new_doc])
                        st.success(f"Added {uploaded_file.name}")

                except Exception as e:
                    st.error(f"Error uploading {uploaded_file.name}: {str(e)}")

            progress_bar.empty()
            status_text.empty()

            if total_files > 1:
                st.info(f"Completed processing {total_files} files. Total documents in system: {len(st.session_state.documents)}")

            # Try to load a cached knowledge graph for these documents
            if GRAPHRAG_AVAILABLE and st.session_state.graph_rag:
                try:
                    import re as _re
                    source_names = set(uf.name for uf in uploaded_files)
                    sanitized = _re.sub(r'[^a-zA-Z0-9]', '_', "_".join(sorted(source_names)))
                    graph_cache_path = os.path.join("data", "graph_cache", f"{sanitized}_graph.json")
                    if os.path.exists(graph_cache_path):
                        st.session_state.graph_rag.load_graph(graph_cache_path)
                        st.success(f"Loaded cached knowledge graph ({len(st.session_state.graph_rag.entities)} entities)")
                except Exception:
                    pass  # Will rebuild on first GraphRAG query

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.query_history = []
                st.rerun()
        with col2:
            if st.button("Clear All Documents"):
                # Properly clear the vector store including persisted data
                if st.session_state.base_rag:
                    st.session_state.base_rag.clear_vector_store()
                st.session_state.documents = []
                # Use the currently selected model, not hardcoded
                st.session_state.base_rag = OllamaRAG(
                    model=st.session_state.selected_model,
                    verbose=False,
                    cache_manager=st.session_state.cache_manager
                )
                st.success("All documents cleared")
                st.rerun()

        st.markdown("---")

        if st.button("Start Fresh Session", help="Clear vector database and all documents for a clean start", type="secondary"):
            if reset_vector_database():
                st.success("Database cleared! Upload new documents to start fresh.")
                st.rerun()
            else:
                st.error("Failed to reset database")

        st.markdown("---")

        # Debug Logging Section
        with st.expander("Debug Logging"):
            st.session_state.debug_enabled = st.checkbox(
                "Enable Debug Logging",
                value=st.session_state.debug_enabled,
                help="Log all queries, retrievals, and responses to debug files"
            )

            if st.session_state.debug_logger:
                log_paths = st.session_state.debug_logger.get_log_paths()

                if log_paths.get("txt"):
                    st.markdown(f"**Log File:** `{log_paths['txt']}`")

                    # Show recent log entries
                    if os.path.exists(log_paths["txt"]):
                        if st.button("View Recent Logs"):
                            try:
                                with open(log_paths["txt"], 'r') as f:
                                    content = f.read()
                                    # Show last 5000 chars
                                    st.text_area("Debug Log (recent)", content[-5000:] if len(content) > 5000 else content, height=300)
                            except Exception as e:
                                st.error(f"Could not read log: {e}")

                        # Download button for full log
                        try:
                            with open(log_paths["txt"], 'r') as f:
                                log_content = f.read()
                            st.download_button(
                                label="Download Full Log",
                                data=log_content,
                                file_name=f"debug_log_{st.session_state.debug_logger.session_id}.txt",
                                mime="text/plain"
                            )
                        except Exception:
                            pass

                if st.button("Start New Debug Session"):
                    st.session_state.debug_logger = init_debug_logger(
                        output_dir="./debug_logs",
                        enabled=True,
                        save_format="both"
                    )
                    st.success(f"New session: {st.session_state.debug_logger.session_id}")
                    st.rerun()

        with st.expander("About"):
            st.markdown("""
**Techniques Implemented:**
- HyDE (Hypothetical Document Embeddings)
- Self-RAG (Reflection Tokens)
- Adaptive Query Routing
- Real-time Streaming
- GraphRAG (Knowledge Graphs)

**Model:** Qwen 2.5 14B (Local)
**Cost:** $0.00 (No API fees)
**Privacy:** Complete (All local)
            """)

    # Main chat interface
    st.header("Chat Interface")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Details"):
                    meta = message["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Strategy:** {get_strategy_badge(meta['strategy'])}", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**Time:** {meta['time']:.2f}s")
                    with col3:
                        st.markdown(f"**Complexity:** {meta['complexity']}/10")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Start debug logging for this query
        debug_log = st.session_state.debug_logger
        if debug_log and st.session_state.debug_enabled:
            debug_log.start_entry(prompt)

        with st.chat_message("assistant"):
            start_time = time.time()

            if strategy_mode == "Verification Mode":
                # Run verification comparison
                with st.spinner("Running verification (comparing RAG vs LLM-only)..."):
                    verification = st.session_state.base_rag.query_with_verification(prompt)

                # Display verification results
                st.markdown("### Verification Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Answer WITH Retrieval (RAG):**")
                    st.markdown(verification["answer_with_retrieval"])
                with col2:
                    st.markdown("**Answer WITHOUT Retrieval (LLM only):**")
                    st.markdown(verification["answer_without_retrieval"])

                # Show verification analysis
                with st.expander("Verification Analysis", expanded=True):
                    st.markdown(f"**Documents Retrieved:** {verification['num_docs_retrieved']}")
                    st.markdown(f"**Context Length:** {verification['context_length']} chars")

                    st.markdown("**Verification Notes:**")
                    for note in verification["verification_notes"]:
                        if "WARNING" in note:
                            st.warning(note)
                        else:
                            st.success(note)

                    if verification["retrieved_docs"]:
                        st.markdown(f"**Retrieved Document Chunks ({len(verification['retrieved_docs'])}):**")
                        for i, doc in enumerate(verification["retrieved_docs"]):
                            page_str = f" | Page {doc['page']}" if doc.get('page') else ""
                            st.markdown(f"**Chunk {i+1}** — `{doc['source']}{page_str}` ({doc['full_length']} chars)")
                            st.text_area(
                                f"verify_doc_{i+1}",
                                doc["content_preview"],
                                height=100,
                                disabled=True,
                                label_visibility="collapsed"
                            )

                response = verification["answer_with_retrieval"]
                selected_strategy = RAGStrategy.BASELINE
                complexity = 0

            elif strategy_mode == "Adaptive (Automatic)":
                with st.spinner("Analyzing query..."):
                    analysis_start = time.time()
                    decision = st.session_state.router.route_query(prompt)
                    selected_strategy = decision.selected_strategy
                    complexity = decision.complexity_score
                    analysis_time = time.time() - analysis_start

                    # Log query analysis
                    if debug_log and st.session_state.debug_enabled:
                        debug_log.log_query_analysis(
                            complexity_score=complexity,
                            complexity_level=decision.complexity_level.value if hasattr(decision, 'complexity_level') else str(complexity),
                            selected_strategy=selected_strategy.value,
                            routing_reasoning=decision.reasoning if hasattr(decision, 'reasoning') else None,
                            analysis_time=analysis_time
                        )

                st.info(f"Selected strategy: **{selected_strategy.value.upper()}** (complexity: {complexity}/10)")
                response = stream_response(prompt, selected_strategy, st.session_state.documents, conversation_history=st.session_state.messages[:-1])
            else:
                strategy_map = {
                    "Baseline RAG": RAGStrategy.BASELINE,
                    "HyDE RAG": RAGStrategy.HYDE,
                    "HyDE + Self-RAG": RAGStrategy.HYDE_SELF_RAG,
                    "Self-RAG Only": RAGStrategy.SELF_RAG,
                    "GraphRAG": RAGStrategy.GRAPHRAG
                }
                selected_strategy = strategy_map.get(manual_strategy, RAGStrategy.BASELINE)
                complexity = 0
                response = stream_response(prompt, selected_strategy, st.session_state.documents, conversation_history=st.session_state.messages[:-1])

            elapsed_time = time.time() - start_time

            st.session_state.total_queries += 1
            st.session_state.total_time += elapsed_time

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "metadata": {
                    "strategy": selected_strategy.value,
                    "time": elapsed_time,
                    "complexity": complexity
                }
            })

            st.session_state.query_history.append({
                "query": prompt,
                "strategy": selected_strategy.value,
                "time": elapsed_time,
                "complexity": complexity,
                "timestamp": time.time()
            })

            # Log retrieval and response to debug logger
            if debug_log and st.session_state.debug_enabled:
                # Log retrieved documents using the deduplicated retrieval method
                try:
                    if hasattr(st.session_state.base_rag, 'retrieve_documents'):
                        retrieved_docs = st.session_state.base_rag.retrieve_documents(prompt, k=10)
                        debug_log.log_retrieval(retrieved_docs, retrieval_time=None)
                except Exception:
                    pass

                # Log Self-RAG reflection if available
                if st.session_state.last_reflection:
                    ref = st.session_state.last_reflection
                    debug_log.log_self_rag_reflection(
                        relevance_token=ref.relevance.value,
                        relevance_score=ref.relevance.score,
                        support_token=ref.support.value,
                        support_score=ref.support.score,
                        utility_token=ref.utility.value,
                        utility_score=ref.utility.score,
                        overall_score=ref.overall_score,
                        reflection_time=ref.reflection_time
                    )

                # Log GraphRAG if used
                if st.session_state.last_reasoning_path:
                    graph_stats = st.session_state.graph_rag.get_graph_stats() if st.session_state.graph_rag else {}
                    debug_log.log_graphrag(
                        entities_used=[step.get('from', '') for step in st.session_state.last_reasoning_path[:5]],
                        relationships_used=[f"{step.get('from', '')} -> {step.get('to', '')}" for step in st.session_state.last_reasoning_path],
                        num_hops=len(st.session_state.last_reasoning_path)
                    )

                # Log final response
                debug_log.log_response(
                    response=response,
                    generation_time=elapsed_time,
                    total_time=elapsed_time
                )

                # Save the entry
                debug_log.save_entry()

    if st.session_state.query_history:
        with st.expander("Query History"):
            for i, q in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                st.markdown(f"""
**{i}.** {q['query'][:60]}{'...' if len(q['query']) > 60 else ''}
- Strategy: {get_strategy_badge(q['strategy'])} | Time: {q['time']:.1f}s | Complexity: {q['complexity']}/10
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
