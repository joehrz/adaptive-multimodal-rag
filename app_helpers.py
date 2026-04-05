"""
Helper functions for the Streamlit UI.
Extracted from app.py for modularity.
"""

import os
import shutil
import streamlit as st
from typing import List

from langchain.schema import Document

from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter, RAGStrategy
from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer
from src.experiments.streaming.ollama_streaming_rag import OllamaStreamingRAG
from src.core.ollama_rag import OllamaRAG
from src.core.config import get_config
from src.core.debug_logger import init_debug_logger

try:
    from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG
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

config = get_config()


def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        import ollama
        models = ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return [config.llm.model]


def initialize_system(model: str = None):
    """Initialize RAG components with selected model."""
    selected_model = model or st.session_state.selected_model or config.llm.model

    if st.session_state.debug_logger is None and st.session_state.debug_enabled:
        st.session_state.debug_logger = init_debug_logger(
            output_dir="./debug_logs",
            enabled=True,
            save_format="both"
        )

    if st.session_state.router is None:
        with st.spinner(f"Initializing Adaptive RAG System with {selected_model}..."):
            try:
                analyzer = OllamaQueryAnalyzer(model=selected_model, verbose=False)
                st.session_state.router = OllamaAdaptiveRouter(
                    query_analyzer=analyzer, verbose=False)
                st.session_state.streaming_rag = OllamaStreamingRAG(
                    model=selected_model, verbose=False)
                st.session_state.base_rag = OllamaRAG(
                    model=selected_model,
                    verbose=False,
                )
                st.session_state.cache_manager = st.session_state.base_rag.cache_manager

                if SELF_RAG_AVAILABLE:
                    st.session_state.self_rag = OllamaSelfRAG(
                        model=selected_model, verbose=False)

                if GRAPHRAG_AVAILABLE:
                    st.session_state.graph_rag = OllamaGraphRAG(
                        model=selected_model, verbose=False)

                if HYDE_AVAILABLE:
                    st.session_state.hyde_rag = OllamaHyDE(
                        model=selected_model, verbose=False)

                if not st.session_state.documents:
                    st.session_state.documents = []

                return True
            except Exception as e:
                st.error(f"Failed to initialize system: {str(e)}")
                return False
    return True


def reset_vector_database():
    """Clear vector database for fresh session."""
    try:
        if hasattr(st.session_state, 'base_rag') and st.session_state.base_rag:
            if hasattr(st.session_state.base_rag, 'clear_vector_store'):
                st.session_state.base_rag.clear_vector_store()
            else:
                if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store is not None:
                    try:
                        st.session_state.base_rag.vector_store.delete_collection()
                    except Exception:
                        pass
                    st.session_state.base_rag.vector_store = None
                st.session_state.base_rag.documents = []

        try:
            cfg = get_config()
            persist_dir = cfg.vector_db.persist_directory
        except Exception:
            persist_dir = "./data/chroma_db_ollama"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        st.session_state.documents = []
        if 'query_history' in st.session_state:
            st.session_state.query_history = []
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_time = 0.0

        if st.session_state.cache_manager:
            st.session_state.cache_manager.clear_all()

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
    """Get HTML badge for strategy."""
    badge_class = strategy.lower().replace(' ', '_').replace('+', '_')
    return f'<span class="strategy-badge {badge_class}">{strategy.upper()}</span>'


def retrieve_context_docs(query: str) -> List[Document]:
    """Retrieve documents used as context for the query."""
    try:
        if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
            return st.session_state.base_rag.retrieve_documents(query, k=10)
    except Exception:
        pass
    return []


def display_retrieved_chunks(retrieved_docs: List[Document]):
    """Display retrieved document chunks in an expander."""
    if not retrieved_docs:
        return

    with st.expander(f"Retrieved Context ({len(retrieved_docs)} chunks)", expanded=False):
        st.caption("These are the actual document chunks retrieved from your uploaded files and fed to the LLM as context.")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            page_str = f" | Page {page}" if page else ""
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            st.markdown(f"**Chunk {i+1}** -`{source}{page_str}`")
            st.text_area(
                f"chunk_{i+1}",
                content_preview,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )


def run_groundedness_check(query: str, response: str, retrieved_docs: List[Document]):
    """Run a lightweight groundedness check comparing RAG vs no-retrieval answers."""
    if not retrieved_docs:
        return

    with st.expander("Groundedness Check", expanded=False):
        st.caption("Compares the RAG answer against an answer generated without document context to verify retrieval is being used.")

        with st.spinner("Generating LLM-only answer for comparison..."):
            llm_only_answer = st.session_state.base_rag._generate_response(query, context="", require_citations=False)

        if llm_only_answer.startswith("Error generating"):
            st.error(f"Could not generate LLM-only answer: {llm_only_answer}")
            return

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
                st.error("HIGH -LLM may be using training data")
            elif overlap > 0.5:
                st.warning("MEDIUM -Partial reliance on training data")
            else:
                st.success("LOW -RAG is providing unique info")

        has_citations = "[Document" in response or "[Doc" in response
        if has_citations:
            st.success("Answer contains document citations -good sign RAG is grounding the response")
        else:
            st.info("No explicit citations found. Consider if the answer references specific details only found in your documents.")
