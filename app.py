"""
Adaptive Multimodal RAG - Streamlit Web UI
Production-ready web interface with all SOTA techniques
"""

import streamlit as st
import sys
import time
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src" / "experiments" / "adaptive_routing"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "experiments" / "streaming"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "experiments" / "graph_reasoning"))

from langchain.schema import Document

# Import components
from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter, RAGStrategy
from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer
from src.experiments.streaming.ollama_streaming_rag import OllamaStreamingRAG
from src.core.ollama_rag import OllamaRAG
from src.core.caching_system import RAGCacheManager

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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'router' not in st.session_state:
    st.session_state.router = None
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


def initialize_system():
    """Initialize RAG components"""
    if st.session_state.router is None:
        with st.spinner("Initializing Adaptive RAG System..."):
            try:
                # Initialize cache manager first
                st.session_state.cache_manager = RAGCacheManager(enable_auto_cleanup=True)

                # Initialize components
                analyzer = OllamaQueryAnalyzer(verbose=False)
                st.session_state.router = OllamaAdaptiveRouter(
                    query_analyzer=analyzer, verbose=False)
                st.session_state.streaming_rag = OllamaStreamingRAG(
                    model="qwen2.5:14b", verbose=False)
                st.session_state.base_rag = OllamaRAG(
                    model="qwen2.5:14b",
                    verbose=False,
                    cache_manager=st.session_state.cache_manager
                )

                # Initialize Self-RAG if available
                if SELF_RAG_AVAILABLE:
                    st.session_state.self_rag = OllamaSelfRAG(
                        model="qwen2.5:14b", verbose=False)

                # Initialize GraphRAG if available
                if GRAPHRAG_AVAILABLE:
                    st.session_state.graph_rag = OllamaGraphRAG(
                        model="qwen2.5:14b", verbose=False)

                # Initialize empty document list
                if not st.session_state.documents:
                    st.session_state.documents = []

                return True
            except Exception as e:
                st.error(f"Failed to initialize system: {str(e)}")
                return False
    return True


def reset_vector_database():
    """Clear vector database for fresh session"""
    try:
        # Clear the vector store if it exists
        if hasattr(st.session_state, 'base_rag') and st.session_state.base_rag:
            if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store is not None:
                try:
                    st.session_state.base_rag.vector_store.delete_collection()
                except Exception:
                    pass  # Collection might not exist

                st.session_state.base_rag.vector_store = None

            st.session_state.base_rag.documents = []

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


def stream_response(query: str, strategy: RAGStrategy, documents: List[Document]):
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
        with debug_placeholder.expander("Debug: Query Processing", expanded=False):
            st.write(f"**Query:** {query}")
            st.write(f"**Strategy:** {strategy.value}")
            st.write(f"**Total documents available:** {len(documents)}")

            try:
                if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                    retrieved_docs = st.session_state.base_rag.vector_store.similarity_search(query, k=5)
                    st.write(f"**Retrieved {len(retrieved_docs)} documents for this query:**")
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get('source', 'Unknown')
                        doc_type = doc.metadata.get('type', 'Unknown')
                        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        st.text_area(f"Retrieved Doc {i+1}: {source} ({doc_type})", content_preview, height=80, disabled=True)
            except Exception as debug_e:
                st.write(f"Could not retrieve debug info: {debug_e}")

        # Handle different strategies
        if strategy == RAGStrategy.BASELINE:
            progress_placeholder.info(stages["generating"])
            response = st.session_state.base_rag.query(query)
            if isinstance(response, dict):
                full_response = response.get('answer', str(response))
            else:
                full_response = str(response)

        elif strategy == RAGStrategy.SELF_RAG and SELF_RAG_AVAILABLE and st.session_state.self_rag:
            progress_placeholder.info(stages["generating"])

            # Get retrieved documents
            retrieved_docs = []
            if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                retrieved_docs = st.session_state.base_rag.vector_store.similarity_search(query, k=5)

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

            # Query the graph
            progress_placeholder.info(stages["graph_traversing"])
            result = st.session_state.graph_rag.query(query)
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

        else:
            # HyDE, HyDE+Self-RAG, or other streaming strategies
            for chunk in st.session_state.streaming_rag.stream_multi_stage_rag(
                query=query,
                documents=documents,
                use_hyde=(strategy in [RAGStrategy.HYDE, RAGStrategy.HYDE_SELF_RAG]),
                on_token=None,
                on_progress=None
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "|")

            # If HyDE+Self-RAG, run reflection on the final answer
            if strategy == RAGStrategy.HYDE_SELF_RAG and SELF_RAG_AVAILABLE and st.session_state.self_rag:
                progress_placeholder.info(stages["reflecting"])
                retrieved_docs = []
                if hasattr(st.session_state.base_rag, 'vector_store') and st.session_state.base_rag.vector_store:
                    retrieved_docs = st.session_state.base_rag.vector_store.similarity_search(query, k=5)

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

        return full_response

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"


def main():
    # Header
    st.markdown('<div class="main-header">Adaptive Multimodal RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Question Answering with State-of-the-Art Techniques</div>', unsafe_allow_html=True)

    # Initialize system
    if not initialize_system():
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        strategy_mode = st.radio(
            "Strategy Selection",
            ["Adaptive (Automatic)", "Manual"],
            help="Adaptive automatically selects the best strategy based on query complexity"
        )

        manual_strategy = None
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

                    if st.button("Clear Graph"):
                        st.session_state.graph_rag.clear_graph()
                        st.success("Graph cleared!")
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

                            full_text = ""
                            for page_num, page in enumerate(reader.pages, 1):
                                page_text = page.extract_text()
                                if page_text.strip():
                                    full_text += f"\n\n--- Page {page_num} ---\n{page_text}"

                            if full_text.strip():
                                doc = Document(
                                    page_content=full_text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "type": "pdf_text",
                                        "pages": len(reader.pages)
                                    }
                                )
                                st.session_state.documents.append(doc)
                                st.session_state.base_rag.add_documents([doc])

                            os.unlink(temp_pdf_path)
                            st.success(f"Processed {uploaded_file.name} - Extracted text from {len(reader.pages)} pages (fast mode)")

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
                            documents = multimodal_rag.process_pdf(temp_pdf_path)

                            if isinstance(documents, list):
                                for doc in documents:
                                    doc.metadata["source"] = uploaded_file.name
                                    st.session_state.documents.append(doc)

                                st.session_state.base_rag.add_documents(documents)

                                images_found = sum(1 for doc in documents if doc.metadata.get('type') == 'image')
                                tables_found = sum(1 for doc in documents if doc.metadata.get('type') == 'table')
                                text_pieces = sum(1 for doc in documents if doc.metadata.get('type') == 'text')

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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.query_history = []
                st.rerun()
        with col2:
            if st.button("Clear All Documents"):
                st.session_state.documents = []
                st.session_state.base_rag = OllamaRAG(model="qwen2.5:14b", verbose=False)
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

        with st.chat_message("assistant"):
            start_time = time.time()

            if strategy_mode == "Adaptive (Automatic)":
                with st.spinner("Analyzing query..."):
                    decision = st.session_state.router.route_query(prompt)
                    selected_strategy = decision.selected_strategy
                    complexity = decision.complexity_score

                st.info(f"Selected strategy: **{selected_strategy.value.upper()}** (complexity: {complexity}/10)")
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

            response = stream_response(prompt, selected_strategy, st.session_state.documents)

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

    if st.session_state.query_history:
        with st.expander("Query History"):
            for i, q in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                st.markdown(f"""
**{i}.** {q['query'][:60]}{'...' if len(q['query']) > 60 else ''}
- Strategy: {get_strategy_badge(q['strategy'])} | Time: {q['time']:.1f}s | Complexity: {q['complexity']}/10
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
