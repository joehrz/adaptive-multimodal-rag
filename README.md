# Adaptive Multimodal RAG

A research project exploring automatic RAG strategy selection based on query complexity. The system analyzes incoming queries, assigns a complexity score, and routes them to the most appropriate retrieval and generation strategy.

## Motivation

Standard RAG systems use a fixed retrieval pipeline regardless of query complexity. Simple factual questions get the same treatment as complex analytical queries. This project explores whether automatic routing between different RAG techniques can improve results.

## Implemented Techniques

### Baseline RAG
Standard retrieve-then-generate pipeline using vector similarity search with cross-encoder reranking.

### HyDE (Hypothetical Document Embeddings)
Based on [Gao et al., 2022](https://arxiv.org/abs/2212.10496). Generates a hypothetical answer first, then uses it for retrieval. This bridges the semantic gap between questions and documents.

### Self-RAG with Reflection Tokens
Based on [Asai et al., 2023](https://arxiv.org/abs/2310.11511). Generates reflection tokens to assess relevance, support, and utility of retrieved documents and generated answers. Regenerates if quality is low.

### GraphRAG
Based on [Edge et al., 2024](https://arxiv.org/abs/2404.16130). Builds a knowledge graph from documents by extracting entities and relationships. For multi-hop queries, traverses the graph to gather connected information before generating.

### Cross-Encoder Reranking
Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to rerank retrieved documents by relevance. The initial retrieval fetches a larger candidate set, then the cross-encoder scores each query-document pair to select the most relevant chunks.

### Multimodal Processing
Uses LLaVA for vision-based question answering on images, figures, and charts extracted from documents.

## Query Routing

The router analyzes incoming queries and assigns a complexity score (0-10):

| Score | Complexity | Strategy | Rationale |
|-------|------------|----------|-----------|
| 0-3 | Simple | Baseline | Direct factual lookups |
| 4-7 | Medium | HyDE | Explanatory queries benefit from hypothetical document generation |
| 8-10 | Complex | HyDE + Self-RAG | Analytical queries need better retrieval and quality checks |
| 6+ | Multi-hop | GraphRAG | Relationship queries need graph traversal |
| any | Visual | Multimodal | Queries about images, figures, or charts |

Summarization queries (summary, overview, key findings, main contribution) are routed to HyDE regardless of complexity score, since they require document synthesis rather than direct lookup.

## Project Structure

```
src/
├── core/
│   ├── ollama_rag.py           # Base RAG with cross-encoder reranking
│   ├── config.py               # Configuration loading
│   └── caching_system.py       # Query and embedding cache
├── experiments/
│   ├── adaptive_routing/       # Query analysis and routing
│   ├── self_reflection/        # Self-RAG implementation
│   ├── graph_reasoning/        # GraphRAG implementation
│   ├── streaming/              # Token streaming for UI
│   └── multimodal/             # LLaVA vision processing
└── ocr/
    └── advanced_ocr_engine.py  # Multi-backend OCR (EasyOCR, Tesseract)
```

## Setup

### Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) for local LLM inference
- 8GB+ RAM (16GB recommended)

### Installation

```bash
git clone https://github.com/joehrz/adaptive-multimodal-rag.git
cd adaptive-multimodal-rag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

### Docker

```bash
docker compose up --build
```

The Docker setup runs both the Ollama server and the Streamlit app. Set `RAG_OLLAMA_URL` to override the Ollama endpoint.

### Running

```bash
# Web interface
streamlit run app.py

# CLI for testing and debugging
python cli_query.py
```

## Configuration

Edit `config.yaml` to customize models, reranking, retrieval parameters, and routing thresholds. Key settings:

- `llm.model`: Ollama model (default: qwen2.5:14b)
- `reranker.enabled`: Toggle cross-encoder reranking (default: true)
- `reranker.candidates`: Number of initial candidates before reranking (default: 30)
- `reranker.top_k`: Number of documents kept after reranking (default: 10)
- `documents.chunk_size`: Document chunk size in characters (default: 1000)

Environment variables `RAG_OLLAMA_URL` and `RAG_OLLAMA_TIMEOUT` override the corresponding config values.

## Testing

```bash
# Unit tests (no Ollama needed)
pytest tests/

# Integration tests against real PDFs (requires Ollama running)
python tests/integration/run_quality_tests.py
```

## Models

Tested with:
- **Language**: Qwen 2.5 (3B, 7B, 14B), Llama 3.1, Mistral
- **Vision**: LLaVA 7B/13B (for multimodal)
- **Embeddings**: all-MiniLM-L6-v2
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

Any Ollama-compatible model should work for language and vision.

## Limitations

- Query complexity scoring is heuristic-based, not learned
- GraphRAG entity extraction depends on LLM quality
- Multimodal processing is slow (requires vision model inference)

## References

- Gao et al. (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- Asai et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- Liu et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- Edge et al. (2024). [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)

## License

MIT
