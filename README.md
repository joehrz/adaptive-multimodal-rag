# Adaptive Multimodal RAG

A research implementation exploring multiple RAG (Retrieval-Augmented Generation) techniques with adaptive query routing. This project investigates how different retrieval and generation strategies perform across varying query complexities, and whether automatic strategy selection can improve overall system performance.

## Motivation

Standard RAG systems use a fixed retrieval-generation pipeline regardless of query complexity. Simple factual questions get the same treatment as complex analytical queries. This project explores:

1. **Can we classify query complexity automatically?**
2. **Do different RAG techniques perform better for different query types?**
3. **Can adaptive routing between techniques improve results?**

## Implemented Techniques

### HyDE (Hypothetical Document Embeddings)
Based on [Gao et al., 2022](https://arxiv.org/abs/2212.10496). Instead of embedding the query directly, we first generate a hypothetical answer, then use that for retrieval. This bridges the semantic gap between questions and documents.

### Self-RAG with Reflection Tokens
Based on [Asai et al., 2023](https://arxiv.org/abs/2310.11511). The system generates reflection tokens to assess:
- **Relevance**: Are the retrieved documents relevant?
- **Support**: Is the generated answer grounded in the documents?
- **Utility**: Does the answer actually address the question?

If quality is low, the system can regenerate.

### GraphRAG
Builds a knowledge graph from documents by extracting entities and relationships. For queries requiring multi-hop reasoning (e.g., "How does X relate to Y?"), the system traverses the graph to gather connected information before generating an answer.

### Baseline RAG
Standard retrieve-then-generate pipeline using vector similarity search.

## Query Routing

The router analyzes incoming queries and assigns a complexity score (0-10):

| Score | Complexity | Strategy | Rationale |
|-------|------------|----------|-----------|
| 0-3 | Simple | Baseline | Direct factual lookups don't need sophisticated retrieval |
| 4-7 | Medium | HyDE | Explanatory queries benefit from hypothetical document generation |
| 8-10 | Complex | HyDE + Self-RAG | Analytical queries need both better retrieval and quality checks |
| - | Multi-hop | GraphRAG | Relationship queries need graph traversal |
| - | Visual | Multimodal | Queries about images/tables need vision processing |

## Project Structure

```
src/
├── core/
│   ├── ollama_rag.py           # Base RAG implementation
│   └── caching_system.py       # Query/embedding cache
├── experiments/
│   ├── adaptive_routing/       # Query analysis and routing
│   ├── self_reflection/        # Self-RAG implementation
│   ├── graph_reasoning/        # GraphRAG implementation
│   ├── streaming/              # Token streaming for UI
│   └── multimodal/             # OCR + LLaVA processing
└── ocr/
    └── advanced_ocr_engine.py  # Multi-backend OCR
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

### Running

```bash
# Web interface
streamlit run app.py

# Or use the startup script
./run_app.sh
```

## Usage

### Web Interface
The Streamlit app provides:
- Document upload (PDF, TXT, MD)
- Automatic or manual strategy selection
- Streaming responses
- Reflection token visualization (for Self-RAG)
- Graph traversal display (for GraphRAG)

### Python API

```python
from src.core.ollama_rag import OllamaRAG
from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG
from src.experiments.graph_reasoning.ollama_graph_rag import OllamaGraphRAG

# Basic RAG
rag = OllamaRAG(model="qwen2.5:3b")
rag.add_documents(documents)
answer = rag.query("What is X?")

# Self-RAG with reflection
self_rag = OllamaSelfRAG(model="qwen2.5:3b")
result = self_rag.query_with_reflection(query, documents)
print(f"Answer: {result.answer}")
print(f"Quality score: {result.reflection.overall_score}")

# GraphRAG
graph_rag = OllamaGraphRAG(model="qwen2.5:3b")
graph_rag.build_graph_from_documents(documents)
result = graph_rag.query("How does X relate to Y?")
print(f"Reasoning path: {result.reasoning_path}")
```

## Testing

```bash
# Quick import test (no Ollama needed)
python test_all_features.py --quick

# Full tests (requires Ollama running)
python test_all_features.py

# Test with a PDF
python test_all_features.py --pdf /path/to/document.pdf
```

## Models

Tested with:
- **Language**: Qwen 2.5 (3B, 7B, 14B), Llama 3.1, Mistral
- **Vision**: LLaVA 7B/13B (for multimodal)
- **Embeddings**: all-MiniLM-L6-v2

Any Ollama-compatible model should work.

## Limitations

- Query complexity scoring is heuristic-based, not learned
- GraphRAG entity extraction depends on LLM quality
- Multimodal processing is slow (requires vision model inference)
- No formal evaluation benchmarks yet

## References

- Gao et al. (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- Asai et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- Liu et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- Edge et al. (2024). [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)

## License

MIT
