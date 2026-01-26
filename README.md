# Adaptive Multimodal RAG

A production-ready Retrieval-Augmented Generation system implementing adaptive query routing, multimodal document processing, and multiple state-of-the-art RAG techniques. Designed for local deployment using Ollama with zero API costs.

## Overview

This system automatically analyzes incoming queries and selects the optimal processing strategy from multiple available techniques. Unlike fixed-strategy RAG systems, it dynamically balances speed and quality based on query complexity and content type.

### Core Capabilities

- **Adaptive Query Routing**: Automatic complexity analysis and strategy selection
- **Multimodal Processing**: OCR and vision-based understanding of documents with images and tables
- **Multiple RAG Strategies**: HyDE, Self-RAG with reflection tokens, and baseline approaches
- **Local Operation**: Complete privacy using Ollama for LLM inference
- **Business Intelligence**: Cost tracking, ROI calculation, and performance monitoring
- **Web Interface**: Streamlit-based UI with real-time response streaming

## Architecture

The system implements five primary processing strategies:

1. **Baseline RAG**: Fast retrieval for simple queries
2. **HyDE (Hypothetical Document Embeddings)**: Enhanced retrieval using generated hypothetical documents
3. **Self-RAG**: Quality control using reflection tokens ([RETRIEVE], [ISREL], [ISSUP], [ISUSE])
4. **HyDE + Self-RAG**: Combined approach for complex analytical queries
5. **Multimodal RAG**: Hybrid OCR + LLaVA vision processing for documents with visual content

### Query Routing Process

```
Query Input → Complexity Analysis (0-10 scoring) → Strategy Selection → Processing → Response
```

- Complexity 0-3: Baseline RAG
- Complexity 4-7: HyDE-enhanced retrieval  
- Complexity 8-10: HyDE + Self-RAG with reflection
- Visual content detected: Multimodal processing

## Installation

### Prerequisites

```bash
# System requirements
# Minimum: 8GB RAM, 4GB storage
# Recommended: 16GB RAM, NVIDIA GPU with 8GB+ VRAM

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen2.5:3b      # Primary language model
ollama pull llava:7b        # Vision model (optional, for multimodal)
```

### Setup

```bash
git clone https://github.com/yourusername/adaptive-multimodal-rag.git
cd adaptive-multimodal-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Web Interface

```bash
./run_app.sh
# Access at http://localhost:8501
```

### Python API

```python
from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter

# Initialize router
router = OllamaAdaptiveRouter()

# Add documents
router.add_documents(["document1.pdf", "document2.txt"])

# Query with automatic strategy selection
result = router.route_and_process("What are the main findings?")

print(f"Strategy used: {result['strategy']}")
print(f"Answer: {result['answer']}")
```

### Manual Strategy Selection

```python
from src.experiments.hyde_retrieval.hyde_retriever import HyDERetriever
from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG

# Use specific techniques directly
hyde_retriever = HyDERetriever()
self_rag = OllamaSelfRAG()
```

## Multimodal Processing

The system includes hybrid OCR + vision processing for documents containing images, charts, and tables.

### Testing Components

```bash
# Test individual components
python tests/manual/test_multimodal_components.py document.pdf --mode ocr
python tests/manual/test_multimodal_components.py document.pdf --mode llava
python tests/manual/test_multimodal_components.py document.pdf --mode hybrid
```

### OCR Engines

Supports multiple OCR engines with automatic fallback:
- EasyOCR
- PaddleOCR  
- Tesseract
- TrOCR (Transformer-based)

### Vision Processing

Uses LLaVA for image understanding with context-aware prompting based on OCR results.

## Configuration

Edit `config.yaml` for system configuration:

```yaml
# Model settings
default_model: "qwen2.5:3b"
llava_model: "llava:7b"
temperature: 0.3

# RAG settings
chunk_size: 500
chunk_overlap: 50
max_retrieval_docs: 5

# Adaptive routing
quality_weight: 0.7  # Balance between quality (1.0) and speed (0.0)
enable_learning: true

# Multimodal settings
enable_multimodal: true
fusion_strategy: "confidence_weighted"
```

## Performance Monitoring

The system includes comprehensive business intelligence features:

### Cost Tracking
- Query processing time measurement
- Manual vs automated time savings calculation
- ROI analysis with configurable hourly productivity rates

### Performance Analytics
- Strategy selection frequency
- Average processing times by strategy
- Query complexity distribution

### Access Reports

```python
from src.core.business_intelligence import BusinessIntelligence

bi = BusinessIntelligence()
report = bi.generate_roi_report()
print(report)
```

## Testing

### Component Tests

```bash
# Run unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# System validation
python scripts/comprehensive_system_check.py
```

### Manual Testing

```bash
# Test specific strategies
python tests/integration/test_hyde_retrieval.py
python tests/integration/test_self_rag.py
python tests/integration/test_adaptive_routing.py
```

## Supported Models

### Language Models (via Ollama)
- Qwen 2.5 (1.5B, 3B, 7B, 14B)
- Llama 3.1
- Mistral
- Any Ollama-compatible model

### Vision Models
- LLaVA 7B/13B/34B
- BakLLaVA (efficient alternative)

### Embedding Models
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2

## Project Structure

```
src/
├── core/                           # Core RAG components
│   ├── ollama_rag.py              # Base Ollama integration
│   ├── document_processor.py      # Document processing pipeline
│   └── business_intelligence.py   # Cost tracking and ROI
├── experiments/
│   ├── adaptive_routing/          # Query routing and complexity analysis
│   ├── hyde_retrieval/            # HyDE implementation
│   ├── self_reflection/           # Self-RAG with reflection tokens
│   ├── multimodal/               # OCR + Vision processing
│   └── streaming/                # Real-time response streaming
├── ocr/
│   └── advanced_ocr_engine.py    # Multi-engine OCR system
└── debugging/
    └── comprehensive_debugger.py  # System analysis and debugging
```

## Troubleshooting

### Memory Issues
```bash
# Use smaller models
ollama pull qwen2.5:1.5b  # Instead of 3B/7B
ollama pull llava:7b      # Instead of 13B/34B

# Reduce processing parameters
chunk_size: 300           # In config.yaml
max_retrieval_docs: 3
```

### Performance Optimization
```bash
# For speed priority
quality_weight: 0.3
enable_multimodal: false

# For quality priority  
quality_weight: 0.9
max_retrieval_docs: 8
```

### GPU Configuration
```bash
# Check GPU availability
nvidia-smi

# Set GPU for Ollama
export CUDA_VISIBLE_DEVICES=0
```

## Limitations

- Performance benchmarks are implementation-dependent and may vary
- Multimodal processing requires significant computational resources
- GraphRAG implementation is basic entity-relationship extraction
- OCR accuracy depends on document quality and engine selection

## Development

### Adding New RAG Techniques

1. Implement technique in `src/experiments/your_technique/`
2. Add routing logic in `src/experiments/adaptive_routing/`
3. Update strategy enum and complexity scoring
4. Add tests in `tests/experiments/`

### Code Standards

- Python 3.8+ with type hints
- Google-style docstrings
- PEP 8 compliance
- Comprehensive error handling

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation builds on research from:
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Self-RAG: Learning to Critique and Correct](https://arxiv.org/abs/2310.11511)
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

Technology stack includes Ollama, LangChain, ChromaDB, and Streamlit.