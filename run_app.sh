#!/bin/bash
# Launch script for Adaptive RAG Web UI

echo "=========================================="
echo "Adaptive Multimodal RAG - Web UI"
echo "=========================================="
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running!"
    echo "Please start Ollama first: ollama serve"
    exit 1
fi

# Check if model is available
if ! ollama list | grep -q "qwen2.5:14b"; then
    echo "WARNING: qwen2.5:14b model not found"
    echo "Pulling model... (this may take a while)"
    ollama pull qwen2.5:14b
fi

echo "Starting Web UI..."
echo "Access the app at: http://localhost:8501"
echo ""

# Launch Streamlit
streamlit run app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false
