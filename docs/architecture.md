# Architecture

Detailed diagrams of the Adaptive Multimodal RAG system internals.

## End-to-End Pipeline

How a query flows through the system from input to response.

```mermaid
flowchart TB
    A[User Query] --> B[Cache Lookup]
    B -->|Cache Hit| C[Return Cached Response]
    B -->|Cache Miss| D[Query Analyzer]

    D --> E[Heuristic Analysis]
    D --> F[LLM Analysis]
    E --> G["Weighted Score (30% heuristic + 70% LLM)"]
    F --> G

    G --> H[Adaptive Router]
    H --> I[Strategy Execution]
    I --> J[Document Retrieval]
    J --> K[Cross-Encoder Reranking]
    K --> L[LLM Generation with Streaming]
    L --> M[Cache Response]
    M --> N[Return Response with Citations]
```

## Query Analysis

The analyzer combines fast heuristic signals with LLM judgment to score query complexity on a 0-10 scale.

```mermaid
flowchart LR
    A[Query] --> B[Heuristic Scoring]
    A --> C[LLM Scoring]

    subgraph Heuristics
        B --> B1[Query length & word count]
        B --> B2[Comparison keywords\ncompare, contrast, vs]
        B --> B3[Analysis keywords\nanalyze, evaluate, impact]
        B --> B4[Simple question patterns\nwhat, who, when, define]
        B --> B5[Technical terms\nalgorithm, architecture]
    end

    subgraph LLM
        C --> C1["Prompt: score 0-10\n0-3 simple, 4-7 medium, 8-10 complex"]
    end

    B1 & B2 & B3 & B4 & B5 --> D[Heuristic Score]
    C1 --> E[LLM Score]
    D -->|Weight: 0.3| F[Final Complexity Score]
    E -->|Weight: 0.7| F
```

## Routing Decision Tree

The router checks query characteristics in priority order. The first matching rule wins.

```mermaid
flowchart TB
    A[Query + Complexity Score] --> B{Contains visual\nkeywords?}
    B -->|"image, figure,\nchart, diagram"| C["Multimodal\n(LLaVA)"]
    B -->|No| D{Relationship\npatterns?}
    D -->|"relationship between,\nhow does X affect Y,\nconnection between"\n+ score >= 6| E[GraphRAG]
    D -->|No| F{Summarization\nkeywords?}
    F -->|"summarize, overview,\nkey findings"| G{Score >= 8?}
    G -->|Yes| H[HyDE + Self-RAG]
    G -->|No| I[HyDE]
    F -->|No| J{Complexity\nScore}
    J -->|0-3| K[Baseline RAG]
    J -->|4-7| I
    J -->|8-10| H

    style C fill:#e1bee7,stroke:#9C27B0
    style E fill:#bbdefb,stroke:#2196F3
    style H fill:#ffcdd2,stroke:#E91E63
    style I fill:#fff9c4,stroke:#FFC107
    style K fill:#c8e6c9,stroke:#4CAF50
```

## Strategy Details

What each RAG strategy does after routing.

```mermaid
flowchart TB
    subgraph Baseline["Baseline RAG"]
        direction TB
        BA1[Query] --> BA2[Vector Similarity Search]
        BA2 --> BA3[Cross-Encoder Reranking]
        BA3 --> BA4[LLM Generation]
    end

    subgraph HyDE["HyDE"]
        direction TB
        HY1[Query] --> HY2[Generate Hypothetical Answer]
        HY2 --> HY3[Embed Hypothetical Document]
        HY3 --> HY4[Retrieve Similar Documents]
        HY4 --> HY5[Cross-Encoder Reranking]
        HY5 --> HY6[LLM Generation]
    end

    subgraph SelfRAG["Self-RAG"]
        direction TB
        SR1[Retrieved Context] --> SR2[Generate Answer]
        SR2 --> SR3[Reflection Tokens]
        SR3 --> SR4{Quality\nPassed?}
        SR4 -->|"Relevance, Support,\nUtility scores OK"| SR5[Return Answer]
        SR4 -->|Below threshold| SR6[Regenerate]
        SR6 --> SR3
    end

    subgraph Graph["GraphRAG"]
        direction TB
        GR1[Documents] --> GR2[Extract Entities\n& Relationships]
        GR2 --> GR3[Build Knowledge Graph]
        GR3 --> GR4[Graph Traversal\nMulti-hop Paths]
        GR4 --> GR5[Collect Connected Context]
        GR5 --> GR6[LLM Generation]
    end

    style Baseline fill:#c8e6c9,stroke:#4CAF50
    style HyDE fill:#fff9c4,stroke:#FFC107
    style SelfRAG fill:#ffcdd2,stroke:#E91E63
    style Graph fill:#bbdefb,stroke:#2196F3
```

## Document Processing Pipeline

How uploaded PDFs become searchable vector embeddings.

```mermaid
flowchart LR
    A[PDF Upload] --> B[Text Extraction]
    B --> C[OCR Fallback\nEasyOCR / Tesseract]
    B --> D{Chunking\nStrategy}
    D -->|Recursive| E[Fixed-Size Chunks\nwith Overlap]
    D -->|Semantic| F[Embedding Similarity\nSplit at Topic Boundaries]
    E & F --> G[Generate Embeddings]
    G --> H{Backend}
    H -->|HuggingFace| I[all-MiniLM-L6-v2]
    H -->|Ollama| J[nomic-embed-text]
    I & J --> K[(ChromaDB\nVector Store)]
```

## Caching Layers

The caching system sits between the user and the pipeline to avoid redundant computation.

```mermaid
flowchart TB
    A[Incoming Query] --> B{Semantic Query\nCache}
    B -->|Hit: similar query\npreviously answered| C[Return Cached Response]
    B -->|Miss| D[Run Full Pipeline]
    D --> E[Retrieve Documents]
    E --> F{Vector Search\nCache}
    F -->|Hit: embeddings\ncached| G[Return Cached Vectors]
    F -->|Miss| H[Compute Embeddings]
    H --> I[Cache Vectors]
    G & I --> J[Continue Pipeline]
    J --> K[Cache Final Response]
    K --> L[Return Response]

    style B fill:#fff9c4,stroke:#FFC107
    style F fill:#bbdefb,stroke:#2196F3
```
