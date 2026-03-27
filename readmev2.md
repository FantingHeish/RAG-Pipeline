# RAG Knowledge Pipeline 
A configurable, multi-source Retrieval-Augmented Generation (RAG) pipeline with adaptive query routing, multi-stage retrieval, and LLM-based evaluation. Built with LangChain + LangGraph, this project extends standard RAG systems by introducing structured routing, explainable retrieval scoring, and measurable evaluation, making it suitable for production-oriented and enterprise AI applications. 

## Overview
In many standard RAG implementations, we observed three recurring issues:
- Opaque query routing decisions — models select tools/sources without exposing confidence  
- Binary retrieval grading — documents are accepted/rejected without clear reasoning  
- Lack of evaluation framework — improvements cannot be measured systematically  
These limitations make it difficult to debug failures or improve system performance in a structured way.

#### Design
We designed an adaptive RAG pipeline with following details
- Structured query routing with confidence scoring
- 3-layer retrieval architecture (recall → precision → quality control)
- LLM-based multi-dimensional scoring instead of binary grading
- Gold-standard evaluation pipeline for quantitative comparison

#### Impact
Enables explainable, robust, and measurable RAG performance across following domains
- More robust handling of ambiguous queries  
- Explainable retrieval decisions (not just pass/fail)  
- Systematic performance improvement via evaluation  
- Modular architecture that scales across domains  

#### Supported use cases
- Enterprise knowledge assistants (internal documents, policies)  
- Domain-specific QA systems (finance, healthcare, legal)  
- Hybrid search systems (internal + external knowledge)  

The pipeline is modular, configurable, and extensible, allowing developers to plug in new data sources, retrieval strategies, and evaluation logic.


## Key Features

<details> <summary>Multi-source Data Ingestion</summary>

- Config-driven ingestion across local, web, and extensible sources  
- Centralized source management for scalable expansion  
- Supports:
  - Local documents  
  - Web content (via search APIs)  
  - Cloud storage (extensible)  
- All sources are centrally defined in `config.py`, allowing new data sources to be added without modifying pipeline logic  
- Improves scalability and maintainability in multi-source environments  

</details>



<details>
<summary>Adaptive Query Routing</summary>

- Structured router with interpretable output:
  - `sources`
  - `confidence`
  - `reasoning`  
- Confidence-aware routing decision with threshold-based fallback  
- Automatically routes queries to:
  - vectorstore (internal knowledge)
  - web search (external knowledge)  
- Handles ambiguous or out-of-domain queries via fallback strategy  
- Improves robustness and makes routing decisions debuggable and explainable  

</details>



<details>
<summary>Multi-source Retrieval</summary>

- Multi-collection retrieval across heterogeneous knowledge bases  
- Each source is independently indexed and dynamically selected  
- Hybrid retrieval combining:
  - dense vector search (semantic similarity)
  - external web search (real-time information)  
- Enables cross-domain querying without hardcoded data sources  
- Improves coverage and recall across diverse knowledge distributions  

</details>



<details>
<summary>Multi-stage Ranking Pipeline</summary>

- 3-layer retrieval architecture:

| Layer   | Method                                                                 | Purpose                     |
|--------|------------------------------------------------------------------------|-----------------------------|
| Layer 1 | Vector Search (Chroma, cosine similarity) <br> / Pointwise scoring        | Fast candidate retrieval    |
| Layer 2 | Cross-Encoder Reranker (BAAI/bge-reranker-base) <br> / Pairwise reranking | Precision ranking           |
| Layer 3 | LLM-as-Judge                                                           | Quality filtering + explainability |

- Each layer optimized for a distinct objective:
  - recall → precision → quality control  
- Improves ranking accuracy while maintaining scalability  

</details>



<details>
<summary>Explainable Scoring & Filtering</summary>

- Replaces binary filtering with multi-dimensional scoring  
- Weighted scoring rubric:
  - factual relevance (0.5)  
  - information sufficiency (0.3)  
  - specificity (0.2)  
- Threshold-based filtering (e.g., score < 3.0 removed)  
- Provides structured scoring logs for each document  
- Enables:
  - explainable filtering decisions  
  - fine-grained tuning of scoring behavior  
  - better debugging and evaluation  

</details>



<details>
<summary>Evaluation & Optimization</summary>

- LLM-based evaluation for:
  - answer quality  
  - hallucination detection  
- Gold-standard evaluation dataset including:
  - query
  - expected routing
  - expected answer signals  
- Metrics:
  - routing accuracy  
  - answer quality  
- Structured logging (`scores_log`) for performance analysis  
- Enables iterative improvement and quantitative comparison across pipeline changes  

</details>
