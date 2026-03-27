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

---

### ▶ Adaptive Query Routing
- Structured router with confidence scoring and reasoning output  
- Confidence-aware fallback strategy for ambiguous queries  

---

### ▶ Multi-source Retrieval
- Multi-collection retrieval across heterogeneous knowledge bases  
- Hybrid retrieval combining vector search and external search  

---

### ▶ Multi-stage Ranking Pipeline
- 3-layer retrieval architecture:
  - Vector search (recall)
  - Cross-encoder reranking (precision)
  - LLM-based filtering (quality control)  
- Multi-stage reranking (pointwise + pairwise + LLM grading)

---

### ▶ Explainable Scoring & Filtering
- Weighted scoring rubric:
  - factual relevance  
  - information sufficiency  
  - specificity  
- Replaces binary filtering with interpretable scoring  

---

### ▶ Evaluation & Optimization
- LLM-based evaluation for answer quality and hallucination detection  
- Gold-standard dataset for routing and response evaluation  
- Structured scoring logs for debugging and iteration 

