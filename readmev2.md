## Overview

In many standard RAG implementations, I observed three recurring issues:

- Opaque query routing decisions — models select tools/sources without exposing confidence  
- Binary retrieval grading — documents are accepted/rejected without clear reasoning  
- Lack of evaluation framework — improvements cannot be measured systematically  

These limitations make it difficult to debug failures or improve system performance in a structured way.


### Decision
<details>
<summary>Designed an adaptive RAG pipeline with structured routing, multi-stage retrieval, and LLM-based evaluation.</summary>

- Structured query routing with confidence scoring
- 3-layer retrieval architecture (recall → precision → quality control)
- LLM-based multi-dimensional scoring instead of binary grading
- Gold-standard evaluation pipeline for quantitative comparison
</details>


### Impact
<details>
<summary>Enables explainable, robust, and measurable RAG performance across domains.</summary>

- More robust handling of ambiguous queries  
- Explainable retrieval decisions (not just pass/fail)  
- Systematic performance improvement via evaluation  
- Modular architecture that scales across domains  
</details>

### Supported use cases
- Enterprise knowledge assistants (internal documents, policies)  
- Domain-specific QA systems (finance, healthcare, legal)  
- Hybrid search systems (internal + external knowledge)  

The pipeline is modular, configurable, and extensible, allowing developers to plug in new data sources, retrieval strategies, and evaluation logic.


## Key Features
<details> <summary>Data Ingestion</summary>

Config-driven multi-source ingestion enabling scalable and extensible data integration.
- Supports:
  - Local documents  
  - Web content (via search APIs)  
  - Cloud storage (extensible)  
- All sources are centrally defined in `config.py`, allowing new data sources to be added without modifying pipeline logic  
- Improves scalability and maintainability in multi-source environments  

</details>

---

### 2.2 Search & Retrieval

<details>
<summary>Hybrid multi-stage retrieval combining structured routing, dense retrieval, and reranking for accuracy and robustness.</summary>

#### 🔹 Multi-collection Retrieval

- Supports querying across multiple knowledge bases  
- Each source is independently indexed and dynamically selected  

Enables flexible retrieval across domains without hardcoding data sources.

#### 🔹 Hybrid Retrieval Strategy

Combines structured routing + dense retrieval + external search to handle different query types.

##### (1) Structured Router with Confidence Score

- Outputs:
  - `sources`  
  - `confidence`  
  - `reasoning`  

- Fallback rule:
  - If confidence < 0.6 → force web search  

Improves robustness for ambiguous queries and makes routing decisions interpretable.

##### (2) 3-Layer Retrieval Architecture

| Layer | Method | Purpose |
|------|--------|--------|
| Layer 1 | Vector Search (Chroma, cosine similarity) | Fast candidate retrieval |
| Layer 2 | Cross-Encoder Reranker (BAAI/bge-reranker-base) | Precision ranking |
| Layer 3 | LLM-as-Judge | Quality filtering + explainability |

Separates concerns across:
- ⚡ Speed  
- 🎯 Accuracy  
- 🔍 Interpretability  


##### (3) Weighted Scoring Rubric

Replaces binary filtering with multi-dimensional scoring:

- factual_relevance (0.5)  
- information_sufficiency (0.3)  
- specificity (0.2)  

Documents below threshold (3.0) are filtered out.

Enables explainable filtering and structured evaluation signals.

##### (4) Multi-stage Reranking

- Pointwise scoring (vector similarity)  
- Pairwise reranking (Cross-Encoder)  
- Final LLM-based grading  

Improves ranking accuracy while maintaining flexibility in retrieval depth.

</details>



### 2.3 Reranking & Accuracy Optimization

<details>
<summary>Enhances retrieval precision while maintaining high recall without aggressive parameter tuning.</summary>

- Cross-Encoder improves semantic matching accuracy  
- Allows larger candidate pools without sacrificing precision  
- Reduces dependency on strict top-k tuning  

Balances recall and precision in a controlled and scalable manner.

</details>

