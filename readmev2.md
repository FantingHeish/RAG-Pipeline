## 1. Overview

In many standard RAG implementations, I observed three recurring issues:

- Opaque query routing decisions — models select tools/sources without exposing confidence  
- Binary retrieval grading — documents are accepted/rejected without clear reasoning  
- Lack of evaluation framework — improvements cannot be measured systematically  

These limitations make it difficult to debug failures or improve system performance in a structured way.


### 1.1 Decision
<details>
<summary>Designed an adaptive RAG pipeline with structured routing, multi-stage retrieval, and LLM-based evaluation.</summary>

- Structured query routing with confidence scoring
- 3-layer retrieval architecture (recall → precision → quality control)
- LLM-based multi-dimensional scoring instead of binary grading
- Gold-standard evaluation pipeline for quantitative comparison
</details>


### 1.2 Impact
<details>
<summary>Enables explainable, robust, and measurable RAG performance across domains.</summary>

- More robust handling of ambiguous queries  
- Explainable retrieval decisions (not just pass/fail)  
- Systematic performance improvement via evaluation  
- Modular architecture that scales across domains  
</details>

### 1.3 Supported use cases
- Enterprise knowledge assistants (internal documents, policies)  
- Domain-specific QA systems (finance, healthcare, legal)  
- Hybrid search systems (internal + external knowledge)  

The pipeline is modular, configurable, and extensible, allowing developers to plug in new data sources, retrieval strategies, and evaluation logic.


## 2.Key Features
### 2.1 Data Ingestion
•	Config-driven multi-source ingestion 
<details>
  <summary>•	Supports: </summary>
o	Local documents 
o	Web content (via search APIs) 
o	Cloud storage (extensible) 
</details>
All sources are centrally defined in config.py, allowing new data sources to be added without modifying pipeline logic, improving scalability and maintainability.

### 2.2 Search & Retrieval
🔹 Multi-collection Retrieval
•	Supports querying across multiple knowledge bases 
•	Each source is independently indexed and dynamically selected 
This enables flexible retrieval across domains without hardcoding data sources.
🔹 Hybrid Retrieval Strategy
Combines structured routing + dense retrieval + external search to handle different query types.
(1) Structured Router with Confidence Score
•	Outputs: 
o	sources 
o	confidence 
o	reasoning 
•	Fallback rule: 
o	If confidence < 0.6 → force web search 
This design improves robustness for ambiguous queries while making routing decisions interpretable and debuggable.
(2) 3-Layer Retrieval Architecture
Layer	Method	Purpose
Layer 1	Vector Search (Chroma, cosine similarity)	Fast candidate retrieval
Layer 2	Cross-Encoder Reranker (BAAI/bge-reranker-base)	Precision ranking
Layer 3	LLM-as-Judge	Quality filtering + explainability
This layered design separates concerns:
•	⚡ Speed (Layer 1) 
•	🎯 Accuracy (Layer 2) 
•	🔍 Interpretability (Layer 3) 
(3) Weighted Scoring Rubric
Replaces binary filtering with multi-dimensional scoring:
•	factual_relevance (0.5)
•	information_sufficiency (0.3) 
•	specificity (0.2) 
Documents below threshold (3.0) are filtered out.
Benefits:
•	Explainable filtering decisions 
•	Tunable scoring system 
•	Enables deeper evaluation and debugging 
This enables more granular and explainable filtering compared to binary grading.
(4) Multi-stage Reranking
•	Pointwise scoring (vector similarity) 
•	Pairwise reranking (Cross-Encoder) 
•	Final LLM-based grading 
This improves ranking accuracy while maintaining flexibility in retrieval depth.
### 2.3 Reranking & Accuracy Optimization
•	Cross-Encoder improves semantic matching accuracy 
•	Allows larger candidate pools without sacrificing precision 
•	Reduces dependency on strict top-k tuning 
This design allows the system to balance recall and precision without aggressive parameter tuning.
<img width="468" height="634" alt="image" src="https://github.com/user-attachments/assets/5588c80d-f60f-4ae3-b381-766fecea1b85" />

