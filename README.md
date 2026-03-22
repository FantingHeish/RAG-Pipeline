# RAG Knowledge Pipeline

Multi-source RAG pipeline with adaptive query routing, 3-layer document retrieval, and LLM-as-Judge scoring.

Built with LangChain + LangGraph. Designed around [Adaptive RAG](https://arxiv.org/pdf/2403.14403.pdf), [CRAG](https://arxiv.org/pdf/2401.15884.pdf), and [Self-RAG](https://arxiv.org/pdf/2310.11511.pdf).

---

## Background

Most RAG tutorials use a binary yes/no grader to filter retrieved documents, which makes it hard to understand why a document gets filtered out or how to improve retrieval quality systematically.

This project extends the baseline in three directions:

- **Query routing** — replaced `bind_tools` with structured JSON output so routing decisions include a confidence score and a reasoning explanation, not just a destination
- **Retrieval** — added a Cross-Encoder reranking layer and replaced the binary grader with a 3-dimension weighted scoring rubric
- **Evaluation** — built a gold standard dataset to measure routing accuracy and answer quality before and after each change

---

## Architecture

```
Query
  ↓
┌─────────────────────────────────────────────────────┐
│ retrieve node                                       │
│                                                     │
│  [Router] structured JSON + confidence score        │
│    ├── confidence < 0.6 → force web_search          │
│    └── confidence ≥ 0.6 → selected source(s)        │
│              ↓                      ↓               │
│    [Layer 1] Vector Search     [Web Search]         │
│    Chroma cosine similarity,                        │
│    k=10 (Pointwise)                                 │
│              ↓                                      │
│    [Layer 2] Cross-Encoder Reranker                 │
│    BAAI/bge-reranker-base,                          │
│    top_n=5 (Pairwise)                               │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ retrieval_grade node                                │
│                                                     │
│  [Layer 3] LLM-as-Judge                             │
│  3-dimension weighted scoring, threshold=3.0        │
│                                                     │
│    ↓ 有通過的文件        ↓ 全部被過濾掉              │
└─────────────────────────────────────────────────────┘
  ↓                             ↓
[rag_generate]        [web_search_fallback]
  ↓                             ↓
  └──────────────────────────── ↓
                       [retrieval_grade]（重新評分）
  ↓
[grade_rag_generation]
  ↓              ↓                    ↓
useful        not_useful          not_supported
  ↓          （答案沒回應問題）    （hallucination）
 END          web_search_fallback   rag_generate
                    ↓                   （重新生成）
            [retrieval_grade]

```

---

## Design Decisions

**Structured Router with Confidence Score**

原本用 `bind_tools` 讓 LLM 直接選工具，沒辦法知道它有多確定這個決定。
改成 `with_structured_output` 後，Router 會輸出三個欄位：`sources`（去哪找）、`confidence`（0–1 的信心分數）、`reasoning`（為什麼這樣選）。

設計了一條 heuristic rule：confidence 低於 0.6 時，不管選了哪個 source，都強制 fallback 到 web search。這讓系統在面對模糊問題時有一個保守但合理的預設行為，不需要依賴模型完全判斷正確。

**3-Layer Retrieval**

| Layer | 方法 | 作用 |
|-------|------|------|
| 1 | Chroma vector search（cosine similarity） | 快速從大量文件召回候選（k=10） |
| 2 | Cross-Encoder reranker（BAAI/bge-reranker-base） | 對問題和文件配對打分，重新排序取 top 5 |
| 3 | LLM-as-Judge（加權評分） | 最終品質把關，過濾不夠相關的文件 |

Layer 2 用本地小模型跑，不花 API。Layer 3 雖然最貴，但它輸出的評分記錄是建 gold standard dataset 的原始資料，讓評估可以量化。

**Weighted Scoring Rubric**

把 binary grader 換成三個維度的評分，每個維度 1–5 分：

```
factual_relevance      × 0.5   # 文件有沒有直接回答這個問題
information_sufficiency × 0.3   # 資訊量夠不夠
specificity            × 0.2   # 夠不夠具體，不是泛泛而談
```

加權分數低於 3.0 的文件會被過濾掉。這個設計的好處是每個維度可以獨立調整，也可以透過分析 scores_log 找出哪個 source 的文件品質比較差。

**Config-driven Multi-source**

四個 knowledge source（technical / business / legal / healthcare）透過 `SOURCE_CONFIG` 和 `INDEX_DESCRIPTIONS` 集中管理。要新增一個 source，只需要在這兩個地方加一個 key，Router 的 prompt 會自動更新，不需要動其他程式碼。

---

## Project Structure

```
RAG-Pipeline/
├── config.py        # 所有設定：API keys、retrieval 參數、SOURCE_CONFIG
├── vectorstore.py   # 文件載入、chunking、Chroma vectorstore、三層 retriever 組裝
├── router.py        # RouteDecision schema、router prompt、confidence-based fallback
├── graders.py       # DocumentScore、加權評分、RAG chain、hallucination/answer grader
├── pipeline.py      # LangGraph pipeline：GraphState、nodes、conditional edges
├── evaluation.py    # evaluate_pipeline()、print_comparison()、save_scores_log()
└── main.py          # 入口點
```

---

## Setup

```bash
pip install langchain langchain-openai chromadb langgraph pypdf \
            langchain_community pydantic gdown requests \
            beautifulsoup4 sentence-transformers
```

在 `config.py` 填入 API keys 和文件路徑：

```python
# config.py
OPENAI_API_KEY = "your-key"
TAVILY_API_KEY = "your-key"

SOURCE_CONFIG = {
    "healthcare": {
        "local_folder": "/path/to/your/docs",
        # 或 "pdf_urls": ["https://..."]
        # 或 "gdrive_folder": "https://drive.google.com/..."
    },
    ...
}
```

執行：

```bash
python main.py
```

---

## Evaluation

用 gold standard dataset 量化改前改後的效果。每筆測試資料包含問題、預期 routing 目標、以及答案中應該出現的關鍵字。

跑完評估後用 `print_comparison()` 印出對比：

```
============================================================
EVALUATION COMPARISON: Baseline vs New
============================================================
Metric                    Baseline        New      Delta
------------------------------------------------------------
route_accuracy              65.0%       82.4%   ↑  17.4%
answer_quality              70.0%       88.2%   ↑  18.2%
============================================================
```

測試題集不放進版本控制，避免 test set leakage 讓評估結果失去客觀性。

---

## Stack

- LangChain / LangGraph
- Chroma（vector store）
- OpenAI Embeddings + GPT-3.5-turbo
- BAAI/bge-reranker-base（Cross-Encoder）
- Tavily Search API
