# RAG Knowledge Pipeline

Multi-source RAG pipeline with adaptive query routing, 3-layer document retrieval, and LLM-as-Judge scoring.
Built with LangChain + LangGraph. Designed around Adaptive RAG, CRAG.

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
│    ↓ 有通過的文件        ↓ 全部被過濾掉                 │
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

當 RAG pipeline 用 `bind_tools` 做 query routing 時，LLM 會直接選接下來回答的工具，但這個做法有一個盲點：我們不知道它有多確定這個決定。遇到問題類型比較模糊的時候，它(LLM)可能選了一個 source 但其實沒把握，導致後面撈到的文件完全不相關。
這邊 `Router` 模組中改成 with_structured_output 後，輸出三個欄位： 1) sources（去哪找）、2) confidence（0–1 的信心分數）、3)reasoning（為什麼這樣選）。當confidence score 低於 threshol 0.6 時，不論目前 LLM 選了什麼工具，設計上都會強制 fallback 到 `web search`。這條 heuristic rule 讓系統對模糊問題有一個保守但可預測的行為，不需要依賴模型每次都判斷正確，也讓 routing 決策變得可解釋、可 debug。

**3-Layer Retrieval**

三層架構的分工如下：
| Layer | 方法 | 作用 |
|-------|------|------|
| 1 | Chroma vector search（cosine similarity） | 快速從大量文件召回候選（k=10） |
| 2 | Cross-Encoder reranker（BAAI/bge-reranker-base） | 對問題和文件配對打分，重新排序取 top 5 |
| 3 | LLM-as-Judge（加權評分） | 最終品質把關，過濾不夠相關的文件並輸出可解釋的評分和過濾理由 |

目前我們的設計中，`Layer 1` 負責速度，`Layer 2` 負責精準度，`Layer 3` 負責可解釋性。
`Layer 2` 用本地模型跑，不花 API，加了這層之後不需要把 k 設很小也能保證最終拿到的文件是相關的。`Layer 3` 除了過濾之外，它輸出的 scores_log 是建立測試集的可參考原始資料，讓評估可以量化，這是單純 binary grader 做不到的。

**Weighted Scoring Rubric**

把 binary grader 換成三個維度的評分，每個維度 1–5 分, 並後續做加權平均：

```
factual_relevance       × 0.5   # 文件有沒有直接回答這個問題
information_sufficiency × 0.3   # 資訊量夠不夠
specificity             × 0.2   # 夠不夠具體，不是泛泛而談
```

若加權分數低於 3.0 的文件會被過濾掉。而這樣設計有兩個具體好處。
- 第一，它讓過濾決策可解釋：與其說「這份文件被過濾掉了」，能說「factual_relevance 只有 2 分，文件提到了相關主題但沒有直接回答問題」。
- 第二，每個維度的權重和 threshold 都是可以調整的參數，好處是每個維度可以獨立調整，也可以透過分析 scores_log 找出哪個 source 的文件品質比較差。

**Config-driven Multi-source**

如果不同的 knowledge source（technical / business / legal / healthcare） 的設定和 `Router` 邏輯是耦合在一起的，每次新增一個 source 就要改 prompt、改 routing function、改 vectorstore 初始化，很容易漏改。
現在的設計做集中管理，把所有 source 的定義集中在 `config.py`(SOURCE_CONFIG（資料在哪裡）和 INDEX_DESCRIPTIONS（這個 source 回答什麼類型的問題）)。新增一個 source 只需要在這兩個 dict 各加一個 key，`Router` 的 prompt 會動態注入 INDEX_DESCRIPTIONS，vectorstore 的建立也是迴圈跑 SOURCE_CONFIG，其他程式碼都不需要動。這讓系統可以在不改架構的情況下擴展到新的 domain。


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

我們可以設計用 gold standard dataset 量化改前改後的效果。每筆測試資料包含問題、預期 routing 目標、以及答案中應該出現的關鍵字。

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

在此我們未將測試題集上傳，避免 test set leakage 讓評估結果失去客觀性。

---

## Stack

- LangChain / LangGraph
- Chroma（vector store）
- OpenAI Embeddings + GPT-3.5-turbo
- BAAI/bge-reranker-base（Cross-Encoder）
- Tavily Search API
