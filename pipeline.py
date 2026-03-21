# pipeline.py
# 負責：GraphState、所有 nodes、conditional edges、build graph、回傳 app

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from config import CONFIDENCE_THRESHOLD, RELEVANCE_THRESHOLD
from router import RouteDecision, build_question_router
from graders import (
    DocumentScore, compute_weighted_score,
    build_retrieval_grader, build_rag_chain,
    build_llm_chain, build_hallucination_grader, build_answer_grader,
)


# ============================================================
# GraphState
# ============================================================

class GraphState(TypedDict):
    question:         str
    generation:       str
    documents:        List[str]
    selected_sources: List[str]   # Router 選了哪些 source
    route_confidence: float        # Router 信心分數
    route_reasoning:  str          # Router 理由
    scores_log:       List[dict]   # 每份文件的三層評分記錄


# ============================================================
# build_pipeline：組裝所有元件，回傳 compiled app
# ============================================================

def build_pipeline(retrievers: dict):
    """
    組裝整個 RAG pipeline，回傳 compiled app。
    retrievers 由 vectorstore.py 的 build_retrievers() 提供。
    """

    # 初始化所有元件
    question_router      = build_question_router()
    retrieval_grader     = build_retrieval_grader()
    rag_chain            = build_rag_chain()
    llm_chain            = build_llm_chain()
    hallucination_grader = build_hallucination_grader()
    answer_grader        = build_answer_grader()
    web_search_tool      = TavilySearchResults()

    # ============================================================
    # Nodes：定義 pipeline 各節點
    # ============================================================

    # --- retrieve（由 Router 決定去哪找資料 + 把資料撈回來）---
    def retrieve(state):
        """
        Router + Layer 1 + Layer 2：
        1. Router 決定去哪個 source
        2. ContextualCompressionRetriever.invoke() 自動執行：
           Layer 1：向量相似度撈 LAYER1_K 份候選
           Layer 2：Cross-Encoder reranker 重排序取 LAYER2_TOP_N 份
        """
        print("---ROUTE & RETRIEVE (Layer 1 + Layer 2)---")
        question = state["question"]

        decision: RouteDecision = question_router.invoke({"question": question})
        print(f"  Selected sources : {decision.sources}")  # Router 選擇的資料來源
        print(f"  Confidence       : {decision.confidence:.2f}")  # Router 信心分數
        # print(f"  Reasoning      : {decision.reasoning}")  # Router 選擇該資料來源原因 # OPTIMIZE

        # Heuristic fallback：confidence 低於設定標準 → web_search
        selected = decision.sources
        if decision.confidence < CONFIDENCE_THRESHOLD:
            print(f"  -LOW CONFIDENCE ({decision.confidence:.2f}), FALLBACK TO WEB SEARCH-")
            selected = ["web_search"]

        all_documents: List[Document] = []

        # confidence 達到標準或以上 → vector retrieval
        for source in selected:
            # web search 不經過 retrieval
            if source == "web_search":
                docs = web_search_tool.invoke({"query": question})
                web_docs = [
                    Document(
                        page_content=d["content"],
                        metadata={"source": "web_search", "url": d.get("url", "")}
                    )
                    for d in docs
                ]
                all_documents.extend(web_docs)
                print(f"  -WEB SEARCH: got {len(web_docs)} docs-")

            # 資料透過 retrieval（Layer 1 + Layer 2 自動在這一行完成）
            elif source in retrievers:
                docs = retrievers[source].invoke(question)
                for doc in docs:
                    doc.metadata["source"] = source
                all_documents.extend(docs)
                print(f"  -{source.upper()}: Layer1+2 done, got {len(docs)} docs-")

            else:
                print(f"  -UNKNOWN SOURCE '{source}', SKIPPED-")

        # 整理輸出
        return {
            "documents":        all_documents,
            "question":         question,
            "selected_sources": selected,
            "route_confidence": decision.confidence,
            "route_reasoning":  decision.reasoning,
            "scores_log":       [],
        }

    # --- retrieval_grade（幫每份文件打分 → 過濾文件）---
    def retrieval_grade(state):
        """
        Layer 3：LLM-as-Judge 多維度評分。
        進到這裡的文件已經過 Layer 1（向量相似度）和 Layer 2（Cross-Encoder）。
        這裡做最終品質把關，同時把三層分數都記錄進 scores_log。
        """
        print("---[Layer 3] LLM-AS-JUDGE GRADING---")
        documents  = state["documents"]
        question   = state["question"]
        scores_log = state.get("scores_log") or []

        filtered_docs = []

        for d in documents:
            # 取出 Layer 2 的 Cross-Encoder 分數（存在 metadata 裡）
            rerank_score = d.metadata.get("relevance_score", None)
            source       = d.metadata.get("source", "unknown")

            # Layer 3：LLM-as-Judge 打分
            llm_score: DocumentScore = retrieval_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            weighted = compute_weighted_score(llm_score)  # 加權分數

            # 記錄 log（包含三層 retrieval 分數）
            # OPTIMIZE: 未來可用來建 gold standard dataset, 調參, eval
            scores_log.append({
                "question":                question,
                "source":                  source,
                "doc_snippet":             d.page_content[:120],
                "reranker_score":          round(rerank_score, 4) if rerank_score is not None else None,
                "factual_relevance":       llm_score.factual_relevance,
                "information_sufficiency": llm_score.information_sufficiency,
                "specificity":             llm_score.specificity,
                "weighted_score":          round(weighted, 3),
                "reasoning":               llm_score.reasoning,
                "passed":                  weighted >= RELEVANCE_THRESHOLD,
            })

            print(f"  Source: {source}")  # TODEBUG
            if rerank_score is not None:
                print(f"  Reranker score (Layer 2): {rerank_score:.4f}")  # Cross-Encoder 打的分數
            print(f"  LLM score (Layer 3): {weighted:.2f} "  # retrieval grader 的分數並 breakdown
                  f"(F:{llm_score.factual_relevance} "
                  f"S:{llm_score.information_sufficiency} "
                  f"Sp:{llm_score.specificity})")
            print(f"  Reasoning: {llm_score.reasoning}")

            # --- filter ---
            if weighted >= RELEVANCE_THRESHOLD:
                print(f"  -PASS-")
                filtered_docs.append(d)
            else:
                print(f"  -FILTERED-")

        return {
            "documents":  filtered_docs,
            "question":   question,
            "scores_log": scores_log,
        }
    # OPTIMIZE: 做 fallback → 如果 Layer3 全 filter 掉：
    '''
    if not filtered_docs:
        return top_k_docs[:2]  # fallback
    '''

    # --- web_search_fallback（如果前面全部被 filter 掉 → 則透過 web search）---
    def web_search_fallback(state):
        """所有文件被過濾後補做一次 web search"""
        print("---WEB SEARCH FALLBACK---")
        question  = state["question"]
        documents = state.get("documents") or []
        docs      = web_search_tool.invoke({"query": question})
        web_docs  = [Document(page_content=d["content"]) for d in docs]
        return {"documents": documents + web_docs, "question": question}

    # --- RAG answer generation（Context + Question → Answer）---
    def rag_generate(state):
        print("---GENERATE IN RAG MODE---")
        question   = state["question"]
        documents  = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    # --- 直接讓 LLM 回答 ---
    def plain_answer(state):
        """
        備用：直接讓 LLM 回答，不查文件（目前未接入主流程）。
        目前所有問題都從 retrieve 進入，web search 已在 fallback 機制裡處理。
        如果未來 Router 需要區分「純閒聊」路徑，可在 retrieve node 裡加分支接入。
        """
        print("---GENERATE PLAIN ANSWER---")
        question   = state["question"]
        generation = llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}

    # ============================================================
    # Conditional Edges
    # ============================================================

    # --- retrieval fail → fallback control ---
    def route_retrieval(state):
        print("---ROUTE RETRIEVAL---")
        # 如果 retrieval grader（Layer 3）把全部文件過濾掉 → fallback to web search
        if not state["documents"]:
            print("  -ALL DOCS FILTERED, FALLBACK TO WEB SEARCH-")
            return "web_search_fallback"
        # retrieval grader 後有可用文件 → 繼續 RAG
        print("  -RELEVANT DOCS FOUND, GENERATE-")
        return "rag_generate"

    # --- 檢查 LLM 產生的答案 ---
    def grade_rag_generation(state):
        # check hallucinations?
        print("---CHECK HALLUCINATIONS---")
        question   = state["question"]
        documents  = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        # 如果沒有 hallucination
        if score.binary_score == "no":
            print("  -GROUNDED IN DOCUMENTS-")
            # 檢查是否回答問題
            score2 = answer_grader.invoke({"question": question, "generation": generation})
            if score2.binary_score == "yes":
                print("  -ADDRESSES QUESTION-")
                return "useful"
            print("  -DOES NOT ADDRESS QUESTION-")
            return "not useful"
        print("  -HALLUCINATION DETECTED, RE-TRY-")
        return "not supported"

    '''
    retrieve
       ↓
    retrieval_grade
       ↓
    route_retrieval ───────→ web_search_fallback
       ↓
    rag_generate
       ↓
    grade_rag_generation
       ↓
       ├── useful → 結束 
       ├── not useful → # 答案沒回應問題 → 補做 web search 再重新生成
       └── hallucination → 重跑
    '''

    # ============================================================
    # Build Graph
    # ============================================================

    # 建立 State Machine（整個 pipeline 傳來傳去的資料，每個 node 都會「讀 state + 改 state」）
    workflow = StateGraph(GraphState)

    # 把 function 建立成 node，放進 graph
    workflow.add_node("retrieve",            retrieve)
    workflow.add_node("retrieval_grade",     retrieval_grade)
    workflow.add_node("web_search_fallback", web_search_fallback)
    workflow.add_node("rag_generate",        rag_generate)
    workflow.add_node("plain_answer",        plain_answer)

    # 設定 pipeline 起點
    workflow.set_entry_point("retrieve")
    # 設定固定流程
    workflow.add_edge("retrieve", "retrieval_grade")

    # --- pipeline 條件分支 ---

    # 分支 1：retrieval_grade → route_retrieval（依評分回傳 web_search_fallback 或 rag_generate）
    workflow.add_conditional_edges(
        "retrieval_grade",
        route_retrieval,
        {
            "web_search_fallback": "web_search_fallback",
            "rag_generate":        "rag_generate",
        },
    )
    # 如果回傳 web_search_fallback，會重新評分新抓的文件（產生 fallback）
    workflow.add_edge("web_search_fallback", "retrieval_grade")

    # 分支 2：rag_generate → grade_rag_generation（判斷 hallucination、有沒有回答問題）
    workflow.add_conditional_edges(
        "rag_generate",
        grade_rag_generation,
        {
            "not supported": "rag_generate",        # rag_generate 重新生成
            "not useful":    "web_search_fallback",  # 重新評分新抓的文件（產生 fallback）
            "useful":        END,
        },
    )
    workflow.add_edge("plain_answer", END)  # 備用出口（plain_answer 目前未接入主流程入口）

    '''
            [retrieve]
                 ↓
        [retrieval_grade]
             ↓       ↓
       有文件       沒文件
         ↓            ↓
    [rag_generate]  [fallback]
         ↓            ↓
         └────→ [retrieval_grade]
               ↓
       [grade_rag_generation]
      ↓           ↓              ↓
   useful    not_useful     not_supported
     ↓       （答案沒回應）  （hallucination）
    END       fallback          retry
    '''

    # compile
    return workflow.compile()
