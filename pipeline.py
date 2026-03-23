# pipeline.py
# 負責：GraphState、所有 nodes、conditional edges、build graph、回傳 app

from typing import List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from config import CONFIDENCE_THRESHOLD, RELEVANCE_THRESHOLD, QUERY_REWRITING_ENABLED
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
    question:          str
    rewritten_question: Optional[str] # [Query Rewriting] 改寫後的問題，用於 retrieval
    generation:        str
    documents:         List[str]
    selected_sources:  List[str] # Router 選了哪些 source
    route_confidence:  float # Router 信心分數
    route_reasoning:   str # Router 理由
    scores_log:        List[dict] # 每份文件的三層評分記錄


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
    # [Query Rewriting] 建立 query rewriter chain
    # ============================================================
    # 參考：Rewrite-Retrieve-Read, Ma et al., 2023 (arxiv 2305.14283)
    # 原始問題可能用詞模糊、口語化、或包含對 retrieval 無用的上下文
    # 改寫後的問題更貼近文件的寫法，讓向量搜尋和 BM25 都能找到更相關的結果
    _rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一個搜尋查詢優化專家。"
            "你的任務是把使用者的問題改寫成更適合在文件資料庫中搜尋的形式。\n\n"
            "改寫原則：\n"
            "1. 移除口語化表達，換成更精確的技術用詞\n"
            "2. 展開縮寫或代名詞，讓問題更完整\n"
            "3. 保留所有重要的關鍵字和概念\n"
            "4. 只輸出改寫後的問題，不要任何說明\n\n"
            "如果問題已經很精確，直接回傳原始問題即可。"
        )),
        ("human", "原始問題：{question}\n\n改寫後的問題："),
    ])
    _rewrite_llm   = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    _rewrite_chain = _rewrite_prompt | _rewrite_llm | StrOutputParser()

    # ============================================================
    # Nodes：定義 pipeline 各節點
    # ============================================================

    # --- retrieve（由 Router 決定去哪找資料 + 把資料撈回來）---
    def retrieve(state):
        """
        [Query Rewriting] → Router → Layer 1（Hybrid Search）→ Layer 2（Reranker）

        流程：
        1. [新增] 如果 QUERY_REWRITING_ENABLED，先對問題做 LLM 改寫
           改寫後的問題用於 retrieval；原始問題留著給最終答案生成用
        2. Router 決定去哪個 source
        3. ContextualCompressionRetriever.invoke() 自動執行：
           Layer 1：Hybrid Search（BM25 + 向量搜尋，RRF fusion）
           Layer 2：Cross-Encoder reranker 重排序取 LAYER2_TOP_N 份
        """
        print("---ROUTE & RETRIEVE (Query Rewriting + Layer 1 Hybrid + Layer 2)---")
        question = state["question"]

        # [Query Rewriting] 改寫問題，讓 retrieval 更準確
        if QUERY_REWRITING_ENABLED:
            rewritten = _rewrite_chain.invoke({"question": question})
            rewritten_question = rewritten.strip()
            print(f"  Original question  : {question}")
            print(f"  Rewritten question : {rewritten_question}")
        else:
            rewritten_question = question
            print(f"  Question (no rewrite): {question}")

        # Router 用改寫後的問題做路由決策
        decision: RouteDecision = question_router.invoke({"question": rewritten_question})
        print(f"  Selected sources : {decision.sources}") # Router 選擇的資料來源
        print(f"  Confidence       : {decision.confidence:.2f}") # Router 信心分數
        # print(f"  Reasoning      : {decision.reasoning}") # Router 選擇該資料來源原因 # OPTIMIZE

        # Heuristic fallback：confidence 低於設定標準 → web_search
        selected = decision.sources
        if decision.confidence < CONFIDENCE_THRESHOLD:
            print(f"  -LOW CONFIDENCE ({decision.confidence:.2f}), FALLBACK TO WEB SEARCH-")
            selected = ["web_search"]

        all_documents: List[Document] = []

        # confidence 達到標準或以上 → vector retrieval（用改寫後的問題搜尋）
        for source in selected:
            # web search 不經過 retrieval（也用改寫後的問題搜尋）
            if source == "web_search":
                docs = web_search_tool.invoke({"query": rewritten_question})
                web_docs = [
                    Document(
                        page_content=d["content"],
                        metadata={"source": "web_search", "url": d.get("url", "")}
                    )
                    for d in docs
                ]
                all_documents.extend(web_docs)
                print(f"  -WEB SEARCH: got {len(web_docs)} docs-")

            # 資料透過 retrieval（Layer 1 Hybrid + Layer 2）
            elif source in retrievers:
                docs = retrievers[source].invoke(rewritten_question)
                for doc in docs:
                    doc.metadata["source"] = source
                all_documents.extend(docs)
                print(f"  -{source.upper()}: Layer1(Hybrid)+2 done, got {len(docs)} docs-")

            else:
                print(f"  -UNKNOWN SOURCE '{source}', SKIPPED-")

        # 整理輸出
        # 注意：rewritten_question 存進 state，但 rag_generate 用的是原始 question
        return {
            "documents":          all_documents,
            "question":           question,           # 原始問題：用於最終答案生成
            "rewritten_question": rewritten_question, # 改寫問題：記錄用，已用於 retrieval
            "selected_sources":   selected,
            "route_confidence":   decision.confidence,
            "route_reasoning":    decision.reasoning,
            "scores_log":         [],
        }

    # --- retrieval_grade（幫每份文件打分 -> 過濾文件）---
    def retrieval_grade(state):
        """
        Layer 3：LLM-as-Judge 多維度評分。
        進到這裡的文件已經過 Layer 1（Hybrid Search）和 Layer 2（Cross-Encoder）。
        這裡做最終品質把關，同時把三層分數都記錄進 scores_log。
        """
        print("---[Layer 3] LLM-AS-JUDGE GRADING---")
        documents  = state["documents"]
        question   = state["question"]   # 用原始問題做 grading
        scores_log = state.get("scores_log") or []

        filtered_docs = []

        for d in documents:
            # 取出 Layer 2 的 Cross-Encoder 分數（存在 metadata 裡）
            rerank_score = d.metadata.get("relevance_score", None)
            source       = d.metadata.get("source", "unknown")
            doc_type     = d.metadata.get("doc_type", "original")

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
                "doc_type":                doc_type,  # original / raptor_summary
                "doc_snippet":             d.page_content[:120],
                "reranker_score":          round(rerank_score, 4) if rerank_score is not None else None,
                "factual_relevance":       llm_score.factual_relevance,
                "information_sufficiency": llm_score.information_sufficiency,
                "specificity":             llm_score.specificity,
                "weighted_score":          round(weighted, 3),
                "reasoning":               llm_score.reasoning,
                "passed":                  weighted >= RELEVANCE_THRESHOLD,
            })

            print(f"  Source: {source} [{doc_type}]")  # TODEBUG
            if rerank_score is not None:
                print(f"  Reranker score (Layer 2): {rerank_score:.4f}")
            print(f"  LLM score (Layer 3): {weighted:.2f} "
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

        # OPTIMIZE: 如果 Layer 3 全部過濾掉，可在這裡加 hard fallback 避免空回傳
        # if not filtered_docs:
        #     filtered_docs = documents[:2] # 例如退而求其次，保留 top 2

        return {
            "documents":  filtered_docs,
            "question":   question,
            "scores_log": scores_log,
        }

    # --- web_search_fallback（如果前面全部被 filter 掉 -> 則透過web search）---
    def web_search_fallback(state):
        """所有文件被過濾後補做一次 web search"""
        print("---WEB SEARCH FALLBACK---")
        question   = state["question"]
        # fallback 用原始問題搜尋，保持與最終答案生成的一致性
        search_q   = state.get("rewritten_question") or question
        documents  = state.get("documents") or []
        docs       = web_search_tool.invoke({"query": search_q})
        web_docs   = [Document(page_content=d["content"]) for d in docs]
        return {"documents": documents + web_docs, "question": question}

    # --- RAG answer generation（Context + Question -> Answer）---
    def rag_generate(state):
        print("---GENERATE IN RAG MODE---")
        question   = state["question"]   # 用原始問題生成答案，不用改寫後的問題
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
        # 如果 retrieval grader（Layer 3）把全部文件過濾掉 -> fallback to web search
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
    retrievel流程
    retrieve（Query Rewriting -> Router -> Layer 1 Hybrid -> Layer 2 Reranker）
        v
    retrieval_grade（Layer 3 LLM-as-Judge）
        v
    route_retrieval -> web_search_fallback
        v
    rag_generate（用原始問題生成答案）
        v
    grade_rag_generation
        v
       |-useful -> 結束
       |- not useful -> 答案沒回應問題 -> web search 再重新生成
       |- hallucination -> 重跑
    '''

    # ============================================================
    # Build Graph
    # ============================================================

    # 建立 State Machine（整個 pipeline 傳來傳去的資料，每個 node 都會 *讀state / 改 state* ）
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
            "not useful":    "web_search_fallback",  # 答案沒回應問題 → web search 再重新生成
            "useful":        END,
        },
    )
    workflow.add_edge("plain_answer", END)  # 備用出口（plain_answer 目前未接入主流程入口）

    '''
            [retrieve]
            (Query Rewriting -> Router -> Layer1 Hybrid -> Layer2 Reranker)
                 v
        [retrieval_grade]
          (Layer 3 LLM-as-Judge)
          v          v
       有文件       沒文件
         v            v
    [rag_generate]  [fallback]
         v            v
         └────→ [retrieval_grade]
               v
       [grade_rag_generation]
      v           v              v
   useful    not_useful     not_supported
     v       （答案沒回應）  （hallucination）
    END       fallback          retry
    '''

    # compile
    return workflow.compile()
