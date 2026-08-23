# pipeline.py
# 負責：GraphState、所有 nodes、conditional edges、build graph、回傳 app
#
# [架構調整] 線上檢索過濾不再呼叫 LLM-as-Judge：
#   - Layer 1 (retrieve)：Hybrid Search，回傳候選文件
#   - Layer 2 (retrieval_grade)：Cross-Encoder rerank，用 relevance_score 門檻過濾
#     （原本這裡是 LLM batch 評分，現在 LLM-as-Judge 只在 offline_evaluate.py 離線用，
#      拿來幫 reranker 產生訓練資料，不再是每個問題都要付的線上成本——見 README 的說明）
#   - 生成後品質檢查合併成一次 LLM 呼叫（graders.build_answer_quality_grader），
#     取代原本 hallucination_grader + answer_grader 兩次呼叫
#   - retry_count：迴圈保護，超過 MAX_RETRIES 強制走 plain_answer / mark_low_confidence
#   - web_search 呼叫全部包 try/except，key 沒設定時自動跳過不中斷

from typing import List, Optional

import numpy as np
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from config import (
    RERANK_SCORE_THRESHOLD, QUERY_REWRITING_ENABLED,
    EMBEDDING_PREFILTER_ENABLED, EMBEDDING_PREFILTER_THRESHOLD,
    MAX_RETRIES, TAVILY_API_KEY,
)
from graders import build_rag_chain, build_llm_chain, build_answer_quality_grader
from vectorstore import build_reranker_model, rerank_documents


# ============================================================
# GraphState
# ============================================================

class GraphState(TypedDict):
    question:           str
    rewritten_question:  Optional[str]
    generation:          str
    documents:           List[Document]
    scores_log:          List[dict]
    retry_count:         int   # 迴圈保護計數器
    quality_check:       Optional[dict]  # {"is_grounded", "addresses_question", "reasoning"}，見 check_answer_quality


# ============================================================
# Helper：embedding 粗篩
# ============================================================

def _embedding_prefilter(question: str, documents: List[Document], embeddings, threshold: float) -> List[Document]:
    """
    送進 Cross-Encoder rerank 之前，先用向量 cosine similarity 濾掉明顯不相關的文件，
    減少 reranker 要處理的量。設計上刻意保守：篩完變成空 list 就直接回傳原始 documents（避免誤殺全部文件）。
    """
    if not embeddings or not documents:
        return documents

    try:
        q_vec = np.array(embeddings.embed_query(question))
        texts = [d.page_content for d in documents]
        doc_vecs = np.array(embeddings.embed_documents(texts))

        q_norm = np.linalg.norm(q_vec) + 1e-8
        doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-8
        sims = (doc_vecs @ q_vec) / (doc_norms * q_norm)

        kept = [d for d, s in zip(documents, sims) if s >= threshold]
        dropped = len(documents) - len(kept)
        if dropped > 0:
            print(f"  [Embedding Prefilter] dropped {dropped}/{len(documents)} docs below similarity {threshold}")

        return kept if kept else documents
    except Exception as e:
        print(f"  [Embedding Prefilter] WARNING: failed ({e}), skipping prefilter for this call.")
        return documents


# ============================================================
# build_pipeline
# ============================================================

def build_pipeline(retriever, embeddings=None, reranker_model=None):
    """
    組裝整個 RAG pipeline，回傳 compiled app。
    retriever：vectorstore.py 的 build_retriever() 提供的 Layer 1 hybrid retriever（不含 rerank）。
    embeddings：可選。有提供的話，retrieval_grade 會先做 embedding 粗篩再送進 reranker。
    reranker_model：可選。不提供的話用 config.RERANKER_MODEL_NAME 現場載入一個
        （要指向微調過的 checkpoint 時，直接把 vectorstore.build_reranker_model(自己的路徑) 傳進來）。
    """

    rag_chain            = build_rag_chain()
    llm_chain             = build_llm_chain()
    answer_quality_grader = build_answer_quality_grader()
    cross_encoder         = reranker_model if reranker_model is not None else build_reranker_model()

    # 沒設定 TAVILY_API_KEY 時，直接不建立 web_search_tool，避免建構時因為缺 key 而丟例外
    # 讓整個 pipeline 組裝失敗；web_search_fallback 節點會檢查這個是不是 None 再決定要不要跳過。
    web_search_tool = None
    if TAVILY_API_KEY:
        try:
            web_search_tool = TavilySearchResults()
        except Exception as e:
            print(f"[WARNING] TavilySearchResults 初始化失敗 ({e})，web_search 相關節點會被自動跳過。")
    else:
        print("[WARNING] TAVILY_API_KEY 未設定，web_search 相關節點會被自動跳過（不會中斷 pipeline）。")

    # ============================================================
    # Query Rewriting chain
    # ============================================================
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
    # Nodes
    # ============================================================

    def retrieve(state):
        """Layer 1：Hybrid Search（BM25 + Vector，EnsembleRetriever 內部用 RRF 融合），回傳候選文件"""
        print("---[Layer 1] HYBRID SEARCH RETRIEVE---")
        question = state["question"]

        if QUERY_REWRITING_ENABLED:
            try:
                rewritten = _rewrite_chain.invoke({"question": question})
                rewritten_question = rewritten.strip()
            except Exception as e:
                print(f"  -QUERY REWRITE ERROR ({e}), 直接用原始問題-")
                rewritten_question = question
            print(f"  Original question  : {question}")
            print(f"  Rewritten question : {rewritten_question}")
        else:
            rewritten_question = question
            print(f"  Question (no rewrite): {question}")

        try:
            docs = retriever.invoke(rewritten_question)
        except Exception as e:
            print(f"  -RETRIEVAL ERROR: {e}-")
            docs = []

        for d in docs:
            d.metadata["source"] = d.metadata.get("source", "local_docs")

        print(f"  -Retrieved {len(docs)} candidate docs-")

        return {
            "documents":          docs,
            "question":           question,
            "rewritten_question": rewritten_question,
            "scores_log":         state.get("scores_log") or [],
            "retry_count":        state.get("retry_count", 0),
        }

    def retrieval_grade(state):
        """
        Layer 2：Cross-Encoder rerank + relevance_score 門檻過濾。
        流程：(可選) embedding 粗篩 -> cross-encoder 重排（自己算分數、寫回 metadata）-> 門檻過濾
        不再呼叫 LLM-as-Judge——那個角色現在只在 offline_evaluate.py 離線跑，
        用來產生訓練資料微調 reranker，不是每個問題都要付的線上成本。
        """
        print("---[Layer 2] CROSS-ENCODER RERANK + THRESHOLD FILTER---")
        documents  = state["documents"]
        question   = state["question"]
        scores_log = state.get("scores_log") or []

        if not documents:
            print("  -NO DOCUMENTS TO RERANK-")
            return {"documents": [], "question": question, "scores_log": scores_log}

        candidates = documents
        if EMBEDDING_PREFILTER_ENABLED and embeddings is not None:
            candidates = _embedding_prefilter(question, documents, embeddings, EMBEDDING_PREFILTER_THRESHOLD)

        try:
            scored_docs = rerank_documents(question, candidates, cross_encoder)
        except Exception as e:
            print(f"  -RERANK ERROR ({e})，這批文件視為未通過（會自然導向 web_search_fallback 補資料）-")
            return {"documents": [], "question": question, "scores_log": scores_log}

        filtered_docs = []
        for d in scored_docs:
            score  = d.metadata.get("relevance_score", 0.0)
            passed = score >= RERANK_SCORE_THRESHOLD
            source = d.metadata.get("source", "unknown")

            scores_log.append({
                "question":        question,
                "source":          source,
                "doc_snippet":     d.page_content[:120],
                "relevance_score": round(score, 4),
                "passed":          passed,
            })
            print(f"  [{source}] relevance_score={score:.3f} {'-PASS-' if passed else '-FILTERED-'}")

            if passed:
                filtered_docs.append(d)

        return {
            "documents":  filtered_docs,
            "question":   question,
            "scores_log": scores_log,
        }

    def web_search_fallback(state):
        """本地文件都被過濾後補做一次 web search。加 try/except + retry_count 遞增"""
        print("---WEB SEARCH FALLBACK---")
        question  = state["question"]
        search_q  = state.get("rewritten_question") or question
        documents = state.get("documents") or []
        retry_count = state.get("retry_count", 0) + 1

        if web_search_tool is None:
            print("  -SKIPPED: web_search_tool not available (no TAVILY_API_KEY / init failed)-")
            return {"documents": documents, "question": question, "retry_count": retry_count}

        try:
            docs = web_search_tool.invoke({"query": search_q})
            web_docs = [
                Document(page_content=d.get("content", ""), metadata={"source": "web_search", "url": d.get("url", "")})
                for d in docs
            ]
        except Exception as e:
            print(f"  -WEB SEARCH ERROR: {e}-")
            web_docs = []

        print(f"  -WEB SEARCH: got {len(web_docs)} docs-")
        return {"documents": documents + web_docs, "question": question, "retry_count": retry_count}

    def rag_generate(state):
        """每次執行(包含重試)都遞增 retry_count，供 route_after_quality_check 判斷是否該強制結束"""
        print("---GENERATE IN RAG MODE---")
        question    = state["question"]
        documents   = state["documents"]
        retry_count = state.get("retry_count", 0) + 1
        try:
            generation = rag_chain.invoke({"documents": documents, "question": question})
        except Exception as e:
            print(f"  -GENERATION ERROR ({e})-")
            generation = ""  # 空字串是給 check_answer_quality 判斷「這次生成失敗，需要補資料」的訊號
        return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count}

    def plain_answer(state):
        """
        直接讓 LLM 回答，不查文件。
        用途：(1) 備用出口 (2) MAX_RETRIES 用盡後的 give_up 路徑（會標註警語）
        這是整個 pipeline 的最後一道防線，就算連這裡的 LLM 呼叫都失敗（例如 API 完全打不通），
        也要回傳一句固定的文字訊息，不能讓例外往上炸到 Streamlit 介面。
        """
        print("---GENERATE PLAIN ANSWER (give up on retrieval/verification loop)---")
        question = state["question"]
        try:
            generation = llm_chain.invoke({"question": question})
        except Exception as e:
            print(f"  -PLAIN ANSWER ERROR ({e})-")
            generation = "抱歉，系統目前暫時無法產生回答（LLM 服務呼叫失敗），請稍後再試一次。"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            generation += "\n\n（提醒：本回答經過多次檢索/驗證仍無法確認品質，以上為 LLM 直接回答，請自行核實。）"
        return {"question": question, "generation": generation}

    def mark_low_confidence(state):
        """
        check_answer_quality/route_after_quality_check 因為撞到 MAX_RETRIES 而放棄繼續重試時，會先經過這個節點。

        這裡分兩種情況處理，不是一律「保留原本的生成內容 + 加警語」：
          - quality_check.is_grounded 明確是 False（最後一次評分認定是幻覺、
            內容沒有文件依據）：這種答案留著弊大於利，直接丟掉，改用純 LLM
            知識重新回答一次（跟 plain_answer 一樣的邏輯），並清楚標註這不是
            根據檢索文件生成的。
          - 其他情況（is_grounded 是 True，只是沒完全回答到問題；或
            is_grounded 是 None，代表根本沒評過分，例如一開始就撞上
            MAX_RETRIES）：保留原本生成內容，加警語即可，因為至少有文件依據，
            比重新生成一個完全不看文件的答案更可靠。
        """
        print("---MARK LOW CONFIDENCE (accepted after retries exhausted)---")
        question   = state["question"]
        generation = state.get("generation", "")
        qc         = state.get("quality_check") or {}

        not_grounded = qc.get("is_grounded") is False  # 明確評過分且判定為幻覺，才視為「不可靠」
        documents_update = {}

        if not generation or not_grounded:
            if not generation:
                print("  -GENERATION STILL EMPTY, FALLING BACK TO PLAIN LLM ANSWER-")
            else:
                print("  -LAST GENERATION WAS NOT GROUNDED (可能是幻覺), DISCARDING, FALLING BACK TO PLAIN LLM ANSWER-")
            try:
                generation = llm_chain.invoke({"question": question})
            except Exception as e:
                print(f"  -PLAIN ANSWER FALLBACK ERROR ({e})-")
                generation = "抱歉，系統目前暫時無法產生回答（LLM 服務呼叫失敗），請稍後再試一次。"
            generation += (
                "\n\n（提醒：多次檢索/生成後仍無法找到可靠依據，以上為 LLM 直接根據自身知識回答，"
                "並非根據檢索到的文件，請自行核實。）"
            )
            # 改用純 LLM 回答後，原本那些「不可靠、答案已經判定跟它們沒有實際關聯」的文件
            # 不該再顯示成這次回答的來源，避免使用者誤以為這個答案是有文件依據的。
            documents_update = {"documents": []}
        else:
            generation += "\n\n（提醒：本回答經過多次生成/驗證仍可能未完整回應問題，請自行核實。）"

        return {"generation": generation, **documents_update}

    # ============================================================
    # Conditional Edges
    # ============================================================

    def route_retrieval(state):
        print("---ROUTE RETRIEVAL---")
        if not state["documents"]:
            if state.get("retry_count", 0) >= MAX_RETRIES:
                print(f"  -MAX_RETRIES({MAX_RETRIES}) REACHED, GIVE UP -> PLAIN ANSWER-")
                return "give_up"
            print("  -ALL DOCS FILTERED, FALLBACK TO WEB SEARCH-")
            return "web_search_fallback"
        print("  -RELEVANT DOCS FOUND, GENERATE-")
        return "rag_generate"

    def check_answer_quality(state):
        """
        生成後的唯一一次線上品質檢查（合併「有沒有幻覺」+「有沒有回答到問題」成一次 LLM 呼叫）。
        這是一個真正的 node（不是 conditional edge 函式），把評分結果寫進
        state["quality_check"]，讓 app.py 之後可以從最終 state 讀到、記錄進
        QUALITY_LOG_PATH——原本這段邏輯直接放在 conditional edge 函式裡，只回傳
        路由用的字串，評分結果算完就丟掉了，從來沒有真的存進 state 過。

        LLM 呼叫本身失敗（API 錯誤/逾時），保守地把 is_grounded/addresses_question
        記成 False，並在 reasoning 註明是呼叫失敗，而不是讓例外往上炸掉整個 pipeline。
        """
        print("---CHECK ANSWER QUALITY---")

        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"  -MAX_RETRIES({MAX_RETRIES}) REACHED, ACCEPT CURRENT ANSWER (低信心)-")
            return {"quality_check": {
                "is_grounded": None, "addresses_question": None,
                "reasoning": f"MAX_RETRIES({MAX_RETRIES}) 已達上限，強制接受目前答案，未實際評分",
            }}

        question   = state["question"]
        documents  = state["documents"]
        generation = state.get("generation", "")

        if not generation:
            # rag_generate 那一步本身就失敗了（generation 是空字串），
            # 不用再浪費一次 LLM call 去評分空答案，直接當作「需要補資料」處理。
            print("  -GENERATION WAS EMPTY (rag_generate failed), TREAT AS NOT SUPPORTED-")
            return {"quality_check": {
                "is_grounded": False, "addresses_question": False,
                "reasoning": "rag_generate 沒有產生任何內容，跳過評分",
            }}

        try:
            result = answer_quality_grader.invoke({
                "documents": documents, "question": question, "generation": generation,
            })
        except Exception as e:
            print(f"  -ANSWER QUALITY GRADER ERROR ({e}), 保守當作需要補資料-")
            return {"quality_check": {
                "is_grounded": False, "addresses_question": False,
                "reasoning": f"評分呼叫失敗：{e}",
            }}

        if not result.is_grounded:
            print(f"  -HALLUCINATION DETECTED ({result.reasoning}), RE-TRY-")
        elif not result.addresses_question:
            print(f"  -DOES NOT ADDRESS QUESTION ({result.reasoning})-")
        else:
            print(f"  -GROUNDED & ADDRESSES QUESTION ({result.reasoning})-")

        return {"quality_check": {
            "is_grounded":        result.is_grounded,
            "addresses_question": result.addresses_question,
            "reasoning":          result.reasoning,
        }}

    def route_after_quality_check(state):
        """
        只負責路由，不呼叫 LLM——check_answer_quality 這個 node 已經把評分結果存進
        state["quality_check"]，這裡單純讀出來決定下一步，邏輯跟原本評分/路由
        合併在同一個函式裡時完全一樣，只是拆成兩個步驟。
        """
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "useful_low_confidence"

        qc = state.get("quality_check") or {}
        if not qc.get("is_grounded"):
            return "not supported"
        if not qc.get("addresses_question"):
            return "not useful"
        return "useful"

    # ============================================================
    # Build Graph
    # ============================================================

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve",             retrieve)
    workflow.add_node("retrieval_grade",      retrieval_grade)
    workflow.add_node("web_search_fallback",  web_search_fallback)
    workflow.add_node("rag_generate",         rag_generate)
    workflow.add_node("check_answer_quality", check_answer_quality)
    workflow.add_node("plain_answer",         plain_answer)
    workflow.add_node("mark_low_confidence",  mark_low_confidence)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "retrieval_grade")

    workflow.add_conditional_edges(
        "retrieval_grade",
        route_retrieval,
        {
            "web_search_fallback": "web_search_fallback",
            "rag_generate":        "rag_generate",
            "give_up":             "plain_answer",
        },
    )
    workflow.add_edge("web_search_fallback", "retrieval_grade")

    workflow.add_edge("rag_generate", "check_answer_quality")
    workflow.add_conditional_edges(
        "check_answer_quality",
        route_after_quality_check,
        {
            # 幻覺 -> 先補一次 web search 拿更多資料，再回頭重新評分/生成，
            # 而不是原地用同一批（可能本來就不夠）的文件再生成一次。
            "not supported":         "web_search_fallback",
            "not useful":            "web_search_fallback",
            "useful":                END,
            "useful_low_confidence": "mark_low_confidence",
        },
    )
    workflow.add_edge("mark_low_confidence", END)
    workflow.add_edge("plain_answer", END)

    return workflow.compile()
