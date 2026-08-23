# pipeline_baseline.py 
'''
兩種 baseline，跟 pipeline.py 的 adaptive RAG 做三欄 A/B/C 對比：

  1) build_no_retrieval_pipeline()：完全不檢索，LLM 純憑自身知識回答。
     用來看「有沒有檢索」這件事本身值不值得——沒有這條線，naive RAG 跟 adaptive RAG
     的分數差異沒辦法告訴你「檢索有沒有用」，只能告訴你「兩種檢索方式誰比較好」。

  2) build_naive_pipeline()：naive RAG baseline，沒有 hybrid search、沒有 rerank、
     沒有 LLM-as-judge 過濾，單純對文件做向量相似度搜尋 top-k，直接生成答案。
     用來看「adaptive 那些機制（hybrid/rerank/門檻過濾/retry/web_search）有沒有讓答案更準」。

三條線一起比較（見 main.py），可以同時看出「檢索有沒有用」跟「檢索做得好不好」兩層差異。
'''

from typing import List

from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from graders import build_rag_chain, build_llm_chain

NAIVE_TOP_K = 5  # 純向量搜尋抓幾份文件


# ============================================================
# A. 完全不檢索：LLM 純憑自身知識回答
# ============================================================

class NoRetrievalState(TypedDict):
    question:   str
    documents:  List[Document]
    generation: str


def build_no_retrieval_pipeline():
    """
    完全不查任何文件，直接把問題丟給 LLM。documents 固定是空 list，
    只是為了讓這條 pipeline 的輸出格式（generation + documents）
    跟另外兩條一致，evaluation.py 的 run()/run_all() 不用特別區分著寫。
    """
    llm_chain = build_llm_chain()

    def no_retrieval_generate(state):
        print("---[NO-RETRIEVAL] PLAIN LLM (完全不查文件)---")
        question = state["question"]
        try:
            generation = llm_chain.invoke({"question": question})
        except Exception as e:
            print(f"  -ERROR: {e}-")
            generation = "抱歉，系統目前暫時無法產生回答（LLM 服務呼叫失敗），請稍後再試一次。"
        return {"documents": [], "question": question, "generation": generation}

    workflow = StateGraph(NoRetrievalState)
    workflow.add_node("no_retrieval_generate", no_retrieval_generate)
    workflow.set_entry_point("no_retrieval_generate")
    workflow.add_edge("no_retrieval_generate", END)

    return workflow.compile()


# ============================================================
# B. Naive RAG：純向量檢索，無 hybrid/rerank/grading
# ============================================================

class BaselineState(TypedDict):
    question:   str
    documents:  List[Document]
    generation: str

# store：vectorstore.py 的 load_existing_store() 回傳的單一 Chroma store。
# 直接對這個 collection 做純向量搜尋，不做任何路由或過濾。
def build_naive_pipeline(store):

    rag_chain = build_rag_chain()

    def naive_retrieve(state):
        print("---[BASELINE] NAIVE VECTOR RETRIEVE (no hybrid/rerank/grading)---")
        question = state["question"]
        try:
            docs = store.similarity_search(question, k=NAIVE_TOP_K)
        except Exception as e:
            print(f"  -ERROR: {e}-")
            docs = []
        print(f"  -got {len(docs)} docs-")
        return {"documents": docs, "question": question}

    def naive_generate(state):
        print("---[BASELINE] GENERATE---")
        question   = state["question"]
        documents  = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    workflow = StateGraph(BaselineState)
    workflow.add_node("naive_retrieve", naive_retrieve)
    workflow.add_node("naive_generate", naive_generate)
    workflow.set_entry_point("naive_retrieve")
    workflow.add_edge("naive_retrieve", "naive_generate")
    workflow.add_edge("naive_generate", END)

    return workflow.compile()
