"""
Adaptive RAG - Multi-Source + Scoring Rubric + Advanced Retrieval
======================================================
  1. Router → structured JSON output + confidence score + heuristic fallback
  2. Retrieval Layer：
       [新增] Query Rewriting（Rewrite-Retrieve-Read, Ma et al., 2023）
       Layer 1：Hybrid Search - BM25 + Vector Search with RRF fusion
                （RRF, Cormack et al., SIGIR 2009）
       Layer 2：Cross-Encoder Reranker（Pairwise）
       Layer 3：LLM-as-Judge 3-dimension weighted scoring（1-5）
     [新增] RAPTOR 多層摘要索引（Sarthi et al., ICLR 2024, arxiv 2401.18059）
  3. Eval: Gold Standard Dataset + evaluate_pipeline() 比較改前改後效果
  4. Multi_source: 支援多個 vectorstore（technical / business / legal / healthcare）
"""

# ============================================================
# Environment
# ============================================================
# pip install langchain langchain-openai chromadb langgraph pypdf
#             langchain_community pydantic gdown requests
#             beautifulsoup4 sentence-transformers
#             rank_bm25 (Hybrid Search (BM25))
#             scikit-learn (RAPTOR clustering)

import os

from langchain_openai.embeddings import OpenAIEmbeddings

from google.colab import drive
drive.mount('/drive')

from config import OPENAI_API_KEY, TAVILY_API_KEY, RAPTOR_ENABLED, QUERY_REWRITING_ENABLED
from vectorstore import build_stores, build_retrievers
from pipeline import build_pipeline
from evaluation import run, evaluate_pipeline, print_comparison

# 設定 API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ============================================================
# 1. 組裝 pipeline
# ============================================================

# 1. prepareing
embeddings = OpenAIEmbeddings()

# 4. Build multi-source vectorstore
# [RAPTOR] 如果 RAPTOR_ENABLED，會在 build_stores() 裡做遞迴摘要索引
# （chunking → RAPTOR cluster/summarize → embed → 存入 Chroma）
stores = build_stores(embeddings)

# 5. Retrieval Layer（Layer 1 Hybrid + Layer 2 Reranker）
retrievers = build_retrievers(stores)

# Build Graph（Query Rewriting + Router + Nodes + Conditional Edges）
app = build_pipeline(retrievers)

# ============================================================
# 測試 & 評估
# ============================================================

if __name__ == "__main__":

    # 印出整個 pipeline 的每一步
    def run_verbose(question: str):
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)
        output = None
        for output in app.stream({"question": question}):
            print()
        if output:
            last = list(output.values())[-1]
            print(f"\n[Answer]\n{last.get('generation', '(no generation)')}")
            if last.get("rewritten_question"):
                print(f"[Rewritten]\n{last.get('rewritten_question')}")
        return output

    # 單問題測試
    run_verbose("What are the clinical guidelines for periodontal disease treatment?")
    run_verbose("How does RAG work in AI systems?")
    run_verbose("Who is the current president of Taiwan?")
    run_verbose("Hello!")

    '''
    # Gold Standard 評估
    try:
        from gold_standard import GOLD_STANDARD
        results = evaluate_pipeline(app, GOLD_STANDARD)
        print(f"Route accuracy : {results['route_accuracy']:.1%} ({results['route_correct']}/{results['total']})")
        print(f"Answer quality : {results['answer_quality']:.1%} ({results['answer_correct']}/{results['total']})")
    except ImportError:
        print("WARNING: gold_standard.py not found, skipping evaluation.")
    '''
