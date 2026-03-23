# vectorstore.py
# 負責：載入文件、chunking、RAPTOR 多層摘要索引、Hybrid Search retriever

# ============================================================
# pip install langchain langchain-openai chromadb langchain_community
#             pypdf gdown requests beautifulsoup4
#             rank_bm25          ← Hybrid Search (BM25) 需要
#             scikit-learn umap-learn  ← RAPTOR clustering 需要
# 切分 → [RAPTOR] → 轉向量 → 存入 Chroma
# ============================================================

import os
import requests
import gdown
import numpy as np
from typing import Dict, List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    PERSIST_DIR,
    LAYER1_K, LAYER1_BM25_K, LAYER2_TOP_N,
    HYBRID_VECTOR_WEIGHT,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SOURCE_CONFIG,
    RAPTOR_ENABLED, RAPTOR_N_LEVELS, RAPTOR_MAX_CLUSTER,
)


# ============================================================
# 3. Chunking → return all doc
# ============================================================

# 利用 LangChain load doc、chunking，並把完成 chunk 的文件加進總文件池 all_docs
def load_source(source_name: str, config: dict) -> list:
    """根據 config 自動從各種來源 load 文件，回傳切好的 Document list"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_docs = []

    # Google Drive 資料夾
    if "gdrive_folder" in config:
        local_dir = f"/tmp/gdrive_{source_name}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"  Downloading from Google Drive...")
        gdown.download_folder(config["gdrive_folder"], output=local_dir, quiet=False)
        loader = DirectoryLoader(local_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        all_docs.extend(splitter.split_documents(loader.load()))

    # 本機資料夾
    if "local_folder" in config:
        print(f"  Loading from local folder: {config['local_folder']}")
        loader = DirectoryLoader(
            config["local_folder"], glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        all_docs.extend(splitter.split_documents(loader.load()))

    # 線上 PDF 連結
    if "pdf_urls" in config:
        for url in config["pdf_urls"]:
            print(f"  Downloading PDF: {url}")
            local_path = f"/tmp/{source_name}_{abs(hash(url))}.pdf"
            response = requests.get(url, timeout=30)
            with open(local_path, "wb") as f:
                f.write(response.content)
            loader = PyPDFLoader(local_path)
            all_docs.extend(loader.load_and_split(splitter))

    # 網頁連結
    if "webpages" in config:
        print(f"  Loading webpages: {config['webpages']}")
        loader = WebBaseLoader(config["webpages"])
        all_docs.extend(splitter.split_documents(loader.load()))

    return all_docs


# ============================================================
# RAPTOR：遞迴 Cluster → Summarize → 建立多層摘要索引
# ============================================================
# 參考：RAPTOR, Sarthi et al., ICLR 2024 (arxiv 2401.18059)
#
# 原始論文流程：
#   1. 把 leaf chunks 做 embedding
#   2. 用 UMAP 降維 + GMM 做 clustering
#   3. 對每個 cluster 的 chunks 做 LLM 摘要，產生摘要 chunk
#   4. 把摘要 chunk 當新的 leaf，遞迴重複直到 cluster 數量 = 1
#   5. 把所有層的 chunk（原始 + 各層摘要）存進同一個 vectorstore
#      （collapsed tree 策略：查詢時一次搜尋全部層）
#
# 這個實作採用 collapsed tree 策略，論文指出這是效果最好的 retrieval 方式。

_raptor_summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一個文件摘要專家。"
        "以下是一組語意相近的文件片段，請為它們產生一段簡潔但完整的摘要，"
        "保留所有重要的技術細節、數據和關鍵概念。"
        "摘要長度約 200-400 字。"
    )),
    ("human", "文件片段：\n\n{text}\n\n請產生摘要："),
])


def _cluster_texts(texts: List[str], embeddings_list: List[List[float]], max_clusters: int) -> Dict[int, List[int]]:
    """
    對文件做 KMeans clustering，回傳 {cluster_id: [doc_indices]} dict。
    採用 sklearn KMeans（比 GMM 更穩定，適合 POC 實作）。
    論文用 UMAP + GMM，但需要額外安裝套件，這裡用更輕量的版本。
    """
    from sklearn.cluster import KMeans
    import numpy as np

    n = len(texts)
    if n <= 1:
        return {0: list(range(n))}

    k = min(max_clusters, max(1, n // 3))  # cluster 數量不超過 doc 數 / 3
    embeddings_array = np.array(embeddings_list)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings_array)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    return clusters


def build_raptor_docs(
    leaf_docs: List[Document],
    embeddings: OpenAIEmbeddings,
    source_name: str,
    n_levels: int = 3,
    max_clusters: int = 10,
) -> List[Document]:
    """
    對 leaf_docs 做 RAPTOR 遞迴摘要，回傳所有層的 Document list。
    回傳的 list 包含：原始 leaf chunks + 各層的摘要 chunks（collapsed tree）。

    Args:
        leaf_docs:    原始切好的 Document list
        embeddings:   OpenAI embeddings，用於計算向量做 clustering
        source_name:  source 名稱，寫進 metadata
        n_levels:     遞迴層數（config 的 RAPTOR_N_LEVELS）
        max_clusters: 每層最多 cluster 數（config 的 RAPTOR_MAX_CLUSTER）

    Returns:
        所有層的 Document list（原始 + 摘要）
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    summarize_chain = _raptor_summarize_prompt | llm | StrOutputParser()

    all_docs = list(leaf_docs)    # collapsed tree：所有層都存進來
    current_texts = [d.page_content for d in leaf_docs]

    for level in range(1, n_levels + 1):
        if len(current_texts) <= 1:
            print(f"  [RAPTOR] Level {level}: only 1 chunk left, stopping recursion.")
            break

        print(f"  [RAPTOR] Level {level}: embedding {len(current_texts)} chunks for clustering...")
        embeddings_list = embeddings.embed_documents(current_texts)

        clusters = _cluster_texts(current_texts, embeddings_list, max_clusters)
        print(f"  [RAPTOR] Level {level}: {len(clusters)} clusters → generating summaries...")

        summary_texts = []
        for cluster_id, indices in clusters.items():
            cluster_content = "\n\n".join(current_texts[i] for i in indices)
            summary = summarize_chain.invoke({"text": cluster_content})
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "source":       source_name,
                    "raptor_level": level,
                    "cluster_id":   cluster_id,
                    "doc_type":     "raptor_summary",
                }
            )
            all_docs.append(summary_doc)
            summary_texts.append(summary)
            print(f"    Cluster {cluster_id}: {len(indices)} chunks → summary ({len(summary)} chars)")

        current_texts = summary_texts  # 下一層用這層的摘要繼續遞迴

    print(f"  [RAPTOR] Done. Total docs in index: {len(all_docs)} "
          f"(original: {len(leaf_docs)}, summaries: {len(all_docs) - len(leaf_docs)})")
    return all_docs


# ============================================================
# 4. Build multi-source vectorstore
# ============================================================
'''
# 如果 RAPTOR_ENABLED：
#   load_source() -> build_raptor_docs()（加入多層摘要）-> Chroma.from_documents()
# 如果 RAPTOR_ENABLED = False：
#   load_source() -> Chroma.from_documents()（一般 chunking）
#
# Chroma.from_documents 同時做：embed 每個 chunk -> 存入向量資料庫
'''

def build_stores(embeddings: OpenAIEmbeddings) -> Dict[str, Chroma]:
    """建立所有 vectorstore，回傳 {source_name: Chroma} dict"""
    print("Building vectorstores...")
    if RAPTOR_ENABLED:
        print(f"  RAPTOR enabled: n_levels={RAPTOR_N_LEVELS}, max_cluster={RAPTOR_MAX_CLUSTER}")
    stores = {}

    for source_name, config in SOURCE_CONFIG.items():
        print(f"\n[{source_name}] Loading documents...")
        docs = load_source(source_name, config)

        # empty store fallback
        if not docs:
            print(f"[{source_name}] WARNING: No documents loaded. Using empty vectorstore.")
            stores[source_name] = Chroma(
                collection_name=source_name,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIR,
            )
            continue

        for doc in docs:
            doc.metadata["source"]   = source_name
            doc.metadata["doc_type"] = doc.metadata.get("doc_type", "original")
            # OPTIMIZE: metadata 加 source 後續可做 filter retrieval, scoring, explainability

        # [RAPTOR] 如果啟用，對 docs 做遞迴摘要，擴充索引內容
        if RAPTOR_ENABLED:
            print(f"[{source_name}] Running RAPTOR indexing...")
            docs = build_raptor_docs(
                leaf_docs=docs,
                embeddings=embeddings,
                source_name=source_name,
                n_levels=RAPTOR_N_LEVELS,
                max_clusters=RAPTOR_MAX_CLUSTER,
            )

        print(f"[{source_name}] {len(docs)} total chunks ready, building index...")
        stores[source_name] = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,          # 轉向量
            collection_name=source_name,   # 存進 vector database（Chroma）
            persist_directory=PERSIST_DIR, # Chroma 把資料存到 persist
        )
        print(f"[{source_name}] Done.")

        # OPTIMIZE: 加 reset / 檢查，避免重複 build 會 duplicate embeddings
        '''
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        '''
        # OPTIMIZE: retrieval 時用 filter，例如只查 original chunk 或只查 summary
        '''
        retriever = store.as_retriever(
            search_kwargs={"filter": {"doc_type": "original"}}
        )
        '''

    return stores


# ============================================================
# 5. Retrieval Layer（對 document 做 retrieval）
# ============================================================
'''
# Layer 1 - Hybrid Search（Pointwise）
#   [新增] BM25 + Chroma 向量搜尋，用 EnsembleRetriever 做 RRF fusion
#   → 解決純向量搜尋對專有名詞和精確關鍵字不敏感的問題
#   → BM25 擅長關鍵字匹配，向量搜尋擅長語意理解，兩者互補
#   參考：EnsembleRetriever 使用 RRF (Cormack et al., SIGIR 2009) 做 score fusion
#
# Layer 2 - Cross-Encoder reranker（Pairwise）
#   → 對每份候選文件和問題配對，用小型 BERT-like 模型打分並重排序
#   → 取 top_n=LAYER2_TOP_N，分數存在 doc.metadata["relevance_score"]
#   → 本地執行，不花 API，模型約 280MB（第一次執行自動下載）
#
# Layer 3 - LLM-as-Judge（在 retrieval_grade node 裡執行）
#   → 三維度加權評分，低於 RELEVANCE_THRESHOLD 過濾掉
'''

def build_retrievers(stores: Dict[str, Chroma]) -> dict:
    """
    建立三層 retriever，回傳 {source_name: retriever} dict。
    Layer 1：Hybrid Search（BM25 + Vector，EnsembleRetriever RRF fusion）
    Layer 2：Cross-Encoder Reranker
    Layer 3：LLM-as-Judge（在 pipeline.py 的 retrieval_grade node 裡執行）
    """
    assert LAYER1_K > LAYER2_TOP_N, \
        f"LAYER1_K ({LAYER1_K}) must be > LAYER2_TOP_N ({LAYER2_TOP_N})"

    print("\nSetting up 3-layer retrieval (with Hybrid Search)...")

    # Layer 2：Cross-Encoder reranker（Pairwise）
    # 本地執行，不花 API，模型約 280MB（第一次執行自動下載）
    _rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    _reranker     = CrossEncoderReranker(model=_rerank_model, top_n=LAYER2_TOP_N)

    retrievers = {}

    for name, store in stores.items():
        # --- Layer 1a：Chroma 向量搜尋（Pointwise, Dense） ---
        vector_retriever = store.as_retriever(
            search_kwargs={"k": LAYER1_K}
        )

        # --- Layer 1b：BM25 關鍵字搜尋（Sparse） ---
        # BM25 需要從 store 取出所有 docs 來建 index
        # Chroma 0.4+ 可以用 get() 取出所有文件
        try:
            store_data = store.get()
            bm25_docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(store_data["documents"], store_data["metadatas"])
            ] if store_data["documents"] else []
        except Exception:
            bm25_docs = []

        if bm25_docs:
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = LAYER1_BM25_K

            # --- Layer 1：Hybrid（BM25 + Vector，RRF fusion）---
            # EnsembleRetriever 使用 Weighted RRF 合併兩個 retriever 的結果
            # HYBRID_VECTOR_WEIGHT：向量搜尋的權重（BM25 = 1 - 此值）
            hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[HYBRID_VECTOR_WEIGHT, 1 - HYBRID_VECTOR_WEIGHT],
            )
            print(f"  [{name}] Hybrid Search ready "
                  f"(vector={HYBRID_VECTOR_WEIGHT:.0%}, BM25={1-HYBRID_VECTOR_WEIGHT:.0%})")
        else:
            # BM25 建不起來（空 store）→ fallback 到純向量搜尋
            hybrid_retriever = vector_retriever
            print(f"  [{name}] WARNING: BM25 unavailable (empty store), using vector-only.")

        # --- Layer 1 + Layer 2 組合（Hybrid + Cross-Encoder Reranker）---
        retrievers[name] = ContextualCompressionRetriever(
            base_compressor=_reranker,
            base_retriever=hybrid_retriever,
        )

    print("3-layer retrieval ready.")
    return retrievers
