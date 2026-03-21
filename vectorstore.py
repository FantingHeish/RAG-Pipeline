# vectorstore.py
# 負責：載入文件、chunking、建 vectorstore、三層 retriever

# ============================================================
# pip install langchain langchain-openai chromadb langchain_community
#             pypdf gdown requests beautifulsoup4
# 切分 → 轉向量 → 存入 Chroma
# ============================================================

import os
import requests
import gdown
from typing import Dict

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    PERSIST_DIR,
    LAYER1_K, LAYER2_TOP_N,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SOURCE_CONFIG,
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
# 4. Build multi-source vectorstore
# ============================================================
'''
# Chroma.from_documents 同時做：embed 每個 chunk → 存入向量資料庫
# 1. 把每個 chunk 做 embedding
# 2. 存進 vector database（Chroma）
# 3. 寫到 persist（Colab 環境已設為 Google Drive，避免重啟後消失）
'''

def build_stores(embeddings: OpenAIEmbeddings) -> Dict[str, Chroma]:
    """建立所有 vectorstore，回傳 {source_name: Chroma} dict"""
    print("Building vectorstores...")
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
            doc.metadata["source"] = source_name
            # OPTIMIZE: metadata 加 source 後續可做 filter retrieval, scoring, explainability

        print(f"[{source_name}] {len(docs)} chunks ready, building index...")
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
        # OPTIMIZE: hybrid search: vector search + keyword search（BM25）
        # OPTIMIZE: retrieval 時用 filter
        '''
        retriever = store.as_retriever(
            search_kwargs={"filter": {"source": "papers"}}
        )
        '''

    return stores


# ============================================================
# 5. Retrieval Layer（對 document 做 retrieval）
# ============================================================
'''
# Layer 1 - Chroma 向量相似度（Pointwise）
#   → retriever.invoke(question) 時自動執行：
#     問題轉向量 → 和資料庫所有向量算 cosine similarity → 取 top k=LAYER1_K
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
    """建立三層 retriever，回傳 {source_name: retriever} dict"""
    assert LAYER1_K > LAYER2_TOP_N, \
        f"LAYER1_K ({LAYER1_K}) must be > LAYER2_TOP_N ({LAYER2_TOP_N})"

    print("\nSetting up 3-layer retrieval...")

    # Cross-Encoder reranker（Pairwise）
    # 本地執行，不花 API，模型約 280MB（第一次執行自動下載）
    _rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    _reranker     = CrossEncoderReranker(model=_rerank_model, top_n=LAYER2_TOP_N)
    # top_n=LAYER2_TOP_N：Layer 2 從候選裡選出最相關的幾份，再交給 Layer 3

    # Chroma 向量相似度（Pointwise）+ Cross-Encoder（Pairwise）組合
    retrievers = {
        name: ContextualCompressionRetriever(
            base_compressor=_reranker,  # Cross-Encoder reranker（從 k 份候選裡選出 top_n）
            base_retriever=store.as_retriever(
                search_kwargs={"k": LAYER1_K}  # Vector Search（每個 source 各撈 k 份候選）
            ),
        )
        for name, store in stores.items()
    }

    print("3-layer retrieval ready.")
    return retrievers
