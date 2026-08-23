# vectorstore.py
# 讀取向量資料庫（ingest_helper.py 的 Chroma collection）
# 組出 Layer 1 Hybrid Search retriever + Layer 2 手動 rerank（自己控制分數，不用 LangChain
# 內建的 CrossEncoderReranker——它排序完會把分數丟掉，沒辦法拿來做門檻過濾，見下方 rerank_documents()）
# 另外提供圖片 collection（CLIP embedding）的存取，供「以文搜圖」使用。

import os
# 強制 transformers 只用 PyTorch 路徑，不要去載入 TensorFlow 整合。
# 沒這兩行的話，如果環境裡剛好也裝了 TensorFlow（例如透過 anaconda base env），
# sentence-transformers -> transformers 會嘗試載入 TF 相關程式碼，
# 遇到新版 Keras 3 不相容會直接 crash（我們完全用不到 TF，這裡只是被動被拖下水）。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from config import (
    PERSIST_DIR, COLLECTION_NAME, IMAGE_COLLECTION_NAME,
    LAYER1_K, LAYER1_BM25_K, LAYER2_TOP_N,
    HYBRID_VECTOR_WEIGHT, RERANKER_MODEL_NAME, CLIP_MODEL_NAME,
)


# ============================================================
# 打開 Chroma collection
# ============================================================
# 已經用 ingest_helper.py（ingest_files / ingest_folder / ingest_texts）匯入好的
def load_existing_store(embeddings: OpenAIEmbeddings, collection_name: str = COLLECTION_NAME) -> Chroma:

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    try:
        count = store._collection.count()
    except Exception:
        count = None

    if count == 0:
        print(f"[WARNING] Chroma collection '{collection_name}' 目前是空的！")
        print(f"          請先執行 ingest_helper.py 把文件匯入進去，例如：")
        print(f"          from ingest_helper import ingest_folder")
        print(f"          ingest_folder('./smart_healthcare_docs', source_name='{collection_name}', "
              f"extensions=('.pdf', '.md', '.docx'))")
    elif count is not None:
        print(f"[OK] Chroma collection '{collection_name}' 已載入，共 {count} 筆 chunks。")

    return store


# ============================================================
# Layer 1：Hybrid Search（BM25 + Vector，EnsembleRetriever 內部用 RRF 融合）
# ============================================================
def build_retriever(store: Chroma):
    """回傳 Layer 1 hybrid retriever（不含 rerank）。Rerank 交給下面的 rerank_documents() 明確處理。"""
    print("\nSetting up Layer 1 (Hybrid Search: BM25 + Vector)...")

    vector_retriever = store.as_retriever(search_kwargs={"k": LAYER1_K})

    try:
        store_data = store.get()
        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(store_data["documents"], store_data["metadatas"])
        ] if store_data["documents"] else []
    except Exception as e:
        print(f"  WARNING: 讀取 store 建 BM25 索引失敗 ({e})")
        bm25_docs = []

    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = LAYER1_BM25_K

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[HYBRID_VECTOR_WEIGHT, 1 - HYBRID_VECTOR_WEIGHT],
        )
        print(f"  Hybrid Search ready (vector={HYBRID_VECTOR_WEIGHT:.0%}, BM25={1-HYBRID_VECTOR_WEIGHT:.0%})")
    else:
        hybrid_retriever = vector_retriever
        print("  WARNING: BM25 unavailable (empty store), using vector-only.")

    return hybrid_retriever


# ============================================================
# Layer 2：Cross-Encoder Rerank（自己算分數、自己存進 metadata）
# ============================================================
# OPTIMIZE/ HuggingFaceCrossEncoder 每次呼叫都會重新從硬碟/網路載入模型，在 Streamlit 這種
#          會重跑 script 的環境下很浪費時間。app.py 已經用 st.cache_resource 包住，CLI（main.py）
#          則只在程式啟動時組一次，都不會每個問題重載一次模型。

def build_reranker_model(model_name: str = RERANKER_MODEL_NAME):
    """
    回傳 cross-encoder 模型本身（不是 LangChain 的 CrossEncoderReranker compressor）。
    model_name 可以指向微調過的 checkpoint 路徑（本地資料夾），例如
    finetune_reranker.py 訓練完產出的 config.RERANKER_CHECKPOINT_DIR/reranker-YYYYMMDD/。
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    print(f"  Loading reranker model: {model_name}")
    return HuggingFaceCrossEncoder(model_name=model_name)


def rerank_documents(
    query: str,
    documents: List[Document],
    cross_encoder,
    top_n: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """
    手動重排，並把分數寫回 metadata["relevance_score"]。
    不用 LangChain 內建的 CrossEncoderReranker，因為它排序完會把分數丟掉，
    我們需要保留分數才能做門檻過濾（而不是固定回傳 top-N，不管分數高低）。

    - min_score：分數低於這個門檻的直接濾掉（可能濾到剩 0 筆，由呼叫端決定接下來怎麼處理）
    - top_n：安全上限，避免萬一大部分文件都過了門檻，塞進生成階段的 context 太大
    """
    if not documents:
        return []

    scores = cross_encoder.score([(query, d.page_content) for d in documents])
    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    if min_score is not None:
        scored = [(d, s) for d, s in scored if s >= min_score]
    if top_n is not None:
        scored = scored[:top_n]

    result = []
    for d, s in scored:
        d.metadata["relevance_score"] = float(s)
        result.append(d)
    return result


# ============================================================
# 圖片 / CLIP（以文搜圖，額外的檢索管道）
# ============================================================
# CLIP 的圖片向量跟文字 collection 用的 OpenAIEmbeddings 是兩個不同的向量空間，
# 不能直接混在同一個 collection 裡比大小，所以獨立開一個 collection。
# 圖片本身的文字說明（caption）在 ingest_helper.py 已經另外用 OpenAIEmbeddings 存進主 collection了，
# 主要的問答流程完全不需要碰這裡的 CLIP collection 就能檢索到圖片（靠 caption 文字）。
# 這裡的 CLIP collection 是額外的「以文搜圖」/「以圖搜圖」能力，需要的話再用。

def get_clip_model(model_name: str = CLIP_MODEL_NAME):
    """回傳 sentence-transformers 的 CLIP 模型，文字/圖片都用同一支 encode()。"""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading CLIP model: {model_name}")
    return SentenceTransformer(model_name)


class ClipEmbeddings:
    """
    把 CLIP 包成 LangChain Embeddings 介面（embed_query / embed_documents），
    這樣 Chroma 的 similarity_search() 才能直接拿查詢文字去 CLIP 的文字 encoder 轉向量。
    embed_documents() 只是為了符合介面規格而實作，實際寫入圖片向量走下面的
    add_image_documents()（直接編碼圖片本身，不是編碼文字），不會呼叫到這裡。
    """

    def __init__(self, model=None):
        self.model = model or get_clip_model()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def add_image_documents(image_store: Chroma, image_paths: List[str], metadatas: List[dict], clip_model) -> int:
    """
    直接用 CLIP 編碼圖片本身（不是編碼文字），寫進圖片 collection。
    繞過 Chroma 高階 API 的 add_texts()（那個路徑會把文字送進 embedding_function，
    但我們要編碼的是圖片像素，不是文字），改用底層 collection.add() 自己塞 embeddings。
    """
    from PIL import Image

    embeddings, ids, documents = [], [], []
    for i, (path, meta) in enumerate(zip(image_paths, metadatas)):
        try:
            img = Image.open(path).convert("RGB")
            vec = clip_model.encode(img, convert_to_numpy=True)
        except Exception as e:
            print(f"  WARNING: CLIP 編碼圖片失敗，略過 '{path}' ({e})")
            continue
        embeddings.append(vec.tolist())
        documents.append(os.path.basename(path))  # 只是佔位文字，圖片內容不靠這個欄位檢索
        ids.append(f"img:{meta.get('content_hash', i)}:{i}")

    if not embeddings:
        return 0

    image_store._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    return len(embeddings)


def load_image_store(embeddings_fn=None) -> Optional[Chroma]:
    """
    打開圖片 collection（CLIP embedding）。embeddings_fn 需要提供一個 LangChain Embeddings 相容介面
    （見 ingest_helper.py 的 ClipEmbeddings），沒有的話回傳 None（代表還沒開始用圖片功能）。
    """
    if embeddings_fn is None:
        return None
    store = Chroma(
        collection_name=IMAGE_COLLECTION_NAME,
        embedding_function=embeddings_fn,
        persist_directory=PERSIST_DIR,
    )
    try:
        count = store._collection.count()
        print(f"[OK] 圖片 collection '{IMAGE_COLLECTION_NAME}' 已載入，共 {count} 張圖片。")
    except Exception:
        pass
    return store


def image_similarity_search(query_text: str, image_store: Chroma, k: int = 5) -> List[Document]:
    """
    以文搜圖：直接用 CLIP 的文字 encoder 把查詢字串 embed 進跟圖片一樣的空間去搜尋。
    這是 CLIP 最大的優勢——不需要幫每張圖寫 caption，靠向量空間對齊就能搜到語意相關的圖片。
    """
    if image_store is None:
        return []
    try:
        return image_store.similarity_search(query_text, k=k)
    except Exception as e:
        print(f"  WARNING: 圖片搜尋失敗 ({e})")
        return []
