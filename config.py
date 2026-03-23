# config.py

import os

# ============================================================
# 1. Setting：
# ============================================================

# -- API Keys --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # 填入 OpenAI API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "") # 填入 Tavily API key

# -- Storage --
# Colab 用：PERSIST_DIR = "/drive/MyDrive/chroma_db"
PERSIST_DIR = "/drive/MyDrive/chroma_db"  # NOTE: 本機暫存 -> PERSIST_DIR = "./chroma_db"

# ============================================================
# -- Retrieval 設定 --
# ============================================================

# [Hybrid Search] Layer 1：BM25 + Vector Search，用 RRF 做 score fusion
# 參考：RRF (Reciprocal Rank Fusion), Cormack et al., SIGIR 2009
LAYER1_K             = 10   # Vector Search：每個 source 各撈 k 份候選
LAYER1_BM25_K        = 10   # BM25 Search：每個 source 各撈 k 份候選
HYBRID_VECTOR_WEIGHT = 0.6  # 向量搜尋的 RRF 權重（1 - 此值 = BM25 權重）
# 技術文件類 source 建議調低 Vector 權重（專有名詞多，BM25 更準）
# 語意問題類 source 建議調高 Vector 權重

# Layer 2：Cross-Encoder Reranker
LAYER2_TOP_N        = 5    # reranker 從候選裡選出 top_n 份給 Layer 3

# Layer 3：LLM-as-Judge
RELEVANCE_THRESHOLD = 3.0  # 低於此值過濾

# ============================================================
# -- Query Rewriting 設定 --
# ============================================================
# 參考：Rewrite-Retrieve-Read, Ma et al., 2023 (arxiv 2305.14283)
# 問題進來之前先讓 LLM 改寫成更適合搜尋的形式
# 改寫後的問題用於 retrieval，原始問題用於最終答案生成
QUERY_REWRITING_ENABLED = True  # False 可關閉，直接用原始問題做 retrieval

# ============================================================
# -- RAPTOR 設定 --
# ============================================================
# 參考：RAPTOR, Sarthi et al., ICLR 2024 (arxiv 2401.18059)
# 文件建索引時，遞迴做 cluster → summarize，建立多層摘要
# 把原始 chunk + 各層摘要都存進 vectorstore（collapsed tree 策略）
# 讓 retrieval 可以同時撈到細節（原始 chunk）和高層概念（摘要）
RAPTOR_ENABLED     = True  # False 可關閉，退回一般 chunking
RAPTOR_N_LEVELS    = 3     # 遞迴摘要的層數（層數越多 API 費用越高）
RAPTOR_MAX_CLUSTER = 10    # 每一層最多幾個 cluster（控制摘要數量）

# ============================================================
# -- Router 設定 --
# ============================================================
CONFIDENCE_THRESHOLD = 0.6  # 低於 CONFIDENCE_THRESHOLD → 強制 fallback 到 web_search

# ============================================================
# -- Chunking 設定 --
# ============================================================
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 128

# ============================================================
# 2. Prepare SOURCE_CONFIG：
# ============================================================
# 每個 source 對應的資料在哪裡, 四種key可單獨使用，也可以混合使用
# 依實際情況填入相關路徑或連結 # HACK
SOURCE_CONFIG = {
    "technical": {
        # 技術文件：論文、規格書、API 文件
        # "gdrive_folder": "https://drive.google.com/drive/folders/你的FOLDER_ID",
        # "local_folder":  "/content/technical/",
        # "pdf_urls": ["https://arxiv.org/pdf/1706.03762.pdf"],
        # "webpages": ["https://arxiv.org/abs/1706.03762"],
    },
    "business": {
        # 商業文件：市場報告、財報、競品分析
        # "local_folder": "/content/business/",
        # "webpages": ["https://某個市場報告網址"],
    },
    "legal": {
        # 法規文件：法條、合約、專利
        # "local_folder": "/content/legal/",
        # "pdf_urls": ["https://某個法規PDF"],
    },
    "healthcare": {
        # 醫療文件：臨床指引、醫學研究、病患FAQ
        # "gdrive_folder": "...",
        # "local_folder": "/content/healthcare/",
        # "pdf_urls": ["https://某個醫療指引PDF"],
        # "webpages": ["https://某個醫療資訊網址"],
    },
}

INDEX_DESCRIPTIONS = {
    "technical": (
        "Technical documents including research papers, API specs, "
        "architecture docs, and engineering guidelines"
    ),
    "business": (
        "Business documents including market reports, financial analyses, "
        "competitor research, and strategic plans"
    ),
    "legal": (
        "Legal documents including regulations, compliance requirements, "
        "contracts, patents, and IP filings"
    ),
    "healthcare": (
        "Healthcare documents including clinical guidelines, medical research, "
        "treatment protocols, patient FAQs, and disease management resources"
    ),
}
