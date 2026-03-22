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

# -- Retrieval 設定 --
LAYER1_K            = 10  # Layer 1：每個 source 各撈 k 份候選給 Layer 2 排序
LAYER2_TOP_N        = 5   # Layer 2：reranker 從 k 份候選裡選出 top_n 份給 Layer 3
RELEVANCE_THRESHOLD = 3.0 # Layer 3：低於此值過濾 # OPTIMIZE: 目前出現Retrieval兩次後保留的doc都低於threshold狀況, 會先用web_search解決

# -- Router 設定 --
CONFIDENCE_THRESHOLD = 0.6  # 低於 CONFIDENCE_THRESHOLD → 強制 fallback 到 web_search

# -- Chunking 設定 --
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
