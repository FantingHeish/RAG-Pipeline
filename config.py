# config.py

import os
import sys

from dotenv import load_dotenv

# 從專案根目錄的 .env 檔載入環境變數（.env 不應該被 commit 進版控，
# 請參考 .env.example 建立自己的 .env）
load_dotenv()

# -- API Keys --
# 絕對不要把真正的 key 寫死在程式碼或 comment 裡（就算是「暫時測試用」也不行）——
# 一旦進了 git history 或分享出去，這把 key 就等於外流，必須馬上到對應平台 revoke/重新產生。
# 一律只從環境變數 / .env 讀取。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 必要
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # 沒設定時，web_search 相關功能會被跳過


def validate_api_keys(require_tavily: bool = False, hard_exit: bool = True):
    """
    檢查必要的 API key 是否存在。
    hard_exit=True（CLI 用）：OPENAI_API_KEY 缺少時直接中止程式。
    hard_exit=False（例如 Streamlit UI 用）：只回傳問題清單，不強制中止，
        讓呼叫端自己決定要怎麼呈現錯誤（例如畫面上顯示警告，而不是讓整個 App 掛掉）。
    """
    problems = []
    if not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY 未設定，無法運行")
    if require_tavily and not TAVILY_API_KEY:
        problems.append("TAVILY_API_KEY 未設定，即時 web search 功能無法使用")

    if problems:
        print("=" * 60)
        print("[STARTUP CHECK] 發現以下問題：")
        for p in problems:
            print(f"  - {p}")
        print("=" * 60)
        if hard_exit and not OPENAI_API_KEY:
            sys.exit(1)
    else:
        print("[STARTUP CHECK] API keys OK.")

    return problems


# -- Storage --
# Chroma embedded 向量資料庫
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")  # 本機資料夾路徑，會自動建立

# ============================================================
# -- Retrieval Settings --
# ============================================================

# Hybrid Search：BM25 + Vector Search（EnsembleRetriever 內部用 RRF 做 score fusion）
LAYER1_K = int(os.getenv("LAYER1_K", "10"))
LAYER1_BM25_K = int(os.getenv("LAYER1_BM25_K", "10"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
# OPTIMIZE/ 上面這個權重目前還是憑經驗猜的初始值。用 tune_hybrid_weight.py 掃過
# gold_standard.py 在不同權重下的 RAGAS context_precision/context_recall，
# 挑實際分數最高的值回填這裡，取代用猜的。

# Cross-Encoder Reranker
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
LAYER2_TOP_N = 3

# 線上檢索過濾門檻：直接用 Layer 2 reranker 自己的 relevance_score 做門檻過濾，
# 不再對每個問題都額外呼叫一次 LLM-as-Judge（那個角色改成離線用，見 offline_evaluate.py）。
# bge-reranker 系列輸出的是未經過 sigmoid 的 raw logit，不是 0-1 機率，
# 所以這個門檻要用你自己的資料實際測過幾筆再回填，不能憑感覺猜（可以先跑幾個問題，
# 把 debug 模式下看到的 relevance_score 記下來，抓一個能把明顯不相關文件濾掉的值）。
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))

# 生成後的輕量品質檢查（只保留「有沒有回答到問題」，不再是 hallucination+answer 兩次呼叫）
ANSWER_QUALITY_CHECK_ENABLED = True

# Embedding 粗篩（送進 reranker 之前，先用向量相似度濾掉明顯不相關的，減少 reranker 要處理的量）
EMBEDDING_PREFILTER_ENABLED = True
EMBEDDING_PREFILTER_THRESHOLD = 0.15  # cosine similarity 門檻，低於此值直接濾掉（門檻放寬鬆一點，避免誤殺）

# Retry 次數限制
# retrieval_grade <-> web_search_fallback <-> rag_generate 之間可能反覆繞圈，
# 超過 MAX_RETRIES 次後強制結束，走 plain_answer（LLM 直接用自身知識回答，並標註警語）
MAX_RETRIES = 3

# Query Rewriting
QUERY_REWRITING_ENABLED = True

# ============================================================
# -- 語意化 Key 前綴（hash key + 語意前綴）--
# ============================================================
# 不只用內容 hash 當 chunk id，前面加一段從內容抽出來的關鍵字（例如 "2024-ai-laptop-application"），
# 讓 key 本身就看得懂內容在講什麼，方便用前綴做分類/清理/除錯，不影響 hash 原本的去重功能。
SEMANTIC_KEY_ENABLED = os.getenv("SEMANTIC_KEY_ENABLED", "true").lower() == "true"
SEMANTIC_KEY_BATCH_SIZE = 5  # 每次 LLM call 一次抽幾個 chunk 的關鍵字，降低 API 呼叫次數

# ============================================================
# -- 圖片 / 多模態 --
# ============================================================
IMAGE_INGEST_ENABLED = os.getenv("IMAGE_INGEST_ENABLED", "true").lower() == "true"
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "gpt-4o-mini")  # 要有視覺能力的模型才能看圖
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")  # 用 sentence-transformers 內建的 CLIP
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME", f"{os.getenv('COLLECTION_NAME', 'smart_healthcare')}_images")

# ============================================================
# -- 離線評估 / Reranker 微調 --
# ============================================================
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "./training_data/reranker_pairs.jsonl")
RERANKER_CHECKPOINT_DIR = os.getenv("RERANKER_CHECKPOINT_DIR", "./models")
MIN_NEW_TRAINING_SAMPLES = 50  # 累積到至少這麼多筆新標記資料，才建議觸發重訓（避免資料太少訓出爛模型）

# 真實使用紀錄：app.py 每次有人問問題就會把問題文字（只有文字，不含答案/文件內容）
# 追加寫進這個檔案，offline_evaluate.py 會把這裡的問題跟 gold_standard.py 的題目合併，
# 一起送去產生 reranker 訓練資料——這樣訓練資料才會反映「使用者實際怎麼問」，
# 不會只侷限在我們自己想出來的固定題目。
USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "./training_data/usage_questions.jsonl")

# 品質監控 log：app.py 每次回答完會記錄「有沒有依據文件、有沒有回答到問題」等輕量欄位
# （見 app.py 的 log_answer_quality()），用來事後追蹤系統回答品質的變化趨勢。
QUALITY_LOG_PATH = os.getenv("QUALITY_LOG_PATH", "./training_data/answer_quality.jsonl")

# 離線 LLM-as-Judge 設定（只有 offline_evaluate.py 會用到，線上問答已經不再呼叫這組）
OFFLINE_JUDGE_BATCH_SIZE = 5       # 每次 batch LLM call 最多塞幾份文件評分
OFFLINE_RELEVANCE_THRESHOLD = 3.0  # 加權分數門檻，用來把 doc 標成相關/不相關（訓練資料的 label）

# RAPTOR 設定（目前簡化流程未使用，保留設定供未來需要時開啟）
# OPTIMIZE/ 目前 pipeline.py / vectorstore.py 都沒有實際使用 RAPTOR_* 設定，
#          真的要開啟前需要先實作對應的 clustering + summarization 流程
RAPTOR_ENABLED = os.getenv("RAPTOR_ENABLED", "false").lower() == "true"
RAPTOR_N_LEVELS = 3
RAPTOR_MAX_CLUSTER = 10

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ============================================================
# 單一文件集合設定
# ============================================================
# DOCS_FOLDER：放文件的資料夾，可以混著 .pdf / .docx / .md / .txt，不需要分類子資料夾
# COLLECTION_NAME：Chroma 的 collection 名稱，ingest_helper.py 匯入時跟這裡查詢要用同一個名字
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./smart_healthcare_docs")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "smart_healthcare")

# ============================================================
# Router（目前沒有任何地方 import router.py，見該檔案開頭說明）
# ============================================================
# OPTIMIZE/ 若之後要重新啟用多來源 router，把 INDEX_DESCRIPTIONS 改成實際對應到的
# 多個 collection/來源說明（現在專案是單一 collection，router 沒有實際用途）。
CONFIDENCE_THRESHOLD = 0.7
INDEX_DESCRIPTIONS = {
    "healthcare": "醫療、健康照護相關的政策、法規、臨床應用資料",
}
