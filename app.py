
# app.py
# Streamlit 介面
# 執行：streamlit run app.py
#
# 呼叫 vectorstore.py / pipeline.py / ingest_helper.py 的函式
# 用 st.cache_resource 只組一次

import json
import os
from datetime import datetime

import streamlit as st

from config import (
    validate_api_keys, DOCS_FOLDER, COLLECTION_NAME, MAX_RETRIES, QUERY_REWRITING_ENABLED,
    USAGE_LOG_PATH, QUALITY_LOG_PATH,
)

st.set_page_config(page_title="Adaptive RAG PoC", page_icon="🩺", layout="wide")
st.title("🩺 Adaptive RAG PoC")
st.caption(f"Collection: `{COLLECTION_NAME}`　·　MAX_RETRIES={MAX_RETRIES}　·　Query Rewriting={'on' if QUERY_REWRITING_ENABLED else 'off'}")

# ============================================================
# API Key 檢查
# ============================================================
problems = validate_api_keys(require_tavily=True, hard_exit=False)
missing_openai_key = any("OPENAI_API_KEY" in p for p in problems)

for p in problems:
    st.warning(p)
if problems:
    st.info("請在專案根目錄建立 `.env`（可參考 `.env.example`），設定好 key 後重新整理頁面。")

if missing_openai_key:
    st.stop()  # 沒有 OPENAI_API_KEY 就不繼續往下組 pipeline 了（TAVILY 缺少只是功能降級，可以繼續）


# ============================================================
# 重物件：只組一次，之後每次互動都重複使用（st.cache_resource）
# ============================================================

@st.cache_resource(show_spinner="正在載入 embeddings / vectorstore ...")
def get_embeddings():
    from langchain_openai.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings()


@st.cache_resource(show_spinner="正在開啟既有的 Chroma collection ...")
def get_store(_embeddings):
    from vectorstore import load_existing_store
    return load_existing_store(_embeddings)


@st.cache_resource(show_spinner="正在組裝 Hybrid Search + Rerank retriever（第一次會比較久，要下載 cross-encoder 模型）...")
def get_retriever(_store):
    from vectorstore import build_retriever
    return build_retriever(_store)


@st.cache_resource(show_spinner="正在組裝 Adaptive RAG pipeline ...")
def get_pipeline(_retriever, _embeddings):
    from pipeline import build_pipeline
    return build_pipeline(_retriever, embeddings=_embeddings)


@st.cache_resource(show_spinner="正在組裝 baseline pipeline ...")
def get_baseline_pipeline(_store):
    from pipeline_baseline import build_naive_pipeline
    return build_naive_pipeline(_store)


def get_chunk_count(store) -> int:
    try:
        return store._collection.count()
    except Exception:
        return -1


# ============================================================
# Sidebar：資料匯入 / collection 狀態
# ============================================================

with st.sidebar:
    st.header("文件庫")
    embeddings = get_embeddings()
    store = get_store(embeddings)
    chunk_count = get_chunk_count(store)

    if chunk_count == 0:
        st.error(f"Collection「{COLLECTION_NAME}」目前是空的，請先匯入文件才能問答。")
    elif chunk_count > 0:
        st.success(f"Collection「{COLLECTION_NAME}」已載入，共 {chunk_count} 筆 chunks。")
    else:
        st.info("尚未確認 collection 狀態。")

    st.divider()
    st.subheader("匯入 / 更新文件")
    st.caption(f"預設會掃描資料夾：`{DOCS_FOLDER}`（增量匯入，內容沒變的檔案會自動跳過）")

    if st.button("重新掃描並匯入 DOCS_FOLDER", use_container_width=True):
        from ingest_helper import ingest_folder
        with st.spinner("匯入中 ..."):
            ingest_folder(
                DOCS_FOLDER,
                source_name=COLLECTION_NAME,
                extensions=(".pdf", ".md", ".docx", ".png", ".jpg", ".jpeg"),
            )
        st.cache_resource.clear()  # 文件變了，store / retriever / pipeline 都要重組
        st.success("匯入完成，頁面即將重新整理 ...")
        st.rerun()

    uploaded_files = st.file_uploader(
        "或直接上傳檔案（.pdf / .docx / .md / .txt / .png / .jpg）",
        type=["pdf", "docx", "md", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("匯入上傳的檔案", use_container_width=True):
        import os
        from ingest_helper import ingest_files

        # 存進 DOCS_FOLDER（永久保留），而不是暫存資料夾：
        # 這樣檔案會跟手動放進資料夾的文件一視同仁，之後「重新掃描並匯入」也認得到，
        # ingest_helper.py 的增量追蹤（用檔案路徑 + 內容 hash 判斷有沒有變更）才能正常運作。
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        saved_paths = []
        overwritten = []
        for f in uploaded_files:
            path = os.path.join(DOCS_FOLDER, f.name)
            if os.path.exists(path):
                overwritten.append(f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(path)

        if overwritten:
            st.warning(f"以下檔名跟資料夾裡已有的檔案重複，內容已被覆蓋：{', '.join(overwritten)}")

        with st.spinner("匯入中 ..."):
            ingest_files(saved_paths, source_name=COLLECTION_NAME)
        st.cache_resource.clear()
        st.success(f"已將 {len(saved_paths)} 個檔案存進 `{DOCS_FOLDER}` 並匯入資料庫，頁面即將重新整理 ...")
        st.rerun()

    st.divider()
    mode = st.radio("Pipeline 模式", ["Adaptive RAG", "Naive RAG (baseline)"], index=0)
    show_debug = st.toggle("顯示檢索細節（重寫問題 / 文件 / 分數）", value=True)

if chunk_count == 0:
    st.stop()

retriever = get_retriever(store)
adaptive_app = get_pipeline(retriever, embeddings)

if mode == "Naive RAG (baseline)":
    app = get_baseline_pipeline(store)
else:
    app = adaptive_app


# ============================================================
# 主畫面：問答
# ============================================================

def log_usage_question(question: str, documents=None):
    """
    把使用者實際問過的問題追加寫進 USAGE_LOG_PATH。

    若這次問答已經檢索到文件（documents 不是 None/空），一併把
    page_content + metadata 存進去 —— offline_evaluate.py 的 _load_usage_items()
    會讀這個欄位，有存到就直接拿來當訓練樣本、不用重新檢索一次；
    沒存到（documents=None，例如舊格式紀錄或這次沒有任何檢索結果）
    就退回重新檢索。這裡特意不存 LLM 產生的答案全文，只存檢索到的
    文件，檔案大小可控。
    寫檔失敗不能讓使用者的問答體驗中斷，所以整段包 try/except，静默失敗。
    """
    try:
        os.makedirs(os.path.dirname(USAGE_LOG_PATH) or ".", exist_ok=True)
        record = {
            "question":  question,
            "timestamp": datetime.now().isoformat(),
            "source":    "streamlit",
        }
        if documents:
            record["documents"] = [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in documents
            ]
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARNING] 使用紀錄寫入失敗（不影響問答功能）: {e}")


def log_answer_quality(question: str, last: dict):
    """
    輕量、有界的品質監控 log：只存「這次回答有沒有依據文件、有沒有回答到問題」這幾個
    小欄位 + 文件來源類型（本地/網路搜尋），不存文件全文或答案全文，所以不會像存整份
    檢索結果那樣讓檔案無限長大。用途：
      - 追蹤系統回答品質隨時間的變化（例如算「grounded 比例」畫成趨勢圖）
      - 特別看「source_type=web_search」的那些，了解觸發網路搜尋補資料的問題，
        事後回答品質好不好——網路內容本身是一次性的（URL 可能失效、下次搜同樣問題
        結果也可能不同），不適合存全文當 reranker 訓練資料，但這個品質訊號還是有意義。
    """
    try:
        os.makedirs(os.path.dirname(QUALITY_LOG_PATH) or ".", exist_ok=True)
        docs = last.get("documents") or []
        sources = {d.metadata.get("source", "unknown") for d in docs}
        if "web_search" in sources and len(sources) > 1:
            source_type = "mixed"
        elif "web_search" in sources:
            source_type = "web_search"
        elif sources:
            source_type = "local"
        else:
            source_type = "none"

        qc = last.get("quality_check")  # pipeline.py 的 check_answer_quality 節點存的結果
        record = {
            "question":           question,
            "timestamp":          datetime.now().isoformat(),
            "source_type":        source_type,
            "doc_count":          len(docs),
            "is_grounded":        qc.get("is_grounded") if qc else None,
            "addresses_question": qc.get("addresses_question") if qc else None,
            "reasoning":          (qc.get("reasoning", "")[:150] if qc else None),
        }
        with open(QUALITY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARNING] 品質監控 log 寫入失敗（不影響問答功能）: {e}")


def render_sources(last: dict, show_debug: bool):
    """顯示這次回答用的是資料庫文件還是網路搜尋結果（一律顯示摘要，細節看 show_debug）"""
    docs = last.get("documents") or []
    local_docs = [d for d in docs if d.metadata.get("source") != "web_search"]
    web_docs   = [d for d in docs if d.metadata.get("source") == "web_search"]

    if docs:
        badge_parts = []
        if local_docs:
            badge_parts.append(f"📚 資料庫文件 × {len(local_docs)}")
        if web_docs:
            badge_parts.append(f"🌐 網路搜尋結果 × {len(web_docs)}")
        st.caption("　·　".join(badge_parts))
    else:
        st.caption("⚠️ 這次沒有任何檢索到的文件（純 LLM 回答）")

    if not show_debug:
        return

    if last.get("rewritten_question"):
        st.caption(f"重寫後的問題：{last['rewritten_question']}")

    if local_docs:
        with st.expander(f"📚 資料庫文件（{len(local_docs)} 筆）"):
            for i, d in enumerate(local_docs):
                origin = d.metadata.get("origin_file") or d.metadata.get("source", "unknown")
                st.markdown(f"**[{i}] {origin}**")
                st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
                st.divider()

    if web_docs:
        with st.expander(f"🌐 網路搜尋結果（{len(web_docs)} 筆）", expanded=True):
            for i, d in enumerate(web_docs):
                url = d.metadata.get("url", "")
                if url:
                    st.markdown(f"**[{i}]** [{url}]({url})")
                else:
                    st.markdown(f"**[{i}]** (無網址)")
                st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
                st.divider()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict]: question / answer / last (pipeline 的最終 state) / error

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if not turn.get("error"):
            render_sources(turn.get("last") or {}, show_debug)

question = st.chat_input("輸入你的問題 ...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("思考中 ...")

        final_state = None
        error_message = None
        with st.spinner("執行 pipeline 中 ..."):
            try:
                # 用 invoke() 直接拿完整的最終狀態，不用 stream()。
                # stream() 預設每一步只回傳「這個節點自己更新了哪些欄位」，
                # 如果最後一個節點沒有動到 documents（例如 mark_low_confidence 只回傳 generation），
                # 拿最後一步當作完整狀態就會誤判成「沒有文件」——即使答案其實真的有用到文件。
                final_state = app.invoke({"question": question})
            except Exception as e:
                # 最後一道防線：就算 pipeline 內部所有 try/except 都沒接住（例如非預期的例外型別），
                # 也不要讓使用者看到整頁的 Python traceback，改成顯示一句友善訊息。
                error_message = str(e)

        if error_message:
            answer = f"抱歉，處理這個問題時發生非預期錯誤，請稍後再試一次。\n\n（技術細節：{error_message}）"
            last = {}
        else:
            last = final_state or {}
            answer = last.get("generation", "(沒有產生答案)")
        placeholder.write(answer)

        log_usage_question(question, last.get("documents"))
        if not error_message:
            render_sources(last, show_debug)
            log_answer_quality(question, last)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "last": last,
            "error": bool(error_message),
        })
