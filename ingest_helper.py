# ingest_helper.py
#
# 用途：把你手上的文件放進 Chroma 向量資料庫，並且是「增量」匯入 ——
#       只有新增或內容變更過的檔案才會重新 chunk + embed，沒變的檔案會被跳過，
#       不會每次重跑都把整個資料夾重新處理一次（避免重複浪費 API 額度、
#       也避免 Chroma 裡出現同一份文件的重複 chunk）。
#
# 運作原理：用一個 manifest（PERSIST_DIR/ingest_manifest.json）記錄每個檔案的
# 內容 hash（SHA256）跟它產生的 chunk id list。重跑時：
#   - hash 沒變 -> 跳過，不重新處理
#   - hash 變了（檔案被編輯過）-> 先刪掉舊 chunk，再重新 embed 新內容
#   - 全新檔案 -> 正常處理並記進 manifest
#
# chunk id 格式：doc:{語意前綴}:{內容hash前12碼}:{chunk編號}
#   語意前綴不是隨便取的，是從內容抽出來的關鍵字（config.SEMANTIC_KEY_ENABLED 控制開關）——
#   讓 key 本身就看得懂內容在講什麼（例如 doc:2024-ai-laptop:a1b2c3d4e5f6:0），
#   方便用前綴做分類/除錯；hash 段落還是照樣保證同內容不重複 embed，兩者互不影響。
#   關掉這個開關就退回用檔名當前綴（不用多花 LLM 呼叫）。
#
# 支援四種情境：
#   1) ingest_texts()  ：手上已經是純文字（字串 list，沒有原始檔案，不做增量追蹤）
#   2) ingest_files()   ：手上是實際的 PDF / Word(.docx) / 圖片(.png/.jpg) 檔案路徑 <- 一般情況用這個
#   3) ingest_folder()  ：手上是一整個資料夾，裡面混著各種支援的檔案類型
#   4) 圖片：用有視覺能力的模型生成文字描述（caption），跟一般文字 chunk 走同一條路存進主 collection，
#      另外用 CLIP 把圖片本身也編碼進一個獨立的圖片 collection，供「以文搜圖」使用（見 vectorstore.py）
#
# 用法（增量匯入，之後加新文件重跑同一行指令就好）：
#   from ingest_helper import ingest_folder
#   ingest_folder("./smart_healthcare_docs", source_name="smart_healthcare",
#                 extensions=(".pdf", ".md", ".docx", ".png", ".jpg", ".jpeg"))
#
# 需要額外安裝：pip install docx2txt（讀取 .docx 用）、pillow + sentence-transformers（CLIP 圖片用）

import base64
import os
import glob
import json
import hashlib
import re
import time
from typing import List, Optional

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import (
    PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    SEMANTIC_KEY_ENABLED, SEMANTIC_KEY_BATCH_SIZE,
    IMAGE_INGEST_ENABLED, IMAGE_CAPTION_MODEL,
)

MANIFEST_FILENAME = "ingest_manifest.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ============================================================
# Manifest讀寫：記錄「哪個檔案、什麼內容、產生了哪些 chunk id」
# ============================================================
# 在 PERSIST_DIR(Chroma 資料庫資料夾)下維護一份 JSON 清單
# 每次匯入檔案,先算這個檔案目前的內容 hash,拿去跟 manifest 裡記錄的舊 hash 比對,以此判斷該怎麼處理該檔案


# 回傳 manifest 檔案存在哪裡
def _manifest_path() -> str:
    return os.path.join(PERSIST_DIR, MANIFEST_FILENAME)

# 讀 JSON
def _load_manifest() -> dict:
    """manifest 結構：{ collection_name: { abs_file_path: {hash, chunk_ids, ingested_at} } }"""
    path = _manifest_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: manifest 讀取失敗 ({e})，視為空白重新開始")
    return {}

# 寫 JSON
def _save_manifest(manifest: dict):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# HASH SHA256
def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# 查詢目前 collection 目前記錄了哪些檔案、各幾個 chunk、檔案是否存在
def show_ingest_status(source_name: str):
    manifest = _load_manifest()
    entries = manifest.get(source_name, {})
    if not entries:
        print(f"[status] collection '{source_name}' 目前沒有任何 manifest 記錄。")
        return
    print(f"[status] collection '{source_name}' 已匯入 {len(entries)} 個檔案：")
    for path, info in entries.items():
        exists = "✓" if os.path.exists(path) else "✗ 檔案已不在硬碟上"
        print(f"  - {os.path.basename(path)}: {len(info.get('chunk_ids', []))} chunks  [{exists}]")


# ============================================================
# 語意化 Key 前綴：ingest 時多一次 LLM 呼叫，幫每個 chunk 抽 3-5 個關鍵字當標籤
# ============================================================
# 例如某個 chunk 在講「2024年 AI 在筆電的應用」，key 就會長得像：
#   doc:2024-ai-laptop-application:a1b2c3d4e5f6:0
# 前面的語意前綴不影響去重（hash 段落沒變），只是讓 key 本身看得懂內容在講什麼。
#
# 注意：這是每個 chunk 都會抽一次關鍵字（用 batch 呼叫降低次數，但還是要多花 LLM 額度）。
# 文件量大的時候成本會累加，不需要的話把 config.SEMANTIC_KEY_ENABLED 設成 false 即可，
# 會自動退回用檔名當前綴，不會多花任何 API 呼叫。

def _slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text).strip("-").lower()
    return text[:max_len] or "untitled"


class _ChunkKeywords(BaseModel):
    chunk_id: int = Field(description="對應輸入時的 chunk 編號（從 0 開始）")
    keywords: List[str] = Field(description="3-5 個能代表這段內容主題的關鍵字，短詞不要句子")


class _ChunkKeywordsBatch(BaseModel):
    items: List[_ChunkKeywords] = Field(description="每個 chunk 對應一筆關鍵字結果，數量要跟輸入一致")


def _build_keyword_extractor():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一個關鍵字抽取器。對每一段輸入的文字，抽出 3-5 個最能代表這段內容主題的關鍵字"
            "（例如年份、技術名詞、應用場景），用短詞不要句子，中英文皆可。"
            "輸入會是多個帶編號的 chunk（格式：[chunk N] 內容...），"
            "請針對「每一個」chunk 分別輸出，數量要跟輸入一致，不能省略任何一個。"
        )),
        ("human", "{chunks_block}"),
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return prompt | llm.with_structured_output(_ChunkKeywordsBatch)


def _generate_semantic_slugs(chunks: List[Document], batch_size: int = SEMANTIC_KEY_BATCH_SIZE) -> List[Optional[str]]:
    """對每個 chunk 抽關鍵字、組成 slug；某批呼叫失敗只影響那一批（回傳 None，呼叫端會退回用檔名）。"""
    extractor = _build_keyword_extractor()
    slugs: List[Optional[str]] = [None] * len(chunks)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        block = "\n\n".join(f"[chunk {i}]\n{c.page_content[:300]}" for i, c in enumerate(batch))
        try:
            result = extractor.invoke({"chunks_block": block})
            for item in result.items:
                if 0 <= item.chunk_id < len(batch):
                    slugs[start + item.chunk_id] = _slugify("-".join(item.keywords))
        except Exception as e:
            print(f"  WARNING: 語意關鍵字抽取失敗 ({e})，這批 {len(batch)} 個 chunk 退回用檔名當前綴")

    return slugs


# ============================================================
# 圖片：caption（主要能力，走一般文字流程） + CLIP embedding（額外的以文搜圖能力）
# ============================================================

def _caption_image(path: str, model_name: str = IMAGE_CAPTION_MODEL) -> Optional[str]:
    """
    用有視覺能力的模型（例如 gpt-4o-mini）幫圖片生成文字描述。
    這段文字之後會跟一般文字 chunk 走同一條 embed/檢索流程——這是圖片能被現有 RAG pipeline
    直接讀到的關鍵，因為最終生成答案的 LLM（gpt-3.5-turbo）本身看不懂圖片，只能讀文字。
    """
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "jpeg" if ext == "jpg" else ext
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  [ERROR] 讀取圖片失敗 '{path}': {e}")
        return None

    llm = ChatOpenAI(model=model_name, temperature=0)
    message = HumanMessage(content=[
        {"type": "text", "text": (
            "請用繁體中文詳細描述這張圖片的內容。如果是圖表/示意圖/流程圖，"
            "說明它在呈現什麼資訊、有哪些關鍵數字或趨勢；如果是一般照片，描述畫面內容。"
            "控制在 150 字以內，不要加任何開場白。"
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
    ])
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"  [ERROR] 圖片描述生成失敗 '{path}' ({e})——"
              f"確認 config.IMAGE_CAPTION_MODEL='{model_name}' 是有視覺能力的模型")
        return None


def _add_clip_embedding(path: str, metadata: dict, clip_model):
    """
    額外把圖片本身（不是 caption 文字）編碼進獨立的 CLIP collection，供之後「以文搜圖」使用。
    這步驟失敗不影響 caption 已經存進主資料庫——圖片一樣查得到，只是少了以文搜圖這個加碼功能。
    """
    import vectorstore

    try:
        image_store = vectorstore.load_image_store(vectorstore.ClipEmbeddings(clip_model))
        vectorstore.add_image_documents(image_store, [path], [metadata], clip_model)
    except Exception as e:
        print(f"  WARNING: CLIP 圖片向量寫入失敗，不影響 caption 已存進主資料庫 ({e})")


# ============================================================
# ingest_texts / add_more_texts：純文字, 沒有原始檔案, 不做增量更新
# ============================================================

# 把文字 chunk + embed，存/建立指定 source 的 Chroma
# 純文字沒有hash 會重複更新, 如果資料是檔案, 優先用 ingest_files()
def ingest_texts(
    texts: List[str],
    source_name: str,
    metadatas: Optional[List[dict]] = None,
) -> Chroma:

    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas 長度必須跟 texts 一致")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    docs = []
    for i, text in enumerate(texts):
        meta = dict(metadatas[i]) if metadatas else {}
        meta["source"]   = source_name
        meta["doc_type"] = meta.get("doc_type", "original")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=meta))

    print(f"[ingest_texts] {len(texts)} 篇原始文字 -> 切成 {len(docs)} 個 chunks")

    embeddings = OpenAIEmbeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=source_name,
        persist_directory=PERSIST_DIR,
    )
    print(f"[ingest_texts] 已存入 Chroma collection '{source_name}'（路徑：{PERSIST_DIR}）")
    return store

# 加純文字到既有 collection
def add_more_texts(store: Chroma, texts: List[str], source_name: str, metadatas: Optional[List[dict]] = None):
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas 長度必須跟 texts 一致")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_chunks, all_metas = [], []
    for i, text in enumerate(texts):
        meta = dict(metadatas[i]) if metadatas else {}
        meta["source"]   = source_name
        meta["doc_type"] = meta.get("doc_type", "original")
        for chunk in splitter.split_text(text):
            all_chunks.append(chunk)
            all_metas.append(meta)

    store.add_texts(texts=all_chunks, metadatas=all_metas)
    print(f"[add_more_texts] 追加了 {len(all_chunks)} 個 chunks 到既有 collection")


# ============================================================
# ingest_files / ingest_folder(增量更新)
# ============================================================

# 依檔案類型load，並切 chunk
def _load_and_split_file(path: str, splitter) -> Optional[List[Document]]:

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".doc":
        print(f"  [SKIP] '{path}' 是舊版 .doc 格式，不支援，請先另存為 .docx 再匯入。")
        return None
    else:
        print(f"  [SKIP] 不支援的副檔名: {path}")
        return None

    try:
        raw_docs = loader.load()
    except Exception as e:
        print(f"  [ERROR] 讀取失敗 '{path}': {e}")
        return None

    if not raw_docs or not any(d.page_content.strip() for d in raw_docs):
        print(f"  [WARNING] '{path}' 讀不到任何文字內容（可能是掃描版 PDF，需要先 OCR）")
        return None

    return splitter.split_documents(raw_docs)


# 將檔案匯入 Chroma 增量更新(目前支援：.pdf .docx .txt .md .png .jpg .jpeg)
# 只有新增或內容變更過的檔案才會被重新處理，沒變的檔案則不會重複處理
def ingest_files(
    file_paths: List[str], # 檔案路徑 list
    source_name: str, # # 存進哪個 collection
    extra_metadata: Optional[dict] = None, # 每份文件都會附加的共同 metadata
    incremental: bool = True, # True= 跳過沒變的檔案, False = 需要增量更新的部分
) -> Optional[Chroma]:

    splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = OpenAIEmbeddings()
    store = Chroma(collection_name=source_name, embedding_function=embeddings, persist_directory=PERSIST_DIR)

    manifest = _load_manifest()
    collection_manifest = manifest.setdefault(source_name, {})

    # 圖片用的 CLIP 模型只在真的有圖片要處理時才載入（第一次用會比較慢，要下載模型）
    has_images = any(os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS for p in file_paths)
    clip_model = None
    if has_images and IMAGE_INGEST_ENABLED:
        import vectorstore
        clip_model = vectorstore.get_clip_model()

    new_docs, new_ids = [], []
    skipped_count, updated_count, new_count, failed_count = 0, 0, 0, 0

    for path in file_paths:
        abs_path = os.path.abspath(path)
        ext = os.path.splitext(path)[1].lower()
        try:
            current_hash = _file_hash(path)
        except Exception as e:
            print(f"  [ERROR] 無法讀取檔案計算 hash '{path}': {e}")
            failed_count += 1
            continue

        prev = collection_manifest.get(abs_path)

        if incremental and prev and prev.get("hash") == current_hash:
            skipped_count += 1
            continue

        # 檔案內容變更過, 先刪掉舊 chunk
        if prev and prev.get("chunk_ids"):
            try:
                store.delete(ids=prev["chunk_ids"])
            except Exception as e:
                print(f"  WARNING: 刪除 '{os.path.basename(path)}' 舊 chunks 失敗（可能本來就不存在）: {e}")

        fallback_slug = _slugify(os.path.splitext(os.path.basename(path))[0])

        # ---- 圖片：caption 進主 collection + CLIP 進圖片 collection ----
        if ext in IMAGE_EXTENSIONS:
            if not IMAGE_INGEST_ENABLED:
                print(f"  [SKIP] 圖片匯入功能關閉（IMAGE_INGEST_ENABLED=false），略過 '{path}'")
                continue

            caption = _caption_image(path)
            if caption is None:
                failed_count += 1
                continue

            meta = {
                "source":      source_name,
                "doc_type":    "image_caption",
                "modality":    "image",
                "origin_file": os.path.basename(path),
                "image_path":  abs_path,
            }
            if extra_metadata:
                meta.update(extra_metadata)

            doc = Document(page_content=caption, metadata=meta)
            chunk_ids = [f"doc:image-{fallback_slug}:{current_hash[:12]}:0"]

            new_docs.append(doc)
            new_ids.extend(chunk_ids)

            if clip_model is not None:
                _add_clip_embedding(path, dict(meta), clip_model)

            collection_manifest[abs_path] = {
                "hash": current_hash, "chunk_ids": chunk_ids, "ingested_at": time.time(),
            }
            if prev:
                updated_count += 1
                print(f"  [UPDATE-IMAGE] {os.path.basename(path)} -> caption 已重新生成")
            else:
                new_count += 1
                print(f"  [NEW-IMAGE] {os.path.basename(path)} -> caption: {caption[:60]}...")
            continue

        # ---- 一般文字檔案（PDF / docx / txt / md）----
        chunks = _load_and_split_file(path, splitter)
        if chunks is None:
            failed_count += 1
            continue

        if SEMANTIC_KEY_ENABLED:
            slugs = _generate_semantic_slugs(chunks)
        else:
            slugs = [None] * len(chunks)

        chunk_ids = [
            f"doc:{slugs[i] or fallback_slug}:{current_hash[:12]}:{i}"
            for i in range(len(chunks))
        ]
        for c, cid in zip(chunks, chunk_ids):
            c.metadata["source"]      = source_name 
            c.metadata["doc_type"]    = c.metadata.get("doc_type", "original")
            c.metadata["origin_file"] = os.path.basename(path)
            if extra_metadata: 
                c.metadata.update(extra_metadata)

        new_docs.extend(chunks)
        new_ids.extend(chunk_ids)

        collection_manifest[abs_path] = {
            "hash":        current_hash,
            "chunk_ids":   chunk_ids,
            "ingested_at": time.time(),
        }

        if prev:
            updated_count += 1
            print(f"  [UPDATE] {os.path.basename(path)} -> {len(chunks)} chunks（內容已變更，已重新處理）")
        else:
            new_count += 1
            print(f"  [NEW] {os.path.basename(path)} -> {len(chunks)} chunks")

    if new_docs:
        store.add_documents(documents=new_docs, ids=new_ids)

    _save_manifest(manifest)

    print(f"[ingest_files] 完成：新增 {new_count} 個檔案、更新 {updated_count} 個檔案、"
          f"跳過 {skipped_count} 個未變更的檔案、失敗 {failed_count} 個。")
    if skipped_count > 0:
        print(f"  （跳過的檔案內容沒變，不會重新 embed，這是正常且預期的行為）")

    return store

# 掃資料夾裡指定副檔名的檔案,轉呼叫 ingest_files()增量更新
def ingest_folder(
    folder_path: str,
    source_name: str,
    extensions: tuple = (".pdf", ".docx"),
    extra_metadata: Optional[dict] = None,
    incremental: bool = True, # # True= 跳過沒變的檔案, False = 需要增量更新的部分
    remove_missing: bool = False, #True:manifest 裡記錄過、但現在資料夾裡已經找不到的檔案, False:不刪除, 預設 False
) -> Optional[Chroma]:

    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))

    print(f"[ingest_folder] 在 '{folder_path}' 找到 {len(file_paths)} 個檔案（副檔名：{extensions}）")
    if not file_paths:
        print("[ingest_folder] 沒找到任何符合的檔案，請確認路徑跟副檔名是否正確。")
        return None

    store = ingest_files(
        file_paths, source_name=source_name, extra_metadata=extra_metadata, incremental=incremental
    )

    if remove_missing:
        manifest = _load_manifest()
        collection_manifest = manifest.get(source_name, {})
        current_abs_paths = {os.path.abspath(p) for p in file_paths}
        missing = [p for p in collection_manifest if p not in current_abs_paths]

        if missing:
            print(f"[ingest_folder] 偵測到 {len(missing)} 個檔案已從資料夾移除，清除對應的向量資料...")
            for path in missing:
                info = collection_manifest.pop(path, None)
                if info and info.get("chunk_ids") and store is not None:
                    try:
                        store.delete(ids=info["chunk_ids"])
                        print(f"  [REMOVED] {os.path.basename(path)}")
                    except Exception as e:
                        print(f"  WARNING: 刪除 '{os.path.basename(path)}' 失敗: {e}")
            manifest[source_name] = collection_manifest
            _save_manifest(manifest)

    return store


# main() 範例
if __name__ == "__main__":
    # 範例 1：整個資料夾一次匯入
    # ingest_folder("./smart_healthcare_docs", source_name="smart_healthcare",
    #               extensions=(".pdf", ".md", ".docx"))

    # 範例 2：直接指定檔案路徑
    # ingest_files(
    #     ["/mnt/user-data/uploads/report.pdf", "/mnt/user-data/uploads/spec.docx"],
    #     source_name="smart_healthcare",
    # )

    # 範例 3：查看目前已經匯入了哪些檔案
    # show_ingest_status("smart_healthcare")

    # 範例 4：手上已經是純文字（沒有原始檔案），沒有增量追蹤
    my_texts = [
        "這是我第一篇要放進去的文章全文，內容隨便填，用來測試流程是否正常運作。",
        "這是第二篇文章，可以是任何長度的純文字，函式會自動幫你切成適合檢索的 chunk。",
    ]
    ingest_texts(my_texts, source_name="smart_healthcare")
