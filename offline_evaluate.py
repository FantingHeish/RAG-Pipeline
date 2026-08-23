# offline_evaluate.py
#
# 離線版 LLM-as-Judge：不再是每個線上問題都要付的成本，改成批次跑一次，
# 針對 gold_standard.py + 真實使用紀錄（USAGE_LOG_PATH）裡的每一題：
#   1. 有帶已檢索的文件（來自 app.py 的使用紀錄）就直接用，沒有才用 Layer 1 Hybrid Search 重新撈
#   2. 用 LLM-as-Judge（graders.build_batch_retrieval_grader，跟以前線上用的是同一套邏輯，
#      但這裡是「重新」獨立判斷，不是採用線上 reranker 當時自己的分數——用同一個模型的舊分數
#      當訓練標籤等於自我循環，reranker 學不到新東西，一定要靠獨立的裁判）
#      幫每份候選文件標記「相關 / 不相關」
#   3. 把 (query, doc文字, label) 存成訓練資料（config.TRAINING_DATA_PATH，JSONL 格式）
#
# 這份訓練資料就是 finetune_reranker.py 拿去微調 cross-encoder reranker 的原料。
# 對應簡報畫的：「識別好壞 doc → 產出訓練資料 → 每個 query 配對相關/不相關」。
#
# 用法：
#   python offline_evaluate.py

import json
import os
from typing import List, Optional

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from config import (
    PERSIST_DIR, COLLECTION_NAME, TRAINING_DATA_PATH, OFFLINE_RELEVANCE_THRESHOLD,
    OFFLINE_JUDGE_BATCH_SIZE, USAGE_LOG_PATH,
)
from vectorstore import load_existing_store, build_retriever
from graders import build_batch_retrieval_grader, compute_weighted_score


def _grade_documents_in_batches(question: str, documents, batch_grader, batch_size: int) -> dict:
    """把 documents 切成每 batch_size 份一組，每組只呼叫一次 LLM，回傳 {index -> DocScoreItem}"""
    results = {}
    for start in range(0, len(documents), batch_size):
        chunk = documents[start:start + batch_size]
        block = "\n\n".join(f"[文件 {i}]\n{d.page_content}" for i, d in enumerate(chunk))
        try:
            batch_result = batch_grader.invoke({"question": question, "documents_block": block})
        except Exception as e:
            print(f"  WARNING: batch 評分呼叫失敗 ({e})，這批 {len(chunk)} 份文件略過")
            continue
        for item in batch_result.scores:
            if 0 <= item.doc_id < len(chunk):
                results[start + item.doc_id] = item
    return results


def generate_training_data(items: List[dict], append: bool = True) -> int:
    """
    items：每筆是 {"question": str, "documents": Optional[List[Document]]}。
    - documents 有提供（例如從 Streamlit 使用紀錄帶過來，當次已經檢索過）：直接用，不重新檢索。
    - documents 是 None（例如 gold_standard.py 的題目）：呼叫 retriever 重新檢索一次。
    append=True：累加在既有檔案後面（正常使用情境——訓練資料應該隨時間持續累積）。
    append=False：覆蓋重寫（測試或想整批重新標記時用）。
    回傳這次新增了幾筆訓練樣本。
    """
    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)
    retriever = build_retriever(store)
    batch_grader = build_batch_retrieval_grader()

    os.makedirs(os.path.dirname(TRAINING_DATA_PATH) or ".", exist_ok=True)
    mode = "a" if append else "w"

    new_samples = 0
    reused_retrieval = 0
    with open(TRAINING_DATA_PATH, mode, encoding="utf-8") as f:
        for item in items:
            question = item["question"]
            docs = item.get("documents")

            print(f"\n[offline_evaluate] 問題: {question}")

            if docs:
                reused_retrieval += 1
                print(f"  -沿用已檢索的 {len(docs)} 份文件（來自使用紀錄，跳過重新檢索）-")
            else:
                try:
                    docs = retriever.invoke(question)
                except Exception as e:
                    print(f"  -RETRIEVAL ERROR ({e})，略過這題-")
                    continue

            if not docs:
                print("  -沒有檢索到任何文件，略過-")
                continue

            score_map = _grade_documents_in_batches(question, docs, batch_grader, OFFLINE_JUDGE_BATCH_SIZE)

            for idx, d in enumerate(docs):
                score_item = score_map.get(idx)
                if score_item is None:
                    continue
                weighted = compute_weighted_score(score_item)
                label = 1.0 if weighted >= OFFLINE_RELEVANCE_THRESHOLD else 0.0

                record = {
                    "question": question,
                    "document": d.page_content,
                    "label": label,
                    "weighted_score": round(weighted, 3),
                    "source": d.metadata.get("source", "unknown"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_samples += 1
                print(f"  [{'相關' if label == 1.0 else '不相關'}] weighted={weighted:.2f}  "
                      f"{d.page_content[:60]}...")

    print(f"\n[offline_evaluate] 完成，新增 {new_samples} 筆訓練樣本 -> {TRAINING_DATA_PATH}"
          f"（其中 {reused_retrieval} 題沿用了使用紀錄裡已檢索的文件，省了重新檢索）")
    return new_samples


def _load_usage_items(path: str) -> List[dict]:
    """
    讀 app.py 累積下來的真實使用紀錄（USAGE_LOG_PATH），回傳
    [{"question": str, "documents": List[Document] 或 None（沒存到文件的舊紀錄）}, ...]
    """
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                q = record.get("question", "").strip()
                if not q:
                    continue
                docs: Optional[List[Document]] = None
                raw_docs = record.get("documents")
                if raw_docs:
                    docs = [
                        Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                        for d in raw_docs
                    ]
                items.append({"question": q, "documents": docs})
            except Exception as e:
                print(f"  WARNING: 跳過解析失敗的一行使用紀錄 ({e})")
    return items


if __name__ == "__main__":
    from gold_standard import GOLD_STANDARD

    gold_items = [
        {"question": case["question"], "documents": None}
        for case in GOLD_STANDARD if isinstance(case, dict) and "question" in case
    ]
    usage_items = _load_usage_items(USAGE_LOG_PATH)

    seen = set()
    all_items = []
    for it in gold_items + usage_items:
        if it["question"] not in seen:
            seen.add(it["question"])
            all_items.append(it)

    print(f"[offline_evaluate] gold_standard.py: {len(gold_items)} 題，"
          f"真實使用紀錄: {len(usage_items)} 題，合併去重後共 {len(all_items)} 題")
    generate_training_data(all_items, append=True)