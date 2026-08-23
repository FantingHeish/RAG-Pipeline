# tune_layer1_k.py
#
# LAYER1_K（向量搜尋 top-k）跟 LAYER1_BM25_K（BM25 top-k）決定 RRF 融合前，
# 各自先撈幾份候選文件。目前都設 5。這支腳本用 reranker_pairs.jsonl 的標記資料
# （跟 tune_bm25_params.py 同一批），分別測向量搜尋、BM25 各自單獨的 recall@k，
# 看調高 k 有沒有實際幫助——如果某個 k 值下 recall 已經接近飽和（再往上加也不太
# 提升），代表目前的 k 已經夠用；如果還在明顯往上爬，代表調高有意義。
#
# 不呼叫任何 LLM（只用 embedding 搜尋 + BM25 打分），跑起來快。
#
# 用法：
#   python tune_layer1_k.py

import json
from collections import defaultdict

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME, TRAINING_DATA_PATH
from vectorstore import load_existing_store

CANDIDATE_KS = [5, 10, 15, 20]


def load_relevant_pairs(path: str) -> dict:
    relevant = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if float(record["label"]) == 1.0:
                    relevant[record["question"]].append(record["document"])
            except Exception as e:
                print(f"  WARNING: 跳過解析失敗的一行 ({e})")
    return relevant


def main():
    relevant_by_question = load_relevant_pairs(TRAINING_DATA_PATH)
    if not relevant_by_question:
        print(f"[tune_layer1_k] {TRAINING_DATA_PATH} 裡沒有 label=1 的資料，無法評估。")
        return
    print(f"[tune_layer1_k] 讀到 {len(relevant_by_question)} 題、"
          f"共 {sum(len(v) for v in relevant_by_question.values())} 筆標記為相關的文件\n")

    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)
    store_data = store.get()
    bm25_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(store_data["documents"], store_data["metadatas"])
    ]

    print(f"{'k':>4}  {'向量 recall@k':>14}  {'BM25 recall@k':>14}")
    print("-" * 38)
    for k in CANDIDATE_KS:
        # 每個 k 值只建一次 retriever，重複用在所有題目上，不要每題重建一次索引
        vector_retriever = store.as_retriever(search_kwargs={"k": k})
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = k

        hits_v, hits_b, total = 0, 0, 0
        for question, relevant_docs in relevant_by_question.items():
            v_texts = {d.page_content for d in vector_retriever.invoke(question)}
            b_texts = {d.page_content for d in bm25_retriever.invoke(question)}
            for rd in relevant_docs:
                total += 1
                if rd in v_texts:
                    hits_v += 1
                if rd in b_texts:
                    hits_b += 1

        vr = hits_v / total if total else 0.0
        br = hits_b / total if total else 0.0
        print(f"{k:>4}  {vr:>14.3f}  {br:>14.3f}")

    print("\n[說明] 觀察哪個 k 之後 recall 幾乎不再上升（飽和點），")
    print("       建議把 LAYER1_K / LAYER1_BM25_K 設在飽和點附近，不用設更大（候選變多但沒實際幫助，")
    print("       只會讓 reranker 要多算幾份、拖慢速度）。")


if __name__ == "__main__":
    main()
