# tune_bm25_params.py
#
# BM25 本身沒有可訓練的權重，但有兩個超參數 k1（詞頻飽和速度）、b（文件長度正規化程度），
# 目前 langchain 的 BM25Retriever 用的是 rank_bm25 的預設值（k1=1.5, b=0.75），從沒用你的
# 資料實際調過。這支腳本重複利用 finetune_reranker.py 同一批 reranker_pairs.jsonl 標記資料
# （question, document, label），對每組候選 (k1, b) 做網格搜尋：
#   - 對每一題，用 BM25 對「全部文件庫」檢索出 LAYER1_BM25_K 筆
#   - 檢查這批標記資料裡「label=1（相關）」的文件，有多少真的出現在 BM25 撈回來的結果裡
#   - recall@k 越高，代表這組 (k1, b) 讓 BM25 越容易把真正相關的文件排進前 k 名
#
# 注意：這是 recall-based 的網格搜尋，不是梯度訓練——BM25 沒有「學習」這回事，
# 這裡做的事情跟 tune_hybrid_weight.py 調 HYBRID_VECTOR_WEIGHT 是同一種性質。
#
# 用法：
#   python tune_bm25_params.py
# 跑完印出建議的 (k1, b)，手動更新 config.py 的 BM25_K1 / BM25_B（目前這兩個常數還不存在，
# 需要自己在 config.py 加，或直接在 vectorstore.py 建立 BM25Retriever 時的 bm25_params 帶入）。

import json
from collections import defaultdict

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME, LAYER1_BM25_K, TRAINING_DATA_PATH
from vectorstore import load_existing_store

CANDIDATE_K1 = [1.2, 1.5, 1.8, 2.2]
CANDIDATE_B = [0.5, 0.75, 0.9]


def load_relevant_pairs(path: str) -> dict:
    """讀 reranker_pairs.jsonl，回傳 {question: [相關文件的 page_content, ...]}（只取 label=1 的）。"""
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
        print(f"[tune_bm25_params] {TRAINING_DATA_PATH} 裡沒有任何 label=1 的資料，無法評估，先累積訓練資料。")
        return
    print(f"[tune_bm25_params] 讀到 {len(relevant_by_question)} 題、"
          f"共 {sum(len(v) for v in relevant_by_question.values())} 筆標記為相關的文件")

    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)
    store_data = store.get()
    bm25_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(store_data["documents"], store_data["metadatas"])
    ]

    print(f"\n{'k1':>6}  {'b':>6}  {'recall@' + str(LAYER1_BM25_K):>10}")
    print("-" * 30)

    results = {}
    for k1 in CANDIDATE_K1:
        for b in CANDIDATE_B:
            retriever = BM25Retriever.from_documents(bm25_docs, bm25_params={"k1": k1, "b": b})
            retriever.k = LAYER1_BM25_K

            hits, total = 0, 0
            for question, relevant_docs in relevant_by_question.items():
                retrieved_texts = {d.page_content for d in retriever.invoke(question)}
                for rd in relevant_docs:
                    total += 1
                    if rd in retrieved_texts:
                        hits += 1

            recall = hits / total if total else 0.0
            results[(k1, b)] = recall
            print(f"{k1:>6.2f}  {b:>6.2f}  {recall:>10.3f}")

    best_k1, best_b = max(results, key=results.get)
    print("\n" + "=" * 40)
    print(f"[建議] k1={best_k1}, b={best_b}（recall@{LAYER1_BM25_K} = {results[(best_k1, best_b)]:.3f}）")
    print("       套用方式：vectorstore.py 建立 BM25Retriever 的地方加上")
    print(f"       bm25_params={{'k1': {best_k1}, 'b': {best_b}}}")
    print("=" * 40)


if __name__ == "__main__":
    main()
