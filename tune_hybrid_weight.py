# tune_hybrid_weight.py
#
# config.py 的 HYBRID_VECTOR_WEIGHT 目前是憑經驗猜的 0.6，這支腳本用 gold_standard.py
# 當驗證集，掃過幾組候選權重，各自算 RAGAS 的 context_precision / context_recall，
# 挑實際分數最高的權重，取代用猜的。
#
# 只測 Layer 1（Hybrid Search）本身的檢索品質，不跑後面的 rerank/生成/LLM 判斷，
# 這樣才不會被其他層的效果混在一起，看不出 Hybrid 權重本身的影響。
#
# 用法：
#   python tune_hybrid_weight.py
# 跑完會印出每個權重的分數，並建議一個值——手動回填到 .env 的 HYBRID_VECTOR_WEIGHT，
# 不會自動幫你改設定檔（這種會影響全系統行為的參數，改之前你應該自己看過數字再決定）。

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME, LAYER1_K, LAYER1_BM25_K
from vectorstore import load_existing_store
from gold_standard import GOLD_STANDARD
from evaluation import get_ragas_embeddings, extract_metric_mean

CANDIDATE_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _build_hybrid_retriever(store, vector_weight: float):
    vector_retriever = store.as_retriever(search_kwargs={"k": LAYER1_K})
    store_data = store.get()
    bm25_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(store_data["documents"], store_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = LAYER1_BM25_K
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[vector_weight, 1 - vector_weight])


def main():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall

    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)

    print(f"{'weight':>8}  {'context_precision':>18}  {'context_recall':>15}  {'sum':>6}")
    print("-" * 55)

    results = {}
    for w in CANDIDATE_WEIGHTS:
        retriever = _build_hybrid_retriever(store, w)
        rows = []
        for case in GOLD_STANDARD:
            try:
                docs = retriever.invoke(case["question"])
            except Exception as e:
                print(f"  WARNING: 檢索失敗 ({e})，這題略過")
                docs = []
            contexts = [d.page_content for d in docs] or [""]
            rows.append({
                "question": case["question"],
                "contexts": contexts,
                "ground_truth": case.get("ground_truth", ""),
                "answer": "",  # context_precision/recall 不需要真的生成答案
            })

        dataset = Dataset.from_list(rows)
        kwargs = {"metrics": [context_precision, context_recall]}
        embeddings_for_ragas = get_ragas_embeddings()
        if embeddings_for_ragas is not None:
            kwargs["embeddings"] = embeddings_for_ragas
        result = evaluate(dataset, **kwargs)
        precision = extract_metric_mean(result, "context_precision")
        recall = extract_metric_mean(result, "context_recall")
        results[w] = (precision, recall)
        print(f"{w:>8.1f}  {precision:>18.3f}  {recall:>15.3f}  {precision + recall:>6.3f}")

    best_weight = max(results, key=lambda w: sum(results[w]))
    print("\n" + "=" * 55)
    print(f"[建議] HYBRID_VECTOR_WEIGHT = {best_weight}"
          f"（context_precision + context_recall 加總最高：{sum(results[best_weight]):.3f}）")
    print("       手動更新 .env 的 HYBRID_VECTOR_WEIGHT，或直接改 config.py 的預設值。")
    print("=" * 55)


if __name__ == "__main__":
    main()
