# calibrate_rerank_threshold.py
#
# RERANK_SCORE_THRESHOLD 目前是預設值 0.0，從沒用實際資料校準過（見 config.py 的註解：
# bge-reranker 輸出的是原始 logit，不是 0~1 機率，0.0 這個值只是隨便選的起點）。
#
# 這支腳本不呼叫任何 LLM（不花 OpenAI 額度），只做 Layer 1 檢索 + Layer 2 rerank 打分，
# 拿兩組問題的實際 relevance_score 分布來對照：
#   - IN_DOMAIN：本地文件庫真的有涵蓋的主題（從 gold_standard.py 抽幾題）
#   - OUT_OF_DOMAIN：本地文件庫確定沒有涵蓋的主題（例如你之前測過的 Neuralink）
# 理想情況下兩組分數會有清楚的區隔，門檻就設在中間；如果兩組分數混在一起，
# 代表 reranker 本身（不是門檻）分辨力不夠，那是另一個要解決的問題，不能只靠調門檻。
#
# 用法：
#   python calibrate_rerank_threshold.py

from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME
from vectorstore import load_existing_store, build_retriever, build_reranker_model, rerank_documents
from gold_standard import GOLD_STANDARD

# 從 gold_standard.py 抽幾題當作「本地文件庫真的有」的代表（前 5 題）
IN_DOMAIN_QUESTIONS = [case["question"] for case in GOLD_STANDARD[:5]]

# 確定本地文件庫沒有涵蓋的主題，換成你自己測過確定會觸發 web search 的問題也可以
OUT_OF_DOMAIN_QUESTIONS = [
    "馬斯克 ai 人腦 臨床案例?",
    "Neuralink 有哪些研究與應用方向",
    "SpaceX 最新的火箭發射計畫",
]


def collect_scores(questions, retriever, cross_encoder, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    all_scores = []
    for q in questions:
        docs = retriever.invoke(q)
        if not docs:
            print(f"\nQ: {q}\n  -Layer 1 就沒撈到任何候選文件-")
            continue
        scored = rerank_documents(q, docs, cross_encoder)  # 不設 min_score，看完整分數
        print(f"\nQ: {q}")
        for d in scored:
            score = d.metadata.get("relevance_score", 0.0)
            all_scores.append(score)
            origin = d.metadata.get("origin_file") or d.metadata.get("source", "?")
            print(f"  score={score:>8.3f}  {origin}  | {d.page_content[:50]}...")
    return all_scores


def main():
    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)
    retriever = build_retriever(store)
    cross_encoder = build_reranker_model()

    in_scores = collect_scores(IN_DOMAIN_QUESTIONS, retriever, cross_encoder, "IN-DOMAIN（本地文件庫真的有涵蓋）")
    out_scores = collect_scores(OUT_OF_DOMAIN_QUESTIONS, retriever, cross_encoder, "OUT-OF-DOMAIN（本地文件庫確定沒有）")

    print(f"\n{'=' * 70}\n統計摘要\n{'=' * 70}")
    if in_scores:
        print(f"IN-DOMAIN     : min={min(in_scores):.3f}  max={max(in_scores):.3f}  "
              f"avg={sum(in_scores)/len(in_scores):.3f}  n={len(in_scores)}")
    if out_scores:
        print(f"OUT-OF-DOMAIN : min={min(out_scores):.3f}  max={max(out_scores):.3f}  "
              f"avg={sum(out_scores)/len(out_scores):.3f}  n={len(out_scores)}")

    if in_scores and out_scores:
        if min(in_scores) > max(out_scores):
            suggested = (min(in_scores) + max(out_scores)) / 2
            print(f"\n[建議] 兩組分數完全分開，可以把 RERANK_SCORE_THRESHOLD 設成約 {suggested:.3f}")
            print(f"       （IN-DOMAIN 最低分 {min(in_scores):.3f} 到 OUT-OF-DOMAIN 最高分 {max(out_scores):.3f} 之間）")
        else:
            print(f"\n[警告] 兩組分數有重疊（IN-DOMAIN 最低 {min(in_scores):.3f} <= "
                  f"OUT-OF-DOMAIN 最高 {max(out_scores):.3f}），沒有一個門檻能完美分開兩組。")
            print("       這代表光調門檻沒辦法完全解決問題，reranker 本身對這類問題的分辨力有限，")
            print("       之後 reranker 微調累積更多資料、涵蓋更多主題後，應該能改善這個分辨力。")
            print(f"       先抓一個折衷值，例如 OUT-OF-DOMAIN 的平均分數 {sum(out_scores)/len(out_scores):.3f} 附近，")
            print("       盡量濾掉大部分不相關文件，同時接受可能還是會有一些誤判。")

    print("\n把上面建議的數字填進 .env：")
    print("RERANK_SCORE_THRESHOLD=<建議值>")


if __name__ == "__main__":
    main()
