"""
Adaptive RAG PoC
======================================================
  1. 資料庫建立：先用 ingest_helper.py 把 PDF/Word/Markdown/圖片匯入 Chroma
     （見下方「執行前準備」，或直接參考 ingest_helper.py 裡的範例）
  2. Retrieval Layer
       Query Rewriting
       Layer 1: Hybrid Search - BM25 + Vector Search with RRF fusion
       Layer 2: Cross-Encoder Reranker(Pairwise) + relevance_score 門檻過濾
     線上不再呼叫 LLM-as-Judge 逐題評分——那套邏輯搬到 offline_evaluate.py 離線跑，
     用來產生訓練資料微調 reranker（見 finetune_reranker.py），不是每個問題都要付的成本。
  3. 生成後品質檢查：合併「有沒有幻覺」+「有沒有回答到問題」成一次 LLM 呼叫
  4. Retry: 迴圈超過 MAX_RETRIES 強制走 plain_answer / mark_low_confidence
  5. Eval: keyword 版 + RAGAS(faithfulness / answer_relevancy / context_precision / context_recall)，
     兩者共用同一次 pipeline 執行結果（run_full_evaluation()），不會讓同一組問題被重跑兩次
"""

from langchain_openai.embeddings import OpenAIEmbeddings

from config import validate_api_keys, QUERY_REWRITING_ENABLED, MAX_RETRIES, COLLECTION_NAME
from vectorstore import load_existing_store, build_retriever
from pipeline import build_pipeline
from pipeline_baseline import build_naive_pipeline, build_no_retrieval_pipeline
from evaluation import run_full_evaluation, evaluate_with_ragas_from_results, print_comparison, print_ragas_comparison, print_answers

# ============================================================
# 啟動檢查
# ============================================================
validate_api_keys(require_tavily=True)  # OPENAI_API_KEY 缺少會直接中止；TAVILY 缺少只警告（不中止）

print(f"[CONFIG] QUERY_REWRITING_ENABLED={QUERY_REWRITING_ENABLED}  MAX_RETRIES={MAX_RETRIES}  "
      f"COLLECTION_NAME={COLLECTION_NAME}")

# ============================================================
# 組裝 pipeline
# ============================================================

embeddings = OpenAIEmbeddings()

# 打開已經用 ingest_helper.py 匯入好的 Chroma collection
store = load_existing_store(embeddings)

# Retrieval Layer
retriever = build_retriever(store)

# Adaptive RAG（Query Rewriting + LLM-as-Judge + retry 保護）
app = build_pipeline(retriever, embeddings=embeddings)

# Naive RAG baseline（A/B Test，純向量檢索，無 hybrid/rerank/grading）
baseline_app = build_naive_pipeline(store)

# 完全不檢索的 baseline（A/B/C Test 的第三欄，純 LLM 憑自身知識回答）
no_retrieval_app = build_no_retrieval_pipeline()

# ============================================================
# 測試 & 評估
# ============================================================

if __name__ == "__main__":

    def run_verbose(question: str, compare_baseline: bool = False):
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)
        final_state = None
        # stream_mode="values"：每一步拿到的都是「目前為止累積的完整狀態」，
        # 不是預設 stream() 那種「這個節點自己更新了哪些欄位」的 delta。
        # 用預設模式的話，最後一步如果剛好是只回傳 generation 的節點（例如 mark_low_confidence），
        # documents 這類其他欄位就會憑空消失，變成「有答案但看不到來源文件」的誤導畫面。
        for state in app.stream({"question": question}, stream_mode="values"):
            final_state = state
            print()
        if final_state:
            print(f"\n[Adaptive Answer]\n{final_state.get('generation', '(no generation)')}")
            if final_state.get("rewritten_question"):
                print(f"[Rewritten]\n{final_state.get('rewritten_question')}")
            docs = final_state.get("documents") or []
            if docs:
                sources = ", ".join(sorted({d.metadata.get("source", "unknown") for d in docs}))
                print(f"[Adaptive Sources] {len(docs)} docs -> {sources}")

        if compare_baseline:
            print("\n--- Baseline (Naive RAG，無 hybrid/rerank/web_search) ---")
            baseline_state = None
            for state in baseline_app.stream({"question": question}, stream_mode="values"):
                baseline_state = state
            if baseline_state:
                print(f"[Baseline Answer]\n{baseline_state.get('generation', '(no generation)')}")
                baseline_docs = baseline_state.get("documents") or []
                print(f"[Baseline Sources] {len(baseline_docs)} docs（naive RAG 沒有 web_search fallback，"
                      f"文件庫沒有的問題通常會直接編答案或說不知道，可以對照 adaptive 版本的差異）")

        return final_state

    # ---- 評估：三欄比較 no-retrieval / naive RAG / adaptive RAG ----
    # keyword + RAGAS 共用同一次 pipeline 執行結果（run_full_evaluation()），
    # 每題每條 pipeline 只跑一次，不會被 keyword eval 跟 RAGAS 各自重跑一次。
    #
    # 三欄各自回答不同問題：
    #   no-retrieval → naive RAG：檢索這件事本身有沒有用（純 LLM vs 隨便查一下文件）
    #   naive RAG    → adaptive RAG：檢索做得好不好有沒有差（hybrid/rerank/門檻過濾/retry/web_search）
    try:
        from gold_standard import GOLD_STANDARD
    except ImportError:
        print("WARNING: gold_standard.py not found, skipping evaluation.")
    else:
        try:
            print("\n[Eval] Running No-Retrieval vs Naive RAG vs Adaptive RAG"
                  "（每題每條 pipeline 只執行一次，keyword + RAGAS 共用結果）...")
            no_retrieval_eval = run_full_evaluation(no_retrieval_app, GOLD_STANDARD)
            baseline_eval      = run_full_evaluation(baseline_app, GOLD_STANDARD)
            adaptive_eval      = run_full_evaluation(app, GOLD_STANDARD)

            # 先印出每題的實際回答內容，再印分數摘要——想知道「答了什麼」看這段，
            # 想知道「答得好不好」看下面的 keyword/RAGAS 對比表。
            print_answers({
                "No-Retrieval": no_retrieval_eval["results"],
                "Naive RAG":    baseline_eval["results"],
                "Adaptive RAG": adaptive_eval["results"],
            })

            keyword_results = {
                "No-Retrieval": no_retrieval_eval["keyword"],
                "Naive RAG":    baseline_eval["keyword"],
                "Adaptive RAG": adaptive_eval["keyword"],
            }
            for name, r in keyword_results.items():
                print(f"{name:<14} answer quality : {r['answer_quality']:.1%} ({r['answer_correct']}/{r['total']})")
            print_comparison(keyword_results)
        except Exception as e:
            # gold_standard.py 如果有語法錯誤（SyntaxError）或其他問題，也不該讓整個 main.py 崩潰，
            # 至少要跳過這段、讓後面還有機會執行（雖然 RAGAS 這次也拿不到 results 了，一起跳過）。
            print(f"WARNING: 評估失敗 ({type(e).__name__}: {e})，跳過。")
        else:
            # ---- RAGAS：直接用上面已經跑好的結果算分數，不重新呼叫 pipeline ---- # TODO: for DEMO
            # 確認有 pip install ragas datasets，且 gold_standard.py 的 ground_truth 欄位要填實際內容 # TODO: for DEMO
            try:
                print("\n[RAGAS] Scoring already-collected results（不重跑 pipeline）...")
                ragas_results = {
                    "No-Retrieval": evaluate_with_ragas_from_results(no_retrieval_eval["results"]),
                    "Naive RAG":    evaluate_with_ragas_from_results(baseline_eval["results"]),
                    "Adaptive RAG": evaluate_with_ragas_from_results(adaptive_eval["results"]),
                }
                print_ragas_comparison(ragas_results)
            except ImportError as e:
                print(f"\n[RAGAS] SKIPPED: 套件未安裝 ({e})。請執行 `pip install ragas datasets` 後再跑一次。")
            except Exception as e:
                print(f"\n[RAGAS] ERROR: {e}")
