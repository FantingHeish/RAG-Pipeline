# evaluation.py
# 負責：run()、evaluate_pipeline()、RAGAS 評估、A/B 對比、save_scores_log()
#
# [重要] run_all() / run_full_evaluation()：keyword eval 跟 RAGAS 共用同一次 pipeline 執行結果，
# 不要分別各自呼叫一次 —— 8 題 gold_standard、2 條 pipeline，如果 keyword eval 跟 RAGAS
# 各自重新呼叫一次，就是 32 次完整 pipeline 執行（每次都要打好幾次 LLM API）；
# 共用一次執行結果可以砍半到 16 次。main.py 用的是 run_full_evaluation()，不是舊版分開呼叫的寫法。

import json

# ============================================================
# run() / run_all()
# ============================================================

def run(app, question: str) -> dict:
    """跑一個問題，回傳含 generation / documents 的 dict。documents 拿來餵給 RAGAS 當 contexts 用"""
    inputs = {"question": question}
    # 用 invoke() 拿完整的最終狀態，不要用 stream() 的最後一步——
    # stream() 預設只回傳每個節點「自己更新了哪些欄位」，如果最後一個節點沒有動到 documents
    # （例如 pipeline.py 的 mark_low_confidence 只回傳 generation），
    # 這裡的 documents 就會被誤判成空的，連帶讓 RAGAS 的 context_precision/context_recall 算錯。
    final_state = app.invoke(inputs)
    if not final_state:
        return {"generation": "", "documents": []}
    return {
        "generation": final_state.get("generation", ""),
        "documents":  final_state.get("documents", []),
    }


def run_all(app, test_cases: list) -> list:
    """
    對每一題只呼叫一次 app（透過 run()），回傳完整結果 list。
    keyword eval（score_keyword_eval）跟 RAGAS（evaluate_with_ragas_from_results）
    都從這份結果算分數，不用各自重新執行一次 pipeline。
    """
    results = []
    for i, case in enumerate(test_cases):
        if not isinstance(case, dict) or "question" not in case:
            # gold_standard.py 裡如果有一筆資料格式不對（例如不小心放了一個字串進 list，
            # 不是預期的 {"question": ..., ...} 字典），跳過這筆並印警告，
            # 不要讓整組評估因為一筆資料壞掉就整個中斷。
            print(f"  WARNING: test_cases[{i}] 不是預期的格式（應該是包含 'question' 的 dict，"
                  f"實際拿到 {type(case).__name__}: {case!r}），已跳過這筆。")
            continue
        output = run(app, case["question"])
        results.append({
            "question":        case["question"],
            "generation":      output.get("generation", ""),
            "documents":       output.get("documents", []),
            "answer_keywords": case.get("answer_keywords", []),
            "ground_truth":    case.get("ground_truth", ""),
        })
    return results


# ============================================================
# 關鍵字版評估：從 run_all() 的結果算分數，不重新呼叫 pipeline
# ============================================================

def score_keyword_eval(results: list) -> dict:
    answer_correct = 0
    details = []
    for r in results:
        gen       = r["generation"].lower()
        answer_ok = any(kw.lower() in gen for kw in r["answer_keywords"])
        answer_correct += int(answer_ok)
        details.append({
            "question":   r["question"],
            "answer_ok":  answer_ok,
            "generation": r["generation"][:200],
        })

    total = len(results)
    return {
        "total":          total,
        "answer_quality": answer_correct / total if total else 0.0,
        "answer_correct": answer_correct,
        "details":        details,
    }


def evaluate_pipeline(app, test_cases: list) -> dict:
    """
    舊介面，保留給只想單獨跑 keyword eval 的情境用（例如只想看關鍵字分數，不需要 RAGAS）。
    內部呼叫 run_all()，如果你同時也需要 RAGAS，改用 run_full_evaluation()，
    不要分別呼叫 evaluate_pipeline() + evaluate_with_ragas()，那樣同一組問題會被重跑一次。
    """
    return score_keyword_eval(run_all(app, test_cases))


# ============================================================
# A/B(/C...) test
# ============================================================
# 對比多條 pipeline 的 keyword 分數，欄位數不固定（2 欄 A/B 對比、3 欄 A/B/C 對比都可以）
def print_comparison(results: dict):
    """
    results：{"顯示名稱": keyword_score_dict, ...}，順序就是印出來的欄位順序。
    例如 print_comparison({"No-Retrieval": r1, "Naive RAG": r2, "Adaptive RAG": r3})
    """
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON (keyword-based)")
    print("=" * 60)
    names = list(results.keys())
    header = f"{'Metric':<16}" + "".join(f"{name:>16}" for name in names)
    print(header)
    print("-" * len(header))
    values = [results[name]["answer_quality"] for name in names]
    row = f"{'answer_quality':<16}" + "".join(f"{v:>15.1%} " for v in values)
    print(row)
    if len(values) >= 2:
        delta = values[-1] - values[0]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"\n{names[0]} → {names[-1]}: {arrow} {abs(delta):.1%}")
    print("=" * 60)


# ============================================================
# RAGAS 評估
# ============================================================
'''
需要：pip install ragas datasets

RAGAS 的四個 metric：
  faithfulness : 生成答案有沒有超出 context 亂編（對應 hallucination grader 想抓的問題）
  answer_relevancy : 答案有沒有真的回應問題（對應 answer grader 想抓的問題）
  context_precision : 檢索回來的文件裡，有用的比例高不高（對應 Layer3 想過濾的問題）
  context_recall : 檢索回來的文件，涵蓋了多少「應該要有」的資訊（需要 ground_truth）
'''

def build_ragas_dataset_from_results(results: list):
    """從 run_all() 的結果組 RAGAS Dataset，不重新呼叫 pipeline。"""
    from datasets import Dataset

    rows = []
    for r in results:
        contexts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in r["documents"]
        ]
        rows.append({
            "question":     r["question"],
            "answer":       r["generation"],
            "contexts":     contexts if contexts else [""],
            "ground_truth": r["ground_truth"],
        })
    return Dataset.from_list(rows)


def build_ragas_dataset(app, test_cases: list):
    """舊介面（會重新呼叫一次 pipeline）。已經有 run_all() 結果的話用 build_ragas_dataset_from_results()。"""
    return build_ragas_dataset_from_results(run_all(app, test_cases))


def get_ragas_embeddings():
    """
    ragas 0.4.3 有個內部不一致的問題（親自用 raise_exceptions=True 跑出完整 traceback 確認過）：
    `answer_relevancy` 這個指標的原始碼（ragas/metrics/_answer_relevance.py）還在呼叫
    舊版介面的 `self.embeddings.embed_query(...)`，但如果你傳新版原生的
    `ragas.embeddings.OpenAIEmbeddings`（只有 `embed_text()`，沒有 `embed_query()`），
    就會直接 AttributeError——這是 ragas 自己新舊版本交接沒做乾淨，不是我們這邊的問題。
    解法：優先用舊版的 LangchainEmbeddingsWrapper，它的存在目的就是提供 embed_query() 這個
    相容介面，剛好是 answer_relevancy 內部需要的方法名稱。只有這個 wrapper 不存在時，
    才退回用新版原生寫法（雖然那條路線在 answer_relevancy 上會失敗，但至少其他指標還能算）。
    """
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai.embeddings import OpenAIEmbeddings as LCEmbeddings
        return LangchainEmbeddingsWrapper(LCEmbeddings())
    except ImportError:
        pass

    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        return RagasOpenAIEmbeddings(client=AsyncOpenAI(), model="text-embedding-3-small")
    except ImportError:
        return None  # 都抓不到的話，讓 ragas 用它自己的預設值（原本的行為）


def evaluate_with_ragas_from_results(results: list, debug: bool = False):
    """
    對已經跑好的結果算 RAGAS 分數，不重新呼叫 pipeline。
    debug=True：加上 raise_exceptions=True，遇到錯誤會丟出完整 traceback，而不是
    ragas 預設那種吞掉細節、只印一行「Exception raised in Job[N]: ...」的摘要——
    用來實際定位問題出在 ragas 原始碼的哪一行，而不是靠猜的。
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = build_ragas_dataset_from_results(results)
    kwargs = {"metrics": [faithfulness, answer_relevancy, context_precision, context_recall]}

    embeddings = get_ragas_embeddings()
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    if debug:
        kwargs["raise_exceptions"] = True

    return evaluate(dataset, **kwargs)


def evaluate_with_ragas(app, test_cases: list):
    """
    舊介面，保留給只想單獨跑 RAGAS 的情境用（例如 finetune_reranker.py 比較新舊 reranker，
    只呼叫一次、沒有跟 keyword eval 重複執行的問題）。
    main.py 這種同時要 keyword + RAGAS 的情境，改用 run_full_evaluation()。
    """
    return evaluate_with_ragas_from_results(run_all(app, test_cases))


def run_full_evaluation(app, test_cases: list) -> dict:
    """
    只呼叫一次 pipeline（run_all），同時準備好 keyword 分數跟 RAGAS 需要的原始結果。
    回傳 {"keyword": {...}, "results": [...]}——"results" 直接丟給
    evaluate_with_ragas_from_results() 算 RAGAS，不會再重新執行一次 pipeline。
    """
    results = run_all(app, test_cases)
    return {"keyword": score_keyword_eval(results), "results": results}


def print_answers(results_by_pipeline: dict):
    """
    印出每一題在各條 pipeline 下實際生成的答案內容（不是分數），
    直接複用 run_all()/run_full_evaluation() 已經跑好的結果，不會多呼叫一次 pipeline。

    results_by_pipeline：{"顯示名稱": results_list, ...}，results_list 是
    run_all(app, test_cases) 或 run_full_evaluation(app, test_cases)["results"] 的回傳值。
    """
    names = list(results_by_pipeline.keys())
    if not names:
        return

    # 用第一條 pipeline 的題目順序當基準；用「問題文字」對應，不是用 index——
    # run_all() 遇到格式錯的資料會跳過，不同 pipeline 的 results 筆數理論上該一致，
    # 但用文字比對比較保險，不會因為某條 pipeline 少一筆就整個對錯位。
    questions_seen = []
    for r in results_by_pipeline[names[0]]:
        if r["question"] not in questions_seen:
            questions_seen.append(r["question"])

    lookup = {
        name: {r["question"]: r["generation"] for r in results}
        for name, results in results_by_pipeline.items()
    }

    print("\n" + "=" * 70)
    print("詳細回答內容（每題各條 pipeline 的實際答案）")
    print("=" * 70)
    for i, q in enumerate(questions_seen, 1):
        print(f"\n[{i}] Question: {q}")
        for name in names:
            ans = lookup[name].get(q, "(這條 pipeline 沒有這題的結果)")
            print(f"  [{name}]\n  {ans}\n")
    print("=" * 70)


def extract_metric_mean(result, metric_name: str) -> float:
    """
    不同版本的 ragas，result[metric_name] 回傳的型態不一樣：
    - 舊版：直接是一個算好平均值的 float
    - 新版（例如 0.4.x）：是一整串每一筆資料的分數 list（可能包含 NaN，代表那筆算失敗了）
    這裡統一轉成一個平均值的 float，兩種版本都能用。
    """
    import math

    value = result[metric_name]
    if isinstance(value, (int, float)):
        return float(value)

    # list/array 的情況：濾掉 NaN 或 None 再取平均，避免一筆失敗就讓整體變成 NaN
    valid = [v for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        print(f"  WARNING: metric '{metric_name}' 這批全部算失敗（可能是 ragas 版本相容性問題），當作 0 處理")
        return 0.0
    return sum(valid) / len(valid)


# 印出多條 pipeline 的 RAGAS 分數對比表，欄位數不固定
def print_ragas_comparison(results: dict):
    """results：{"顯示名稱": ragas_result, ...}，順序就是印出來的欄位順序。"""
    print("\n" + "=" * 62)
    print("RAGAS COMPARISON")
    print("=" * 62)
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    names = list(results.keys())
    header = f"{'Metric':<20}" + "".join(f"{name:>16}" for name in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        values = [extract_metric_mean(results[name], m) for name in names]
        row = f"{m:<20}" + "".join(f"{v:>15.3f} " for v in values)
        print(row)
    print("=" * 62)


# run evaluate_with_ragas() & print_ragas_comparison()
# A/B Test baseline_result v.s. adaptive_result
# 舊介面，保留給只想單獨跑 RAGAS（不需要 keyword eval）的情境用，會各自重新呼叫一次 pipeline。
def compare_with_ragas(baseline_app, adaptive_app, test_cases: list):
    print("Running RAGAS on baseline (naive RAG)...")
    baseline_result = evaluate_with_ragas(baseline_app, test_cases)

    print("Running RAGAS on adaptive RAG...")
    adaptive_result = evaluate_with_ragas(adaptive_app, test_cases)

    print_ragas_comparison({"Baseline (naive RAG)": baseline_result, "Adaptive RAG": adaptive_result})
    return baseline_result, adaptive_result

# 把評分結果記錄到JSON 後續作為  gold standard ＆ debug
# state：pipeline 執行完的完整最終狀態（例如 app.invoke({...}) 的回傳值），
# 不是 stream() 預設模式那種 {node_name: node_delta} 的巢狀結構。
def save_scores_log(state: dict, filepath: str = "scores_log.json"):
    logs = state.get("scores_log", [])
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Scores log saved to {filepath} ({len(logs)} entries)")
