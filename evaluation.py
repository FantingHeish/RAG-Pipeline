# evaluation.py
# 負責：run()、evaluate_pipeline()、print_comparison()、save_scores_log()

# ============================================================
# [Eval] evaluate_pipeline()
# ============================================================
'''
# 用途：量化「改前 vs 改後」的效果。
# 欄位說明：
#   question          → 測試問題
#   expected_sources  → 預期應該 route 到哪個 source（可多個）
#   answer_keywords   → 答案中應該出現的關鍵字（任一即可）
#
# 使用方式（兩個版本各自跑，手動對比數字）：
#   results = evaluate_pipeline(app, GOLD_STANDARD)
#   print(f"Route accuracy: {results['route_accuracy']:.1%}")
#   print(f"Answer quality: {results['answer_quality']:.1%}")
'''

import json


# 跑 graph
def run(app, question: str) -> dict:
    """跑一個問題，回傳含 selected_sources 和 generation 的 dict"""
    inputs = {"question": question}
    output = None

    for output in app.stream(inputs):
        pass
    if output is None:
        return {"selected_sources": [], "generation": ""}
    last_state = list(output.values())[-1]
    return {
        "selected_sources": last_state.get("selected_sources", []),
        "generation":       last_state.get("generation", ""),
    }


# --- route 是否正確 ---
def evaluate_pipeline(app, test_cases: list) -> dict:
    """
    跑所有 test case，計算：
      route_accuracy : selected_sources 和 expected_sources 是否有交集
      answer_quality : generation 是否包含任一 answer_keyword
    """
    route_correct  = 0
    answer_correct = 0
    details        = []

    for case in test_cases:
        output    = run(app, case["question"])
        selected  = output.get("selected_sources", [])
        gen       = output.get("generation", "").lower()
        expected  = case["expected_sources"]
        route_ok  = bool(set(selected) & set(expected))
        answer_ok = any(kw.lower() in gen for kw in case["answer_keywords"])
        route_correct  += int(route_ok)
        answer_correct += int(answer_ok)
        details.append({
            "question":         case["question"],
            "expected_sources": expected,
            "selected_sources": selected,
            "route_ok":         route_ok,
            "answer_ok":        answer_ok,
            "generation":       output.get("generation", "")[:200],
        })

    total = len(test_cases)
    return {
        "total":          total,
        "route_accuracy": route_correct  / total,
        "answer_quality": answer_correct / total,
        "route_correct":  route_correct,
        "answer_correct": answer_correct,
        "details":        details,
    }


def print_comparison(baseline: dict, new: dict):
    """印出改前 vs 改後的對比"""
    print("\n" + "="*60)
    print("EVALUATION COMPARISON: Baseline vs New")
    print("="*60)
    print(f"{'Metric':<25} {'Baseline':>10} {'New':>10} {'Delta':>10}")
    print("-"*60)
    for metric in ["route_accuracy", "answer_quality"]:
        b     = baseline[metric]
        n     = new[metric]
        delta = n - b
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"{metric:<25} {b:>9.1%} {n:>10.1%} {arrow} {abs(delta):>7.1%}")
    print("="*60)


def save_scores_log(state_output: dict, filepath: str = "scores_log.json"):
    """把三層評分記錄存到 JSON，方便後續建 gold standard"""
    logs = []
    for node_state in state_output.values():
        if "scores_log" in node_state:
            logs.extend(node_state["scores_log"])
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Scores log saved to {filepath} ({len(logs)} entries)")
