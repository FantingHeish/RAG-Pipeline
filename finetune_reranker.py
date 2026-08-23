# finetune_reranker.py
#
# 業界常見的離線、批次微調流程：
#   1. 讀 offline_evaluate.py 累積下來的訓練資料（config.TRAINING_DATA_PATH）
#   2. 資料量沒到 config.MIN_NEW_TRAINING_SAMPLES 就不訓練（資料太少訓出來的模型不可靠）
#   3. 用 sentence-transformers 的 CrossEncoder 在現有 reranker 基礎上繼續訓練幾個 epoch
#   4. 存成新 checkpoint（config.RERANKER_CHECKPOINT_DIR/reranker-YYYYMMDD_HHMMSS/）
#   5. 在 gold_standard.py 這個 held-out 驗證集上，用 RAGAS 的
#      context_precision / context_recall 比較「舊 reranker vs 新 reranker」
#   6. 新的沒有比較好就不建議上線——不是自動部署，是印出建議，你自己決定要不要
#      把 config.py 的 RERANKER_MODEL_NAME 改指向新 checkpoint
#
# 什麼時候該跑這支：累積到 MIN_NEW_TRAINING_SAMPLES 筆新資料，或排定每週/每月固定跑一次
#（例如用 cron：0 3 * * 0 表示每週日凌晨三點跑一次 `python offline_evaluate.py && python finetune_reranker.py`）。
#
# 用法：
#   python offline_evaluate.py   # 先確保有累積訓練資料
#   python finetune_reranker.py  # 再跑這支做微調 + 比較
#
# 注意：sentence-transformers 的 CrossEncoder 微調 API 在不同版本之間有調整過
# （新版一些改成 Trainer-based API）。這裡用的是長期以來最常見、文件最齊全的
# `CrossEncoder.fit(train_dataloader, ...)` 寫法；如果你裝的版本這個 API 已經被換掉，
# 錯誤訊息裡通常會提示新的用法，去查 sentence-transformers 官方文件的 Cross-Encoder 訓練章節即可。

import json
import os
from datetime import datetime
from typing import List, Tuple

from config import (
    TRAINING_DATA_PATH, RERANKER_CHECKPOINT_DIR, RERANKER_MODEL_NAME,
    MIN_NEW_TRAINING_SAMPLES, COLLECTION_NAME,
)


def _load_training_data(path: str) -> List[Tuple[str, str, float]]:
    if not os.path.exists(path):
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                samples.append((record["question"], record["document"], float(record["label"])))
            except Exception as e:
                print(f"  WARNING: 跳過解析失敗的一行 ({e})")
    return samples


def finetune(epochs: int = 2, batch_size: int = 16, samples: List[Tuple[str, str, float]] = None) -> str:
    """
    訓練新 checkpoint，回傳存檔路徑。資料不夠會直接 raise，不會硬訓一個不可靠的模型。

    samples：正常用法不傳，會自動從 TRAINING_DATA_PATH 讀取真實累積的訓練資料。
        傳入時（例如 --smoke-test）直接用給定的假資料，略過檔案讀取跟
        MIN_NEW_TRAINING_SAMPLES 筆數門檻——這是刻意的，smoke test 的目的是快速驗證
        「訓練 -> 存檔 -> 讀回」這條路徑通不通，不是真的要訓出一個能用的模型。
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    if samples is None:
        samples = _load_training_data(TRAINING_DATA_PATH)
        print(f"[finetune_reranker] 讀到 {len(samples)} 筆訓練樣本（{TRAINING_DATA_PATH}）")

        if len(samples) < MIN_NEW_TRAINING_SAMPLES:
            raise RuntimeError(
                f"訓練樣本只有 {len(samples)} 筆，低於 MIN_NEW_TRAINING_SAMPLES={MIN_NEW_TRAINING_SAMPLES}。"
                f"先跑 python offline_evaluate.py 多累積一些資料再回來訓練，避免資料太少訓出不可靠的模型。"
            )
    else:
        print(f"[finetune_reranker] 使用外部傳入的 {len(samples)} 筆樣本（略過 MIN_NEW_TRAINING_SAMPLES 門檻）")

    train_examples = [InputExample(texts=[q, doc], label=label) for q, doc, label in samples]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print(f"[finetune_reranker] 從 '{RERANKER_MODEL_NAME}' 繼續訓練 {epochs} epochs ...")
    model = CrossEncoder(RERANKER_MODEL_NAME, num_labels=1)

    checkpoint_name = f"reranker-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(RERANKER_CHECKPOINT_DIR, checkpoint_name)
    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=max(10, len(train_dataloader) // 10),
        output_path=output_path,
    )
    # OPTIMIZE/ 新版 sentence-transformers 的 CrossEncoder.fit() 底層改用 Trainer-based API，
    # 觀察到它在 output_path 留下的檔案有時候不完整（缺 model_type 等欄位，會導致
    # HuggingFaceCrossEncoder/AutoConfig.from_pretrained() 讀不回來）。
    # 明確再呼叫一次 save_pretrained()，強制用標準 HF 格式把完整檔案（含正確的 config.json）寫出來，
    # 不依賴 fit() 內部的存檔行為是否完整。
    model.save_pretrained(output_path)
    print(f"[finetune_reranker] 訓練完成，新 checkpoint 存在: {output_path}")
    return output_path


def compare_old_vs_new(new_checkpoint_path: str, gold_standard: list = None):
    """
    在 gold_standard.py 上，用 RAGAS context_precision/context_recall 比較舊 vs 新 reranker。

    gold_standard：正常用法不傳，會用完整的 GOLD_STANDARD（19 題）。
        傳入時（例如 --smoke-test 只傳 1 題）用給定的清單——一樣是刻意的，
        目的是快速驗證這整條路徑（載入 store -> 組 pipeline -> RAGAS -> 印結論）通不通，
        不是真的要拿這個結果來判斷新 checkpoint 好不好。
    """
    from langchain_openai.embeddings import OpenAIEmbeddings
    from vectorstore import load_existing_store, build_retriever, build_reranker_model
    from pipeline import build_pipeline
    from evaluation import evaluate_with_ragas, print_ragas_comparison, extract_metric_mean

    if gold_standard is None:
        from gold_standard import GOLD_STANDARD as gold_standard

    embeddings = OpenAIEmbeddings()
    store = load_existing_store(embeddings, collection_name=COLLECTION_NAME)
    retriever = build_retriever(store)

    print(f"\n[finetune_reranker] 用舊 reranker（{RERANKER_MODEL_NAME}）跑 RAGAS ...")
    old_reranker = build_reranker_model(RERANKER_MODEL_NAME)
    old_app = build_pipeline(retriever, embeddings=embeddings, reranker_model=old_reranker)
    old_result = evaluate_with_ragas(old_app, gold_standard)

    print(f"\n[finetune_reranker] 用新 reranker（{new_checkpoint_path}）跑 RAGAS ...")
    new_reranker = build_reranker_model(new_checkpoint_path)
    new_app = build_pipeline(retriever, embeddings=embeddings, reranker_model=new_reranker)
    new_result = evaluate_with_ragas(new_app, gold_standard)

    print_ragas_comparison({"舊 reranker": old_result, "新 checkpoint": new_result})

    # 用 context_precision + context_recall 的加總判斷新的有沒有比較好
    # extract_metric_mean() 相容新舊版 ragas（新版 result[metric] 可能是 list，不是 float）
    old_score = extract_metric_mean(old_result, "context_precision") + extract_metric_mean(old_result, "context_recall")
    new_score = extract_metric_mean(new_result, "context_precision") + extract_metric_mean(new_result, "context_recall")

    print("\n" + "=" * 62)
    if new_score > old_score:
        print(f"[結論] 新 checkpoint 檢索品質更好（{new_score:.3f} > {old_score:.3f}）。")
        print(f"       建議把 .env 的 RERANKER_MODEL_NAME 改成：\n       {new_checkpoint_path}")
    else:
        print(f"[結論] 新 checkpoint 沒有比較好（{new_score:.3f} <= {old_score:.3f}），不建議上線。")
        print(f"       繼續用目前的 {RERANKER_MODEL_NAME}，累積更多/更多樣的訓練資料後再試一次。")
    print("=" * 62)


def _smoke_test_samples() -> List[Tuple[str, str, float]]:
    """給 --smoke-test 用的極少量假訓練資料（2 筆），用來快速驗證訓練+存檔路徑，不是真的訓練。"""
    return [
        ("台灣的健康台灣深耕計畫預算是多少？", "健康台灣深耕計畫總經費約新台幣489億元。", 1.0),
        ("台灣的健康台灣深耕計畫預算是多少？", "貓咪喜歡在陽光下睡覺。", 0.0),
    ]


if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv:
        # 用假資料 + 1 epoch + 只取 gold_standard 第 1 題，把 finetune() 跟
        # compare_old_vs_new() 這兩個「正式函式」完整跑一遍，幾十秒到一兩分鐘內
        # 就能確認訓練/存檔/讀回/RAGAS 整條路徑通不通，不用等完整流程 30-40 分鐘。
        print("[finetune_reranker] === SMOKE TEST 模式：用假資料快速驗證整條路徑 ===\n")
        from gold_standard import GOLD_STANDARD

        smoke_path = finetune(epochs=1, batch_size=2, samples=_smoke_test_samples())
        compare_old_vs_new(smoke_path, gold_standard=GOLD_STANDARD[:1])

        print("\n[finetune_reranker] ✅ SMOKE TEST 完成：整條路徑（訓練 -> 存檔 -> 讀回 -> RAGAS -> 印結論）都跑得通。")
        print("[finetune_reranker] 可以放心跑正式流程：python finetune_reranker.py")
    else:
        new_path = finetune()
        compare_old_vs_new(new_path)
