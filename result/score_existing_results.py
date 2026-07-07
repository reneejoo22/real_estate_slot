"""
LLM을 다시 호출하지 않고, 이미 쌓인 eval_results.csv만 읽어서
슬롯별 Accuracy / Precision / Recall / F1(macro)과 전체 정답률을 계산하는 스크립트.

evaluate_pipeline.py가 API 요청 제한(429) 등으로 중간에 멈췄을 때,
지금까지 쌓인 결과만으로 채점하고 싶을 때 사용한다.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

EXACT_SLOT_NAMES = ["지역", "거래유형", "주거유형", "층조건", "근접시설", "가격_최소", "가격_최대", "면적_최소"]
SIMILARITY_SLOT_NAMES = ["기타"]

RESULT_CSV = "result/eval_results.csv"
SUMMARY_CSV = "result/eval_summary.csv"


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in EXACT_SLOT_NAMES:
        true_col = f"true_{name}"
        pred_col = f"pred_{name}"
        if true_col not in results_df.columns or pred_col not in results_df.columns:
            continue

        y_true = results_df[true_col].fillna("NONE").astype(str)
        y_pred = results_df[pred_col].fillna("NONE").astype(str)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        rows.append({
            "slot": name,
            "accuracy": round(acc, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "n_samples": len(results_df),
        })

    # 기타 슬롯: evaluate_pipeline.py가 이미 계산해둔 similarity_기타 / match_기타 컬럼을 그대로 집계
    # (여기서는 임베딩 모델을 다시 로드하지 않아 빠르고 가벼움)
    for name in SIMILARITY_SLOT_NAMES:
        match_col = f"match_{name}"
        sim_col = f"similarity_{name}"
        if match_col not in results_df.columns:
            continue
        match_rate = results_df[match_col].astype(str).str.lower().eq("true").mean()
        avg_similarity = results_df[sim_col].mean() if sim_col in results_df.columns else None
        rows.append({
            "slot": f"{name} (유사도 기반)",
            "accuracy": round(match_rate, 4),
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": round(avg_similarity, 4) if avg_similarity is not None else None,  # f1 자리에 평균 유사도
            "n_samples": len(results_df),
        })

    if "exact_match_all_slots" in results_df.columns:
        overall_acc = results_df["exact_match_all_slots"].astype(str).str.lower().eq("true").mean()
        rows.append({
            "slot": "ALL_SLOTS_EXACT_MATCH (엄격: 모든 슬롯 일치해야 정답)",
            "accuracy": round(overall_acc, 4),
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "n_samples": len(results_df),
        })

    # 더 관대한 지표: 슬롯별 accuracy의 평균 (한두 개 슬롯 틀렸다고 전부 0점 처리하지 않음)
    per_slot_acc = [r["accuracy"] for r in rows if r["accuracy"] is not None and "ALL_SLOTS" not in r["slot"]]
    if per_slot_acc:
        rows.append({
            "slot": "AVERAGE_PER_SLOT_ACCURACY (관대: 슬롯별 정확도/매치율 평균)",
            "accuracy": round(sum(per_slot_acc) / len(per_slot_acc), 4),
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "n_samples": len(results_df),
        })

    return pd.DataFrame(rows)


def main():
    results_df = pd.read_csv(RESULT_CSV)
    print(f"현재까지 채점된 샘플 수: {len(results_df)}건\n")

    metrics_df = compute_metrics(results_df)

    print("=" * 60)
    print("슬롯별 평가 결과 (현재까지 쌓인 데이터 기준)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {SUMMARY_CSV}")


if __name__ == "__main__":
    main()