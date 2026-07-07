"""
LLM을 다시 호출하지 않고, 이미 쌓인 eval_results.csv만 읽어서
슬롯별 Accuracy / Precision / Recall / F1(macro)과 전체 정답률을 계산하는 스크립트.

evaluate_pipeline.py가 API 요청 제한(429) 등으로 중간에 멈췄을 때,
지금까지 쌓인 결과만으로 채점하고 싶을 때 사용한다.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SLOT_NAMES = ["지역", "거래유형", "주거유형", "층조건", "근접시설", "가격_최소", "가격_최대", "면적_최소"]

RESULT_CSV = "eval_results.csv"
SUMMARY_CSV = "eval_summary.csv"


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    # CSV의 match_ 컬럼을 믿지 않고 pred/true에서 직접 재계산.
    # 이렇게 하면 나중에 실행분과 이전 실행분의 컬럼 구조가 달라도 안전하게 채점된다.
    for name in SLOT_NAMES:
        results_df[f"match_{name}"] = (
            results_df[f"pred_{name}"].fillna("NONE").astype(str)
            == results_df[f"true_{name}"].fillna("NONE").astype(str)
        )
    results_df["exact_match_all_slots"] = results_df[
        [f"match_{n}" for n in SLOT_NAMES]
    ].all(axis=1)

    rows = []
    for name in SLOT_NAMES:
        y_true = results_df[f"true_{name}"].fillna("NONE").astype(str)
        y_pred = results_df[f"pred_{name}"].fillna("NONE").astype(str)

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

    overall_acc = results_df["exact_match_all_slots"].mean()
    rows.append({
        "slot": "ALL_SLOTS_EXACT_MATCH (엄격: 8개 슬롯 전부 일치해야 정답)",
        "accuracy": round(overall_acc, 4),
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "n_samples": len(results_df),
    })

    per_slot_acc = [r["accuracy"] for r in rows if "ALL_SLOTS" not in r["slot"]]
    rows.append({
        "slot": "AVERAGE_PER_SLOT_ACCURACY (관대: 슬롯별 정확도 평균)",
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