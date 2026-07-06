"""
평가 파이프라인

test_cases.csv (utterance + true_슬롯들)를 하나씩 읽어서:
  1) 슬롯 추출 (langchain_pipeline.extract_slots) → 예측 슬롯
  2) listings_augmented.csv에서 pandas 필터링 → 상위 3개 매물
  3) 결과 포맷팅 → 최종 답변 텍스트
  4) 예측 슬롯 vs 정답 슬롯 비교 → 슬롯별 일치 여부, 전체 정확도
  5) 모든 항목(입력/예측/정답/일치여부/상위3개/최종답변)을 한 행으로 CSV에 누적 저장
  6) 전체 테스트가 끝나면 슬롯별 Accuracy, Precision/Recall/F1(macro) 계산 후
     콘솔 출력 + eval_summary.csv로 저장
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from langchain_pipeline import extract_slots, search_listings, format_results, SlotFilling

SLOT_NAMES = ["지역", "거래유형", "주거유형", "층조건", "근접시설", "가격_최소", "가격_최대", "면적_최소"]

RESULT_CSV = "eval_results.csv"
SUMMARY_CSV = "eval_summary.csv"


def normalize(value):
    """None/NaN을 비교 가능한 하나의 값으로 통일."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def slots_to_dict(slots: SlotFilling) -> dict:
    return {name: normalize(getattr(slots, name)) for name in SLOT_NAMES}


def evaluate_single(row: pd.Series, listings_df: pd.DataFrame) -> dict:
    utterance = row["utterance"]

    # 1) 슬롯 추출 (예측)
    predicted = extract_slots(utterance)
    pred_dict = slots_to_dict(predicted)

    # 2) 정답 슬롯
    true_dict = {name: normalize(row.get(f"true_{name}")) for name in SLOT_NAMES}

    # 3) pandas 필터링 → 상위 3개
    top3_df = search_listings(listings_df, predicted).head(3)

    def _price_text(r):
        if r.get("거래유형") == "전세":
            return f"전세 {r.get('보증금', 0)}만원"
        return f"월세 {r.get('월세', 0)}만원"

    top3_summary = "; ".join(
        f"{r['지역']} {_price_text(r)}" for _, r in top3_df.iterrows()
    ) if not top3_df.empty else "결과 없음"

    # 4) 최종 답변 텍스트
    final_answer = format_results(top3_df)

    # 5) 슬롯별 일치 여부
    slot_match = {f"match_{name}": (pred_dict[name] == true_dict[name]) for name in SLOT_NAMES}
    exact_match_all = all(slot_match.values())

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": utterance,
    }
    for name in SLOT_NAMES:
        record[f"pred_{name}"] = pred_dict[name]
    for name in SLOT_NAMES:
        record[f"true_{name}"] = true_dict[name]
    record.update(slot_match)
    record["exact_match_all_slots"] = exact_match_all
    record["top3_listings"] = top3_summary
    record["final_answer"] = final_answer.replace("\n", " | ")

    return record


def append_to_csv(record: dict, path: str = RESULT_CSV) -> None:
    """한 건씩 CSV에 누적 저장 (파일 없으면 헤더 포함 생성, 있으면 이어붙임)."""
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """슬롯별 Accuracy / Precision / Recall / F1(macro)을 계산."""
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

    overall_exact_match_acc = results_df["exact_match_all_slots"].mean()
    rows.append({
        "slot": "ALL_SLOTS_EXACT_MATCH (엄격: 8개 슬롯 전부 일치해야 정답)",
        "accuracy": round(overall_exact_match_acc, 4),
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "n_samples": len(results_df),
    })

    avg_per_slot_acc = sum(r["accuracy"] for r in rows) / len(rows)
    rows.append({
        "slot": "AVERAGE_PER_SLOT_ACCURACY (관대: 슬롯별 정확도 평균)",
        "accuracy": round(avg_per_slot_acc, 4),
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "n_samples": len(results_df),
    })

    return pd.DataFrame(rows)


def run_evaluation(test_cases_path: str = "test_cases.csv",
                    listings_path: str = "synthetic_listings.csv",
                    max_samples: int = None,
                    sleep_seconds: float = 15.0) -> None:
    test_df = pd.read_csv(test_cases_path)
    listings_df = pd.read_csv(listings_path)

    if max_samples is not None:
        test_df = test_df.head(max_samples)
        print(f"[테스트 모드] 상위 {max_samples}개만 실행합니다.\n")

    # 기존 결과 파일이 있으면 이번 실행 전에 초기화하고 싶으면 아래 주석 해제
    # if os.path.exists(RESULT_CSV):
    #     os.remove(RESULT_CSV)

    all_records = []
    for i, row in test_df.iterrows():
        record = evaluate_single(row, listings_df)
        append_to_csv(record)
        all_records.append(record)
        print(f"[{i+1}/{len(test_df)}] 입력: {record['input']}")
        print(f"    예측: {[record[f'pred_{n}'] for n in SLOT_NAMES]}")
        print(f"    정답: {[record[f'true_{n}'] for n in SLOT_NAMES]}")
        print(f"    전체일치: {record['exact_match_all_slots']} | 상위3개: {record['top3_listings']}")
        print(f"    답변: {record['final_answer']}\n")

        # 무료 티어 분당/일일 요청 제한(429)을 피하기 위한 대기.
        # 마지막 건 처리 후에는 굳이 기다릴 필요 없으므로 스킵.
        is_last = (row.name == test_df.index[-1])
        if not is_last:
            time.sleep(sleep_seconds)

    results_df = pd.DataFrame(all_records)
    metrics_df = compute_metrics(results_df)

    print("=" * 60)
    print("슬롯별 평가 결과")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 개별 결과 -> {RESULT_CSV}")
    print(f"[저장 완료] 평가 요약 -> {SUMMARY_CSV}")


if __name__ == "__main__":
    # 테스트 삼아 10개만 먼저 돌려보고 싶으면 max_samples=10
    # 무료 티어 제한이 여유로워지면 max_samples=None으로 전체 실행
    run_evaluation(max_samples=15, sleep_seconds=15.0)