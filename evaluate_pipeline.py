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
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from langchain_pipeline import extract_slots, search_listings, format_results, SlotFilling, get_embedding_model

# 정확 일치로 채점하는 슬롯 (범주형 + 숫자형)
EXACT_SLOT_NAMES = ["지역", "거래유형", "주거유형", "층조건", "근접시설", "가격_최소", "가격_최대", "면적_최소"]
# 유사도로 채점하는 슬롯 (자유 텍스트라 정확 일치가 의미 없음)
SIMILARITY_SLOT_NAMES = ["기타"]
SLOT_NAMES = EXACT_SLOT_NAMES + SIMILARITY_SLOT_NAMES

# 기타 슬롯을 "맞았다"고 인정할 코사인 유사도 임계값.
# 둘 다 None(둘 다 언급 안 함)이면 유사도 계산 없이 자동으로 일치 처리.
SIMILARITY_MATCH_THRESHOLD = 0.5

RESULT_CSV = "result/eval_results.csv"
SUMMARY_CSV = "result/eval_summary.csv"


def normalize(value):
    """None/NaN을 비교 가능한 하나의 값으로 통일."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def slots_to_dict(slots: SlotFilling) -> dict:
    return {name: normalize(getattr(slots, name)) for name in SLOT_NAMES}


def compute_text_similarity(text_a, text_b) -> float:
    """
    두 자유 텍스트 사이의 코사인 유사도(0~1)를 로컬 임베딩 모델로 계산.
    둘 중 하나라도 None이면 다음 규칙으로 처리:
      - 둘 다 None -> 1.0 (둘 다 '언급 없음'이므로 완전 일치로 간주)
      - 한쪽만 None -> 0.0 (한쪽은 언급했는데 한쪽은 안 했으므로 불일치)
    """
    if text_a is None and text_b is None:
        return 1.0
    if text_a is None or text_b is None:
        return 0.0

    model = get_embedding_model()
    vecs = model.encode([text_a, text_b], normalize_embeddings=True)
    similarity = float(vecs[0] @ vecs[1])
    return similarity


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
    final_answer = format_results(top3_df, slots=predicted)

    # 5) 슬롯별 일치 여부
    #    - 정확 일치 슬롯: pred == true
    #    - 기타(유사도) 슬롯: 코사인 유사도가 임계값 이상이면 일치로 간주
    slot_match = {f"match_{name}": (pred_dict[name] == true_dict[name]) for name in EXACT_SLOT_NAMES}

    similarity_scores = {}
    for name in SIMILARITY_SLOT_NAMES:
        sim = compute_text_similarity(pred_dict[name], true_dict[name])
        similarity_scores[f"similarity_{name}"] = round(sim, 4)
        slot_match[f"match_{name}"] = (sim >= SIMILARITY_MATCH_THRESHOLD)

    exact_match_all = all(slot_match.values())

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": utterance,
    }
    for name in SLOT_NAMES:
        record[f"pred_{name}"] = pred_dict[name]
    for name in SLOT_NAMES:
        record[f"true_{name}"] = true_dict[name]
    record.update(similarity_scores)
    record.update(slot_match)
    record["exact_match_all_slots"] = exact_match_all
    record["top3_listings"] = top3_summary
    record["final_answer"] = final_answer.replace("\n", " | ")

    return record


def append_to_csv(record: dict, path: str = RESULT_CSV) -> None:
    """한 건씩 CSV에 누적 저장 (파일 없으면 헤더 포함 생성, 있으면 이어붙임)."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    정확 일치 슬롯: Accuracy / Precision / Recall / F1(macro)
    기타(유사도) 슬롯: 평균 코사인 유사도 + 임계값 기준 매치율
    """
    rows = []
    for name in EXACT_SLOT_NAMES:
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

    for name in SIMILARITY_SLOT_NAMES:
        match_col = f"match_{name}"
        sim_col = f"similarity_{name}"
        match_rate = results_df[match_col].astype(str).str.lower().eq("true").mean() \
            if match_col in results_df.columns else None
        avg_similarity = results_df[sim_col].mean() if sim_col in results_df.columns else None
        rows.append({
            "slot": f"{name} (유사도 기반, 임계값 {SIMILARITY_MATCH_THRESHOLD})",
            "accuracy": round(match_rate, 4) if match_rate is not None else None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": round(avg_similarity, 4) if avg_similarity is not None else None,  # f1 자리에 평균 유사도 기록
            "n_samples": len(results_df),
        })

    overall_exact_match_acc = results_df["exact_match_all_slots"].mean()
    rows.append({
        "slot": "ALL_SLOTS_EXACT_MATCH (엄격: 모든 슬롯 일치해야 정답, 기타는 유사도 임계값 기준)",
        "accuracy": round(overall_exact_match_acc, 4),
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "n_samples": len(results_df),
    })

    per_slot_acc = [r["accuracy"] for r in rows if r["accuracy"] is not None and "ALL_SLOTS" not in r["slot"]]
    avg_per_slot_acc = sum(per_slot_acc) / len(per_slot_acc) if per_slot_acc else None
    rows.append({
        "slot": "AVERAGE_PER_SLOT_ACCURACY (관대: 슬롯별 정확도 평균, 기타는 매치율로 포함)",
        "accuracy": round(avg_per_slot_acc, 4) if avg_per_slot_acc is not None else None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "n_samples": len(results_df),
    })

    return pd.DataFrame(rows)


def run_evaluation(test_cases_path: str = "db/test_cases.csv",
                    listings_path: str = "db/synthetic_listings.csv",
                    max_samples: int = None,
                    sleep_seconds: float = 15.0) -> None:
    os.makedirs("result", exist_ok=True)
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
    print("슬롯별 평가 결과 (이번 실행분만 기준)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 개별 결과 -> {RESULT_CSV}")
    print(f"[저장 완료] 평가 요약 -> {SUMMARY_CSV}")

    # ── 다 끝나고 자동으로 score_existing_results.py 실행 ──
    # (result/eval_results.csv에 누적된 전체 데이터 기준으로 다시 채점)
    run_score_existing_results()


def run_score_existing_results() -> None:
    """
    score_existing_results.py를 찾아서 자동으로 실행한다.
    같은 폴더(프로젝트 루트) 또는 result/ 폴더 어디에 있든 찾아서 실행.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "score_existing_results.py"),
        os.path.join(this_dir, "result", "score_existing_results.py"),
    ]
    script_path = next((p for p in candidates if os.path.exists(p)), None)

    if script_path is None:
        print("\n[경고] score_existing_results.py를 찾지 못해 자동 실행을 건너뜁니다. "
              "직접 실행해주세요.")
        return

    print("\n" + "=" * 60)
    print("전체 누적 결과 기준 재채점 (score_existing_results.py 자동 실행)")
    print("=" * 60)
    subprocess.run([sys.executable, script_path], cwd=this_dir)


if __name__ == "__main__":
    # 테스트 삼아 10개만 먼저 돌려보고 싶으면 max_samples=10
    # 무료 티어 제한이 여유로워지면 max_samples=None으로 전체 실행
    run_evaluation(max_samples=10, sleep_seconds=15.0)