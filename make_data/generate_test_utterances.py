"""
synthetic_listings.csv(generate_synthetic_data.py 결과)를 바탕으로,
각 매물의 슬롯 값 중 일부만 무작위로 골라 자연어 발화를 만들고,
그 발화에 실제로 '언급된' 슬롯 값만 정답(ground truth)으로 저장하는 스크립트.

핵심 포인트: 매물의 전체 속성이 아니라, 발화에 실제 등장한 슬롯만 정답으로 삼는다.
언급 안 된 슬롯은 정답도 None이어야 "슬롯 추출기가 억지로 채워넣지 않는지"를
올바르게 평가할 수 있다.

기타(자유 텍스트) 슬롯은 매물의 실제 속성과 무관하게, 반려동물/주차/분위기 등
흔한 자유 조건 패턴을 별도 풀에서 뽑아 문장에 섞는다.
"""

import random
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "db")

random.seed(7)

TEMPLATES = [
    {"text": "{지역} {주거유형} {거래유형} {가격조건}", "required": ["지역", "주거유형", "거래유형"]},
    {"text": "{지역}에서 {거래유형} {가격조건} {주거유형} 찾아줘", "required": ["지역", "거래유형", "주거유형"]},
    {"text": "{주거유형} 구하는데 {지역} {근접시설}이면 좋겠어요", "required": ["주거유형", "지역", "근접시설"]},
    {"text": "{가격조건} {주거유형} {지역} 있나요", "required": ["주거유형", "지역"]},
    {"text": "{지역}로 집 좀 찾아줘", "required": ["지역"]},
    {"text": "{거래유형}로 {가격조건} 방 있어?", "required": ["거래유형"]},
    {"text": "{지역} {층조건} {주거유형} {거래유형}으로 구해요", "required": ["지역", "층조건", "주거유형", "거래유형"]},
    # ── 기타(자유 텍스트) 슬롯이 포함된 템플릿 ──
    {"text": "{지역} {주거유형} {거래유형} 구하는데 {기타절} 곳이면 좋겠어요", "required": ["지역", "주거유형", "거래유형", "기타"]},
    {"text": "{지역}에서 {기타절} 집 있나요", "required": ["지역", "기타"]},
    {"text": "{기타절} {주거유형} {가격조건} 찾아줘", "required": ["주거유형", "기타"]},
]

# 슬롯 이름 -> 그 슬롯이 실제로 들어갈 수 있는 템플릿 플레이스홀더 문자열
SLOT_TO_PLACEHOLDER = {
    "층조건": "{층조건}",
    "근접시설": "{근접시설}",
    "가격_최대": "{가격조건}",
}

PRICE_PHRASES_MAX = ["{}만원 이하로", "{}만원 안쪽으로", "{}만원 넘지 않게"]

# ── 기타(자유 텍스트) 슬롯 테스트용 (문장에 들어갈 절, 정답 라벨) 쌍 ──
# generate_synthetic_data.py의 매물 설명 문구들과 의미가 겹치도록 구성해서
# 임베딩 유사도 검색이 실제로 걸리는지 검증할 수 있게 함.
OTHER_CLAUSE_PAIRS = [
    ("강아지랑 같이 살 수 있는", "반려동물 동반 가능한 곳"),
    ("고양이 키워도 되는", "반려동물 동반 가능한 곳"),
    ("주차 가능한", "주차 가능한 곳"),
    ("차 댈 곳 있는", "주차 가능한 곳"),
    ("조용한", "조용한 동네 분위기"),
    ("주변이 시끄럽지 않은", "조용한 동네 분위기"),
    ("채광 좋은", "채광이 좋은 곳"),
    ("햇빛 잘 드는", "채광이 좋은 곳"),
    ("신축인", "신축 건물"),
    ("최근에 지어진", "신축 건물"),
    ("바로 입주할 수 있는", "즉시 입주 가능한 곳"),
    ("당장 이사 들어갈 수 있는", "즉시 입주 가능한 곳"),
    ("풀옵션인", "옵션 완비된 곳"),
    ("가구, 가전 다 있는", "옵션 완비된 곳"),
    ("보안이 철저한", "보안이 잘 되어 있는 곳"),
    ("치안 좋은", "보안이 잘 되어 있는 곳"),
]


def make_price_phrase(max_price) -> str:
    if pd.isna(max_price):
        return ""
    return random.choice(PRICE_PHRASES_MAX).format(int(max_price))


def generate_test_case(row: pd.Series) -> dict:
    """
    row(매물 1건)에서 템플릿을 먼저 고르고, 그 템플릿이 요구하는 필수 슬롯은 반드시 채운다.
    추가로 선택 슬롯 0~2개를 무작위로 더 붙여 문장을 다양화한다.
    실제로 문장에 들어간 슬롯만 정답으로 남긴다.
    기타(자유 텍스트) 슬롯은 매물의 실제 속성과 무관하게 별도 풀에서 뽑는다.
    """
    slot_values = {
        "지역": row.get("지역"),
        "거래유형": row.get("거래유형"),
        "주거유형": row.get("주거유형", "원룸"),
        "층조건": row.get("층조건"),
        "근접시설": row.get("근접시설"),
        "가격_최대": row.get("월세") if row.get("거래유형") == "월세" else None,
    }

    template = random.choice(TEMPLATES)
    picked = set(template["required"])

    # 이 템플릿이 기타 슬롯을 요구하면, 자유 텍스트 절/정답 쌍을 하나 뽑는다.
    기타절_text = ""
    true_기타 = None
    if "기타" in template["required"]:
        clause, label = random.choice(OTHER_CLAUSE_PAIRS)
        기타절_text = clause
        true_기타 = label

    # 이 템플릿의 텍스트에 실제로 등장하는 선택 슬롯만 후보로 삼는다.
    # (텍스트에 없는 슬롯을 정답에만 몰래 채워넣는 버그 방지)
    remaining = [
        s for s, placeholder in SLOT_TO_PLACEHOLDER.items()
        if placeholder in template["text"]
        and s not in picked
        and pd.notna(slot_values.get(s))
    ]
    extra_n = random.randint(0, min(2, len(remaining)))
    picked |= set(random.sample(remaining, k=extra_n)) if extra_n else set()

    fill = {
        "지역": slot_values["지역"] if "지역" in picked else "",
        "거래유형": slot_values["거래유형"] if "거래유형" in picked else "",
        "주거유형": slot_values["주거유형"] if "주거유형" in picked else "",
        "층조건": slot_values["층조건"] if "층조건" in picked else "",
        "근접시설": slot_values["근접시설"] if "근접시설" in picked else "",
        "가격조건": make_price_phrase(slot_values["가격_최대"]) if "가격_최대" in picked else "",
        "기타절": 기타절_text,
    }

    utterance = template["text"].format(**fill)
    utterance = " ".join(utterance.split())  # 중복 공백 정리

    return {
        "utterance": utterance,
        "true_지역": slot_values["지역"] if "지역" in picked else None,
        "true_거래유형": slot_values["거래유형"] if "거래유형" in picked else None,
        "true_주거유형": slot_values["주거유형"] if "주거유형" in picked else None,
        "true_층조건": slot_values["층조건"] if "층조건" in picked else None,
        "true_근접시설": slot_values["근접시설"] if "근접시설" in picked else None,
        "true_가격_최소": None,
        "true_가격_최대": slot_values["가격_최대"] if "가격_최대" in picked else None,
        "true_면적_최소": None,
        "true_기타": true_기타,
    }


def generate_test_set(listings_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    need_replace = n > len(listings_df)
    sample = listings_df.sample(n=n, random_state=7, replace=need_replace)
    cases = [generate_test_case(row) for _, row in sample.iterrows()]
    return pd.DataFrame(cases)


if __name__ == "__main__":
    listings = pd.read_csv(os.path.join(DB_DIR, "synthetic_listings.csv"))
    test_df = generate_test_set(listings, n=15)
    test_df.to_csv(os.path.join(DB_DIR, "test_cases.csv"), index=False, encoding="utf-8-sig")
    print(f"[저장 완료] test_cases.csv ({len(test_df)}행)")
    print(test_df[["utterance", "true_기타"]].head(10))