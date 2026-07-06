"""
인풋 데이터와 정답지 만들기 
slot_schema.json에 정의된 범주형 슬롯의 모든 조합(카테시안 곱)에 대해
매물 데이터를 생성하고, 비범주형 슬롯(가격, 평수)은 지역/유형에 맞는
현실적인 랜덤 값으로 채우는 스크립트.
"""

import json
import itertools
import random

import pandas as pd

random.seed(42)

# ── slot_schema.json 로드 ────────────────────────────────────────
with open("slot_schema.json", encoding="utf-8") as f:
    schema = json.load(f)["slot_schema"]

REGIONS = ["강남구", "성동구", "마포구", "관악구", "노원구", "송파구"]
TRADE_TYPES = schema["거래유형"]["allowed_values"]          # ["전세", "월세"]
ROOM_TYPES = schema["주거유형"]["allowed_values"]            # ["원룸", "투룸", "오피스텔"]
FLOOR_CONDS = schema["층조건"]["allowed_values"]             # ["반지하", "1층", "저층", "중층", "고층"]
NEAR_FACILITIES = schema["근접시설"]["allowed_values"] + [None]  # None = 언급 없음도 실제 데이터엔 존재

# ── 지역별 시세 базline (지역마다 가격대가 다르게 보이도록) ─────────
# 월세 기준 (만원), 평수 기준 범위. 실제 시세와 정확히 일치할 필요는 없음(합성이므로).
REGION_PRICE_LEVEL = {
    "강남구": (60, 120),
    "성동구": (45, 90),
    "마포구": (50, 95),
    "관악구": (35, 65),
    "노원구": (30, 55),
    "송파구": (45, 85),
}
DEPOSIT_LEVEL_WOLSE = (500, 3000)     # 월세의 보증금 범위 (만원)
DEPOSIT_LEVEL_JEONSE = (8000, 30000)  # 전세 보증금 범위 (만원)

ROOM_SIZE_RANGE = {
    "원룸": (5, 9),
    "투룸": (10, 16),
    "오피스텔": (7, 13),
}


def sample_price(region: str, trade_type: str) -> tuple[int, int]:
    """(보증금, 월세) 튜플. 전세면 월세=0."""
    low, high = REGION_PRICE_LEVEL[region]
    if trade_type == "전세":
        deposit = random.randint(*DEPOSIT_LEVEL_JEONSE)
        return deposit, 0
    else:
        monthly = random.randint(low, high)
        deposit = random.randint(*DEPOSIT_LEVEL_WOLSE)
        return deposit, monthly


def sample_size(room_type: str) -> float:
    low, high = ROOM_SIZE_RANGE[room_type]
    return round(random.uniform(low, high), 1)


def generate_full_combination_dataset(rows_per_combo: int = 1) -> pd.DataFrame:
    """
    범주형 슬롯 5개(지역, 거래유형, 주거유형, 층조건, 근접시설)의
    모든 조합에 대해 rows_per_combo개씩 매물을 생성한다.
    """
    combos = list(itertools.product(
        REGIONS, TRADE_TYPES, ROOM_TYPES, FLOOR_CONDS, NEAR_FACILITIES
    ))

    rows = []
    for region, trade_type, room_type, floor_cond, near in combos:
        for _ in range(rows_per_combo):
            deposit, monthly = sample_price(region, trade_type)
            size = sample_size(room_type)
            rows.append({
                "지역": region,
                "거래유형": trade_type,
                "주거유형": room_type,
                "층조건": floor_cond,
                "근접시설": near,
                "보증금": deposit,
                "월세": monthly,
                "평수": size,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_full_combination_dataset(rows_per_combo=1)
    print(f"생성된 조합 수(=행 수): {len(df)}")
    print(df.head(10))
    df.to_csv("synthetic_listings.csv", index=False, encoding="utf-8-sig")
