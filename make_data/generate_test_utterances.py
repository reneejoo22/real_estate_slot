"""
listings_augmented.csv(augment_data.py 결과)를 바탕으로,
각 매물의 슬롯 값 중 일부만 무작위로 골라 자연어 발화를 만들고,
그 발화에 실제로 '언급된' 슬롯 값만 정답(ground truth)으로 저장하는 스크립트.

핵심 포인트: 매물의 전체 속성이 아니라, 발화에 실제 등장한 슬롯만 정답으로 삼는다.
언급 안 된 슬롯은 정답도 None이어야 "슬롯 추출기가 억지로 채워넣지 않는지"를
올바르게 평가할 수 있다.

변경 사항:
- 템플릿을 훨씬 다양하게 확장 (단순 나열 → 구어체/문어체/복합 조건 등)
- 슬롯 3개 이상인 케이스가 전체의 80% 이상이 되도록 강제
- 슬롯 1~2개짜리 '쉬운' 케이스는 20% 이하로 제한
"""

import random
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "db")

random.seed(7)

# ---------------------------------------------------------------------------
# 템플릿 정의
# ---------------------------------------------------------------------------
# "slots": 이 템플릿에서 사용 가능한 슬롯 이름 목록
# "text": 포맷 문자열 (슬롯 플레이스홀더 포함)
# "required": 반드시 채워야 하는 슬롯 (이 슬롯이 row에 없으면 템플릿 선택 제외)
# "optional": 선택적으로 넣을 수 있는 추가 슬롯
#
# 슬롯 종류: 지역 / 거래유형 / 주거유형 / 층조건 / 근접시설 / 가격조건
# ---------------------------------------------------------------------------

TEMPLATES = [
    # ── 슬롯 4~5개 (복합 조건) ──────────────────────────────────────────────
    {
        "text": "{지역}에서 {거래유형}로 {가격조건} {주거유형} 구하는데 {근접시설} 조건도 맞으면 좋겠어요",
        "required": ["지역", "거래유형", "가격조건", "주거유형", "근접시설"],
    },
    {
        "text": "{지역} {층조건} {주거유형} {거래유형} {가격조건} 구해요",
        "required": ["지역", "층조건", "주거유형", "거래유형", "가격조건"],
    },
    {
        "text": "{지역} {거래유형} {주거유형} 알아보는데 {층조건}이고 {근접시설}이면 완벽해요",
        "required": ["지역", "거래유형", "주거유형", "층조건", "근접시설"],
    },
    {
        "text": "{가격조건} {주거유형} {지역}에서 찾고 있어요. {층조건}이고 {근접시설} 있으면 좋겠어요",
        "required": ["가격조건", "주거유형", "지역", "층조건", "근접시설"],
    },
    {
        "text": "{지역} {주거유형} {거래유형} 매물 중에 {층조건}이면서 {가격조건} 짜리 있나요",
        "required": ["지역", "주거유형", "거래유형", "층조건", "가격조건"],
    },
    {
        "text": "{지역} {거래유형} {가격조건} {주거유형} {층조건} 조건으로 내놓은 매물 있어요?",
        "required": ["지역", "거래유형", "가격조건", "주거유형", "층조건"],
    },
    {
        "text": "{주거유형} 구하는데 {지역}이고 {거래유형} {가격조건}이에요. {근접시설} 있으면 더 좋고요",
        "required": ["주거유형", "지역", "거래유형", "가격조건", "근접시설"],
    },
    {
        "text": "{지역} {주거유형} 중에 {거래유형} {가격조건}이면서 {층조건}인 매물 찾아줘",
        "required": ["지역", "주거유형", "거래유형", "가격조건", "층조건"],
    },

    # ── 슬롯 3개 ────────────────────────────────────────────────────────────
    {
        "text": "{지역} {주거유형} {거래유형}으로 구해요",
        "required": ["지역", "주거유형", "거래유형"],
    },
    {
        "text": "{지역}에서 {거래유형} {가격조건} {주거유형} 찾아줘",
        "required": ["지역", "거래유형", "주거유형"],
    },
    {
        "text": "{주거유형} 구하는데 {지역} {근접시설}이면 좋겠어요",
        "required": ["주거유형", "지역", "근접시설"],
    },
    {
        "text": "{지역} {층조건} {주거유형} {거래유형}으로 구해요",
        "required": ["지역", "층조건", "주거유형", "거래유형"],
    },
    {
        "text": "{가격조건} {주거유형} {지역} 있나요",
        "required": ["가격조건", "주거유형", "지역"],
    },
    {
        "text": "{거래유형}로 {가격조건} 방 {지역}에 있어?",
        "required": ["거래유형", "가격조건", "지역"],
    },
    {
        "text": "{지역} {주거유형} 매물인데 {거래유형} {층조건}으로 찾아요",
        "required": ["지역", "주거유형", "거래유형", "층조건"],
    },
    {
        "text": "{지역}에서 {주거유형} {거래유형} {가격조건}으로 알아보고 있어요",
        "required": ["지역", "주거유형", "거래유형", "가격조건"],
    },
    {
        "text": "{거래유형} {주거유형} 중에 {지역}이고 {근접시설} 조건 되는 거 있나요?",
        "required": ["거래유형", "주거유형", "지역", "근접시설"],
    },
    {
        "text": "{지역} 근처 {주거유형} {거래유형}으로 {가격조건} 선에서 구합니다",
        "required": ["지역", "주거유형", "거래유형", "가격조건"],
    },
    {
        "text": "{층조건} {주거유형} {지역} {거래유형}으로 나온 매물 있나요?",
        "required": ["층조건", "주거유형", "지역", "거래유형"],
    },
    {
        "text": "{지역} {근접시설} {주거유형} {거래유형} 구해요",
        "required": ["지역", "근접시설", "주거유형", "거래유형"],
    },
    {
        "text": "{지역} {거래유형}로 {주거유형} 보고 있는데 {층조건} 가능해요?",
        "required": ["지역", "거래유형", "주거유형", "층조건"],
    },
    {
        "text": "{주거유형} {거래유형}로 {지역}에서 {가격조건} 구하고 싶어요",
        "required": ["주거유형", "거래유형", "지역", "가격조건"],
    },

    # ── 슬롯 1~2개 (전체 20% 이하 목적, 개수는 적게) ──────────────────────
    {
        "text": "{지역}로 집 좀 찾아줘",
        "required": ["지역"],
    },
    {
        "text": "{거래유형}로 {가격조건} 방 있어?",
        "required": ["거래유형"],
    },
    {
        "text": "{지역} {거래유형} 매물 있나요?",
        "required": ["지역", "거래유형"],
    },
]

# 슬롯 이름 → 포맷 문자열 키 매핑
SLOT_TO_KEY = {
    "지역":     "지역",
    "거래유형": "거래유형",
    "주거유형": "주거유형",
    "층조건":   "층조건",
    "근접시설": "근접시설",
    "가격조건": "가격조건",  # 가격_최대 → 가격조건 문자열로 변환
}

PRICE_PHRASES_MAX = [
    "{}만원 이하로",
    "{}만원 안쪽으로",
    "{}만원 넘지 않게",
    "최대 {}만원",
    "{}만원 이내",
]


def make_price_phrase(max_price) -> str:
    if pd.isna(max_price):
        return ""
    return random.choice(PRICE_PHRASES_MAX).format(int(max_price))


def slots_in_template(template: dict) -> set:
    """템플릿 required 슬롯 집합 반환."""
    return set(template["required"])


def slot_count(template: dict) -> int:
    return len(set(template["required"]))


def generate_test_case(row: pd.Series, force_rich: bool = True) -> dict:
    """
    force_rich=True이면 슬롯 3개 이상인 템플릿만 후보로 삼는다.
    force_rich=False이면 모든 템플릿이 후보가 된다.
    """
    slot_values = {
        "지역":     row.get("지역"),
        "거래유형": row.get("거래유형"),
        "주거유형": row.get("주거유형", "원룸"),
        "층조건":   row.get("층조건"),
        "근접시설": row.get("근접시설"),
        "가격조건": make_price_phrase(
            row.get("월세") if row.get("거래유형") == "월세" else None
        ),
        "가격_최대": row.get("월세") if row.get("거래유형") == "월세" else None,
    }

    # 사용 가능한 슬롯 (값이 있는 것)
    available = {k for k, v in slot_values.items()
                 if v and not (isinstance(v, float) and pd.isna(v))}

    # 템플릿 후보: required 슬롯이 모두 available 안에 있어야 함
    # 단, "가격조건"은 slot_values["가격조건"]이 빈 문자열이 아닌 경우에만 OK
    def template_ok(t):
        for s in t["required"]:
            key = s  # required에 쓰인 슬롯명 = slot_values 키명 일치
            val = slot_values.get(key, "")
            if not val or (isinstance(val, float) and pd.isna(val)):
                return False
        return True

    if force_rich:
        candidates = [t for t in TEMPLATES
                      if slot_count(t) >= 3 and template_ok(t)]
        # 3개짜리 템플릿이 아예 없으면 fallback
        if not candidates:
            candidates = [t for t in TEMPLATES if template_ok(t)]
    else:
        candidates = [t for t in TEMPLATES if template_ok(t)]

    if not candidates:
        candidates = TEMPLATES  # 최후 fallback

    template = random.choice(candidates)
    picked_slots = set(template["required"])

    # 포맷 딕셔너리 구성 (미사용 슬롯은 빈 문자열)
    fill = {
        "지역":     slot_values["지역"]     if "지역"     in picked_slots else "",
        "거래유형": slot_values["거래유형"] if "거래유형" in picked_slots else "",
        "주거유형": slot_values["주거유형"] if "주거유형" in picked_slots else "",
        "층조건":   slot_values["층조건"]   if "층조건"   in picked_slots else "",
        "근접시설": slot_values["근접시설"] if "근접시설" in picked_slots else "",
        "가격조건": slot_values["가격조건"] if "가격조건" in picked_slots else "",
    }

    utterance = template["text"].format(**fill)
    utterance = " ".join(utterance.split())  # 중복 공백 정리

    return {
        "utterance":      utterance,
        "true_지역":      slot_values["지역"]     if "지역"     in picked_slots else None,
        "true_거래유형":  slot_values["거래유형"] if "거래유형" in picked_slots else None,
        "true_주거유형":  slot_values["주거유형"] if "주거유형" in picked_slots else None,
        "true_층조건":    slot_values["층조건"]   if "층조건"   in picked_slots else None,
        "true_근접시설":  slot_values["근접시설"] if "근접시설" in picked_slots else None,
        "true_가격_최소": None,
        "true_가격_최대": slot_values["가격_최대"] if "가격조건" in picked_slots else None,
        "true_면적_최소": None,
    }


def generate_test_set(listings_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    전체 n개 중 80%는 슬롯 3개 이상인 케이스(force_rich=True),
    나머지 20%는 제약 없이 생성.
    """
    need_replace = n > len(listings_df)
    sample = listings_df.sample(n=n, random_state=7, replace=need_replace)

    n_rich = int(n * 0.8)   # 80%: 3+ 슬롯
    n_easy = n - n_rich      # 20%: 제약 없음

    rows = list(sample.iterrows())
    random.shuffle(rows)

    cases = []
    for i, (_, row) in enumerate(rows):
        force = (i < n_rich)
        cases.append(generate_test_case(row, force_rich=force))

    return pd.DataFrame(cases)


if __name__ == "__main__":
    listings = pd.read_csv(os.path.join(DB_DIR, "synthetic_listings.csv"))
    test_df = generate_test_set(listings, n=30)

    # 슬롯 개수 통계 출력
    slot_cols = ["true_지역", "true_거래유형", "true_주거유형",
                 "true_층조건", "true_근접시설", "true_가격_최대"]
    test_df["_slot_count"] = test_df[slot_cols].notna().sum(axis=1)
    rich_ratio = (test_df["_slot_count"] >= 3).mean()
    print(f"[슬롯 3개 이상 비율] {rich_ratio:.1%}")
    print(test_df["_slot_count"].value_counts().sort_index())
    test_df = test_df.drop(columns=["_slot_count"])

    test_df.to_csv(os.path.join(DB_DIR, "test_cases.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] test_cases.csv ({len(test_df)}행)")
    print(test_df.head(5))