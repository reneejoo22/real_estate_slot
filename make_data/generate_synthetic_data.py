"""
slot_schema.json에 정의된 범주형 슬롯의 모든 조합(카테시안 곱)에 대해
매물 데이터를 생성하고, 비범주형 슬롯(가격, 평수)은 지역/유형에 맞는
현실적인 랜덤 값으로 채우는 스크립트.
"""

import json
import itertools
import random
import os

import pandas as pd

random.seed(42)

# 이 스크립트 파일 위치를 기준으로 db 폴더를 찾는다.
# make_data/ 안에서 실행하든, 프로젝트 루트에서 실행하든 항상 동작한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "db")

# ── slot_schema.json 로드 ────────────────────────────────────────
with open(os.path.join(DB_DIR, "slot_schema.json"), encoding="utf-8") as f:
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

# ── 임베딩 검색(기타 슬롯)을 위한 자유 텍스트 매물 설명 속성 풀 ──────
# 각 속성은 (긍정 문구, 긍정 확률) 형태. 확률을 다르게 줘서
# 매물마다 조합이 다양하게 나오도록 함.
PET_PHRASES = [
    "반려동물과 함께 지내실 수 있습니다",
    "소형견 정도는 키우셔도 괜찮은 집이에요",
    "반려동물 동반 입주 가능합니다",
    "사랑스러운 반려동물과 함께 행복한 일상을 시작해 보세요",
    "동물 사육이 허용된 매물이라 눈치 보지 않고 지내실 수 있습니다",
    "댕댕이나 냥이와 함께 입주할 수 있는 귀한 방이랍니다"
]

NO_PET_PHRASES = [
    "다만 반려동물은 함께하실 수 없는 점 양해 부탁드립니다",
    "반려동물 동반은 어려운 매물입니다",
    "건물 규정상 아쉽게도 반려동물과 함께 거주하는 것은 제한됩니다",
    "쾌적한 건물 관리를 위해 동물 사육은 금지되어 있으니 참고해 주세요",
    "반려동물 키우시는 분들은 입주가 어려운 매물인 점 양해 구합니다"
]

MOVE_IN_PHRASES = [
    "즉시 입주가 가능해서 바로 짐을 옮기실 수 있어요",
    "협의 후 빠른 입주가 가능합니다",
    "다음 달 초부터 입주하실 수 있습니다",
    "현재 공실 상태라 원하시는 날짜에 언제든 입주하실 수 있어요",
    "기존 세입자분과 일정 조율 후 신속한 입주가 가능합니다",
    "입주 시기는 자유롭게 조절 가능하니 편하게 말씀해 주세요"
]

PARKING_PHRASES = [
    "건물 내 주차 공간이 마련되어 있어 차량 소유자분들도 편하게 지내실 수 있습니다",
    "지상 주차 1대 가능합니다",
    "지정 주차가 가능하여 퇴근 후에도 주차 스트레스가 전혀 없습니다",
    "자주식 주차 공간이 넉넉하게 확보되어 있어 초보 운전자분도 편리합니다",
    "세대당 1대 무료 주차가 제공되는 큰 장점이 있는 집입니다"
]

NO_PARKING_PHRASES = [
    "다만 별도 주차 공간은 없는 점 참고 부탁드립니다",
    "인근 공영주차장을 이용하셔야 하므로 차량이 없으신 분들께 추천해 드립니다",
    "지정 주차 구역이 따로 없으니 대중교통을 주로 이용하시는 분이 오시면 좋겠어요",
    "주차가 다소 불편한 환경이라 차량 미소유자분께 우선 권해드립니다"
]

BUILDING_AGE_PHRASES = [
    "최근에 지어진 신축 건물이라 시설이 깨끗합니다",
    "지은 지 얼마 안 된 건물이라 내부가 깔끔한 편입니다",
    "구축이지만 리모델링을 마쳐서 생활하기엔 문제없습니다",
    "첫 입주하시는 신축 급 컨디션으로 모든 시설이 최상급입니다",
    "연식은 조금 되었지만 관리가 워낙 잘 되어 세월의 흔적이 느껴지지 않아요",
    "내외부 모두 깔끔하게 유지보수되어 손볼 곳 없이 바로 생활 가능합니다"
]

MOOD_PHRASES = [
    "주변이 조용해서 아늑하게 지내기 좋은 동네입니다",
    "번화가와 가까워 생활 편의시설을 이용하기 편리합니다",
    "골목 안쪽에 위치해 있어 밤에도 비교적 조용한 편이에요",
    "단지 주변이 한적하고 치안이 좋아 여성분들도 안심하고 거주할 수 있습니다",
    "젊은 활기가 느껴지는 상권 인근이면서도 한 블록 안쪽이라 소음 걱정이 없어요",
    "녹지 공간과 산책로가 가까워 도심 속에서 여유를 만끽할 수 있는 동네입니다"
]

LIGHT_PHRASES = [
    "남향이라 채광이 좋아 낮에 불을 안 켜도 될 정도입니다",
    "창이 커서 햇빛이 잘 들어오는 방입니다",
    "막힘없는 조망권을 자랑하며, 하루 종일 화사한 햇살이 가득 들어옵니다",
    "오전부터 오후까지 해가 잘 들어와 집안 전체가 따스하고 포근해요",
    "양창 구조로 설계되어 채광은 물론 맞바람이 불어 환기에도 탁월합니다"
]

INTRO_PHRASES = [
    "{room_type} 매물을 소개해드립니다.",
    "깔끔하게 관리된 {room_type}입니다.",
    "{near_text}",
    "첫눈에 마음에 쏙 드실 만한 {room_type}을 들고 왔습니다.",
    "{near_text} 위치하여 직장인과 학생분들께 강력히 추천하는 매물입니다.",
    "가성비와 컨디션을 모두 잡은 보기 드문 {room_type}입니다."
]

# --- 💡 새로 제안드린 추가 아이디어 카테고리 변수입니다 ---

OPTION_PHRASES = [
    "풀옵션(에어컨, 세탁기, 냉장고 등)이 완비되어 있어 몸만 오시면 됩니다",
    "붙박이장과 수납공간이 넉넉하게 짜여 있어 짐이 많으신 분들도 깔끔하게 정리 가능해요",
    "시스템 에어컨과 빌트인 가구로 공간 활용도를 극대화했습니다"
]

TRAFFIC_PHRASES = [
    "지하철역에서 도보 5분 거리인 초역세권이라 출퇴근길이 한결 가벼워집니다",
    "집 바로 앞에 버스 정류장이 있어 대중교통 이용이 매우 편리한 입지입니다",
    "주요 간선도로 진입이 수월해 자차로 이동하기에 최적의 위치입니다"
]

SECURITY_PHRASES = [
    "24시간 CCTV 가동 및 공동현관 도어락 설치로 보안이 매우 철저합니다",
    "대로변과 가깝고 가로등이 밝아 늦은 밤 귀가 시간에도 안심할 수 있어요",
    "경비실이 운영되고 있어 택배 분실 걱정 없고 안전하게 지내실 수 있습니다"
]

TARGET_PHRASES = [
    "깔끔하고 아늑한 공간을 찾는 사회초년생이나 대학생분들께 적극 추천합니다",
    "미니멀 라이프를 선호하시는 1인 가구에게 안성맞춤인 집입니다",
    "신혼부부가 첫 살림을 시작하기에 더할 나위 없이 좋은 예쁜 공간이에요"
]


import random

def generate_description(region: str, room_type: str, near: str, floor_cond: str) -> str:
    """
    매물마다 1~4문장짜리 자유 텍스트 설명을 랜덤하게 조합해서 생성한다.
    이 텍스트는 슬롯으로 안 잡히는 자유 조건(반려동물, 입주시기, 주차, 분위기 등)을
    임베딩 유사도 검색으로 찾기 위한 재료로 쓰인다.
    """
    sentences = []

    # 1. 인트로 생성
    near_text = f"{near} 위치라 이동이 편리합니다." if near else ""
    intro = random.choice(INTRO_PHRASES).format(room_type=room_type, near_text=near_text)
    if intro.strip():
        sentences.append(intro)

    # 2. 반려동물 (70% 가능, 30% 불가)
    if random.random() < 0.7:
        sentences.append(random.choice(PET_PHRASES))
    else:
        sentences.append(random.choice(NO_PET_PHRASES))

    # 3. 입주 시기 (50% 확률로 언급)
    if random.random() < 0.5:
        sentences.append(random.choice(MOVE_IN_PHRASES))

    # 4. 주차 (50% 확률로 언급: 그중 60% 가능, 40% 불가)
    if random.random() < 0.5:
        if random.random() < 0.6:
            sentences.append(random.choice(PARKING_PHRASES))
        else:
            sentences.append(random.choice(NO_PARKING_PHRASES))

    # 5. 건물 연식 (40% 확률로 언급)
    if random.random() < 0.4:
        sentences.append(random.choice(BUILDING_AGE_PHRASES))

    # 6. 주변 분위기 (40% 확률로 언급)
    if random.random() < 0.4:
        sentences.append(random.choice(MOOD_PHRASES))

    # 7. 채광 (30% 확률로 언급 / 저층 및 반지하는 강제 스킵)
    if floor_cond not in ("반지하", "1층") and random.random() < 0.3:
        sentences.append(random.choice(LIGHT_PHRASES))

    # 8. [신규] 옵션 및 수납 (40% 확률로 언급)
    if random.random() < 0.4:
        sentences.append(random.choice(OPTION_PHRASES))

    # 9. [신규] 교통 편의성 (40% 확률로 언급)
    if random.random() < 0.4:
        sentences.append(random.choice(TRAFFIC_PHRASES))

    # 10. [신규] 보안 및 안전 (30% 확률로 언급)
    if random.random() < 0.3:
        sentences.append(random.choice(SECURITY_PHRASES))

    # 11. [신규] 추천 대상 (30% 확률로 언급)
    if random.random() < 0.3:
        sentences.append(random.choice(TARGET_PHRASES))

    # --- 문장 제한 및 셔플 로직 ---
    # 인트로(첫 문장)를 제외한 나머지 요소들을 무작위로 섞습니다.
    body_sentences = sentences[1:]
    random.shuffle(body_sentences)
    
    # 임베딩 검색용이므로 너무 길지 않게 최대 3개의 본문만 선택 (인트로 포함 총 4문장 제한)
    final_sentences = [sentences[0]] + body_sentences[:3]

    return " ".join(final_sentences)

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
            description = generate_description(region, room_type, near, floor_cond)
            rows.append({
                "지역": region,
                "거래유형": trade_type,
                "주거유형": room_type,
                "층조건": floor_cond,
                "근접시설": near,
                "보증금": deposit,
                "월세": monthly,
                "평수": size,
                "설명": description,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_full_combination_dataset(rows_per_combo=1)
    print(f"생성된 조합 수(=행 수): {len(df)}")
    print(df[["지역", "주거유형", "설명"]].head(5))
    df.to_csv(os.path.join(DB_DIR, "synthetic_listings.csv"), index=False, encoding="utf-8-sig")