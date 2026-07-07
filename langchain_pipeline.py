"""
LangChain Function Calling 기반 부동산 검색 챗봇 파이프라인

구조:
  1) 의도 분류 (Text Classification)   - 이 발화가 '매물 검색'인지 아닌지 먼저 판단
  2) 슬롯 추출 (Prompt Engineering + Information Extraction, Function Calling)
  3) pandas 필터링 (일반 코드, AI 아님)
  4) 결과 포맷팅 (템플릿 기반 문자열 생성, LLM 호출 없음)

요구사항의 "NLP 기술 2개 이상"은 1)+2)로 충족됩니다.
LLM 호출은 총 2번(의도분류, 슬롯추출)이지만 결과 설명은 LLM 없이 템플릿으로 처리하여
비용/지연시간/할루시네이션 리스크를 줄였습니다.
"""

import json
import os
from typing import Optional, Literal
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
# 유료 API로 바꾸고 싶으시면 아래로 교체하세요:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

# .env 파일에서 GOOGLE_API_KEY를 읽어와 환경변수로 등록합니다.
# 코드에 키를 직접 적지 않기 위함입니다.
load_dotenv()

# 로컬 임베딩 모델 (무료, API 호출/요청 제한 없음, 최초 1회만 다운로드).
# 다국어 지원 모델이라 한국어 문장도 잘 처리됨.
_EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 이 스크립트 위치 기준으로 프로젝트 안에 고정된 캐시 폴더를 둔다.
# 매번 사용자 홈 폴더의 숨겨진 캐시를 찾아 헤매지 않도록 명시적으로 지정.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CACHE_DIR = os.path.join(_BASE_DIR, "models_cache")
_MODEL_LOCAL_PATH = os.path.join(_MODEL_CACHE_DIR, _EMBED_MODEL_NAME.replace("/", "_"))


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    임베딩 모델을 최초 1회만 로드/다운로드하고, 프로세스 안에서는 캐시된 걸 재사용.
    - 최초 실행: 인터넷에서 받아서 _MODEL_CACHE_DIR에 저장 (models_cache/ 폴더, 약 470MB)
    - 이후 실행: 이미 저장된 폴더에서 그대로 불러오기만 함 (인터넷 확인 없이 즉시 로드)
    """
    if os.path.exists(_MODEL_LOCAL_PATH):
        # 이미 받아둔 모델이 있으면 완전 오프라인으로 즉시 로드 (허깅페이스 서버 접속 자체를 안 함)
        os.environ["HF_HUB_OFFLINE"] = "1"
        return SentenceTransformer(_MODEL_LOCAL_PATH)

    # 최초 1회: 다운로드 후 지정 경로에 저장
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    model = SentenceTransformer(_EMBED_MODEL_NAME)
    model.save(_MODEL_LOCAL_PATH)
    return model


# ── 1. 슬롯 스키마 (slot_schema.json과 대응) ─────────────────────
class SlotFilling(BaseModel):
    """사용자 발화에서 추출한 부동산 검색 조건. 언급되지 않은 값은 반드시 null로 둔다."""

    지역: Optional[str] = Field(None, description="구/동/역 단위 지역명, 예: 강남구, 홍대")
    거래유형: Optional[Literal["전세", "월세"]] = Field(None, description="전세 또는 월세")
    주거유형: Optional[Literal["원룸", "투룸", "오피스텔"]] = Field(None, description="주거 형태")
    층조건: Optional[Literal["반지하", "1층", "저층", "중층", "고층"]] = Field(None)
    근접시설: Optional[Literal["역세권", "대학가 인근", "학교 근처", "마트 도보 5분", "버스정류장 인접"]] = Field(None)
    가격_최소: Optional[int] = Field(None, description="만원 단위 정수")
    가격_최대: Optional[int] = Field(None, description="만원 단위 정수")
    면적_최소: Optional[float] = Field(None, description="평수")
    기타: Optional[str] = Field(
        None,
        description=(
            "위 정형 슬롯(지역/거래유형/주거유형/층조건/근접시설/가격/면적)에 해당하지 않는 "
            "자유로운 조건. 반려동물 동반 가능 여부, 입주 시기, 주차 가능 여부, 채광, "
            "동네 분위기(조용함/번화가) 등. 사용자의 표현을 요약하되 핵심 의미는 그대로 유지한다. "
            "예: '강아지 키우는데'-> '반려동물 동반 가능한 곳', '주차 되나요' -> '주차 가능'"
        ),
    )


# ── 2. 의도 분류 (Text Classification) ───────────────────────────
class IntentClassification(BaseModel):
    """사용자 발화의 의도를 분류한다."""

    intent: Literal["매물검색", "조건변경", "재검색", "잡담", "기타"] = Field(
        description="매물검색: 새로운 조건으로 집을 찾는 발화. "
                    "조건변경: 기존 검색에 조건을 추가/수정하는 발화. "
                    "재검색: 조건 없이 다시 찾아달라는 발화. "
                    "잡담: 부동산 검색과 무관한 발화. 기타: 위에 해당하지 않는 경우."
    )


# ── 3. LLM 클라이언트 ────────────────────────────────────────────
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # 무료 티어 요청 한도를 더 넉넉하게 쓰고 싶으시면 (품질은 살짝 낮음):
    # return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    # 유료 API 사용 시:
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def classify_intent(user_input: str) -> IntentClassification:
    llm = get_llm().with_structured_output(IntentClassification)
    return llm.invoke(
        f"다음 사용자 발화의 의도를 분류하세요.\n\n발화: \"{user_input}\""
    )


FEW_SHOT_EXAMPLES = """
아래는 발화와 정답 슬롯 예시입니다. 형식과 판단 기준을 참고하세요.

예시 1)
발화: "성동구에서 월 50에 우리 강아지랑 살만한 곳 있을까요"
정답: {"지역": "성동구", "거래유형": "월세", "가격_최대": 50, "기타": "반려동물 동반 가능한 곳"}
(설명: '강아지랑 살만한 곳'은 정형 슬롯이 아니므로 기타에 요약해서 넣는다. 주거유형/근접시설 등 언급 없는 건 전부 null.)

예시 2)
발화: "회사가 강남이라 걸어서 출퇴근할 수 있는 오피스텔 찾는데 조용한 동네였으면 좋겠어요"
정답: {"지역": "강남구", "주거유형": "오피스텔", "기타": "조용한 동네 분위기"}
(설명: '걸어서 출퇴근'은 구체적인 역명이나 "역세권"이라는 표현이 없으므로 근접시설을 추측해서 채우지 않는다. 애매하면 null이 정답에 가깝다.)

예시 3)
발화: "신혼집으로 쓸 투룸인데 주차 되는 곳으로, 노원구나 근처면 좋겠고 전세로요"
정답: {"지역": "노원구", "거래유형": "전세", "주거유형": "투룸", "기타": "주차 가능한 곳"}

예시 4)
발화: "그냥 아무 방이나 있나요"
정답: {}
(설명: 아무 조건도 명시하지 않았으므로 모든 슬롯을 null로 둔다.)
"""


def extract_slots(user_input: str) -> SlotFilling:
    llm = get_llm().with_structured_output(SlotFilling)
    system = (
        "당신은 부동산 검색 챗봇의 슬롯 추출기입니다. "
        "사용자 발화에서 검색 조건만 정확히 추출하세요. "
        "언급되지 않은 항목은 절대 추측하지 말고 null로 두세요. "
        "가격은 항상 만원 단위 정수로 변환하세요 (예: '50만원' -> 50). "
        "지역/거래유형/주거유형/층조건/근접시설/가격/면적에 해당하지 않는 자유로운 조건"
        "(반려동물, 입주시기, 주차, 분위기, 채광 등)은 '기타' 슬롯에 짧게 요약해서 넣으세요. "
        "애매하면 억지로 채우지 말고 null로 두는 것을 우선하세요."
        + FEW_SHOT_EXAMPLES
    )
    return llm.invoke([("system", system), ("user", user_input)])


# ── 4. pandas 필터링 + 점수 기반 랭킹 (+ 기타 슬롯 임베딩 유사도) ──
def search_listings(df: pd.DataFrame, slots: SlotFilling) -> pd.DataFrame:
    """
    지역/거래유형/주거유형/가격/면적: 필수 조건 (하나라도 안 맞으면 후보에서 제외)
    층조건/근접시설: 가산점 조건 (맞으면 +1점, 안 맞아도 탈락시키지 않음)
    기타: 위 키워드 기반 필터를 먼저 통과한 후보들 안에서만,
          로컬 임베딩 모델로 '설명' 컬럼과의 코사인 유사도를 계산해 가산점에 추가.

    결과는 점수 내림차순으로 정렬되어, "조건에 완벽히 맞는 매물이 없어도
    가장 비슷한 매물"을 보여줄 수 있다.
    """
    result = df.copy()

    # ── 필수 조건 (키워드 기반, AI 아님) ──
    if slots.지역:
        query = slots.지역.replace("서울", "").strip()
        result = result[result["지역"].astype(str).apply(
            lambda v: query in v.replace("서울", "").strip() or v.replace("서울", "").strip() in query
        )]
    if slots.거래유형:
        result = result[result["거래유형"] == slots.거래유형]
    if slots.주거유형:
        result = result[result["주거유형"] == slots.주거유형]
    if slots.가격_최대 is not None:
        result = result[result["월세"] <= slots.가격_최대]
    if slots.가격_최소 is not None:
        result = result[result["월세"] >= slots.가격_최소]
    if slots.면적_최소 is not None:
        result = result[result["평수"] >= slots.면적_최소]

    if result.empty:
        return result

    # ── 가산점 조건 (탈락 없음, 점수만 부여) ──
    def compute_score(row) -> float:
        score = 0.0
        if slots.층조건 and row.get("층조건") == slots.층조건:
            score += 1
        if slots.근접시설 and row.get("근접시설") == slots.근접시설:
            score += 1
        return score

    result = result.copy()
    result["_match_score"] = result.apply(compute_score, axis=1)

    # ── 기타 슬롯: 키워드 기반 하드 신호 우선 + 임베딩 유사도는 보조 ──
    if slots.기타:
        기타_text = slots.기타

        # 반려동물/주차처럼 "가능/불가능"이 명확히 갈리는 속성은
        # 임베딩보다 실제 True/False 컬럼을 우선으로 쓴다.
        # (임베딩은 '가능'과 '불가능'을 헷갈리기 쉬움 — 공유 단어가 많아서)
        PET_KEYWORDS = ["반려동물", "강아지", "고양이", "펫", "댕댕이", "냥이", "애완동물"]
        PARKING_KEYWORDS = ["주차"]

        hard_signal_applied = False

        if any(k in 기타_text for k in PET_KEYWORDS) and "반려동물가능" in result.columns:
            # 부정 표현("못 키우는", "안 되는" 등)까지는 아직 정교하게 처리 안 함 -> 기본은 '가능'을 원한다고 가정
            wants_negative = any(neg in 기타_text for neg in ["안 되", "불가", "못 키우", "금지"])
            desired = not wants_negative
            result["_hard_bonus"] = result["반려동물가능"].apply(
                lambda v: 1.0 if v == desired else (-1.0 if pd.notna(v) else 0.0)
            )
            hard_signal_applied = True

        elif any(k in 기타_text for k in PARKING_KEYWORDS) and "주차가능" in result.columns:
            wants_negative = any(neg in 기타_text for neg in ["안 되", "불가", "없어도", "필요 없"])
            desired = not wants_negative
            result["_hard_bonus"] = result["주차가능"].apply(
                lambda v: 1.0 if v == desired else (-1.0 if pd.notna(v) else 0.0)
            )
            hard_signal_applied = True
        else:
            result["_hard_bonus"] = 0.0

        # 임베딩 유사도는 항상 보조 지표로 같이 계산 (하드 신호가 없는 자유 표현엔 이게 메인이 됨)
        if "설명" in result.columns:
            model = get_embedding_model()
            query_vec = model.encode([기타_text], normalize_embeddings=True)[0]
            desc_vecs = model.encode(result["설명"].fillna("").tolist(), normalize_embeddings=True)
            similarities = desc_vecs @ query_vec
            result["_similarity"] = similarities
        else:
            result["_similarity"] = 0.0

        # 하드 신호가 적용된 경우: 하드 신호 비중을 크게, 임베딩은 미세 조정 정도로만
        # 하드 신호가 없는 경우: 임베딩 유사도가 그대로 점수가 됨
        if hard_signal_applied:
            result["_match_score"] = result["_match_score"] + result["_hard_bonus"] * 3 + result["_similarity"] * 0.3
        else:
            result["_match_score"] = result["_match_score"] + result["_similarity"]
    else:
        result["_similarity"] = 0.0
        result["_hard_bonus"] = 0.0

    # 점수 높은 순 -> 동점이면 가격 낮은 순으로 정렬 (가격 낮은 매물을 우선 추천)
    result = result.sort_values(by=["_match_score", "월세"], ascending=[False, True])

    return result.drop(columns=["_match_score", "_similarity", "_hard_bonus"], errors="ignore")


# ── 5. 결과 포맷팅 (템플릿 기반, LLM 호출 없음) ────────────────────
def format_results(result_df: pd.DataFrame, top_n: int = 3, slots: "SlotFilling" = None) -> str:
    if result_df.empty:
        return "조건에 맞는 매물을 찾지 못했습니다. 조건을 조금 완화해서 다시 찾아드릴까요?"

    lines = [f"조건에 맞는 방 {min(len(result_df), top_n)}개를 찾았어요:"]
    for i, (_, row) in enumerate(result_df.head(top_n).iterrows(), start=1):
        near = f", {row['근접시설']}" if pd.notna(row.get("근접시설")) else ""
        거래유형 = row.get("거래유형", "")
        if 거래유형 == "전세":
            price_text = f"전세 {row.get('보증금', 0)}만원"
        else:
            price_text = f"월세 {row.get('월세', 0)}만원 (보증금 {row.get('보증금', 0)}만원)"

        # 가산점 조건(층조건/근접시설)을 요청했는데 이 매물이 못 맞춘 부분이 있으면 안내
        unmatched = []
        if slots is not None:
            if slots.층조건 and row.get("층조건") != slots.층조건:
                unmatched.append(f"층조건은 '{slots.층조건}'이 아닌 '{row.get('층조건')}'")
            if slots.근접시설 and row.get("근접시설") != slots.근접시설:
                unmatched.append(f"근접시설은 '{slots.근접시설}' 아님")
        note = f" (※ {', '.join(unmatched)})" if unmatched else ""

        lines.append(
            f"① {row['지역']} {row.get('주거유형', '')} {price_text}{near}{note}"
        )
        if slots is not None and slots.기타 and pd.notna(row.get("설명")):
            lines.append(f"   └ {row['설명']}")
    return "\n".join(lines)


# ── 6. 전체 파이프라인 ────────────────────────────────────────────
def run_pipeline(user_input: str, df: pd.DataFrame) -> str:
    intent = classify_intent(user_input)

    if intent.intent == "잡담":
        return "저는 부동산 매물 검색을 도와드리는 챗봇입니다. 원하시는 지역이나 조건을 말씀해주세요!"

    slots = extract_slots(user_input)
    result_df = search_listings(df, slots)
    return format_results(result_df, slots=slots)


if __name__ == "__main__":
    # 테스트용 더미 데이터 (실제로는 augment_data.py의 결과 CSV를 로드)
    dummy_df = pd.DataFrame([
        {"지역": "서울 강남구 역삼동", "거래유형": "월세", "주거유형": "원룸",
         "층조건": "저층", "근접시설": "역세권", "월세": 48, "평수": 7.0},
        {"지역": "서울 성동구 성수동", "거래유형": "월세", "주거유형": "오피스텔",
         "층조건": "고층", "근접시설": None, "월세": 55, "평수": 9.2},
    ])

    # OPENAI_API_KEY 환경변수 설정 필요
    # print(run_pipeline("강남 원룸 월세 50만원 이하로 역세권 찾아줘", dummy_df))
    print("스키마 로드 확인:", SlotFilling.model_json_schema()["title"])