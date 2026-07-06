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

import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# 유료 API로 바꾸고 싶으시면 아래로 교체하세요:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

# .env 파일에서 GOOGLE_API_KEY를 읽어와 환경변수로 등록합니다.
# 코드에 키를 직접 적지 않기 위함입니다.
load_dotenv()


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
    # 하루 20개 최대이 
    # return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # 무료 티어 요청 한도를 더 넉넉하게 쓰고 싶으시면 (품질은 살짝 낮음):
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    # 유료 API 사용 시:
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def classify_intent(user_input: str) -> IntentClassification:
    llm = get_llm().with_structured_output(IntentClassification)
    return llm.invoke(
        f"다음 사용자 발화의 의도를 분류하세요.\n\n발화: \"{user_input}\""
    )


def extract_slots(user_input: str) -> SlotFilling:
    llm = get_llm().with_structured_output(SlotFilling)
    system = (
        "당신은 부동산 검색 챗봇의 슬롯 추출기입니다. "
        "사용자 발화에서 검색 조건만 정확히 추출하세요. "
        "언급되지 않은 항목은 절대 추측하지 말고 null로 두세요. "
        "가격은 항상 만원 단위 정수로 변환하세요 (예: '50만원' -> 50)."
    )
    return llm.invoke([("system", system), ("user", user_input)])


# ── 4. pandas 필터링 (AI 아님, 결정론적 코드) ─────────────────────
def search_listings(df: pd.DataFrame, slots: SlotFilling) -> pd.DataFrame:
    result = df.copy()

    if slots.지역:
        query = slots.지역.replace("서울", "").strip()
        result = result[result["지역"].astype(str).apply(
            lambda v: query in v.replace("서울", "").strip() or v.replace("서울", "").strip() in query
        )]
    if slots.거래유형:
        result = result[result["거래유형"] == slots.거래유형]
    if slots.주거유형:
        result = result[result["주거유형"] == slots.주거유형]
    if slots.층조건:
        result = result[result["층조건"] == slots.층조건]
    if slots.근접시설:
        result = result[result["근접시설"] == slots.근접시설]
    if slots.가격_최대 is not None:
        result = result[result["월세"] <= slots.가격_최대]
    if slots.가격_최소 is not None:
        result = result[result["월세"] >= slots.가격_최소]
    if slots.면적_최소 is not None:
        result = result[result["평수"] >= slots.면적_최소]

    return result


# ── 5. 결과 포맷팅 (템플릿 기반, LLM 호출 없음) ────────────────────
def format_results(result_df: pd.DataFrame, top_n: int = 3) -> str:
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
        lines.append(
            f"① {row['지역']} {row.get('주거유형', '')} {price_text}{near}"
        )
    return "\n".join(lines)


# ── 6. 전체 파이프라인 ────────────────────────────────────────────
def run_pipeline(user_input: str, df: pd.DataFrame) -> str:
    intent = classify_intent(user_input)

    if intent.intent == "잡담":
        return "저는 부동산 매물 검색을 도와드리는 챗봇입니다. 원하시는 지역이나 조건을 말씀해주세요!"

    slots = extract_slots(user_input)
    result_df = search_listings(df, slots)
    return format_results(result_df)


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