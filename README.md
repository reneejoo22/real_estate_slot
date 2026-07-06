# 부동산 매물 검색 챗봇 (자연어 슬롯 필링 기반)

자연어 입력에서 부동산 검색 조건(슬롯)을 추출하고, 조건에 맞는 매물을 찾아 응답하는 챗봇 프로젝트입니다.
LangChain + Gemini function calling으로 슬롯을 추출하고, 매물 검색은 pandas 필터링(결정론적 코드)으로 처리합니다.

## 파이프라인 구조

```
사용자 발화
   │
   ▼
① 의도 분류 (Text Classification, LLM)
   │  잡담이면 여기서 종료
   ▼
② 슬롯 추출 (Prompt Engineering + Information Extraction, LLM Function Calling)
   │  지역 / 거래유형 / 주거유형 / 층조건 / 근접시설 / 가격_최소 / 가격_최대 / 면적_최소
   ▼
③ pandas 필터링 (일반 코드, AI 아님)
   ▼
④ 결과 포맷팅 — 상위 3개 매물 (템플릿 기반, LLM 호출 없음)
   ▼
최종 답변 출력
```

## 파일별 설명

### 데이터 정의
- **`slot_schema.json`** — 슬롯 스키마 정의 파일. 각 슬롯의 타입(범주형/비범주형), 허용 값, 설명을 담고 있음. `langchain_pipeline.py`의 `SlotFilling` 클래스와 1:1 대응.

### 데이터 생성
- **`generate_synthetic_data.py`** — `slot_schema.json`의 범주형 슬롯(지역/거래유형/주거유형/층조건/근접시설) 조합을 바탕으로 매물 데이터를 합성 생성. 지역별로 다른 가격대를 반영해 그럴듯한 보증금/월세/평수를 랜덤 배정. 결과물: `synthetic_listings.csv`
- **`synthetic_listings.csv`** — 위 스크립트의 산출물. 챗봇이 실제로 검색 대상으로 사용하는 매물 DB.
- **`generate_test_utterances.py`** — `synthetic_listings.csv`의 매물 정보를 바탕으로, 슬롯 일부만 언급하는 자연어 발화와 그 발화에 실제로 포함된 슬롯 값(정답)을 함께 생성. 슬롯 추출 성능을 평가하기 위한 테스트셋을 만드는 용도. 결과물: `test_cases.csv`
- **`test_cases.csv`** — 위 스크립트의 산출물. `utterance`(입력 문장)와 `true_*`(정답 슬롯 값) 컬럼으로 구성.

### 핵심 파이프라인
- **`langchain_pipeline.py`** — 메인 파이프라인. 의도 분류(`classify_intent`), 슬롯 추출(`extract_slots`), pandas 필터링(`search_listings`), 결과 포맷팅(`format_results`) 함수를 포함. LLM은 Gemini(`gemini-2.5-flash`)를 사용하며 `.env`의 `GOOGLE_API_KEY`로 인증.

### 평가
- **`evaluate_pipeline.py`** — `test_cases.csv`의 각 발화에 대해 실제로 슬롯 추출 → 검색 → 결과 생성까지 실행하고, 예측 슬롯과 정답 슬롯을 비교해 채점하는 스크립트. 슬롯별 Accuracy / Precision / Recall / F1(macro)과 전체 정확한 일치율을 계산. Gemini 무료 티어의 요청 제한(분당/일일)을 피하기 위해 호출 사이에 `time.sleep`을 두고, 테스트용으로 상위 N개만 실행하는 옵션(`max_samples`)도 지원.
- **`score_existing_results.py`** — LLM을 다시 호출하지 않고, 이미 쌓인 `eval_results.csv`만 읽어서 채점 지표를 재계산하는 스크립트. API 요청 제한으로 평가가 중간에 끊겼을 때, 지금까지의 결과만으로 지표를 확인하고 싶을 때 사용.
- **`eval_results.csv`** — `evaluate_pipeline.py` 실행 결과가 건별로 누적 저장되는 파일. 입력 문장, 예측 슬롯, 정답 슬롯, 슬롯별 일치 여부, 상위 3개 매물, 최종 답변 텍스트를 담고 있음.
- **`eval_summary.csv`** — 슬롯별 Accuracy / Precision / Recall / F1과, 전체 일치율(엄격) 및 슬롯별 평균 정확도(관대) 두 가지 종합 지표를 담은 요약 파일.

### 환경 설정
- **`.env.example`** — 필요한 API 키 목록 예시 파일. 실제 사용 시 `.env`로 복사하고 값을 채워넣어야 함 (`GOOGLE_API_KEY`, 필요 시 `PUBLIC_DATA_API_KEY` 등). `.env` 파일 자체는 git에 커밋하지 않음.

## 실행 순서

```bash
# 1. 매물 데이터 생성
python generate_synthetic_data.py

# 2. 평가용 테스트 발화 생성
python generate_test_utterances.py

# 3. 파이프라인 평가 실행 (기본값: 10개만, 호출 간 15초 대기)
python evaluate_pipeline.py

# (선택) API 호출 없이 기존 결과만 재채점
python score_existing_results.py
```

## 평가지표 해석 시 참고사항

- **ALL_SLOTS_EXACT_MATCH**: 슬롯 8개가 전부 정답과 일치해야 정답으로 인정하는 엄격한 지표. 슬롯 하나만 틀려도 0점 처리되므로 낮게 나오는 것이 자연스러움.
- **AVERAGE_PER_SLOT_ACCURACY**: 슬롯별 정확도의 평균. 일부 슬롯만 틀렸을 때의 실제 성능을 더 잘 보여주는 지표.
- 슬롯 중 `층조건`, `근접시설`은 문장에서 명시적으로 언급되지 않는 경우가 많아 다른 슬롯보다 정확도가 낮게 나올 수 있음.

결과 
============================================================  
슬롯별 평가 결과(현재까지 쌓인 데이터 15개 기준, gemini-flash-lite-latest)    
============================================================  
                                        slot  accuracy  precision_macro  recall_macro  f1_macro  n_samples  
                                          지역       1.0              1.0           1.0       1.0         15  
                                        거래유형       1.0              1.0           1.0       1.0         15  
                                        주거유형       1.0              1.0           1.0       1.0         15  
                                         층조건       1.0              1.0           1.0       1.0         15  
                                        근접시설       1.0              1.0           1.0       1.0         15  
                                       가격_최소       1.0              1.0           1.0       1.0         15  
                                       가격_최대       1.0              1.0           1.0       1.0         15  
                                       면적_최소       1.0              1.0           1.0       1.0         15  
ALL_SLOTS_EXACT_MATCH (엄격: 8개 슬롯 전부 일치해야 정답)       1.0              NaN           NaN       NaN         15  
  AVERAGE_PER_SLOT_ACCURACY (관대: 슬롯별 정확도 평균)       1.0              NaN           NaN       NaN         15  

[저장 완료] 개별 결과 -> eval_results.csv  
[저장 완료] 평가 요약 -> eval_summary.csv  

## 알려진 제약사항

- Gemini API 무료 티어는 분당/일일 요청 횟수 제한이 있어, 대량의 테스트를 한 번에 돌리기 어려움.
- 부정 표현("반지하 말고" 등)은 현재 슬롯 스키마에서 별도로 처리하지 않음 (v2 확장 과제로 `slot_schema.json`에 명시).
