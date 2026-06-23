# 로컬 LLM 기반 금융 의사결정 지원 시스템 (LLM vs RAG vs Agent)

## 프로젝트 배경

산업공학 데이터사이언스 분야에서 최근 핵심 화두인 **LLM과 Agentic AI**를 직접 다뤄보기 위해 시작한 프로젝트입니다. Linux OS에서 Ollama + Gemma 3를 구축하여, 금융 의사결정 상황에서 **LLM 단독 / RAG / Agent** 세 가지 방식의 정확도와 신뢰성을 비교 분석한 프로젝트입니다.

단순히 클라우드 API를 호출하는 것이 아니라, 로컬 환경에 LLM 서버를 직접 구축하여 인프라 설계부터 실험까지 전 과정을 다뤘다는 점에 초점을 맞췄습니다.

## 시스템 구성

| 구성 요소 | 사용 기술 |
|---|---|
| 실행 환경 | Ubuntu Linux 24.04 |
| LLM 서버 | Ollama |
| LLM 모델 | Gemma 3 4B |
| RAG 프레임워크 | LangChain + FAISS |
| 임베딩 모델 | snunlp/KR-SBERT-V40K-klueNLI-augSTS |
| Agent | ReAct 패턴 구현 (재무데이터검색 Tool + 웹검색 Tool) |
| 데이터 출처 | 금융감독원 Open DART OpenAPI |

## 데이터

- 대상 기업: 삼성전자, SK하이닉스, 현대자동차, LG에너지솔루션, NAVER (5개사)
- 기간: 2022~2024년 사업보고서 (연결재무제표 기준)
- 항목: 매출액, 영업이익, 자산총계, 부채총계, 자본총계 등 14개 재무 항목

> 본 프로젝트의 재무 데이터는 금융감독원 Open DART(https://opendart.fss.or.kr) OpenAPI를 통해 수집되었습니다. 공시정보는 공시제출 기업의 책임 하에 작성된 것으로, 본 프로젝트는 학습 및 연구 목적으로 이를 가공하여 분석에 활용하였습니다.

## 프로젝트 구조

```
financial_rag_agent/
├── README.md
├── .gitignore
│
├── data/
│   ├── .json                       # DART 원본 공시 데이터 (5개 기업 × 3개년)
│   └── faiss_index/                # FAISS 벡터 DB
│
├── 1_data/
│   ├── get_corp_codes.py           # 기업 고유번호 전체 목록 다운로드
│   ├── find_corp_code.py           # 분석 대상 기업 코드 조회
│   └── collect_reports.py          # DART API로 재무 데이터 수집
│
├── 2_RAG/
│   ├── build_vectorstore.py        # 벡터 DB 구축 (회사명/연도/항목명 메타데이터 포함)
│   └── test_retrieval.py           # 검색 동작 테스트
│
├── 3_agent/
│   └── agent_answer.py             # ReAct 패턴 구현 (RAG Tool + 웹검색 Tool)
│
├── 4_evaluate/
│   ├── final_comparison.py         # LLM 단독 / RAG / Agent 3단계 비교 실행
│   ├── evaluate_final.py           # 재무수치 질문 자동 채점
│   └── view_final.py               # 결과 확인용 출력 스크립트
│
└── results/
    ├── comparison__scored.csv           # 1차 실험 결과 (LLM vs RAG, 15문항)
    └── final_comparison__scored.csv     # 2차 실험 결과 (LLM vs RAG vs Agent, 10문항)
```
## 실험 설계

같은 질문을 세 가지 조건에 동일하게 입력하여 비교했습니다.

```
조건 1: LLM 단독       질문 → Gemma 3 → 답변
조건 2: LLM + RAG      질문 → DART 데이터 검색 → Gemma 3 → 답변
조건 3: LLM + RAG + Agent   질문 → Agent가 Tool 선택(재무데이터검색/웹검색) → 답변
```
질문은 두 가지 유형으로 구성했습니다.

- **재무수치 질문**: DART에 존재하는 정형 데이터로 답할 수 있는 질문 (예: "2024년 삼성전자 매출액이 얼마야?")
- **뉴스/동향 질문**: DART에는 없는 최신 정보가 필요한 질문 (예: "삼성전자 최근 반도체 업황 관련 뉴스 알려줘")

## 실험 결과

### 1차 실험 — 재무수치 질문 15개 (LLM 단독 vs RAG)

| 조건 | 정답 수 | 정확도 |
|---|---|---|
| LLM 단독 | 0/15 | 0% |
| RAG | 15/15 | 100% |

### 2차 실험 — 재무수치 질문 5개 (LLM 단독 vs RAG vs Agent)

| 조건 | 정답 수 | 정확도 |
|---|---|---|
| LLM 단독 | 0/5 | 0% |
| RAG | 5/5 | 100% |
| Agent | 5/5 | 100% |

### 2차 실험 — 뉴스/동향 질문 5개 (정성 평가)

| 조건 | 특징 |
|---|---|
| LLM 단독 | 그럴듯하지만 부정확한 정보 생성 (예: 존재하지 않는 출처 URL, 미묘하게 틀린 고유명사 — 예: "클로바"를 "클로베"로 잘못 표기) |
| RAG | "제공된 정보에 없습니다"라고 정직하게 한계를 인정 |
| Agent | 웹검색 Tool을 활용해 실제 검증 가능한 구체적 최신 정보 제공 (예: 실제 데이터센터명, 구체적 발표 내용) |

## 개발 과정에서 발견한 주요 이슈와 해결

프로젝트의 핵심은 단순히 시스템을 구현한 것이 아니라, 구현 과정에서 발견한 문제들을 진단하고 해결한 과정입니다.

| 문제 | 원인 | 해결 방법 |
|---|---|---|
| 검색 정확도 저하 | 여러 재무 항목이 한 청크에 뭉쳐서 임베딩됨 | 사실(fact) 단위로 청크를 분리 |
| 다른 기업 데이터가 섞여서 검색됨 | 템플릿화된 짧은 문장에서 일반 STS 임베딩 모델이 기업명을 잘 구분하지 못함 | 회사명/항목명을 메타데이터로 저장하고 필터링 검색 적용 |
| 필터링 조건에 맞는 문서가 누락됨 | FAISS의 `fetch_k`(필터링 전 후보군 크기) 기본값이 작아 후보군에서 제외됨 | `fetch_k` 값을 전체 문서 수 이상으로 확대 |
| Agent가 Tool 실행 없이 결과를 지어냄 | LLM이 Stop 없이 Action/Observation까지 한 번에 생성 | LLM 호출 시 `stop=["\nObservation:"]` 지정 |
| Agent가 Action 없이 바로 답변함 | 소형 모델(4B)이 복잡한 형식 지시를 안정적으로 따르지 못함 | Few-shot 예시 1개와 명시적 금지 문구를 프롬프트에 추가 |

## 한계

- 5개 기업, 15개(1차) / 10개(2차) 질문이라는 제한된 규모로 진행되었으며, 이는 통계적 일반화가 아닌 기술적 의사결정과 그 효과 검증에 초점을 맞췄기 때문입니다.
- 질문 유형이 주로 정형화된 재무 수치 확인에 집중되어 있어, RAG의 효과가 가장 두드러지게 나타나는 영역을 다뤘다는 한계가 있습니다.
- 단일 임베딩 모델(KR-SBERT), 단일 LLM 크기(Gemma 3 4B)로만 검증했습니다.
- 멀티 에이전트 구조는 고려했으나, 하드웨어 제약(CPU only, 8GB RAM)과 과제 복잡도를 고려하여 단일 Agent + Tool 2개로 범위를 한정했습니다.

## 향후 과제

- 더 다양한 기업/질문 유형으로 실험 확장
- 여러 임베딩 모델 및 LLM 크기 비교 (예: Gemma 3 4B vs 12B)
- 하이브리드 검색(BM25 + 임베딩) 비교 실험
- 멀티 에이전트 구조 적용 가능성 검토

## 실행 방법

```bash
# 1. Ollama 설치 및 모델 다운로드
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b

# 2. 가상환경 및 패키지 설치
python3 -m venv rag_env
source rag_env/bin/activate
pip install langchain langchain-ollama langchain-community faiss-cpu sentence-transformers ddgs

# 3. 데이터 수집 (DART API 키 필요)
export DART_API_KEY="발급받은키"
python 1_data/get_corp_codes.py
python 1_data/collect_reports.py

# 4. 벡터 DB 구축
python 2_RAG/build_vectorstore.py

# 5. Agent 구축
python 3_agent/agent_answer.py

# 6. 비교 실험 실행
python 4_evaluate/final_comparison.py
python 4_evaluate/evaluate_final.py
```

## 기술 스택

Python · LangChain · FAISS · Ollama · Gemma 3 · DART OpenAPI · Linux (Ubuntu)
