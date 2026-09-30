.PHONY: verify lint types test tables ocr-name clips clip-labels kspo medical embed-model serve serve-prod probe

## 가상환경을 켜 두었으면 그대로, 아니면 `make verify PY=.venv/bin/python`.
PY ?= python

## CI 가 도는 것과 같다
verify: lint types test

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

types:
	$(PY) -m mypy src scripts

test:
	$(PY) -m pytest -q

## ── 표 만들기 ──────────────────────────────────────────────────────────
## 서비스는 원자료를 읽지 않는다. 여기서 만든 data/release 표만 읽는다.
## 차례가 있다: tables → clips → clip-labels.

DATA_DIR ?= data/raw

## 원자료 → 또래 분포·인증 기준·등급 분포 세 장.
## 백분위·점수·등급·궤적이 전부 여기서 나온다. 몇 분 걸린다.
tables:
	$(PY) -m family_fitness_ai.stats.build --data-dir "$(DATA_DIR)"

## 제목·자막 칸에 운동 이름이 안 걸리는 영상만 이름표 자리를 따로 읽는다.
## 성인 「4주 프로그램」이 그렇다 — 이름표가 왼쪽 아래에 뜨는데 자막 칸이 비껴간다.
## 받아 둔 프레임을 읽을 뿐 유튜브에 다시 가지 않는다. 영상 하나에 30초쯤.
ADULT_PROGRAM ?= IhShIA-WJNE yrxN8UyUXoc PTxUuItSAtY lVd366kW7KI
ocr-name:
	$(PY) -m family_fitness_ai.video.ocr $(ADULT_PROGRAM)

## 읽어 둔 화면 글자 → 클립(시작·끝이 있는 한 동작).
## 영상 한 편에 운동이 여럿이라 통째로는 못 쓴다. 유튜브에 다시 가지 않는다.
clips:
	$(PY) -m family_fitness_ai.video.clips

## 클립 이름 → 처방 어휘·체력요인·단계·조건.
## 글자가 같으면 그대로, 임베딩이 0.90 넘으면 그것으로, 그 아래는 LLM 이 후보
## 중에서 고른다(--llm). 이미 붙여 둔 이름은 다시 묻지 않는다.
## 사람이 고친 줄(source=human)은 --refresh 를 줘도 지킨다.
clip-labels:
	$(PY) -m family_fitness_ai.video.labels --llm

## 국민체력100 동영상(공공데이터) → data/release/kspo_videos.csv.
## 한 편이 한 동작이라 끊지 않고, 연령대·요인·수준·도구를 표에서 그대로 옮긴다.
## 받아 둔 원자료(data/raw/kspo)가 있으면 API 를 부르지 않는다. 다시 받으려면
## `make kspo KSPO_ARGS=--fetch` (DATA_GO_KR_KEY 가 있어야 한다).
kspo:
	$(PY) -m family_fitness_ai.video.kspo $(KSPO_ARGS)

## 의료 질의를 얼마나 맞게 가리는지 잰다 (오탐·누락).
medical:
	$(PY) -m family_fitness_ai.rag.medical

## ── 띄우고 보기 ────────────────────────────────────────────────────────
## 임베딩 모델(bge-m3-Q8_0.gguf, 605 MB)을 한 번 받아 두고, 인덱스에 든 벡터와
## 같게 나오는지 잰다. 받아 두면 serve 는 임베딩 서버 없이 뜬다.
## 밖에 띄운 서버를 쓰려면 .env 에 EMBEDDING_BACKEND=http.
embed-model:
	$(PY) -m family_fitness_ai.rag.embed

## 개발용. 코드를 고칠 때마다 다시 뜨니 모델은 처음 쓸 때 올린다(EMBEDDING_WARMUP=0).
serve:
	EMBEDDING_WARMUP=0 $(PY) -m uvicorn family_fitness_ai.api.app:app --reload --port 8000

## 운영용. --reload 없이 띄우고, 모델은 띄울 때 올린다(EMBEDDING_WARMUP 기본 켜짐).
## 인증이 없으니 AI_HOST 는 BE 만 닿는 주소로 둔다 — 공개 포트에 붙이지 않는다.
## 워커는 하나다. 실행(/v1/coach/runs)은 프로세스 메모리에만 있어서, 워커가 둘이면
## BE 의 폴링이 다른 워커로 가 404 run_not_found 가 나고 BE 는 대체 편성으로 빠진다.
AI_HOST ?= 127.0.0.1
AI_PORT ?= 8000
serve-prod:
	$(PY) -m uvicorn family_fitness_ai.api.app:app --host $(AI_HOST) --port $(AI_PORT) --workers 1

## 돌고 있는 서비스에 요청을 보내 눈으로 확인한다. `make serve` 를 먼저 띄운다.
probe:
	$(PY) scripts/probe.py
