.PHONY: verify lint types test tables clips clip-labels medical serve probe

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

## 의료 질의를 얼마나 맞게 가리는지 잰다 (오탐·누락).
medical:
	$(PY) -m family_fitness_ai.rag.medical

## ── 띄우고 보기 ────────────────────────────────────────────────────────
## coach/messages 와 검색을 쓰려면 임베딩 서버도 띄운다:
##   llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082
serve:
	$(PY) -m uvicorn family_fitness_ai.api.app:app --reload --port 8000

## 돌고 있는 서비스에 요청을 보내 눈으로 확인한다. `make serve` 를 먼저 띄운다.
probe:
	$(PY) scripts/probe.py
