.PHONY: verify lint types test wire-check distribution prescription-cells exercises segments missions loadout medical serve probe

## CI가 도는 것과 같다
verify: lint types test wire-check

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

types:
	mypy src scripts

test:
	pytest -q

## 우리가 내는 이름이 백엔드 AiWire.kt 와 같은지 본다.
## 백엔드 저장소가 없으면 skip 하고 통과한다 — CI 에서도 돈다.
wire-check:
	python scripts/wire_check.py

## 원자료 → 연령 구간별 점수 분포. DATA_DIR 은 zip 을 푼 디렉터리다.
DATA_DIR ?= data/raw
distribution:
	python -m family_fitness_ai.stats.distribution --data-dir "$(DATA_DIR)"

## 원자료 → data/release/prescription_cells.csv (처방 칸 708행).
## distribution 과 따로 돈다 — 처방 컬럼만 읽고 기준표를 쓰지 않는다.
prescription-cells:
	python -m family_fitness_ai.mission.cells --data-dir "$(DATA_DIR)"

## data/release/exercises.csv — 미션의 운동 축 (①).
## 합치는 근거는 「같은 처방 칸에 함께 나오지 않는다」다. 어휘·처방 칸이 먼저 있어야 한다.
exercises:
	python -m family_fitness_ai.mission.exercises

## data/release/video_segments.csv 와 구간 통계.
## data/interim/video_labeling.csv (labeling.label) 와 청크가 먼저 있어야 한다.
segments:
	python -m family_fitness_ai.mission.segments

## data/release/missions.csv — **추천의 검색 대상** (미션 = 운동 × 영상 구간 × 라벨).
## prescription-cells 와 segments 가 먼저 있어야 한다.
missions:
	python -m family_fitness_ai.mission.build

## data/release/video_labels.csv 와 멱등 SQL. 백엔드 담당이 그 SQL 을 넣는다.
## 운영의 exercise_videos 가 비어 있으면 미션의 video 가 항상 null 이다.
loadout:
	python -m family_fitness_ai.mission.loadout

## 의료 질의 낱말 목록과 오탐·누락 측정 (docs/04 §4.1).
medical:
	python -m family_fitness_ai.rag.medical

## 서비스를 띄운다. coach/messages 를 쓰려면 임베딩 서버도 띄운다:
##   llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082
serve:
	uvicorn family_fitness_ai.api.app:app --reload --port 8000

## 돌고 있는 서비스에 요청을 보내 눈으로 확인한다. `make serve` 를 먼저 띄운다.
## 인자를 주려면 스크립트를 직접 부른다 — `python scripts/probe.py --help`
probe:
	python scripts/probe.py
