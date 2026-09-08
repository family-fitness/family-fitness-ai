.PHONY: verify lint types test distribution

## CI가 도는 것과 같다
verify: lint types test

lint:
	ruff check src tests
	ruff format --check src tests

types:
	mypy src

test:
	pytest -q

## 원자료 → 연령 구간별 점수 분포. DATA_DIR 은 zip 을 푼 디렉터리다.
DATA_DIR ?= data/raw
distribution:
	python -m family_fitness_ai.stats.distribution --data-dir "$(DATA_DIR)"
