.PHONY: verify lint types test

## CI가 도는 것과 같다
verify: lint types test

lint:
	ruff check src tests
	ruff format --check src tests

types:
	mypy src

test:
	pytest -q
