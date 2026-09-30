"""운영에서 띄우는 명령은 워커 하나로 띄운다.

실행(/v1/coach/runs)은 그 프로세스 메모리(coach/runs.py Store)에만 있다. 워커가 둘이면
폴링이 다른 워커로 가 404 run_not_found 가 나고, BE 는 대체 편성으로 빠진다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _recipe(target: str) -> str:
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{target}:")
    return "\n".join(line for line in lines[start + 1 :] if line.startswith("\t")).split("\n")[0]


def test_serve_prod_pins_one_worker() -> None:
    assert "--workers 1" in _recipe("serve-prod")


def test_readme_run_command_pins_one_worker() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--port 8000 --workers 1" in readme
