"""패키지가 설치되고 각 구역이 임포트되는지만 본다."""

import importlib

import pytest

MODULES = [
    "family_fitness_ai",
    "family_fitness_ai.api",
    "family_fitness_ai.graph",
    "family_fitness_ai.rag",
    "family_fitness_ai.stats",
    "family_fitness_ai.ingest",
    "family_fitness_ai.labeling",
    "family_fitness_ai.common",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    assert importlib.import_module(name) is not None
