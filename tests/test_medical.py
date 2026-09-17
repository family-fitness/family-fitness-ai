import csv
from pathlib import Path

import pytest

from family_fitness_ai.rag.medical import is_medical

FIXTURE = Path(__file__).parent / "fixtures" / "medical_queries.csv"


def _queries():
    with FIXTURE.open(encoding="utf-8") as fh:
        return [(row["question"], row["medical"] == "1") for row in csv.DictReader(fh)]


@pytest.mark.parametrize(("question", "medical"), _queries())
def test_medical_queries(question: str, medical: bool):
    assert is_medical(question) is medical
