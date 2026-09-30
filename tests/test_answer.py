"""질문에 답하는 쪽. 검색은 흉내 내고, 답이 근거가 말한 만큼만 말하는지 본다."""

from __future__ import annotations

import pytest

from family_fitness_ai.coach import answer as coach_answer
from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import Hit, Result

#: 처방 청크는 「그 수준인 사람들이 받은 본운동 처방 전체」다. 유연성 운동 목록이 아니다.
#: 글머리와 인용 이름은 요인별 등급을 걷어 낸 모습이다(rag.index 가 읽을 때 바꾼다).
PRESCRIPTION = Chunk(
    chunk_id="prescription:유소년-12-M-유연성-1",
    source="prescription",
    text=(
        "유소년 12세 남자 중 유연성 수준이 비슷한 1,170명에게 처방된 본운동: "
        "왕복달리기(6%), 엎드려 팔 대고 버티기(4%), 1단 줄넘기(4%), 팔굽혀펴기(3%)"
    ),
    citation_label="국민체력100 운동처방 · 유연성이 비슷한 유소년 12세",
    citation_url="",
    age_group="유소년",
    factors=("유연성",),
    grade="1",
)
CRITERIA = Chunk(
    chunk_id="criteria:유소년-유연성",
    source="criteria",
    text="유소년 유연성은 앉아 윗몸 앞으로 굽히기로 잰다",
    citation_label="국민체력100 인증 기준 · 유소년",
    citation_url="",
    age_group="유소년",
    factors=("유연성",),
    grade="",
)


@pytest.fixture
def found(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[int, str]]]:
    """검색은 처방 한 건과 기준 한 건을 찾는다. LLM 에 넘긴 근거를 적어 둔다."""
    result = Result(hits=[Hit(PRESCRIPTION, 0.8), Hit(CRITERIA, 0.7)], filtered_out={})
    monkeypatch.setattr(coach_answer, "search", lambda *a, **k: result)
    handed: list[list[tuple[int, str]]] = []

    def write_answer(question: str, passages: list[tuple[int, str]]) -> None:
        handed.append(passages)
        return None  # LLM 이 꺼진 것처럼 — 규칙 문장으로 내려간다

    monkeypatch.setattr(coach_answer.copywriter, "write_answer", write_answer)
    return handed


QUESTION = "유소년 유연성을 기르는 운동이 궁금해요"


def test_the_llm_is_told_a_prescription_is_not_a_list_for_that_factor(found):
    coach_answer.answer(QUESTION, "유소년")
    (passages,) = found
    prescription, criteria = (text for _, text in passages)
    assert prescription.startswith(PRESCRIPTION.text)
    assert "처방 전체" in prescription
    assert "유연성만을 위한 운동 목록이 아니다" in prescription
    assert criteria == CRITERIA.text


def test_the_rule_sentence_says_who_received_the_prescription(found):
    written = coach_answer.answer(QUESTION, "유소년")["answer"]
    assert "유연성 수준이 비슷한 1,170명에게 처방된 본운동" in written
    assert "왕복달리기" in written
    assert "기르" not in written and "위해" not in written


def test_the_answer_prompt_forbids_adding_an_effect():
    from family_fitness_ai.coach import llm

    assert "처방된 운동" in llm.ANSWER_SYSTEM
    assert "효과" in llm.ANSWER_SYSTEM
