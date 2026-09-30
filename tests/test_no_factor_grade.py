"""요인별 등급이 화면에 나가지 않는다.

화면에 나오는 등급은 국민체력100 등급 카드(한 사람에 하나)뿐이다. 처방표는 요인마다
「심폐지구력 2등급」 칸으로 나뉘어 있어, 칸 이름을 그대로 쓰면 인용 이름 · 코치
설명에 요인별 등급이 보였다. 인덱스는 그대로 두고 읽을 때 바꾼다.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.coach import answer as coach_answer
from family_fitness_ai.coach import compose
from family_fitness_ai.coach import llm as coach_llm
from family_fitness_ai.common import copy as words
from family_fitness_ai.rag import index
from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import Hit, Result

_GRADE = re.compile(r"\d\s*등급|참가|미달")


def _prescription(grade: str, sex: str = "M") -> Chunk:
    shown = f"{grade}등급" if grade.isdigit() else "참가"
    return Chunk(
        chunk_id=f"prescription:유소년-11-{sex}-심폐지구력-{grade}",
        source="prescription",
        text=index._without_grade(
            "prescription",
            f"유소년 11세 남자 중 심폐지구력 {shown}인 1,170명에게 처방된 본운동: "
            "왕복달리기(6%), 1단 줄넘기(4%), 팔굽혀펴기(3%)",
        ),
        citation_label=index._without_grade_label(
            "prescription", f"국민체력100 운동처방 · 유소년 11세 심폐지구력 {shown}"
        ),
        citation_url="",
        age_group="유소년",
        factors=("심폐지구력",),
        grade=grade,
    )


# ── 인용 이름 · 청크 글 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "국민체력100 운동처방 · 유소년 11세 심폐지구력 2등급",
            "국민체력100 운동처방 · 심폐지구력이 비슷한 유소년 11세",
        ),
        (
            "국민체력100 운동처방 · 유아기 60개월 협응력 참가",
            "국민체력100 운동처방 · 협응력이 비슷한 유아기 60개월",
        ),
        # 요인을 가리지 않은 칸은 등급도 없다 — 그대로 둔다.
        ("국민체력100 운동처방 · 성인 19세", "국민체력100 운동처방 · 성인 19세"),
    ],
)
def test_the_citation_label_names_no_grade(before: str, after: str):
    assert index._without_grade_label("prescription", before) == after


def test_the_chunk_text_says_similar_level_instead_of_the_grade():
    text = index._without_grade(
        "prescription",
        "성인 19세 여자 중 근력 1등급인 2,826명에게 처방된 본운동: 달리기(9%), 스쿼트 1등급(1%)",
    )
    assert text.startswith("성인 19세 여자 중 근력 수준이 비슷한 2,826명에게 처방된 본운동: ")
    # 글머리만 바꾼다. 운동 목록은 자료 그대로다.
    assert text.endswith("스쿼트 1등급(1%)")


def test_other_sources_keep_their_grades():
    """인증 기준은 국민체력100 등급 그 자체의 기준표다. 요인별 등급이 아니다."""
    text = "국민체력100 인증 기준 · 윗몸말아올리기(회) · 유소년 여자: 11~11세 1등급 36.0회"
    assert index._without_grade("criteria", text) == text


def test_the_subject_particle_follows_the_last_sound():
    assert words.with_subject("유연성") == "유연성이"
    assert words.with_subject("자세") == "자세가"


@needs_index
def test_no_loaded_prescription_shows_a_grade():
    index.corpus.cache_clear()
    try:
        prescriptions = [c for c in index.corpus().chunks if c.source == "prescription"]
    finally:
        index.corpus.cache_clear()
    assert prescriptions
    for chunk in prescriptions:
        assert not _GRADE.search(chunk.citation_label), chunk.citation_label
        assert not _GRADE.search(chunk.text.split(":", 1)[0]), chunk.text[:80]


# ── 편성: 이름이 같은 처방은 인용 하나 ──────────────────────────────────────


def test_prescriptions_with_the_same_label_share_one_citation():
    """2등급 칸과 3등급 칸은 이제 이름이 같다. 목록에 같은 이름이 둘 나오지 않게 합친다."""
    citations = compose.Citations()
    first = citations.add(_prescription("2"))
    second = citations.add(_prescription("3"))
    assert first == second == 1
    assert [c["label"] for c in citations.dump()] == [
        "국민체력100 운동처방, 심폐지구력이 비슷한 유소년 11세"
    ]


# ── 편성: 코치가 등급을 쓰면 그 칸은 규칙 문구 ─────────────────────────────


def test_the_plan_prompt_forbids_factor_grades():
    for system in (coach_llm.PLAN_SYSTEM, coach_llm.SYSTEM, coach_llm.ANSWER_SYSTEM):
        assert "요인별 등급" in system


def test_a_field_naming_a_grade_falls_back():
    tally = compose.Tally()
    text = compose._checked_text(
        {
            "title": "유연성 기르기",
            "child": "몸을 늘여 볼까요",
            "parent": "하루 15분이면 충분합니다",
            "reason": "유소년 심폐지구력 2, 3등급에서 가장 처방되는 동작입니다 [1].",
        },
        {"title": "T", "child": "C", "parent": "P", "reason": "R"},
        set(),
        None,
        tally,
    )
    assert text == {
        "title": "유연성 기르기",
        "child": "몸을 늘여 볼까요",
        "parent": "하루 15분이면 충분합니다",
        "reason": "R",
    }
    assert tally.rewritten == 1


def test_the_copy_writer_rejects_a_grade():
    row = {"title": "유연성 기르기", "child": "같이 해 볼까요", "parent": "2등급 수준입니다"}
    assert not coach_llm._acceptable(row)


@needs_release
@needs_index
def test_rule_plan_text_names_no_grade():
    child = compose.RunProfile(ref="p", role="주행자", age=11, age_unit="세", sex="M")
    plan = compose.build([child], date(2026, 10, 5), 1, compose.Constraints())
    assert plan.proposal is not None
    for citation in plan.proposal["citations"]:
        assert not _GRADE.search(str(citation["label"])), citation["label"]
    for mission in plan.proposal["missions"]:
        for text in (mission["title"], mission["reason"], *mission["copy"].values()):
            assert not re.search(r"\d\s*등급", text), text


# ── 질문 답 ──────────────────────────────────────────────────────────────────


@pytest.fixture
def two_grades(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[int, str]]]:
    """검색이 같은 이름이 된 처방 두 칸을 찾는다. LLM 에 넘긴 근거를 적어 둔다."""
    result = Result(
        hits=[Hit(_prescription("2"), 0.8), Hit(_prescription("3"), 0.7)], filtered_out={}
    )
    monkeypatch.setattr(coach_answer, "search", lambda *a, **k: result)
    handed: list[list[tuple[int, str]]] = []

    def write_answer(question: str, passages: list[tuple[int, str]]) -> str:
        handed.append(passages)
        return "심폐지구력 2등급인 아이들이 받은 본운동 처방에는 왕복달리기가 많았습니다 [1]."

    monkeypatch.setattr(coach_answer.copywriter, "write_answer", write_answer)
    return handed


def test_the_answer_merges_citations_with_the_same_label(two_grades):
    result = coach_answer.answer("심폐지구력 운동", "유소년")
    assert [c["label"] for c in result["citations"]] == [
        "국민체력100 운동처방, 심폐지구력이 비슷한 유소년 11세"
    ]
    (passages,) = two_grades
    assert [number for number, _ in passages] == [1]


def test_an_answer_naming_a_grade_the_sources_do_not_falls_back_to_the_rule(two_grades):
    written = str(coach_answer.answer("심폐지구력 운동", "유소년")["answer"])
    assert not re.search(r"\d\s*등급", written), written
    assert "심폐지구력 수준이 비슷한 1,170명에게 처방된 본운동" in written
