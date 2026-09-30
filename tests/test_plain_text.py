"""응답 글에 가운데 점과 긴 대시가 남지 않는지.

사용자 결정이다. 사이트 안의 글에서 가운데 점(·)과 긴 대시(—, –)를 없앤다. AI 가 쓴
티가 난다. AI 가 만든 글은 사이트에 그대로 나가서 여기서 먼저 걷어 낸다. 코치(LLM)가
섞어 돌려줘도, 원자료(공단 표, 처방 표)의 이름에 들어 있어도 내보내기 전에 바꾼다.

LLM 은 부르지 않는다. 가짜 코치가 가운데 점과 대시를 일부러 섞어 돌려준다.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from conftest import needs_embedder, needs_index, needs_release

from family_fitness_ai.coach import answer as coach_answer
from family_fitness_ai.coach import compose, runs
from family_fitness_ai.common import copy as words
from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import Hit, Result

#: 걸러야 하는 글자. 한글 자판의 「ㆍ」 와 일본식 「・」 도 화면에서는 같은 점이다.
MARKS = "·ㆍ・‧∙—–―‒"

#: 글이 아니라 식별자인 칸. 여기 든 값은 바꾸지 않는다.
IDS = {"url", "video_id", "chunk_id", "ref", "profile_ref", "run_id", "source"}


def _texts(value: Any, key: str = "") -> list[str]:
    if isinstance(value, str):
        return [] if key in IDS else [value]
    if isinstance(value, dict):
        return [text for k, v in value.items() for text in _texts(v, str(k))]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _texts(item, key)]
    return []


def _marked(value: Any) -> list[str]:
    return [text for text in _texts(value) if any(mark in text for mark in MARKS)]


# ── 바꾸는 규칙 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("국민체력100 운동처방동영상 · 걷기", "국민체력100 운동처방동영상, 걷기"),
        ("근력·유연성", "근력, 유연성"),
        ("진단ㆍ치료", "진단, 치료"),
        ("7–10세", "7~10세"),
        ("7 — 10세", "7~10세"),
        ("편성은 끝났다 — 너는 말만 쓴다", "편성은 끝났다, 너는 말만 쓴다"),
        ("오늘은 쉬어요 —", "오늘은 쉬어요"),
        ("— 오늘은 쉬어요", "오늘은 쉬어요"),
        ("늘여 봐요 — [1].", "늘여 봐요 [1]."),
        ("T-W-Y-A 동작", "T-W-Y-A 동작"),
        ("", ""),
    ],
)
def test_plain_replaces_dots_and_dashes(before: str, after: str) -> None:
    assert words.plain(before) == after


def test_plain_leaves_nothing_behind() -> None:
    messy = "A · B ㆍ C ・ D — E – F ― G ‒ H 3–5회 · [2]"
    assert not any(mark in words.plain(messy) for mark in MARKS)


# ── 코치에게 주는 지시문 ────────────────────────────────────────────────────


def test_the_coach_is_told_not_to_use_dots_or_dashes() -> None:
    from family_fitness_ai.coach import llm

    for prompt in (llm.PLAN_SYSTEM, llm.ANSWER_SYSTEM, llm.SYSTEM):
        assert "가운데 점" in prompt and "긴 대시" in prompt
        # 지시문이 스스로 쓰면 코치가 따라 쓴다.
        assert not any(mark in prompt for mark in MARKS), prompt


# ── 고정 문구 ──────────────────────────────────────────────────────────────


def test_fixed_copy_has_no_dots_or_dashes() -> None:
    fixed = [
        words.DISCLAIMER,
        words.TRAJECTORY_NOTICE,
        *words.BAND_COPY.values(),
        *words.FOCUS_COPY.values(),
    ]
    assert _marked(fixed) == []


# ── 질문 답 ────────────────────────────────────────────────────────────────

PRESCRIPTION = Chunk(
    chunk_id="prescription:유소년-12-M-유연성-1",
    source="prescription",
    text="유소년 12세 남자 중 유연성 수준이 비슷한 1,170명에게 처방된 본운동: 왕복달리기(6%)",
    citation_label="국민체력100 운동처방 · 유연성이 비슷한 유소년 12세",
    citation_url="",
    age_group="유소년",
    factors=("유연성", "근력"),
    grade="1",
)
VIDEO = Chunk(
    chunk_id="video:abc",
    source="video",
    text="국민체력100 운동영상 · 몸풀기 · 연령대 유소년 · 나오는 운동: 나비자세(02:24)",
    citation_label="국민체력100 운동영상 · 몸풀기",
    citation_url="",
    age_group="유소년",
    factors=(),
    grade="",
)


@pytest.fixture
def searched(monkeypatch: pytest.MonkeyPatch) -> None:
    result = Result(hits=[Hit(PRESCRIPTION, 0.8), Hit(VIDEO, 0.7)], filtered_out={})
    monkeypatch.setattr(coach_answer, "search", lambda *a, **k: result)


def test_the_coachs_answer_loses_its_dots_and_dashes(
    searched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        coach_answer.copywriter,
        "write_answer",
        lambda question, passages: "왕복달리기 · 줄넘기가 많았습니다 — 7–10세도 같아요 [1].",
    )
    body = coach_answer.answer("유연성 운동이 궁금해요", "유소년")
    assert body["refused"] is False
    assert body["answer"] == "왕복달리기, 줄넘기가 많았습니다, 7~10세도 같아요 [1]."
    assert _marked(body) == []


def test_the_rule_answer_and_citation_names_have_no_dots(
    searched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coach_answer.copywriter, "write_answer", lambda q, p: None)
    body = coach_answer.answer("유연성 운동이 궁금해요", "유소년")
    assert body["refused"] is False
    assert _marked(body) == []
    labels = [citation["label"] for citation in body["citations"]]  # type: ignore[union-attr]
    assert labels == [
        "국민체력100 운동처방, 유연성이 비슷한 유소년 12세",
        "국민체력100 운동영상, 몸풀기",
    ]


def test_the_passage_names_factors_without_a_dot() -> None:
    assert "유연성, 근력만을 위한 운동 목록이 아니다" in coach_answer._passage(PRESCRIPTION)


# ── 한 주 편성 ───────────────────────────────────────────────────────────────

CHILD = compose.RunProfile(
    ref="p_c7a91f",
    role="주행자",
    age=11,
    age_unit="세",
    sex="F",
    input_level="L2",
    measurements={"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
)
EIGHT = compose.RunProfile(ref="p_8", role="주행자", age=8, age_unit="세", sex="M")


def _messy_coach(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """가운데 점과 대시를 칸마다 섞어 돌려준다."""
    days = []
    for slot in payload["자리"]:
        used: set[str] = set()
        clips = []
        for clip in payload["클립"]:
            if clip["이름"] in used or clip["영상"] in used:
                continue
            used |= {clip["이름"], clip["영상"]}
            clips.append({"id": clip["id"], "phase": clip["단계"]})
        days.append(
            {
                "day_offset": slot["day_offset"],
                "title": "늘이기·버티기",
                "child": "몸을 늘여 볼까요 — 천천히요",
                "parent": "하루 15분 — 3–5회면 됩니다",
                "reason": "또래 처방 · 근거에 나온 동작입니다 [1].",
                "clips": clips[:7],
            }
        )
    return days


@needs_release
@needs_index
@needs_embedder
def test_the_coachs_plan_loses_its_dots_and_dashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", _messy_coach)
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints(weekly_minutes=30))
    assert plan.proposal is not None
    mission = plan.proposal["missions"][0]
    assert mission["title"] == "늘이기, 버티기"
    assert mission["copy"]["child"] == "몸을 늘여 볼까요, 천천히요"
    assert mission["copy"]["parent"] == "하루 15분, 3~5회면 됩니다"
    assert mission["reason"] == "또래 처방, 근거에 나온 동작입니다 [1]."
    assert _marked(plan.proposal) == []
    assert _marked([step.dict() for step in plan.steps]) == []


@needs_release
@needs_index
@needs_embedder
@pytest.mark.parametrize("profile", [CHILD, EIGHT])
def test_the_rule_plan_notices_and_citations_have_no_dots(
    monkeypatch: pytest.MonkeyPatch, profile: compose.RunProfile
) -> None:
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: False)
    plan = compose.build([profile], date(2026, 9, 7), 1, compose.Constraints(weekly_minutes=30))
    assert plan.proposal is not None
    assert plan.proposal["citations"]
    assert _marked(plan.proposal) == []
    assert _marked([step.dict() for step in plan.steps]) == []


@needs_release
@needs_index
@needs_embedder
def test_kspo_citation_names_use_a_comma(monkeypatch: pytest.MonkeyPatch) -> None:
    from family_fitness_ai.video import catalog

    kspo = next(clip for clip in catalog.clips() if clip.source == "kspo")
    chunk = catalog.citation_for(kspo.video_id)
    assert chunk is not None
    label = str(chunk.citation(1)["label"])
    assert label.startswith("국민체력100 운동처방")
    assert ", " in label
    assert _marked(label) == []


def test_step_summaries_from_the_run_store_have_no_dots() -> None:
    run = runs.Run(run_id="cr_x", profile_refs=("p",))
    plan = compose.Plan(
        steps=[
            compose.Step(1, "assess", "ok", "유연성 백분위 10 · 대상 요인 = 유연성"),
            compose.Step(2, "retrieve", "ok", "처방 자료 2건"),
            compose.Step(3, "compose", "ok", "미션 1건"),
        ],
        proposal={"missions": [], "citations": []},
        refused=False,
        refusal_reason=None,
    )
    runs.Store()._finish(run, plan, {"유소년"})
    assert _marked(run.dict()) == []


@needs_release
@needs_index
@needs_embedder
def test_video_search_citation_names_have_no_dots() -> None:
    from family_fitness_ai.video.videos import search_videos

    found = search_videos("유소년", ("유연성",), ("스트레칭",), 10)
    assert found["hits"]
    assert _marked(found) == []


@needs_release
def test_assessment_and_trajectory_text_have_no_dots() -> None:
    from family_fitness_ai.stats.assess import Profile, assessment, trajectory

    child = Profile("p", 11, "세", "F", 148.0, 41.0, {"028": 41.3, "020": 70, "009": 30})
    assert _marked(assessment(child)) == []
    assert _marked(trajectory(child, "028", 3)) == []


# ── 유튜브 영상 근거 이름 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("title", "after"),
    [
        (
            "[👦🏻유소년] 성장기 학생들을 위한 체력향상 운동프로그램 (30min)",
            "성장기 학생들을 위한 체력향상 운동프로그램(30분)",
        ),
        (
            "🦖유아기 체력 향상 (복합운동) | EP03.알록달록 붙여요! (20miin)",
            "유아기 체력 향상 (복합운동), EP03.알록달록 붙여요!(20분)",
        ),
        (
            "[성인/1주차] 딱 4주만 같이 해봐요💪｜1주일만 해도 체지방 쫙!",
            "딱 4주만 같이 해봐요, 1주일만 해도 체지방 쫙!",
        ),
        ("🦖유아기 복합 지각능력  향상 활동", "유아기 복합 지각능력 향상 활동"),
        ("[유소년]", "[유소년]"),
    ],
)
def test_video_titles_lose_emoji_bracket_heads_and_min(title: str, after: str) -> None:
    assert words.video_title(title) == after


def test_a_youtube_citation_is_named_like_the_rule_planner() -> None:
    chunk = Chunk(
        chunk_id="video:abc",
        source="video",
        text="",
        citation_label=(
            "국민체력100 운동영상 · [👦🏻유소년] 성장기 학생들을 위한 체력향상 운동프로그램 (30min)"
        ),
        citation_url="https://www.youtube.com/watch?v=abc",
        age_group="유소년",
        factors=(),
        grade="",
    )
    assert chunk.citation(1)["label"] == (
        "국민체력100 운동영상, 성장기 학생들을 위한 체력향상 운동프로그램(30분)"
    )
