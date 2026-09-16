"""반환 직전 검사 (docs/03 §5.4).

**부분 통과는 없다.** 넷 중 하나라도 걸리면 거부다. 검사가 제안을 고쳐서
통과시키는 길이 없는지도 함께 잠근다.
"""

from __future__ import annotations

import datetime as dt

import pytest

from family_fitness_ai.api.coach_schemas import (
    Citation,
    Mission,
    MissionCopy,
    MissionPeriod,
    Participant,
    Proposal,
    Session,
    Video,
)
from family_fitness_ai.coach.verify import check_proposal, cite_numbers

DAY = dt.date(2026, 9, 21)
CHILD = {"p_child": ("유소년", 11, "세")}
TODDLER_VIDEO = {"v_toddler": "유아기"}
YOUTH_VIDEO = {"v_youth": "유소년"}
ADULT_VIDEO = {"v_adult": "성인"}


def _mission(
    *,
    title: str = "왕복달리기",
    child_copy: str = "오늘은 왕복달리기 한 번 해볼까요",
    parent_copy: str = "또래 유소년 여아 처방에 1번째로 많이 나온 본운동입니다",
    evidence: list[int] | None = None,
    video: Video | None = None,
    ref: str = "p_child",
) -> Mission:
    return Mission(
        title=title,
        period=MissionPeriod(start_date=DAY, end_date=DAY),
        participants=[Participant(ref=ref, role="주행자")],
        sessions=[
            Session(
                day_offset=0,
                exercise_name="왕복달리기",
                duration_min=15,
                video=video,
                evidence=evidence or [1],
            )
        ],
        copy=MissionCopy(child=child_copy, parent=parent_copy),
    )


CELL_CITATION = Citation(
    index=1, label="국민체력100 운동처방 · 유소년 11세", chunk_id="prescription:x"
)


def _proposal(mission: Mission, citations: list[Citation] | None = None) -> Proposal:
    return Proposal(missions=[mission], citations=citations or [CELL_CITATION])


def test_clean_proposal_passes() -> None:
    assert check_proposal(_proposal(_mission()), CHILD, {}) == []


def test_zero_citations_is_refused() -> None:
    """인용 0 은 거부다. `Proposal` 이 최소 1개를 강제하므로 빈 목록을 직접 만든다."""
    proposal = _proposal(_mission())
    proposal.citations = []
    failures = check_proposal(proposal, CHILD, {})
    assert [f.check for f in failures] == ["citation"]


def test_evidence_outside_citation_range_is_caught() -> None:
    failures = check_proposal(_proposal(_mission(evidence=[1, 9])), CHILD, {})
    assert any(f.check == "evidence" and "9" in f.detail for f in failures)


@pytest.mark.parametrize("word", ["부족", "미달", "하위", "열등", "낙제", "미흡"])
def test_forbidden_word_in_copy_is_caught(word: str) -> None:
    failures = check_proposal(_proposal(_mission(parent_copy=f"유연성이 {word}합니다")), CHILD, {})
    assert any(f.check == "forbidden" for f in failures)


def test_forbidden_word_in_title_is_caught() -> None:
    failures = check_proposal(_proposal(_mission(title="근력 부족 보완")), CHILD, {})
    assert any(f.check == "forbidden" for f in failures)


def test_cite_marker_outside_range_in_copy_is_caught() -> None:
    failures = check_proposal(
        _proposal(_mission(parent_copy="또래 처방에 많이 나옵니다 [7]")), CHILD, {}
    )
    assert any(f.check == "evidence" and "7" in f.detail for f in failures)


def test_cite_marker_inside_range_passes() -> None:
    assert check_proposal(_proposal(_mission(parent_copy="또래 처방입니다 [1]")), CHILD, {}) == []


def test_cite_numbers_reads_every_marker() -> None:
    assert cite_numbers("앞 [1] 가운데 [2] 끝 [10]") == [1, 2, 10]
    assert cite_numbers("없다") == []


# ── 연령 검사 (docs/02 §2.3) ───────────────────────────────────────


def test_adult_video_for_a_child_is_caught() -> None:
    mission = _mission(video=Video(video_id="v_adult", start_sec=10), evidence=[1])
    failures = check_proposal(_proposal(mission), CHILD, ADULT_VIDEO)
    assert any(f.check == "age" for f in failures)


def test_different_child_group_video_is_caught_at_eleven() -> None:
    """만 11세는 예외 구간이 아니다 — 유아기 영상이 걸린다."""
    mission = _mission(video=Video(video_id="v_toddler", start_sec=10))
    failures = check_proposal(_proposal(mission), CHILD, TODDLER_VIDEO)
    assert any(f.check == "age" for f in failures)


def test_different_child_group_video_passes_at_eight() -> None:
    """만 7~10세 예외 — 아이 연령대 영상은 연령대와 무관하게 통과한다."""
    mission = _mission(video=Video(video_id="v_toddler", start_sec=10))
    failures = check_proposal(_proposal(mission), {"p_child": ("유소년", 8, "세")}, TODDLER_VIDEO)
    assert failures == []


def test_same_group_video_passes() -> None:
    mission = _mission(video=Video(video_id="v_youth", start_sec=10))
    assert check_proposal(_proposal(mission), CHILD, YOUTH_VIDEO) == []


def test_video_missing_from_the_table_is_caught() -> None:
    """**모르는 영상을 통과시키지 않는다.** 연령을 알 수 없으면 내보낼 근거가 없다."""
    mission = _mission(video=Video(video_id="v_unknown", start_sec=10))
    failures = check_proposal(_proposal(mission), CHILD, YOUTH_VIDEO)
    assert any(f.check == "age" and "영상 표에 없다" in f.detail for f in failures)


def test_participant_missing_from_the_request_is_caught() -> None:
    mission = _mission(video=Video(video_id="v_youth", start_sec=10), ref="p_ghost")
    failures = check_proposal(_proposal(mission), CHILD, YOUTH_VIDEO)
    assert any(f.check == "age" and "요청에 없다" in f.detail for f in failures)


def test_no_video_needs_no_age_check() -> None:
    assert check_proposal(_proposal(_mission(video=None)), CHILD, {}) == []


def test_every_failure_is_reported_not_just_the_first() -> None:
    """부분 통과가 없으니 무엇이 걸렸는지 전부 보여야 고칠 수 있다."""
    mission = _mission(
        title="근력 부족",
        parent_copy="미달입니다 [9]",
        evidence=[1, 8],
        video=Video(video_id="v_adult", start_sec=0),
    )
    checks = {f.check for f in check_proposal(_proposal(mission), CHILD, ADULT_VIDEO)}
    assert checks == {"forbidden", "evidence", "age"}
