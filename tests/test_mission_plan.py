"""후보 고르기의 규칙 (docs/05).

**만들어 넣은 표로만 시험한다.** 산출물(`data/release/`)을 읽는 시험은 데이터를 다시
처리한 뒤에 쓴다 — 지금은 없다.
"""

from __future__ import annotations

import datetime as dt

import pytest

from family_fitness_ai.mission.segments import VideoSegment
from family_fitness_ai.mission.select import (
    Candidate,
    select,
    video_allowed,
)
from family_fitness_ai.rag.prescription import identity

MONDAY = dt.date(2026, 9, 21)
# ── 연령 규칙 (docs/02 §2.3) ────────────────────────────────────────


@pytest.mark.parametrize(
    ("age_group", "age", "unit", "video", "ok"),
    [
        # 같은 연령대는 통과
        ("유소년", 11, "세", "유소년", True),
        ("유아기", 60, "개월", "유아기", True),
        # 연령대가 다르면 막힌다 — 백엔드 AgeRange 도 같다
        ("유소년", 11, "세", "유아기", False),
        ("유소년", 11, "세", "청소년", False),
        ("청소년", 15, "세", "유아기", False),
        # 아이에게 성인·어르신 영상은 막힌다
        ("유소년", 11, "세", "성인", False),
        ("청소년", 15, "세", "어르신", False),
        # 만 7~10세 예외 — 아이 연령대 영상은 연령대와 무관하게 통과
        ("유소년", 8, "세", "유아기", True),
        ("유소년", 8, "세", "청소년", True),
        ("유소년", 10, "세", "유아기", True),
        # 예외라도 성인·어르신·미상은 막힌다
        ("유소년", 8, "세", "성인", False),
        ("유소년", 8, "세", "", False),
        # 연령 미상 영상 — 아이에게 막고 성인에게는 낸다 (모르는 것이지 다른 것이 아니다)
        ("유아기", 60, "개월", "", False),
        ("성인", 41, "세", "", True),
    ],
)
def test_video_allowed(age_group: str, age: int, unit: str, video: str, ok: bool) -> None:
    assert video_allowed(age_group, age, unit, video) is ok


# ── 고르기 ──────────────────────────────────────────────────────────


def _cand(name: str, *, rank: int, video: bool, common: bool = False, phase: str = "본운동"):
    seg = (
        VideoSegment(
            video_id="v" + name,
            start_sec=0,
            end_sec=60,
            label_end_sec=60,
            length_basis="next_start",
            last=False,
            exercise_name=name,
            age_group="유소년",
            source="screen",
            common=common,
            evidence_text="",
            chunk_id="video:v" + name,
            citation_label="영상 " + name,
        )
        if video
        else None
    )
    return Candidate(
        exercise_name=name,
        phase=phase,
        rank=rank,
        count=100 - rank,
        chunk_id="prescription:x",
        citation_label="처방 x",
        segment=seg,
    )


def test_묶음_이름은_순위가_좋아도_뒤로_밀린다() -> None:
    """`맨몸운동  루틴프로그램 15분` 을 받은 사람은 무엇을 할지 알 수 없다.

    묶음 이름은 처방 횟수가 가장 많아 순위로 고르면 상단을 차지한다. 버리지는
    않는다 — 버리면 후보가 얇아진다.
    """
    pool = [
        _cand("맨몸운동  루틴프로그램", rank=1, video=False, phase="본운동"),
        _cand("팔굽혀펴기", rank=40, video=False, phase="본운동"),
    ]

    picked = select(pool, 2)

    assert [c.exercise_name for c in picked] == ["팔굽혀펴기", "맨몸운동  루틴프로그램"]
    assert pool[0].bundle and not pool[1].bundle


def test_묶음보다_영상과_공통여부가_먼저다() -> None:
    """층의 순서는 영상 → 공통 → 묶음 이다. 묶음을 앞세우지 않는다."""
    pool = [
        _cand("전신 루틴 스트레칭", rank=40, video=True, phase="정리운동"),
        _cand("팔굽혀펴기", rank=1, video=False, phase="본운동"),
    ]

    picked = select(pool, 1)

    assert picked[0].exercise_name == "전신 루틴 스트레칭", "영상이 묶음보다 앞선다"


def test_video_beats_phase_and_rank() -> None:
    """**영상이 첫 축이다.** 영상 붙은 준비운동이 영상 없는 본운동 1위보다 먼저다.

    반대로 두면 유소년의 영상 후보가 전부 밀려 빠진다 — 실제로 그랬다.
    """
    pool = [
        _cand("영상없는본운동", rank=1, video=False, phase="본운동"),
        _cand("영상있는준비운동", rank=40, video=True, phase="준비운동"),
    ]
    assert [c.exercise_name for c in select(pool, 1)] == ["영상있는준비운동"]


def test_common_segment_ranks_after_plain() -> None:
    pool = [
        _cand("공통", rank=1, video=True, common=True),
        _cand("비공통", rank=30, video=True, common=False),
    ]
    assert [c.exercise_name for c in select(pool, 1)] == ["비공통"]


def test_selection_is_deterministic_and_distinct() -> None:
    pool = [_cand(f"운동{i}", rank=i, video=False) for i in range(1, 10)]
    first = select(pool, 3)
    assert [c.exercise_name for c in first] == [c.exercise_name for c in select(pool, 3)]
    assert len({identity(c.exercise_name) for c in first}) == 3


def test_rotation_stays_inside_its_tier() -> None:
    """회전해도 영상 있는 후보가 먼저다 — 묶음을 넘어 돌리면 우선순위가 뒤집힌다."""
    pool = [_cand(f"영상{i}", rank=i, video=True) for i in range(1, 4)]
    pool += [_cand(f"맨몸{i}", rank=i, video=False) for i in range(1, 9)]
    for rotate in range(12):
        picked = select(pool, 3, rotate=rotate)
        assert all(c.has_video for c in picked), f"rotate={rotate} 에서 영상 없는 것이 섞였다"


def test_rotation_changes_the_set() -> None:
    pool = [_cand(f"운동{i}", rank=i, video=False) for i in range(1, 20)]
    weeks = {tuple(c.exercise_name for c in select(pool, 3, rotate=w)) for w in range(6)}
    assert len(weeks) > 1, "회전이 같은 조합만 낸다"


def test_select_fills_from_other_phases_when_main_runs_out() -> None:
    """세 개를 달라고 했는데 두 개를 내지 않는다."""
    pool = [_cand("본운동하나", rank=1, video=False, phase="본운동")]
    pool += [_cand(f"정리{i}", rank=i, video=False, phase="정리운동") for i in range(1, 5)]
    assert len(select(pool, 3)) == 3


def test_select_zero_or_empty() -> None:
    assert select([], 3) == []
    assert select([_cand("하나", rank=1, video=False)], 0) == []
