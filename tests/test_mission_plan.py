"""후보·고르기·편성 (docs/05 · AI-14 §8 ⑥⑦).

커밋된 산출물만 읽는다 — 원자료는 CI 에 없다.
"""

from __future__ import annotations

import datetime as dt

import pytest

from family_fitness_ai.common.settings import RELEASE_DIR
from family_fitness_ai.common.types import resolve_age_group
from family_fitness_ai.mission import cells as C
from family_fitness_ai.mission import plan as P
from family_fitness_ai.mission import segments as G
from family_fitness_ai.mission.build import Mission as MissionRow
from family_fitness_ai.mission.build import read_missions
from family_fitness_ai.mission.segments import SEGMENTS_FILE, VideoSegment
from family_fitness_ai.mission.select import (
    Candidate,
    candidates,
    in_evidence,
    select,
    video_allowed,
)
from family_fitness_ai.rag.prescription import identity

MONDAY = dt.date(2026, 9, 21)


@pytest.fixture(scope="module")
def cells() -> list[C.Cell]:
    return C.load_cells()


@pytest.fixture(scope="module")
def segments() -> list[VideoSegment]:
    return G.read_segments(RELEASE_DIR / SEGMENTS_FILE)


@pytest.fixture(scope="module")
def mission_set() -> list[MissionRow]:
    """**추천의 검색 대상** (`missions.csv` · AI-14 §4). 커밋된 표를 읽는다."""
    return read_missions()


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


def test_seven_to_ten_opens_more_videos_than_eleven(
    cells: list[C.Cell], segments: list[VideoSegment]
) -> None:
    """만 8세가 만 11세보다 영상 후보가 많다 — 예외가 아이 연령대를 다 열어서다.

    이것이 유소년 구멍(영상 구간 1개)을 실제로 메우는 자리다.
    """
    eight = candidates(C.find_cells(cells, "유소년", 8, "세", "M"), segments, age=8, age_unit="세")
    eleven = candidates(
        C.find_cells(cells, "유소년", 11, "세", "M"), segments, age=11, age_unit="세"
    )
    assert sum(c.has_video for c in eight) > sum(c.has_video for c in eleven)


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


# ── 편성 모양 (AI-11 §5.1) ─────────────────────────────────────────

CHILD = ("p_child", P.DRIVER, 11, "세", "F")
PARENT = ("p_parent", P.COMPANION, 41, "세", "F")
CHEER = ("p_cheer", P.CHEER, 68, "세", "M")


def _build(family, cells, mission_set, days=3, minutes=15):
    return P.build(
        list(family),
        cells,
        mission_set,
        start_date=MONDAY,
        days_per_week=days,
        minutes_per_session=minutes,
    )


def test_daily_missions_are_one_day_one_session(cells, mission_set) -> None:
    proposal, _ = _build([CHILD], cells, mission_set)
    daily = [m for m in proposal.missions if m.period.start_date == m.period.end_date]
    assert len(daily) == 3
    for mission in daily:
        assert len(mission.sessions) == 1
        assert mission.sessions[0].day_offset == 0
        assert [p.ref for p in mission.participants] == ["p_child"]


def test_weekly_mission_spans_six_days(cells, mission_set) -> None:
    proposal, _ = _build([CHILD], cells, mission_set)
    weekly = [m for m in proposal.missions if m.period.start_date != m.period.end_date]
    assert len(weekly) == 1
    assert (weekly[0].period.end_date - weekly[0].period.start_date).days == 6
    assert [s.day_offset for s in weekly[0].sessions] == [0, 2, 4]


def test_weekly_is_the_sum_of_the_dailies(cells, mission_set) -> None:
    """**주간은 그 주 일일의 합이다.** 운동이 다르면 일일만 해도 주간이 완료로 뜬다 —
    `TIMER_MINUTES` 가 기간 안 활동 합계라서다 (`MissionCompletionPolicy`)."""
    proposal, _ = _build([CHILD], cells, mission_set)
    daily = [m for m in proposal.missions if m.period.start_date == m.period.end_date]
    weekly = next(m for m in proposal.missions if m.period.start_date != m.period.end_date)
    assert [s.exercise_name for m in daily for s in m.sessions] == [
        s.exercise_name for s in weekly.sessions
    ]
    # 영상도 같은 구간을 가리킨다 — 일일과 주간이 다른 영상을 주면 같은 운동이 아니다
    assert [
        (s.video.video_id, s.video.start_sec) if s.video else None
        for m in daily
        for s in m.sessions
    ] == [(s.video.video_id, s.video.start_sec) if s.video else None for s in weekly.sessions]


def test_daily_dates_match_weekly_offsets(cells, mission_set) -> None:
    proposal, _ = _build([CHILD], cells, mission_set)
    daily = [m for m in proposal.missions if m.period.start_date == m.period.end_date]
    assert [m.period.start_date for m in daily] == [
        MONDAY + dt.timedelta(days=o) for o in P.day_offsets(3)
    ]


def test_cheer_is_left_out_everywhere(cells, mission_set) -> None:
    proposal, planned = _build([CHILD, PARENT, CHEER], cells, mission_set)
    assert "p_cheer" not in {p.ref for m in proposal.missions for p in m.participants}
    assert "p_cheer" not in {m.ref for m in planned}


def test_companion_gets_weekly_only(cells, mission_set) -> None:
    proposal, _ = _build([CHILD, PARENT], cells, mission_set)
    theirs = [m for m in proposal.missions if any(p.ref == "p_parent" for p in m.participants)]
    assert len(theirs) == 1
    assert theirs[0].period.start_date != theirs[0].period.end_date


def test_child_copy_is_empty_for_missions_the_child_is_not_in(cells, mission_set) -> None:
    """동반자만 있는 미션의 아이 문구를 아이 화면에 띄우지 않는다."""
    proposal, _ = _build([CHILD, PARENT], cells, mission_set)
    for mission in proposal.missions:
        refs = {p.ref for p in mission.participants}
        if "p_child" not in refs:
            assert mission.text.child == ""
        else:
            assert mission.text.child


def test_every_session_has_evidence_inside_citations(cells, mission_set) -> None:
    proposal, _ = _build([CHILD, PARENT], cells, mission_set)
    valid = {c.index for c in proposal.citations}
    assert valid
    for mission in proposal.missions:
        for session in mission.sessions:
            assert session.evidence
            assert set(session.evidence) <= valid


def test_citation_label_is_never_empty(cells, mission_set) -> None:
    """`AiWire.CitationBody.label` 은 기본값이 없다 — 비면 역직렬화가 터지고
    그 예외는 503 으로 번역되지 않아 실행이 그냥 FAILED 가 된다."""
    proposal, _ = _build([CHILD, PARENT], cells, mission_set)
    assert all(c.label.strip() for c in proposal.citations)


def test_picked_exercise_is_inside_the_cited_chunk(cells, mission_set) -> None:
    """고른 운동이 인용한 청크 본문 안에 있다 — 청크는 칸당 30개로 잘린다."""
    _, planned = _build([CHILD, PARENT], cells, mission_set)
    for member in planned:
        for pick in member.picks:
            cell = next(c for c in member.match.cells if c.chunk_id == pick.chunk_id)
            assert in_evidence(cell, pick.exercise_name)


def test_citation_age_group_matches_the_asker(cells, mission_set) -> None:
    """**근거는 묻는 사람의 칸이어야 한다.**

    미션 표의 `cell_chunk_id` 는 그 운동이 가장 앞섰던 칸이라 요청자와 연령대가
    다를 수 있다 — 그것을 그대로 쓰면 유소년 11세 아이에게 「성인 62세 처방」이
    근거로 붙는다 (2026-09-16 실측 · 처방 인용 12건 중 6건이 어긋났다).
    """
    for family in (
        [CHILD],
        [("p", P.DRIVER, 8, "세", "M")],
        [("p", P.DRIVER, 60, "개월", "F")],
        [("p", P.DRIVER, 15, "세", "M")],
        [("p", P.DRIVER, 70, "세", "M")],
    ):
        proposal, planned = _build(family, cells, mission_set)
        want = planned[0].match.age_group
        for citation in proposal.citations:
            if not citation.chunk_id.startswith("prescription:"):
                continue
            got = citation.chunk_id.split(":", 1)[1].split("-")[0]
            assert got == want, f"{want} 에게 {got} 처방이 근거로 붙었다 — {citation.label}"


def test_every_pick_is_in_the_askers_own_cell(cells, mission_set) -> None:
    """표가 통과시켜도 그 사람의 칸 청크에 이름이 없으면 근거를 댈 수 없다."""
    _, planned = _build([CHILD, PARENT], cells, mission_set)
    for member in planned:
        for pick in member.picks:
            cell = next(c for c in member.match.cells if c.chunk_id == pick.chunk_id)
            assert in_evidence(cell, pick.exercise_name)


def test_title_has_no_period_prefix(cells, mission_set) -> None:
    """「오늘 · 」/「이번 주 · 」 접두어를 붙이지 않는다 (AI-11 §5.1)."""
    proposal, _ = _build([CHILD], cells, mission_set)
    for mission in proposal.missions:
        assert not mission.title.startswith(("오늘", "이번 주"))


def test_video_is_age_allowed_for_every_participant(cells, mission_set, segments) -> None:
    groups = {s.video_id: s.age_group for s in segments}
    for family in ([CHILD], [("p", P.DRIVER, 8, "세", "M")], [("p", P.DRIVER, 60, "개월", "F")]):
        proposal, _ = _build(family, cells, mission_set)
        for mission in proposal.missions:
            for session in mission.sessions:
                if session.video is None:
                    continue
                group = groups[session.video.video_id]
                for participant in mission.participants:
                    ref, _role, age, unit, _sex = next(f for f in family if f[0] == participant.ref)
                    assert video_allowed(resolve_age_group(age, unit), age, unit, group), (
                        f"{ref} 에게 {group} 영상이 나갔다"
                    )


@pytest.mark.parametrize("days", [1, 2, 3, 4, 5, 6, 7])
def test_every_days_per_week_works(cells, mission_set, days: int) -> None:
    proposal, _ = _build([CHILD], cells, mission_set, days=days)
    daily = [m for m in proposal.missions if m.period.start_date == m.period.end_date]
    weekly = [m for m in proposal.missions if m.period.start_date != m.period.end_date]
    assert len(daily) == days
    assert len(weekly) == 1
    assert len(weekly[0].sessions) == days


def test_refusal_when_no_cell_matches(cells, mission_set) -> None:
    """만 3세는 칸이 없다. 예외가 아니라 거부다."""
    with pytest.raises(P.Refusal) as caught:
        _build([("p", P.DRIVER, 3, "세", "F")], cells, mission_set)
    assert caught.value.reason == "no_candidate"


def test_missions_work_with_no_video_at_all(cells, mission_set) -> None:
    """**영상이 하나도 없어도 미션이 난다** (docs/03 §5.7 — `video: null` 은 결함이 아니다)."""
    import dataclasses

    stripped = [dataclasses.replace(m, segment=None) for m in mission_set]
    proposal, _ = _build([CHILD, PARENT], cells, stripped)
    assert proposal.missions
    assert all(s.video is None for m in proposal.missions for s in m.sessions)
    assert proposal.citations


def test_missions_come_from_the_mission_set(cells, mission_set) -> None:
    """**편성은 미션 집합에서 찾는다** — 요청마다 처방 칸에서 조립하지 않는다.

    집합을 비우면 편성할 것이 없어야 한다. 비었는데도 미션이 나오면 다른 자리에서
    후보를 만들고 있다는 뜻이다.
    """
    with pytest.raises(P.Refusal):
        _build([CHILD], cells, [])

    picked = {
        s.exercise_name for m in _build([CHILD], cells, mission_set)[0].missions for s in m.sessions
    }
    assert picked <= {m.exercise_name for m in mission_set}


# ── 미션 카드 문구 — LLM 이 쓰고 규칙이 대체한다 ─────────────────────


class _CopyWriter:
    """가짜 LLM. 쓸 두 줄을 시험이 정한다."""

    def __init__(self, child: str = "오늘 해보자", parent: str = "또래 처방에 많이 나왔습니다"):
        self.child, self.parent = child, parent
        self.calls = 0
        self.prompts: list[str] = []

    def version(self) -> str:
        return "llm:가짜"

    def write(self, question: str, hits) -> str:  # noqa: ANN001, ARG002
        raise AssertionError("미션 문구는 write_copy 를 쓴다")

    def write_copy(self, system: str, prompt: str) -> str:  # noqa: ARG002
        self.calls += 1
        self.prompts.append(prompt)
        return f"child: {self.child}\nparent: {self.parent}"


def _with_writer(cells, mission_set, writer, days=3):
    return P.build(
        [CHILD],
        cells,
        mission_set,
        start_date=MONDAY,
        days_per_week=days,
        minutes_per_session=15,
        writer=writer,
    )


def test_llm_이_카드_문구를_쓴다(cells, mission_set) -> None:
    writer = _CopyWriter()
    proposal, _ = _with_writer(cells, mission_set, writer)
    assert writer.calls == len(proposal.missions)
    for mission in proposal.missions:
        assert mission.text.child == "오늘 해보자"
        assert mission.text.parent == "또래 처방에 많이 나왔습니다"


def test_llm_이_없으면_규칙_문구다(cells, mission_set) -> None:
    """`writer` 를 주지 않으면 외부 호출이 0이다."""
    proposal, _ = _with_writer(cells, mission_set, None)
    assert all(m.text.parent for m in proposal.missions)
    assert any("또래" in m.text.parent for m in proposal.missions)


def test_llm_이_실패하면_규칙_문구로_강등한다(cells, mission_set) -> None:
    class Broken(_CopyWriter):
        def write_copy(self, system: str, prompt: str) -> str:  # noqa: ARG002
            raise RuntimeError("시험용 실패")

    proposal, _ = _with_writer(cells, mission_set, Broken())
    assert all(m.text.parent for m in proposal.missions), "강등이지 거부가 아니다"
    assert any("또래" in m.text.parent for m in proposal.missions)


def test_금지_어휘를_쓰면_규칙_문구로_강등한다(cells, mission_set) -> None:
    writer = _CopyWriter(parent="유연성이 부족합니다")
    proposal, _ = _with_writer(cells, mission_set, writer)
    assert all("부족" not in m.text.parent for m in proposal.missions)


def test_주당_횟수를_지어내면_강등한다(cells, mission_set) -> None:
    """일일 카드에 「주 1회」, 주 3회인데 「주 7회」가 실렸다 (2026-09-16 실측)."""
    proposal, _ = _with_writer(cells, mission_set, _CopyWriter(parent="주 7회 해 주세요"), days=3)
    assert all("주 7회" not in m.text.parent for m in proposal.missions)

    daily_claim = _CopyWriter(parent="회당 15분씩 주 1회 진행합니다")
    proposal, _ = _with_writer(cells, mission_set, daily_claim, days=3)
    daily = [m for m in proposal.missions if P.is_daily(m)]
    assert all("주 1회" not in m.text.parent for m in daily)


def test_맞는_주당_횟수는_통과한다(cells, mission_set) -> None:
    writer = _CopyWriter(parent="이번 주 3회, 회당 15분")
    proposal, _ = _with_writer(cells, mission_set, writer, days=3)
    weekly = [m for m in proposal.missions if not P.is_daily(m)]
    assert weekly and all(m.text.parent == "이번 주 3회, 회당 15분" for m in weekly)


def test_아이가_없는_미션에는_아이_문구를_넣지_않는다(cells, mission_set) -> None:
    writer = _CopyWriter()
    proposal, _ = P.build(
        [CHILD, PARENT],
        cells,
        mission_set,
        start_date=MONDAY,
        days_per_week=3,
        minutes_per_session=15,
        writer=writer,
    )
    for mission in proposal.missions:
        if "p_child" not in {p.ref for p in mission.participants}:
            assert mission.text.child == ""


def test_문구_프롬프트가_운동을_바꾸지_못하게_담는다(cells, mission_set) -> None:
    """LLM 은 문구만 쓴다 — 프롬프트에 고른 운동과 그 수치가 그대로 있어야 한다."""
    writer = _CopyWriter()
    _, planned = _with_writer(cells, mission_set, writer)
    prompt = writer.prompts[-1]
    for pick in planned[0].picks:
        assert pick.exercise_name in prompt
        assert f"{pick.count:,}" in prompt
