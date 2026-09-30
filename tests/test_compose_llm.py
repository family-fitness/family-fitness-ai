"""LLM 이 짠 한 주를 우리가 어떻게 받아들이나.

여기서는 LLM 을 부르지 않는다. 대신 「이렇게 답했다 치고」를 넣어 두고, 우리가
그 답을 곧이곧대로 쓰는지 아니면 확인하고 쓰는지를 본다. 지어낸 id 가 섞여도
그 자리만 버리고 나머지는 서야 한다.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import needs_embedder, needs_index, needs_release

from family_fitness_ai.coach import compose
from family_fitness_ai.video import catalog

pytestmark = [needs_release, needs_index, needs_embedder]

CHILD = compose.RunProfile(
    ref="p_c7a91f",
    role="주행자",
    age=11,
    age_unit="세",
    sex="F",
    input_level="L2",
    measurements={"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
)


def _fake_plan(rows):
    def plan(payload):
        _fake_plan.seen = payload
        return rows

    return plan


def test_the_plan_may_only_use_clips_we_handed_over(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def plan(payload):
        captured.update(payload)
        # 이름이 겹치지 않는 클립 셋을 고른다 — 같은 이름은 하루에 한 번뿐이다.
        seen: set[str] = set()
        ids = []
        for clip in payload["클립"]:
            if clip["이름"] in seen:
                continue
            seen.add(clip["이름"])
            ids.append(clip["id"])
            if len(ids) == 3:
                break
        captured["ids"] = ids
        return [
            {
                "day_offset": 0,
                "title": "월요일 늘이기",
                "child": "오늘은 몸을 길게 늘여 볼까요",
                "parent": "하루 15분이면 충분합니다",
                "reason": "또래 처방에 나온 동작입니다 [1].",
                "clips": [
                    {"id": ids[0], "phase": "준비운동"},
                    {"id": "지어낸id", "phase": "본운동"},
                    {"id": ids[1], "phase": "본운동"},
                    {"id": ids[2], "phase": "정리운동"},
                ],
            }
        ]

    monkeypatch.setattr(compose.coach_llm, "plan_week", plan)
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)

    plan_result = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints())
    assert plan_result.proposal is not None
    missions = plan_result.proposal["missions"]
    assert len(missions) == 1
    # 지어낸 id 한 자리만 빠지고 코치가 고른 셋은 선다. 모자란 편수는 목록에서 채운다.
    sessions = missions[0]["sessions"]
    names = {clip["id"]: clip["이름"] for clip in captured["클립"]}  # type: ignore[union-attr]
    picked = {names[i] for i in captured["ids"]}  # type: ignore[union-attr]
    assert picked <= {s["exercise_name"] for s in sessions}
    assert len(sessions) == sum(catalog.clip_counts(15).values())
    assert missions[0]["title"] == "월요일 늘이기"
    phases = [s["phase"] for s in sessions]
    assert phases == sorted(phases, key=catalog.PHASES.index)
    # 고를 수 있는 것은 우리가 준다 — 근거와 클립이 프롬프트에 실려 있어야 한다.
    assert captured["근거"] and captured["클립"]
    assert captured["참여자"]["연령대"] == "유소년"


def test_a_day_of_only_made_up_ids_is_dropped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(
        compose.coach_llm,
        "plan_week",
        _fake_plan(
            [
                {
                    "day_offset": 0,
                    "title": "허공",
                    "child": "",
                    "parent": "",
                    "reason": "",
                    "clips": [{"id": "없는것", "phase": "본운동"}],
                }
            ]
        ),
    )
    result = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints())
    assert result.proposal is not None
    # 그 날은 버려지고 규칙 편성이 대신 선다.
    assert result.proposal["missions"]
    assert result.steps[2].status == "partial"


def test_falling_back_to_rules_is_said_out_loud(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", lambda payload: None)
    result = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints())
    assert result.proposal is not None
    assert result.steps[2].status == "partial"
    assert "규칙" in result.steps[2].summary


# ── 일간과 주간 ────────────────────────────────────────────────────────────


def _family():
    mom = compose.RunProfile(ref="mom", role="동반자", age=41, age_unit="세", sex="F")
    return [CHILD, mom]


def test_weekly_mission_covers_the_week_and_names_no_day():
    """주간은 그 주 안에 아무 때나 한다. 날짜를 박지 않는다."""
    plan = compose.build(
        _family(),
        date(2026, 9, 21),
        1,
        compose.Constraints(days_per_week=3, minutes_per_session=15, weekly_minutes=30),
    )
    assert plan.proposal is not None
    kinds = [m["kind"] for m in plan.proposal["missions"]]
    assert kinds.count("일간") == 3
    assert kinds.count("주간") == 1

    weekly = next(m for m in plan.proposal["missions"] if m["kind"] == "주간")
    assert weekly["period"] == {"start_date": "2026-09-21", "end_date": "2026-09-27"}
    assert weekly["duration_min"] == 30


def test_the_companion_joins_the_weekly_not_the_daily():
    """동반자는 제 몫을 따로 받지 않고 주간에 함께한다 — 「동반자」가 그런 뜻이다."""
    plan = compose.build(
        _family(),
        date(2026, 9, 21),
        1,
        compose.Constraints(days_per_week=3, minutes_per_session=15, weekly_minutes=30),
    )
    assert plan.proposal is not None
    for mission in plan.proposal["missions"]:
        roles = {p["role"] for p in mission["participants"]}
        assert roles == ({"주행자", "동반자"} if mission["kind"] == "주간" else {"주행자"})


def test_no_weekly_when_it_is_not_asked_for():
    plan = compose.build(_family(), date(2026, 9, 21), 1, compose.Constraints())
    assert plan.proposal is not None
    assert all(m["kind"] == "일간" for m in plan.proposal["missions"])


def test_the_same_exercise_does_not_come_back_in_the_same_week():
    plan = compose.build(
        _family(),
        date(2026, 9, 21),
        1,
        compose.Constraints(days_per_week=3, minutes_per_session=15, weekly_minutes=30),
    )
    assert plan.proposal is not None
    names = [s["exercise_name"] for m in plan.proposal["missions"] for s in m["sessions"]]
    assert len(names) == len(set(names))


# ── 보호자가 키워 주고 싶은 역량 ──────────────────────────────────────────


def _other_than_lowest() -> tuple[str, str]:
    """측정으로 고른 가장 낮은 요인과, 그것과 다른 요인 하나."""
    lowest, _, _, rows = compose.target_factor(CHILD)
    measured = [str(row["factor"]) for row in rows if row["percentile"] is not None]
    other = next(factor for factor in measured if factor != lowest)
    return lowest, other


def test_the_guardians_focus_comes_before_the_lowest_factor():
    lowest, focus = _other_than_lowest()
    factor, band, percentile, _ = compose.target_factor(CHILD, focus)
    assert factor == focus != lowest
    # 잰 요인이면 그 요인의 band·백분위를 함께 낸다.
    assert band and percentile is not None


def test_a_focus_that_was_not_measured_still_leads():
    factor, band, percentile, _ = compose.target_factor(CHILD, "평형성")
    assert (factor, band, percentile) == ("평형성", "", None)


def test_the_rule_plan_follows_the_focus():
    _, focus = _other_than_lowest()
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints(focus_factor=focus))
    assert plan.proposal is not None
    assert f"대상 요인 = {focus}(보호자가 키워 주고 싶은 역량)" in plan.steps[0].summary
    assert all(focus in m["title"] for m in plan.proposal["missions"])


def test_the_coach_is_told_the_focus_and_the_companion(monkeypatch: pytest.MonkeyPatch):
    _, focus = _other_than_lowest()
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", _fake_plan(None))
    compose.build(
        [CHILD],
        date(2026, 9, 7),
        1,
        compose.Constraints(focus_factor=focus, with_companion=True),
    )
    seen = _fake_plan.seen
    assert seen["참여자"]["대상_체력요인"] == focus
    assert seen["참여자"]["대상_요인을_고른_까닭"] == "보호자가 키워 주고 싶은 역량"
    assert seen["조건"]["보호자도_함께"] is True


def test_without_a_focus_the_lowest_factor_stays(monkeypatch: pytest.MonkeyPatch):
    lowest, _ = _other_than_lowest()
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", _fake_plan(None))
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints())
    assert _fake_plan.seen["참여자"]["대상_체력요인"] == lowest
    assert _fake_plan.seen["참여자"]["대상_요인을_고른_까닭"] == "지금 키우기 좋은 영역"
    assert "보호자가 키워 주고 싶은 역량" not in plan.steps[0].summary
    assert f"대상 요인 = {lowest}(지금 키우기 좋은 영역)" in plan.steps[0].summary


def test_the_focus_is_the_drivers_not_the_companions(monkeypatch: pytest.MonkeyPatch):
    reads: list[compose.Read] = []
    original = compose.read_profile

    def spy(profile, focus=None):
        read = original(profile, focus)
        reads.append(read)
        return read

    monkeypatch.setattr(compose, "read_profile", spy)
    compose.build(_family(), date(2026, 9, 21), 1, compose.Constraints(focus_factor="평형성"))
    by_role = {read.profile.role: read for read in reads}
    assert by_role["주행자"].focused and by_role["주행자"].factor == "평형성"
    assert not by_role["동반자"].focused


def test_a_focus_without_prescriptions_widens_the_factor_before_the_sex():
    """만 11세 여자아이의 평형성 처방은 없다. 요인을 먼저 넓히고, 성별은 끝까지 지킨다.

    여자아이 처방이 있는데 남자아이 처방을 근거로 들면 안 된다.
    """
    read = compose.read_profile(CHILD, "평형성")
    assert read.chunks
    assert all("-F-" in chunk.chunk_id for chunk in read.chunks), [
        chunk.chunk_id for chunk in read.chunks
    ]
    assert all("-11-" in chunk.chunk_id for chunk in read.chunks)
    assert read.notices and "평형성" in read.notices[0]
