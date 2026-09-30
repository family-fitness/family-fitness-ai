"""규칙 편성이 쓰는 보호자 문구의 「언제 얼마나」.

BE 는 하루를 짤 때 days_per_week 1 · weeks 1 로 부른다. 그 편성은 한 번뿐이라
「주 1회 20분」이라고 쓰면 받는 사람이 한 주 계획으로 읽는다. 표와 색인이 없어도 돈다.
"""

from __future__ import annotations

from family_fitness_ai.coach import compose

CHILD = compose.RunProfile(ref="p", role="주행자", age=15, age_unit="세", sex="F")
ME = [{"ref": "p", "role": "주행자"}]


def _read(factor: str = "유연성", band: str = "steady", focused: bool = False) -> compose.Read:
    return compose.Read(CHILD, factor, band, 50, [], [], [], focused)


def _day(minutes: int = 20) -> compose.Slot:
    return compose.Slot("일간", 0, minutes, ME)


def test_one_day_plan_says_one_day_not_once_a_week() -> None:
    copy = compose._rule_copy(_read(), compose.Constraints(days_per_week=1), _day(), 1)
    assert "주 1회" not in copy["parent"]
    assert copy["parent"].endswith("하루 20분이면 충분합니다")


def test_one_day_plan_focused_and_unmeasured_also_say_one_day() -> None:
    focused = compose._rule_copy(
        _read(focused=True, band=""), compose.Constraints(days_per_week=1), _day(), 1
    )
    unmeasured = compose._rule_copy(
        _read(factor="", band=""), compose.Constraints(days_per_week=1), _day(), 1
    )
    assert "하루 20분" in focused["parent"] and "주 1회" not in focused["parent"]
    assert "하루 20분" in unmeasured["parent"] and "주 1회" not in unmeasured["parent"]


def test_unmeasured_plan_does_not_say_factor_to_the_guardian() -> None:
    """「요인」 은 코드 쪽 말이다. 보호자에게는 무엇을 키우면 좋을지로 말한다."""
    unmeasured = compose._rule_copy(
        _read(factor="", band=""), compose.Constraints(days_per_week=1), _day(), 1
    )
    assert "요인" not in unmeasured["parent"]
    assert unmeasured["parent"].endswith("무엇을 키우면 좋을지 알려 드릴게요")


def test_weekly_rhythm_still_says_times_per_week() -> None:
    three = compose._rule_copy(_read(), compose.Constraints(days_per_week=3), _day(15), 1)
    once_for_two_weeks = compose._rule_copy(
        _read(), compose.Constraints(days_per_week=1), _day(15), 2
    )
    assert "주 3회 15분" in three["parent"]
    assert "주 1회 15분" in once_for_two_weeks["parent"]


def test_weekly_slot_keeps_once_a_week() -> None:
    slot = compose.Slot("주간", compose.WEEKLY_SLOT, 30, ME)
    copy = compose._rule_copy(_read(), compose.Constraints(days_per_week=1), slot, 1)
    assert "주에 한 번 30분" in copy["parent"]


# ── 키울 요인을 부르는 말 ─────────────────────────────────────────────────
# 이번 편성이 키울 요인은 고른 까닭으로 부른다. 백분위 구간 문구(「꾸준히 하고 있는
# 영역」)는 그 요인의 수준을 말할 때만 쓴다 — 키울 요인을 「꾸준히 하고 있는 영역」
# 이라고 부르면 왜 그걸 하라는지 읽히지 않는다.


def test_the_lowest_factor_is_called_the_one_to_grow_now() -> None:
    for band in ("growth", "steady", "strength"):
        copy = compose._rule_copy(_read(band=band), compose.Constraints(), _day(), 1)
        assert "유연성은 지금 키우기 좋은 영역입니다" in copy["parent"]
        assert "꾸준히 하고 있는 영역" not in copy["parent"]
        assert "잘하고 있는 영역" not in copy["parent"]


def test_the_guardians_focus_is_called_what_the_guardian_wants_to_grow() -> None:
    for band in ("", "steady"):
        copy = compose._rule_copy(_read(band=band, focused=True), compose.Constraints(), _day(), 1)
        assert "보호자가 키워 주고 싶은 역량" in copy["parent"]
        assert "꾸준히 하고 있는 영역" not in copy["parent"]


def test_the_coach_is_told_how_to_call_the_factor() -> None:
    from family_fitness_ai.coach import llm

    assert "지금 키우기 좋은 영역" in llm.PLAN_SYSTEM
    assert "보호자가 키워 주고 싶은 역량" in llm.PLAN_SYSTEM
    focused = compose._brief(_read(focused=True))
    lowest = compose._brief(_read())
    assert focused["대상_요인을_고른_까닭"] == "보호자가 키워 주고 싶은 역량"
    assert lowest["대상_요인을_고른_까닭"] == "지금 키우기 좋은 영역"
