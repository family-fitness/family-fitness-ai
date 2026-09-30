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
