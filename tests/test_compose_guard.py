"""코치가 짠 한 주를 우리가 어디까지 손보나.

LLM 은 부르지 않는다. 「이렇게 답했다 치고」를 넣어, 코치가 편수를 모자라게
고르거나 이미 쓴 동작을 또 고르거나 글을 길게 써도 나가는 편성은 규칙을 지키는지
본다. 검색을 거치지 않아 임베딩이 없어도 돈다 — CI 에서도 돈다.
"""

from __future__ import annotations

import collections
from datetime import date

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.coach import compose
from family_fitness_ai.common import copy as words
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import catalog

pytestmark = [needs_release, needs_index]

MONDAY = date(2026, 10, 5)
CHILD = compose.RunProfile(ref="p", role="주행자", age=11, age_unit="세", sex="F")
FIFTEEN = catalog.clip_counts(15)  # 준비 2 · 본 4 · 정리 1


def _day(offset: int) -> compose.Slot:
    return compose.Slot("일간", offset, 15, [{"ref": "p", "role": "주행자"}])


def _ids(payload, phase: str, n: int, taken: set[str] | None = None) -> list[str]:
    """그 단계에서 앞에서부터 n 개. 늘리는 동작은 준비·정리 두 단계에 다 있어서,
    다른 단계에서 이미 고른 이름은 건너뛴다 — 같은 동작을 하루에 두 번 고르지 않게."""
    taken = taken if taken is not None else set()
    out = []
    for clip in payload["클립"]:
        if clip["단계"] == phase and clip["이름"] not in taken and len(out) < n:
            taken.add(clip["이름"])
            out.append(clip["id"])
    return out


def _full(payload) -> list[dict[str, str]]:
    """단계별 편수대로 앞에서부터 고른다. 하루 안에서 이름이 겹치지 않는다."""
    taken: set[str] = set()
    return [
        {"id": clip_id, "phase": phase}
        for phase, n in FIFTEEN.items()
        for clip_id in _ids(payload, phase, n, taken)
    ]


def _plan(offset: int, clips, **text) -> dict:
    return {
        "day_offset": offset,
        "title": text.get("title", "몸 늘이기"),
        "child": text.get("child", "몸을 길게 늘여 볼까요"),
        "parent": text.get("parent", "하루 15분이면 충분합니다"),
        "reason": text.get("reason", "또래 처방에 나온 동작입니다 [1]."),
        "clips": clips,
    }


def _run(monkeypatch: pytest.MonkeyPatch, plan, slots):
    monkeypatch.setattr(compose.coach_llm, "plan_week", plan)
    chunk = next(
        c for c in corpus().chunks if c.source == "prescription" and c.age_group == "유소년"
    )
    citations = compose.Citations()
    evidence = [citations.add(chunk)]
    pool, _ = catalog.pool("유소년", factor="유연성")
    read = compose.Read(CHILD, "유연성", "growth", 24, [], [chunk], [])
    tally = compose.Tally()
    missions = compose._by_llm(
        read, pool, slots, compose.Constraints(), evidence, citations, MONDAY, 1, tally
    )
    assert missions is not None
    return missions, tally


def _phases(mission) -> collections.Counter:
    return collections.Counter(s["phase"] for s in mission["sessions"])


# ── ① 이름이 같은 클립 ────────────────────────────────────────────────────


def test_the_pool_holds_each_exercise_once():
    """「엉덩이 스트레칭」이 20개 따로 잘려 있다. 코치에게는 하나로 보인다."""
    pool, _ = catalog.pool("성인", factor="근력")
    keys = [(clip.title, clip.phase) for clip in pool]
    assert len(keys) == len(set(keys))


def test_merging_names_does_not_pull_in_other_ages():
    """합친 뒤의 수로 재면 조건을 켠 유소년이 60 밑이라 다른 연령대가 섞인다.

    또래가 모자란지는 전처럼 클립 수로 본다.
    """
    everything = catalog.Conditions(quiet=True, small_space=True, no_props=True)
    pool, notice = catalog.pool("유소년", conditions=everything, factor="유연성")
    assert notice == ""
    assert {clip.age_group for clip in pool} == {"유소년"}


# ── ② 모자란 편수와 한 주 안 반복 ──────────────────────────────────────────


def test_a_short_pick_is_filled_from_the_list(monkeypatch: pytest.MonkeyPatch):
    picked: list[str] = []

    def plan(payload):
        taken: set[str] = set()
        clips = [
            {"id": _ids(payload, phase, 1, taken)[0], "phase": phase} for phase in catalog.PHASES
        ]
        picked.extend(
            next(c["이름"] for c in payload["클립"] if c["id"] == row["id"]) for row in clips
        )
        return [_plan(0, clips)]

    missions, tally = _run(monkeypatch, plan, [_day(0)])
    assert _phases(missions[0]) == collections.Counter(FIFTEEN)
    # 코치가 고른 셋은 그대로 있고, 모자란 넷을 채웠다.
    assert set(picked) <= {s["exercise_name"] for s in missions[0]["sessions"]}
    assert tally.filled == sum(FIFTEEN.values()) - 3


def test_the_same_pick_twice_still_fills_the_day(monkeypatch: pytest.MonkeyPatch):
    def plan(payload):
        again = _ids(payload, "본운동", 1)[0]
        return [_plan(0, [{"id": again, "phase": "본운동"}] * 4)]

    missions, _ = _run(monkeypatch, plan, [_day(0)])
    names = [s["exercise_name"] for s in missions[0]["sessions"]]
    assert _phases(missions[0]) == collections.Counter(FIFTEEN)
    assert len(names) == len(set(names))


def test_a_move_used_earlier_in_the_week_is_swapped(monkeypatch: pytest.MonkeyPatch):
    """코치가 이틀을 똑같이 짰다. 유소년 유연성은 목록이 넉넉해 겹치지 않게 된다."""

    def plan(payload):
        same = _full(payload)
        return [_plan(0, same), _plan(2, same)]

    missions, tally = _run(monkeypatch, plan, [_day(0), _day(2)])
    first = {s["exercise_name"] for s in missions[0]["sessions"]}
    second = {s["exercise_name"] for s in missions[1]["sessions"]}
    assert not first & second
    assert tally.filled == sum(FIFTEEN.values())


# ── ③ 칸마다 잰다 ──────────────────────────────────────────────────────────


def test_a_field_that_breaks_the_rules_falls_back_on_its_own(monkeypatch: pytest.MonkeyPatch):
    def plan(payload):
        return [_plan(0, _full(payload), title="가" * 17, child="오늘은 조금 부족해도 괜찮아요")]

    missions, tally = _run(monkeypatch, plan, [_day(0)])
    mission = missions[0]
    assert mission["title"] == "월요일 유연성 기르기"  # 17자 → 규칙 제목
    assert mission["copy"]["child"] == words.FOCUS_COPY["유연성"]  # 금지 어휘 → 규칙 문구
    assert mission["copy"]["parent"] == "하루 15분이면 충분합니다"  # 맞는 칸은 그대로
    assert tally.rewritten == 2


def test_text_naming_a_move_we_took_out_is_replaced(monkeypatch: pytest.MonkeyPatch):
    """둘째 날의 동작을 바꿨으니, 바뀐 동작을 이름으로 부르는 글은 못 쓴다."""
    named: list[str] = []

    def plan(payload):
        same = _full(payload)
        name = next(c["이름"] for c in payload["클립"] if c["id"] == same[0]["id"])
        named.append(name)
        reason = f"{name}로 몸을 풉니다 [1]."
        return [_plan(0, same, reason=reason), _plan(2, same, reason=reason)]

    missions, _ = _run(monkeypatch, plan, [_day(0), _day(2)])
    assert named[0] in missions[0]["reason"]  # 첫날은 그 동작이 있다
    assert missions[1]["reason"] == compose._rule_reason([1])  # 둘째 날은 없다


def test_each_slot_tells_the_coach_its_date_and_weekday(monkeypatch: pytest.MonkeyPatch):
    """자리에 day_offset 만 주면 코치가 요일을 짐작한다. 주간은 날을 정하지 않는다."""
    seen: list[dict] = []

    def plan(payload):
        seen.append(payload)
        return [_plan(0, _full(payload))]

    weekly = compose.Slot("주간", compose.WEEKLY_SLOT, 30, [{"ref": "p", "role": "주행자"}])
    _run(monkeypatch, plan, [_day(0), _day(2), weekly])
    slots = seen[0]["자리"]
    assert [(s["날짜"], s["요일"]) for s in slots] == [
        ("2026-10-05", "월요일"),
        ("2026-10-07", "수요일"),
        (None, None),
    ]


def test_a_title_with_another_weekday_falls_back(monkeypatch: pytest.MonkeyPatch):
    """수요일 편성 제목이 「유연성을 키우는 월요일」로 온 적이 있다. 그 칸만 규칙 제목으로."""

    def plan(payload):
        return [
            _plan(0, _full(payload), title="월요일 몸 늘이기"),
            _plan(2, _full(payload), title="유연성 키우는 월요일"),
        ]

    missions, tally = _run(monkeypatch, plan, [_day(0), _day(2)])
    assert missions[0]["title"] == "월요일 몸 늘이기"  # 맞는 요일은 그대로
    assert missions[1]["title"] == "수요일 유연성 기르기"
    assert missions[1]["copy"]["child"] == "몸을 길게 늘여 볼까요"  # 다른 칸은 그대로
    assert tally.rewritten == 1


def test_weekday_words_in_the_weekly_mission_fall_back(monkeypatch: pytest.MonkeyPatch):
    """주간은 그 주 안에 아무 때나 한다. 요일을 박은 글은 못 쓴다."""

    def plan(payload):
        return [
            _plan(
                compose.WEEKLY_SLOT,
                _full(payload),
                title="토요일 다 같이",
                parent="토요일에 온 가족이 함께 해 보세요",
            )
        ]

    weekly = compose.Slot("주간", compose.WEEKLY_SLOT, 15, [{"ref": "p", "role": "주행자"}])
    missions, tally = _run(monkeypatch, plan, [weekly])
    assert missions[0]["title"] == "이번 주 함께 유연성 기르기"
    assert "토요일" not in missions[0]["copy"]["parent"]
    assert tally.rewritten == 2


# ── ④ 차례 ────────────────────────────────────────────────────────────────


def test_missions_come_out_by_date_with_the_weekly_last():
    def mission(kind: str, day: str, ref: str) -> dict:
        return {"kind": kind, "period": {"start_date": day}, "participants": [{"ref": ref}]}

    answered = [
        mission("주간", "2026-10-05", "a"),
        mission("일간", "2026-10-09", "a"),
        mission("일간", "2026-10-05", "a"),
        mission("일간", "2026-10-05", "b"),
        mission("일간", "2026-10-07", "b"),
    ]
    order = [
        (m["kind"], m["period"]["start_date"], m["participants"][0]["ref"])
        for m in compose._in_order(answered)
    ]
    assert order == [
        ("일간", "2026-10-05", "a"),
        ("일간", "2026-10-05", "b"),
        ("일간", "2026-10-07", "b"),
        ("일간", "2026-10-09", "a"),
        ("주간", "2026-10-05", "a"),
    ]
