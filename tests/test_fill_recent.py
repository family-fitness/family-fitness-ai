"""코치가 고른 뒤 한 회를 채울 때(_fill) 최근에 받은 영상과 한 영상 몰아 쓰기를 보는지.

코치에게 「최근 이 true 면 모자랄 때만 고른다」고 부탁해도 LLM 은 최근 영상을 그대로
고른다. 그러면 같은 아이에게 같은 영상이 날마다 나왔다(서준 p12bb2zMebw 8일 모두).
한 회 안에서도 한 영상의 클립이 일곱 칸 중 다섯 칸을 채웠다.

클립 표 · 인덱스 · LLM 없이 돈다. 클립을 여기서 만든다.
"""

from __future__ import annotations

import collections
from datetime import date, timedelta

from family_fitness_ai.coach import compose
from family_fitness_ai.video import catalog

WANT = {"준비운동": 0, "본운동": 3, "정리운동": 0}


def _clip(video_id: str, name: str, phase: str = "본운동", start: int = 0) -> catalog.Clip:
    return catalog.Clip(
        video_id=video_id,
        name=name,
        exercise_name="",
        fitness_factor="심폐지구력",
        phase=phase,
        start_sec=start,
        end_sec=start + 40,
        age_group="유소년",
        quiet=True,
        home_ok=True,
        needs_props=False,
    )


def _offered(**phases: list[catalog.Clip]) -> dict[str, list[catalog.Clip]]:
    out: dict[str, list[catalog.Clip]] = {phase: [] for phase in catalog.PHASES}
    for phase, clips in (("본운동", phases.get("main", [])), ("정리운동", phases.get("cool", []))):
        out[phase] = clips
    return out


def test_a_recent_video_the_coach_picked_goes_behind_fresh_ones():
    old = _clip("p12bb2zMebw", "제자리 뛰기")
    fresh = [
        _clip("00401", "앞뒤 뛰기"),
        _clip("00404", "옆으로 뛰기"),
        _clip("00405", "무릎 올리기"),
    ]
    chosen = compose._fill(
        _offered(main=[old]),
        [old, *fresh],
        WANT,
        set(),
        compose.Tally(),
        recent=frozenset({"p12bb2zMebw"}),
    )
    assert [clip.video_id for _, clip in chosen] == ["00401", "00404", "00405"]


def test_a_recent_video_is_used_again_when_nothing_fresh_is_left():
    old = [_clip("p12bb2zMebw", f"동작{n}", start=n * 60) for n in range(3)]
    chosen = compose._fill(
        _offered(main=old[:1]),
        old,
        WANT,
        set(),
        compose.Tally(),
        recent=frozenset({"p12bb2zMebw"}),
    )
    # 비워 두지 않는다.
    assert len(chosen) == 3


def test_one_video_does_not_fill_several_slots_of_one_mission():
    same = [_clip("p12bb2zMebw", f"동작{n}", start=n * 60) for n in range(3)]
    other = [_clip("00401", "앞뒤 뛰기"), _clip("00404", "옆으로 뛰기")]
    chosen = compose._fill(_offered(main=same), [*same, *other], WANT, set(), compose.Tally())
    videos = [clip.video_id for _, clip in chosen]
    assert len(videos) == 3
    assert videos.count("p12bb2zMebw") == 1


def test_one_video_fills_several_slots_when_there_is_nothing_else():
    # 유아기처럼 후보가 한 영상에서만 나오면 지금처럼 다시 쓴다.
    same = [_clip("gOrg8Lva-8A", f"동작{n}", start=n * 60) for n in range(3)]
    chosen = compose._fill(_offered(main=same[:1]), same, WANT, set(), compose.Tally())
    assert len(chosen) == 3


def test_an_all_recent_phase_does_not_give_the_same_clip_every_day():
    """유아기 정리운동 후보 일곱이 모두 한 영상(gOrg8Lva-8A)에서 나온다. 그 영상이 늘
    최근이라 순서가 그대로 남아 14일 중 13일 같은 클립이 나왔다. 코치가 날마다 같은
    클립을 골라도 날짜로 섞어 돌린다."""
    cool = [_clip("gOrg8Lva-8A", f"스트레칭{n}", "정리운동", n * 30) for n in range(7)]
    want = {"준비운동": 0, "본운동": 0, "정리운동": 1}
    days = []
    for n in range(14):
        chosen = compose._fill(
            _offered(cool=cool[:1]),
            cool,
            want,
            set(),
            compose.Tally(),
            recent=frozenset({"gOrg8Lva-8A"}),
            seed=(date(2026, 9, 30) + timedelta(days=n)).isoformat(),
        )
        days.append(chosen[0][1].name)
    assert max(collections.Counter(days).values()) <= 4
    assert len(set(days)) >= 4


def test_an_all_recent_phase_in_the_rule_order_moves_with_the_date():
    cool = [_clip("gOrg8Lva-8A", f"스트레칭{n}", "정리운동", n * 30) for n in range(7)]
    days = [
        catalog._defer_recent(
            cool,
            "",
            frozenset({"gOrg8Lva-8A"}),
            (date(2026, 9, 30) + timedelta(days=n)).isoformat(),
        )[0].name
        for n in range(14)
    ]
    assert max(collections.Counter(days).values()) <= 4
