"""보호자가 키워 주고 싶은 역량(focus_factor)을 골랐으면 본운동 칸의 4분의 3 이상이 그 역량인지.

네 칸이면 세 칸, 칸 수가 다르면 올림이다. 전에는 처방에 나온 동작이 요인보다 앞서
네 칸 가운데 두 칸만 그 역량이었다. 그 역량 후보가 모자라면 최근에 받은 영상 ·
이번 주에 쓴 동작이라도 그 역량으로 먼저 채우고, 그래도 모자라면 다른 요인으로
채운다. 준비 · 정리운동은 바꾸지 않는다. 고르지 않았으면 전과 같다.

_fill 시험은 클립을 여기서 만들어 표 없이 돈다. routine · build 시험은 실제 표를 쓴다.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from conftest import needs_embedder, needs_index, needs_release

from family_fitness_ai.coach import compose
from family_fitness_ai.video import catalog

WANT = {"준비운동": 0, "본운동": 4, "정리운동": 0}


def _clip(video_id: str, name: str, factor: str, start: int = 0) -> catalog.Clip:
    return catalog.Clip(
        video_id=video_id,
        name=name,
        exercise_name="",
        fitness_factor=factor,
        phase="본운동",
        start_sec=start,
        end_sec=start + 40,
        age_group="유소년",
        quiet=True,
        home_ok=True,
        needs_props=False,
    )


def _main(clips: list[catalog.Clip]) -> dict[str, list[catalog.Clip]]:
    return {"준비운동": [], "본운동": clips, "정리운동": []}


def _factors(chosen: list[tuple[str, catalog.Clip]]) -> list[str]:
    return [clip.fitness_factor for phase, clip in chosen if phase == "본운동"]


@pytest.mark.parametrize(("slots", "quota"), [(1, 1), (3, 3), (4, 3), (5, 4), (6, 5)])
def test_the_share_is_three_quarters_rounded_up(slots: int, quota: int):
    assert catalog.focus_quota(slots) == quota


# ── 코치 편성의 채우기(_fill) ─────────────────────────────────────────────


COACH = [_clip(f"s{n}", f"근력 동작{n}", "근력") for n in range(4)]
FLEX = [_clip(f"f{n}", f"유연성 동작{n}", "유연성") for n in range(4)]


def test_the_fill_swaps_the_coachs_other_factors_for_the_focus():
    tally = compose.Tally()
    chosen = compose._fill(_main(COACH), [*COACH, *FLEX], WANT, set(), tally, focus="유연성")
    assert _factors(chosen).count("유연성") == 3
    # 나머지 한 칸은 코치가 고른 것이다.
    assert [clip for _, clip in chosen if clip.fitness_factor == "근력"] == COACH[:1]
    assert tally.filled == 3


def test_the_focus_comes_before_fresh_videos():
    recent = frozenset(clip.video_id for clip in FLEX)
    chosen = compose._fill(
        _main(COACH), [*COACH, *FLEX], WANT, set(), compose.Tally(), recent, focus="유연성"
    )
    assert _factors(chosen).count("유연성") == 3


def test_the_focus_comes_before_this_weeks_exercises():
    week = {clip.title for clip in FLEX}
    chosen = compose._fill(
        _main(COACH), [*COACH, *FLEX], WANT, week, compose.Tally(), focus="유연성"
    )
    assert _factors(chosen).count("유연성") == 3


def test_the_focus_comes_before_one_video_per_slot():
    same = [_clip("one", f"유연성 동작{n}", "유연성", start=n * 60) for n in range(3)]
    chosen = compose._fill(
        _main(COACH), [*COACH, *same], WANT, set(), compose.Tally(), focus="유연성"
    )
    assert _factors(chosen).count("유연성") == 3


def test_when_the_focus_runs_out_other_factors_fill_the_rest():
    chosen = compose._fill(
        _main(COACH), [*COACH, FLEX[0]], WANT, set(), compose.Tally(), focus="유연성"
    )
    assert sorted(_factors(chosen)) == ["근력", "근력", "근력", "유연성"]


def test_without_a_focus_the_coachs_picks_stay():
    tally = compose.Tally()
    chosen = compose._fill(_main(COACH), [*COACH, *FLEX], WANT, set(), tally)
    assert [clip for _, clip in chosen] == COACH
    assert tally.filled == 0


# ── 규칙 편성(catalog.routine) ────────────────────────────────────────────


def _main_factors(picked: dict[str, list[catalog.Clip]]) -> list[str]:
    return [clip.fitness_factor for clip in picked["본운동"]]


@needs_release
@pytest.mark.parametrize(
    ("age_group", "focus", "minutes"),
    [
        ("유소년", "유연성", 15),
        ("청소년", "협응력", 15),
        ("성인", "심폐지구력", 30),
        ("유아기", "근력", 45),
    ],
)
def test_the_rule_routine_fills_three_quarters_with_the_focus(
    age_group: str, focus: str, minutes: int
):
    picked = catalog.routine(age_group, minutes, factor=focus, focus=focus, seed="2026-09-07")
    main = _main_factors(picked)
    assert len(main) == catalog.clip_counts(minutes)["본운동"]
    assert main.count(focus) >= catalog.focus_quota(len(main)), main


@needs_release
def test_a_week_with_few_candidates_reuses_the_focus_before_other_factors():
    """유아기 유연성 본운동은 네 동작뿐이다. 사흘에 세 칸씩이면 지난날 쓴 것을 다시 쓴다."""
    used: set[str] = set()
    for day in range(3):
        picked = catalog.routine(
            "유아기", 15, factor="유연성", focus="유연성", exclude=used, seed=f"2026-09-0{day + 1}"
        )
        assert _main_factors(picked).count("유연성") >= 3, (day, _main_factors(picked))
        used |= {clip.title for clips in picked.values() for clip in clips}


@needs_release
def test_recent_focus_videos_come_before_fresh_other_factors():
    recent = frozenset(
        clip.video_id
        for clip in catalog.clips()
        if clip.age_group == "유소년" and clip.fitness_factor == "유연성"
    )
    picked = catalog.routine(
        "유소년", 15, factor="유연성", focus="유연성", recent=recent, seed="2026-09-07"
    )
    assert _main_factors(picked).count("유연성") >= 3


@needs_release
def test_a_focus_with_one_exercise_fills_the_rest_with_other_factors():
    """유소년 평형성 본운동은 한 동작뿐이다. 한 회에 같은 동작을 두 번 넣지 않는다."""
    picked = catalog.routine("유소년", 15, factor="평형성", focus="평형성", seed="2026-09-07")
    main = _main_factors(picked)
    assert main.count("평형성") == 1
    assert len(main) == 4


@needs_release
def test_warm_up_and_cool_down_do_not_change_with_the_focus():
    plain = catalog.routine("유소년", 15, factor="유연성", seed="2026-09-07")
    focused = catalog.routine("유소년", 15, factor="유연성", focus="유연성", seed="2026-09-07")
    assert plain["준비운동"] == focused["준비운동"]
    assert plain["정리운동"] == focused["정리운동"]


# ── 편성 전체(compose.build) ──────────────────────────────────────────────

CHILD = compose.RunProfile(
    ref="p_c7a91f",
    role="주행자",
    age=11,
    age_unit="세",
    sex="F",
    input_level="L2",
    measurements={"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
)


def _mission_mains(plan: compose.Plan) -> list[list[str]]:
    assert plan.proposal is not None
    return [
        [s["fitness_factor"] for s in m["sessions"] if s["phase"] == "본운동"]
        for m in plan.proposal["missions"]
    ]


@needs_release
@needs_index
@needs_embedder
def test_the_rule_plan_fills_three_quarters_with_the_focus():
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints(focus_factor="유연성"))
    mains = _mission_mains(plan)
    assert mains
    for main in mains:
        assert main.count("유연성") >= catalog.focus_quota(len(main)), main


def _coach_picks_other_factors(focus: str) -> Any:
    """본운동을 모두 그 역량이 아닌 클립으로 고르는 가짜 코치."""

    def plan(payload: dict[str, Any]) -> list[dict[str, Any]]:
        plan.seen = payload  # type: ignore[attr-defined]
        clips = payload["클립"]
        warm = next(c for c in clips if c["단계"] == "준비운동")
        cool = next(c for c in clips if c["단계"] == "정리운동")
        # 한 영상에서 하나씩만 — 한 회 안의 같은 영상 규칙으로 바뀌지 않게.
        seen_videos: set[str] = set()
        others = []
        for c in clips:
            if c["단계"] == "본운동" and c["요인"] != focus and c["영상"] not in seen_videos:
                seen_videos.add(c["영상"])
                others.append(c)
        plan.mains = []  # type: ignore[attr-defined]
        days = []
        for index, slot in enumerate(payload["자리"]):
            mains = others[index * 4 : index * 4 + 4]
            rows = [warm, *mains, cool]
            plan.mains.append([c["이름"] for c in mains])  # type: ignore[attr-defined]
            days.append(
                {
                    "day_offset": slot["day_offset"],
                    "title": "",
                    "child": "",
                    "parent": "",
                    "reason": "",
                    "clips": [{"id": c["id"], "phase": c["단계"]} for c in rows],
                }
            )
        return days

    return plan


@needs_release
@needs_index
@needs_embedder
def test_the_coach_plan_is_filled_to_three_quarters_with_the_focus(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _coach_picks_other_factors("유연성")
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", fake)
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints(focus_factor="유연성"))
    assert "코치가 편성" in plan.steps[2].summary
    mains = _mission_mains(plan)
    assert len(mains) == 3
    for main in mains:
        assert main.count("유연성") >= catalog.focus_quota(len(main)), main
    # 코치에게도 몇 칸을 그 역량으로 채울지 알려 준다.
    assert all(slot["본운동_대상_요인_편수"] == 3 for slot in fake.seen["자리"])
    assert "4분의 3" in fake.seen["요청"]


@needs_release
@needs_index
@needs_embedder
def test_without_a_focus_the_coach_plan_is_not_swapped(monkeypatch: pytest.MonkeyPatch):
    lowest, *_ = compose.target_factor(CHILD)
    fake = _coach_picks_other_factors(lowest)
    monkeypatch.setattr(compose.coach_llm, "enabled", lambda: True)
    monkeypatch.setattr(compose.coach_llm, "plan_week", fake)
    plan = compose.build([CHILD], date(2026, 9, 7), 1, compose.Constraints())
    assert plan.proposal is not None
    names = [
        [s["exercise_name"] for s in m["sessions"] if s["phase"] == "본운동"]
        for m in plan.proposal["missions"]
    ]
    # 코치가 고른 본운동이 그대로 나간다 — 대상 요인 클립으로 바꾸지 않는다.
    assert names == fake.mains
    assert all("본운동_대상_요인_편수" not in slot for slot in fake.seen["자리"])
    assert "4분의 3" not in fake.seen["요청"]
