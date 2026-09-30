"""같은 아이에게 날마다 같은 영상이 나오지 않는지.

BE 는 하루씩 편성을 부른다(days_per_week 1 · weeks 1). 같은 아이 · 같은 조건이면
예전에는 순위가 늘 같아 날마다 같은 영상이 나왔다. 두 가지로 바꾼다.

- recent_video_ids — 그 사람이 최근 14일 동안 받은 영상. 뒤로 미룬다. 같은 요인 ·
  단계 · 연령의 다른 후보가 모자랄 때만 다시 쓴다.
- start_date — 같은 순위끼리는 날짜를 시드로 삼아 섞는다. recent 가 없어도 날마다
  다른 것이 앞에 온다.

LLM 과 임베딩 없이 돈다. 클립 표와 코퍼스만 있으면 된다.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.api import app as api
from family_fitness_ai.api import schemas
from family_fitness_ai.coach import compose
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import catalog

DAY_ONE = date(2026, 10, 5)
DAY_TWO = date(2026, 10, 6)
CHILD = compose.RunProfile(ref="p", role="주행자", age=11, age_unit="세", sex="F")
ME = [{"ref": "p", "role": "주행자"}]


def _day_plan(start: date, recent: tuple[str, ...] = ()) -> list[str]:
    """BE 의 하루 편성처럼 한 날 한 회를 규칙으로 짜고, 나온 영상 id 를 차례대로."""
    chunk = next(
        c for c in corpus().chunks if c.source == "prescription" and c.age_group == "유소년"
    )
    citations = compose.Citations()
    evidence = [citations.add(chunk)]
    read = compose.Read(CHILD, "유연성", "growth", 24, [], [chunk], [])
    constraints = compose.Constraints(days_per_week=1, recent_video_ids=recent)
    slots = [compose.Slot("일간", 0, 15, ME)]
    missions = compose._by_rule(read, slots, constraints, evidence, citations, start, 1)
    return [s["video"]["video_id"] for m in missions for s in m["sessions"]]


@needs_release
@needs_index
def test_yesterdays_videos_do_not_lead_today():
    yesterday = _day_plan(DAY_ONE)
    today = _day_plan(DAY_TWO, recent=tuple(yesterday))
    assert yesterday and today
    assert not set(today) & set(yesterday)


@needs_release
@needs_index
def test_without_recent_a_new_day_still_changes_the_order():
    # 같은 순위끼리 날짜로 섞는다. 7일 가운데 하루라도 첫날과 다르면 된다 — 순위가
    # 확실히 앞선 영상은 날마다 앞에 와도 된다.
    first = _day_plan(DAY_ONE)
    others = [_day_plan(date(2026, 10, 5 + n)) for n in range(1, 7)]
    assert any(day != first for day in others)
    # 같은 날짜로 다시 부르면 같다 — 되짚을 수 있어야 한다.
    assert _day_plan(DAY_ONE) == first


@needs_release
@needs_index
def test_the_pool_order_moves_with_the_date_too():
    one, _ = catalog.pool("유소년", factor="유연성", seed=DAY_ONE.isoformat())
    days = [
        catalog.pool("유소년", factor="유연성", seed=date(2026, 10, 5 + n).isoformat())[0]
        for n in range(1, 7)
    ]
    assert any([c.video_id for c in day] != [c.video_id for c in one] for day in days)


@needs_release
@needs_index
def test_recent_videos_are_reused_when_nothing_else_is_left():
    # 그 연령대 정리운동을 전부 최근에 받았다고 하자. 비워 두지 않고 다시 쓴다.
    cooldown = {
        clip.video_id
        for clip in catalog.clips()
        if clip.age_group == "유소년" and clip.phase == "정리운동"
    }
    picked = catalog.routine("유소년", 15, factor="유연성", recent=frozenset(cooldown))
    assert picked["정리운동"]


@needs_release
@needs_index
def test_recent_videos_go_behind_fresh_ones_of_the_same_factor_in_the_pool():
    pool, _ = catalog.pool("유소년", factor="유연성")
    main = [clip for clip in pool if clip.phase == "본운동" and clip.fitness_factor == "유연성"]
    assert len(main) >= 2
    recent = frozenset({main[0].video_id})
    again, _ = catalog.pool("유소년", factor="유연성", recent=recent)
    order = [clip for clip in again if clip.phase == "본운동" and clip.fitness_factor == "유연성"]
    fresh = [clip for clip in order if clip.video_id not in recent]
    stale = [clip for clip in order if clip.video_id in recent]
    # 같은 요인의 다른 후보가 모두 앞선다. 최근 영상은 빠지지 않고 뒤에 선다.
    assert stale and order[: len(fresh)] == fresh


def test_the_coach_sees_which_clips_are_recent(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}

    def plan(payload):
        seen.update(payload)
        return None

    monkeypatch.setattr(compose.coach_llm, "plan_week", plan)
    clip = catalog.Clip(
        video_id="0AUDLJ08S_00351",
        name="a",
        exercise_name="",
        fitness_factor="유연성",
        phase="본운동",
        start_sec=0,
        end_sec=40,
        age_group="유소년",
        quiet=True,
        home_ok=True,
        needs_props=False,
    )
    read = compose.Read(CHILD, "유연성", "growth", 24, [], [], [])
    constraints = compose.Constraints(recent_video_ids=("0AUDLJ08S_00351",))
    compose._by_llm(
        read,
        [clip],
        [compose.Slot("일간", 0, 15, ME)],
        constraints,
        [],
        compose.Citations(),
        DAY_ONE,
        1,
        compose.Tally(),
    )
    assert seen["클립"][0]["최근"] is True


def test_the_request_takes_recent_video_ids(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class Started:
        run_id = "r"
        status = "running"

    def start(profiles, start_date, weeks, constraints):
        captured["constraints"] = constraints
        return Started()

    monkeypatch.setattr(api.store, "start", start)
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    many = [f"v{n}" for n in range(200)]
    response = client.post(
        "/v1/coach/runs",
        json={
            "profile_refs": [{"ref": "p", "role": "주행자", "age": 11, "sex": "F"}],
            "period": {"start_date": "2026-10-05"},
            "constraints": {"days_per_week": 1, "recent_video_ids": ["0AUDLJ08S_00351", *many]},
        },
    )
    assert response.status_code == 202
    recent = captured["constraints"].recent_video_ids
    # 최근 것부터 150개(RECENT_LIMIT)까지만 본다. 모르는 id 는 거르지 않고 그대로 둔다
    # — 고를 때 안 걸릴 뿐이다.
    assert recent[0] == "0AUDLJ08S_00351" and len(recent) == schemas.RECENT_LIMIT == 150


def test_the_request_without_recent_video_ids_still_works(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class Started:
        run_id = "r"
        status = "running"

    def start(profiles, start_date, weeks, constraints):
        captured["constraints"] = constraints
        return Started()

    monkeypatch.setattr(api.store, "start", start)
    from fastapi.testclient import TestClient

    response = TestClient(api.app).post(
        "/v1/coach/runs",
        json={
            "profile_refs": [{"ref": "p", "role": "주행자", "age": 11, "sex": "F"}],
            "period": {"start_date": "2026-10-05"},
        },
    )
    assert response.status_code == 202
    assert captured["constraints"].recent_video_ids == ()
