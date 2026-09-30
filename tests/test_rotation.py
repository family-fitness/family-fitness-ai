"""같은 아이에게 날마다 같은 영상이 돌지 않는지 — 최근 영상끼리의 차례.

BE 는 그 사람이 최근 14일 동안 받은 영상 id 를 최근 것부터 보낸다(recent_video_ids).
예전에는 이 목록을 집합으로만 봐서 어느 것이 오래됐는지 몰랐다. 요인이 맞는 새 후보가
떨어지면 요인이 맞는 최근 영상을 순위 그대로 다시 써, 늘 같은 상위 몇 편이 앞에 섰다.
실제로 14일을 돌려 보니 12세 순발력 본운동은 9일 내내 같은 넷, 5세 민첩성 본운동도
세 편만 돌았다. 5세는 한 회 안에서도 같은 영상이 두 번씩 나왔다(14일 중 13일).

이제 최근 영상끼리는 오래전에 받은 것부터 앞에 세운다. 같으면 날짜로 섞는다.
"""

from __future__ import annotations

import collections
from datetime import date, timedelta

from conftest import needs_index, needs_release

from family_fitness_ai.api import schemas
from family_fitness_ai.coach import compose
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import catalog

START = date(2026, 9, 30)
DAYS = 14
ME = [{"ref": "p", "role": "주행자"}]
#: 어제와 열흘 전 사이에 받은 영상들. 하루 7칸씩 아흐레.
FILLER = tuple(f"f{n}" for n in range(catalog.DAY_IDS * 9 - 3))


def _clip(video_id: str, name: str, factor: str = "순발력", start: int = 0) -> catalog.Clip:
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


# ── 차례 한 가지 ────────────────────────────────────────────────────────────


def test_the_least_recent_matching_video_comes_first_in_the_rule_order():
    """순위가 높은 요인 클립 셋을 어제 받았고 순위가 낮은 요인 클립은 열흘 전에 받았다.
    새 후보가 없으면 열흘 전 것이 앞에 선다 — 순위 그대로 두면 어제 셋이 또 나온다."""
    top = [_clip(f"top{n}", f"상위{n}") for n in range(3)]
    old = _clip("old", "오래전")
    other = _clip("x", "다른 요인", factor="근력")
    # 최근 것부터: 어제 셋 → 아흐레치 → 열흘 전 old.
    recent = ("top0", "top1", "top2", *FILLER, "old")
    ranked = catalog._defer_recent([*top, old, other], "순발력", recent, "2026-10-05")
    assert ranked[0].video_id == "old"
    # 요인이 맞는 최근 영상은 여전히 요인이 다른 새 후보보다 앞이다.
    assert [c.video_id for c in ranked].index("x") > 3


def test_fresh_matching_videos_still_lead_recent_ones():
    fresh = _clip("new", "새 동작")
    old = _clip("old", "오래전")
    ranked = catalog._defer_recent([old, fresh], "순발력", ("old",), "2026-10-05")
    assert [c.video_id for c in ranked] == ["new", "old"]


def test_the_least_recent_video_comes_first_when_the_coach_order_is_filled():
    """코치 편성 채우기(_fill)도 같다. 코치가 날마다 고르는 어제 영상보다 열흘 전 영상이 먼저다."""
    top = [_clip(f"top{n}", f"상위{n}") for n in range(3)]
    old = _clip("old", "오래전")
    offered: dict[str, list[catalog.Clip]] = {phase: [] for phase in catalog.PHASES}
    offered["본운동"] = top
    want = {"준비운동": 0, "본운동": 1, "정리운동": 0}
    recent = ("top0", "top1", "top2", *FILLER, "old")
    chosen = compose._fill(offered, [*top, old], want, set(), compose.Tally(), recent, "2026-10-05")
    assert [clip.video_id for _, clip in chosen] == ["old"]


def test_a_set_of_recent_ids_still_works():
    """차례를 모르는 집합을 넘기면 모두 같은 때 받은 것으로 보고 날짜로 섞는다."""
    top = [_clip(f"top{n}", f"상위{n}") for n in range(3)]
    ranked = catalog._defer_recent(top, "순발력", frozenset({"top0", "top1", "top2"}), "")
    assert ranked == top


def test_the_request_keeps_fourteen_days_of_recent_ids():
    # 14일 × 하루 칸 수보다 넉넉해야, 목록에서 빠진 오래된 영상이 「새 영상」 으로
    # 보여 먼저 뽑히지 않는다. 60 이면 하루 7칸(20분)에 9일쯤에 넘쳤다. 35분 회(9칸)도 담는다.
    assert 14 * sum(catalog.clip_counts(35).values()) <= schemas.RECENT_LIMIT
    many = [f"v{n}" for n in range(200)]
    kept = schemas.ConstraintsIn(recent_video_ids=many).recent_video_ids
    assert kept == many[: schemas.RECENT_LIMIT]


# ── 14일을 돌려 본다 ─────────────────────────────────────────────────────────


def _fortnight(
    age: int, factor: str, percentile: int, limit: int = schemas.RECENT_LIMIT
) -> list[list[tuple[str, str]]]:
    """BE 처럼 하루씩 규칙 편성을 부르고, 그날 받은 영상을 다음 날의 recent 로 넘긴다.

    recent 는 BE 와 같다: 최근 14일, 최근 미션부터 · 한 미션 안은 칸 차례, 겹치면
    한 번만, limit 개까지. 하루마다 (단계, 영상 id) 를 칸 차례로 돌려준다.
    """
    child = compose.RunProfile(ref="p", role="주행자", age=age, age_unit="세", sex="F")
    group = child.age_group
    chunks = [c for c in corpus().chunks if c.source == "prescription" and c.age_group == group]
    chunk = next((c for c in chunks if factor in c.factors), chunks[0])
    days: list[list[tuple[str, str]]] = []
    for n in range(DAYS):
        seen = [video for day in reversed(days[-14:]) for _, video in day]
        recent = tuple(dict.fromkeys(seen))[:limit]
        citations = compose.Citations()
        evidence = [citations.add(chunk)]
        read = compose.Read(child, factor, "growth", percentile, [], [chunk], [])
        constraints = compose.Constraints(days_per_week=1, recent_video_ids=recent)
        slots = [compose.Slot("일간", 0, 20, ME)]
        missions = compose._by_rule(
            read, slots, constraints, evidence, citations, START + timedelta(days=n), 1
        )
        days.append(
            [
                (s["phase"], s["video"]["video_id"])
                for m in missions
                for s in m["sessions"]
                if s.get("video")
            ]
        )
    return days


def _main(days: list[list[tuple[str, str]]]) -> list[list[str]]:
    return [[video for phase, video in day if phase == "본운동"] for day in days]


def _longest_run(days: list[list[str]]) -> int:
    """한 영상이 며칠 잇달아 나왔나. 가장 긴 것."""
    best = 0
    run: collections.Counter[str] = collections.Counter()
    for day in days:
        today = set(day)
        run = collections.Counter({video: run[video] + 1 for video in today})
        best = max([best, *run.values()])
    return best


@needs_release
@needs_index
def test_twelve_year_old_power_main_rotates_over_fourteen_days():
    main = _main(_fortnight(12, "순발력", 30))
    # 요인이 맞는 새 영상은 첫 이레 안에 떨어진다. 고치기 전에는 그 뒤 이레 동안 같은
    # 상위 몇 편(8편)만 돌았고, 한 영상이 나흘 잇달아 나왔다.
    later = len({video for day in main[7:] for video in day})
    assert later >= 10, later
    assert _longest_run(main) <= 3, main


@needs_release
@needs_index
def test_five_year_old_agility_main_rotates_over_fourteen_days():
    # 고치기 전: 14일 동안 12편, 한 영상이 열흘 잇달아.
    main = _main(_fortnight(5, "민첩성", 34))
    distinct = len({video for day in main for video in day})
    assert distinct >= 13, distinct
    assert _longest_run(main) <= 3, main


@needs_release
@needs_index
def test_a_five_year_old_does_not_get_one_video_twice_in_a_session():
    """다른 영상 후보가 있으면 한 회 안에 같은 영상을 두 번 넣지 않는다.
    고치기 전에는 14일 중 7일, 한 회에 같은 영상이 두 번 나왔다."""
    for day in _fortnight(5, "민첩성", 34):
        videos = [video for _, video in day]
        assert len(videos) == len(set(videos)), day
