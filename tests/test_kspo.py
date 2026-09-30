"""공단 영상 한 편을 클립 하나로 표에 옮기고, 유튜브 클립과 한 목록에서 똑같이 고르나.

API 를 부르지 않는다. 받아 둔 원자료에서 사례마다 한 영상의 장면 줄을 통째로 잘라
둔 것(tests/fixtures/kspo_rows.json)을 넣는다 — 지어내지 않는다. 영상 전체에 한 값인
칸(제목·설명·길이·연령대)은 크기를 줄이려고 첫 줄에만 남겼다. 표준운동은 단계를
빌려 오는 곳이라 「목 스트레칭」 장면만 잘라 두었다.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.coach import compose, verify
from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import catalog, kspo

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "kspo_rows.json").read_text("utf-8"))
CASES = FIXTURE["cases"]
has_table = pytest.mark.skipif(
    not (settings().release_dir / kspo.OUT).exists(), reason="kspo_videos.csv 가 없다"
)


@lru_cache
def _table() -> tuple[tuple[tuple[str, str], ...], ...]:
    table = kspo.build(FIXTURE["rows"], FIXTURE["alive"])
    return tuple(tuple(row.items()) for row in table)


def _rows(case: str) -> list[dict[str, str]]:
    video_id = Path(CASES[case]).stem
    return [dict(row) for row in _table() if dict(row)["video_id"] == video_id]


def _clip(**given) -> catalog.Clip:
    base = dict(
        video_id="v", name="n", exercise_name="", fitness_factor="", phase="본운동",
        start_sec=0, end_sec=60, age_group="청소년", quiet=True, home_ok=True, needs_props=False,
    )  # fmt: skip
    return catalog.Clip(**{**base, **given})


# ── 읽기 ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("written", "read"),
    [("3~5", (3, 5)), ("1~2", (1, 2)), ("3", (3, 3)), ("중급", (2, 4)), ("공통", (1, 5)),
     ("", (1, 5)), ("3~5, 1~5", (1, 5))],
)  # fmt: skip
def test_levels_read_ranges_words_and_blanks(written, read):
    assert kspo.levels(written) == read


@pytest.mark.parametrize(
    ("written", "read"),
    [("준비 운동", "준비운동"), ("본 운동(순환식 근력 운동)", "본운동"), ("정리 운동", "정리운동"),
     ("스트레칭", ""), ("공통", ""), ("", "")],
)  # fmt: skip
def test_the_api_phase_is_read_only_when_it_is_a_phase(written, read):
    assert kspo.api_phase(written) == read


def test_names_keep_their_brackets_but_the_join_key_drops_them():
    # 보이는 이름은 단계 번호만 뗀다. 괄호가 다른 동작을 가른다.
    assert kspo.display("앉았다 일어서기(매트/바벨)-1") == "앉았다 일어서기(매트/바벨)"
    # 다른 조회의 같은 동작은 괄호와 번호를 떼고 찾는다.
    assert kspo.join_key("팔 굽혀 펴기") == kspo.join_key("팔 굽혀 펴기(매트)")
    assert kspo.join_key("목 스트레칭(Neck stretch)") == kspo.join_key("목 스트레칭")


# ── 한 편 = 클립 하나 ──────────────────────────────────────────────────────


@needs_index
def test_every_video_is_one_clip_from_start_to_end():
    """끊지 않는다. 시작은 0, 끝과 길이는 영상 길이 — 서버가 주고받는 모양을 지킨다."""
    rows = [dict(r) for r in _table()]
    assert rows
    for row in rows:
        assert row["seq"] == "1"
        assert row["start_sec"] == "0"
        assert row["end_sec"] == row["duration_sec"]
        assert 0 < int(row["duration_sec"]) < kspo.MAX_SEC
    assert {r["duration_sec"] for r in _rows("가이드 · 요인·수준이 제 칸에")} == {"91"}


@needs_index
def test_a_guide_brings_its_own_factor_and_level():
    rows = _rows("가이드 · 요인·수준이 제 칸에")
    assert {r["fitness_factor"] for r in rows} == {"근력", "근지구력"}  # 「근력/근지구력」
    assert {(r["level_lo"], r["level_hi"]) for r in rows} == {("2", "4")}  # 중급
    assert {r["age_group"] for r in rows} == {"유소년"}
    assert rows[0]["needs_props"] == "False"  # 매트는 집에 있다
    assert rows[0]["url"].startswith("https://")  # API 는 http 로 준다


@needs_index
def test_a_prescription_video_borrows_factor_and_level_from_the_guide_of_the_same_move():
    """운동처방동영상에는 요인·수준 칸이 없다. 같은 운동명의 가이드가 준 값을 쓴다."""
    rows = _rows("처방동영상 · 요인·수준을 같은 이름의 가이드에서")
    assert {r["fitness_factor"] for r in rows} == {"근력", "근지구력"}
    assert {(r["level_lo"], r["level_hi"]) for r in rows} == {("2", "4")}
    assert {r["age_group"] for r in rows} == {"청소년", "성인"}  # 「공통」
    assert rows[0]["exercise_name"] == "팔굽혀펴기"


@needs_index
def test_without_an_api_factor_the_factor_stays_empty():
    """API 어디에도 요인이 없으면 비워 둔다. 짐작해 채우지 않는다."""
    rows = _rows("처방동영상 · API 에 요인이 없음")
    assert {r["fitness_factor"] for r in rows} == {""}
    assert {(r["level_lo"], r["level_hi"]) for r in rows} == {("1", "5")}


@needs_index
def test_a_phase_is_borrowed_from_the_standard_program():
    """짧은 영상 조회에는 단계 칸이 없다. 표준운동이 같은 동작을 정리 운동이라 적었다."""
    rows = _rows("가이드 · 단계를 같은 이름의 표준운동에서")
    assert {r["phase"] for r in rows} == {"정리운동"}
    assert {r["phase_on_video"] for r in rows} == {""}


@needs_index
def test_a_flexibility_move_without_a_borrowed_phase_goes_to_both_ends():
    rows = _rows("가이드 · 유연성 → 준비·정리")
    assert {r["phase"] for r in rows} == {"준비운동", "정리운동"}


@needs_index
def test_jumping_is_not_quiet_and_a_ladder_needs_room():
    row = _rows("뛰는 동작 · 사다리")[0]
    assert (row["quiet"], row["home_ok"], row["needs_props"]) == ("False", "False", "True")


@needs_index
@pytest.mark.parametrize(
    "case", ["변형이 이어지는 짧은 영상 → 제목", "운동명이 어느 장면에도 없음 → 제목"]
)
def test_a_short_video_without_one_move_name_uses_its_title(case):
    """짧으면 운동이 든 영상이다. 한 클립으로 쓰고 제목을 이름으로 한다."""
    rows = _rows(case)
    assert rows
    assert {r["name_on_video"] for r in rows} == {"줄넘기"}


@needs_index
@pytest.mark.parametrize(
    "case",
    [
        "5분 이상",
        "루틴 프로그램 (5분 미만이라도)",
        "이름에 루틴",
        "질환 · 오십견",
        "질환 · 발목염좌 (설명)",
        "헬스장 · 실내자전거",
        "2인 이상",
        "어르신",
        "링크 죽음 (열림을 거짓으로 둠)",
        "짝 운동 (제목)",
    ],
)
def test_what_we_do_not_use_is_left_out(case):
    assert _rows(case) == []


@needs_index
def test_the_table_has_the_clip_and_label_columns():
    """video_clips.csv 와 clip_labels.csv 의 칸이 다 있어야 편성이 같은 길로 읽는다."""
    for row in (dict(r) for r in _table()):
        assert list(row) == list(kspo.COLUMNS)
        assert row["phase"] in kspo.PHASES
        assert row["is_exercise"] == "True"


@needs_index
def test_the_table_does_not_follow_row_order():
    """두 번 돌린 표가 같아야 한다 — 받은 차례가 바뀌어도."""
    flipped = {op: list(reversed(rows)) for op, rows in FIXTURE["rows"].items()}
    assert kspo.build(flipped, FIXTURE["alive"]) == [dict(r) for r in _table()]


# ── 고르기 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("percentile", "level"), [(0, 1), (19, 1), (20, 2), (55, 3), (99, 5)])
def test_a_percentile_maps_to_a_fitness_level(percentile, level):
    assert catalog.level_of(percentile) == level
    assert catalog.level_of(None) is None


def test_a_level_that_does_not_fit_goes_back_not_out():
    """수준이 안 맞는 영상은 뒤로 미룬다. 빼지는 않는다."""
    fits = _clip(level_lo=1, level_hi=2)
    hard = _clip(level_lo=4, level_hi=5)
    assert catalog._rank(fits, "", set(), level=2) < catalog._rank(hard, "", set(), level=2)
    # 측정이 없으면 수준을 보지 않는다.
    assert catalog._rank(fits, "", set()) == catalog._rank(hard, "", set())


def test_both_sources_play_from_the_same_shape():
    youtube = _clip(video_id="p12bb2zMebw", start_sec=1206, end_sec=1274)
    kspo_clip = _clip(source="kspo", url="https://openapi.kspo.or.kr/web/video/x.mp4")
    assert youtube.as_video().keys() == kspo_clip.as_video().keys()
    assert youtube.as_video()["url"].endswith("t=1206s")
    assert kspo_clip.as_video()["source"] == "kspo"


@needs_release
@needs_index
@has_table
def test_both_sources_share_one_list_and_kspo_is_cited_from_the_table():
    clips = catalog.clips()
    assert {c.source for c in clips} == {"youtube", "kspo"}
    kspo_clips = [c for c in clips if c.source == "kspo"]
    # 한 편이 클립 하나다 — 끊지 않았다.
    assert all(c.start_sec == 0 for c in kspo_clips)
    first = kspo_clips[0]
    chunk = catalog.citation_for(first.video_id)
    assert chunk is not None
    assert chunk.chunk_id == f"kspo:{first.video_id}"
    assert chunk.citation_url == first.url


@needs_release
@needs_index
@has_table
def test_a_kspo_pick_passes_verification(monkeypatch: pytest.MonkeyPatch):
    """코퍼스에 없는 영상도 인용이 서야 한다 — 없으면 검증이 제안을 통째로 버린다."""
    pool, _ = catalog.pool("청소년", factor="민첩성")
    ids = {clip: f"c{i}" for i, clip in enumerate(pool)}
    picks = [
        {"id": ids[c], "phase": c.phase} for c in pool if c.source == "kspo" and c.phase == "본운동"
    ][:4]
    assert picks, "청소년 민첩성 후보에 공단 영상이 없다"
    day = {
        "day_offset": 0,
        "title": "빠르게 움직이기",
        "child": "몸을 빠르게 움직여 볼까요",
        "parent": "하루 15분이면 충분합니다",
        "reason": "또래 처방에 나온 동작입니다 [1].",
        "clips": picks,
    }
    monkeypatch.setattr(compose.coach_llm, "plan_week", lambda payload: [day])
    chunk = next(
        c for c in corpus().chunks if c.source == "prescription" and c.age_group == "청소년"
    )
    citations = compose.Citations()
    evidence = [citations.add(chunk)]
    teen = compose.RunProfile(ref="t", role="주행자", age=15, age_unit="세", sex="M")
    read = compose.Read(teen, "민첩성", "growth", 20, [], [chunk], [])
    missions = compose._by_llm(
        read,
        pool,
        [compose.Slot("일간", 0, 15, [{"ref": "t", "role": "주행자"}])],
        compose.Constraints(),
        evidence,
        citations,
        date(2026, 10, 5),
        1,
        compose.Tally(),
    )
    assert missions is not None
    chosen = [s for s in missions[0]["sessions"] if s["video"]["source"] == "kspo"]
    assert chosen
    assert all(s["video"]["start_sec"] == 0 for s in chosen)
    proposal = {"missions": missions, "citations": citations.dump()}
    assert verify.check_proposal(proposal, {"청소년"}) == []
    cited = {c["chunk_id"] for c in citations.dump()}
    assert all(f"kspo:{s['video']['video_id']}" in cited for s in chosen)


# ── 영상 찾기: 한 요청에서 두 풀 ────────────────────────────────────────────


@needs_release
@needs_index
@has_table
def test_video_search_finds_kspo_clips_by_name_and_factor():
    from family_fitness_ai.video.videos import KSPO_BOTH, KSPO_NAME, kspo_hits

    hits = kspo_hits("청소년", (), ("팔 굽혀 펴기",))
    assert hits
    assert all(h["score"] >= KSPO_NAME and h["matched_exercise_names"] for h in hits)
    assert len({h["video_id"] for h in hits}) == len(hits)
    both = kspo_hits("유소년", ("유연성",), ("다리 벌려 앞으로 상체 숙이기",))
    assert max(h["score"] for h in both) == KSPO_BOTH
    # 연령대는 거르는 조건이다.
    assert kspo_hits("유아기", ("유연성",), ()) == []


@needs_release
@needs_index
@has_table
def test_every_video_hit_says_where_to_play_it(monkeypatch: pytest.MonkeyPatch):
    """유튜브와 공단 영상이 한 응답에 섞여 나온다. 어느 쪽이든 source·url 로 튼다."""
    from family_fitness_ai.rag import search as rag_search
    from family_fitness_ai.video import videos

    # 유튜브 쪽 검색은 임베딩이 있어야 돈다. 여기서는 빈 결과로 두고 합치는 길만 본다.
    empty = rag_search.Result(hits=[], filtered_out={})
    monkeypatch.setattr(videos, "search", lambda *a, **k: empty)
    out = videos.search_videos("유소년", ("유연성",), ("다리 벌려 앞으로 상체 숙이기",), k=3)
    assert out["hits"]
    for hit in out["hits"]:
        assert {"source", "video_id", "url", "start_sec", "end_sec", "score"} <= hit.keys()
        assert hit["start_sec"] == 0
    assert out["hits"][0]["source"] == "kspo"
    scores = [h["score"] for h in out["hits"]]
    assert scores == sorted(scores, reverse=True)


# ── 어르신: 성인 영상을 또래로 ──────────────────────────────────────────────
# 팀 결정 — 어르신 전용 영상을 따로 만들지 않고 성인 영상(공단 「공통」 포함)을 똑같이
# 쓴다. 공단 어르신 영상은 싣지 않으므로, 성인을 또래로 치지 않으면 65세 이상은
# 유아기·유소년 영상까지 섞인 후보를 받고, 규칙 편성은 한 편도 못 고른다.


def test_seniors_share_the_adult_age_group():
    assert catalog.ages_for("어르신") == ("어르신", "성인")
    assert catalog.ages_for("유소년") == ("유소년",)


@needs_release
@needs_index
@has_table
def test_seniors_get_adult_clips_first_and_no_notice():
    for factor in ("근력", "유연성", ""):
        pool, notice = catalog.pool("어르신", factor=factor)
        assert pool
        assert notice == ""
        assert {c.age_group for c in pool} <= {"어르신", "성인"}
        assert {c.source for c in pool} == {"youtube", "kspo"}


@needs_release
@needs_index
@has_table
def test_the_rule_plan_fills_a_senior_session():
    picked = catalog.routine("어르신", 15, factor="근력")
    assert {phase: len(clips) for phase, clips in picked.items()} == catalog.clip_counts(15)
    assert all(c.age_group == "성인" for clips in picked.values() for c in clips)


@needs_release
@needs_index
@has_table
def test_a_rule_plan_for_a_seventy_year_old_is_not_refused():
    grandma = compose.RunProfile(ref="g", role="주행자", age=70, age_unit="세", sex="F")
    plan = compose.build([grandma], date(2026, 10, 5), 1, compose.Constraints())
    assert not plan.refused, plan.steps
    assert plan.proposal is not None
    assert "notices" not in plan.proposal or not any(
        "다른 연령대" in n for n in plan.proposal["notices"]
    )


@needs_release
@needs_index
@has_table
def test_video_search_gives_seniors_the_adult_kspo_clips():
    from family_fitness_ai.video.videos import kspo_hits

    senior = kspo_hits("어르신", ("유연성",), ())
    adult = kspo_hits("성인", ("유연성",), ())
    assert senior
    assert {h["video_id"] for h in senior} == {h["video_id"] for h in adult}
