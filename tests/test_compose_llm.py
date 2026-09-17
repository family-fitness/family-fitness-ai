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
    # 지어낸 id 한 자리만 빠지고 나머지 셋은 선다.
    assert len(missions[0]["sessions"]) == 3
    assert missions[0]["title"] == "월요일 늘이기"
    assert [s["phase"] for s in missions[0]["sessions"]] == [
        "준비운동",
        "본운동",
        "정리운동",
    ]
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
