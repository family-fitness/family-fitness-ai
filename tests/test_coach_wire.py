"""백엔드가 보내고 읽는 모양을 못박는다 (docs/03 §5·§6).

여기서 틀리면 뒤 단계가 전부 틀린다. 골든 JSON 이 정본이고, 모델은 그 JSON 을
그대로 왕복해야 한다.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from family_fitness_ai.api.coach_schemas import (
    MAX_COPY,
    MAX_MESSAGE_REFUSAL_REASON,
    MAX_RUN_ID,
    MAX_RUN_REFUSAL_REASON,
    MAX_TITLE,
    CoachMessageResponse,
    CoachRunRequest,
    CoachRunResult,
    MessageRefusalReason,
    Mission,
    Proposal,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# 백엔드가 실제로 보내는 모양 — measurements 는 null 이고 가족 전원이 온다.
REQUEST: dict[str, Any] = {
    "profile_refs": [
        {
            "ref": "p_c7a91f",
            "role": "주행자",
            "age": 11,
            "age_unit": "세",
            "sex": "F",
            "input_level": "L2",
            "height_cm": None,
            "weight_kg": None,
            "measurements": {"028": 41.3, "012": 4.0},
        },
        {
            "ref": "p_3d0b25",
            "role": "동반자",
            "age": 41,
            "age_unit": "세",
            "sex": "F",
            "input_level": "L1",
            "height_cm": 162.0,
            "weight_kg": 57.0,
            "measurements": None,
        },
    ],
    "period": {"start_date": "2026-09-07", "weeks": 1},
    "constraints": {"days_per_week": 3, "minutes_per_session": 15},
}


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def wire(model: BaseModel) -> dict[str, Any]:
    """응답을 낼 때와 같은 방식으로 직렬화한다."""
    return model.model_dump(by_alias=True, mode="json")


def profile(**overrides: Any) -> dict[str, Any]:
    return {**REQUEST["profile_refs"][0], **overrides}


# ── 1. 골든 왕복 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("coach_run_result.json", CoachRunResult),
        ("coach_run_refused.json", CoachRunResult),
        ("coach_message.json", CoachMessageResponse),
    ],
)
def test_골든이_키와_값까지_그대로_왕복한다(name: str, model: type[BaseModel]) -> None:
    original = load(name)
    assert wire(model.model_validate(original)) == original


def test_요청_골든도_왕복한다() -> None:
    assert wire(CoachRunRequest.model_validate(REQUEST)) == REQUEST


# ── 2·3. 닫아 둔 자리 ─────────────────────────────────────────────────


def test_인용_label_이_빈_문자열이면_거부한다() -> None:
    """백엔드 `CitationBody.label` 에 기본값이 없어 빠지면 역직렬화가 터진다.

    그 예외는 503 으로 번역되지 않아 코치 실행이 그냥 FAILED 가 된다.
    """
    body = load("coach_message.json")
    body["citations"][0]["label"] = ""
    with pytest.raises(ValidationError):
        CoachMessageResponse.model_validate(body)


def test_인용_label_키가_없으면_거부한다() -> None:
    body = load("coach_message.json")
    del body["citations"][0]["label"]
    with pytest.raises(ValidationError):
        CoachMessageResponse.model_validate(body)


def test_status_에_없는_값을_넣으면_거부한다() -> None:
    body = load("coach_run_refused.json")
    body["status"] = "queued"
    with pytest.raises(ValidationError):
        CoachRunResult.model_validate(body)


# ── 4·5. 계약보다 넓게 받는다 ─────────────────────────────────────────


def test_가족_6명과_measurements_null_과_weeks_1_이_통과한다() -> None:
    """docs/03 §5.1 은 1~4명이라 적었지만 백엔드는 가족 전원을 보낸다."""
    body = {
        **REQUEST,
        "profile_refs": [profile(ref=f"p_{i}", measurements=None) for i in range(6)],
        "period": {"start_date": "2026-09-07", "weeks": 1},
    }
    parsed = CoachRunRequest.model_validate(body)
    assert len(parsed.profile_refs) == 6
    assert parsed.profile_refs[0].measurements is None
    assert parsed.period.weeks == 1


def test_role_과_input_level_에_모르는_값이_와도_통과한다() -> None:
    """호출자가 고칠 수 없는 400 을 만들지 않는다."""
    body = {
        **REQUEST,
        "profile_refs": [profile(role="관찰자", input_level="L9", sex="X", age_unit="주")],
    }
    parsed = CoachRunRequest.model_validate(body)
    assert parsed.profile_refs[0].role == "관찰자"
    assert parsed.profile_refs[0].input_level == "L9"


def test_프로필_한_명도_없으면_거부한다() -> None:
    with pytest.raises(ValidationError):
        CoachRunRequest.model_validate({**REQUEST, "profile_refs": []})


# ── 6. 와이어에 없는 이름을 내보내지 않는다 ───────────────────────────


def keys_everywhere(value: Any) -> set[str]:
    """중첩 어디에든 이 키가 있으면 잡는다."""
    found: set[str] = set()
    if isinstance(value, dict):
        found |= set(value)
        for item in value.values():
            found |= keys_everywhere(item)
    elif isinstance(value, list):
        for item in value:
            found |= keys_everywhere(item)
    return found


def test_직렬화에_end_sec_와_week_와_cadence_가_없다() -> None:
    """`VideoBody` 에는 `end_sec` 자리가 없고 `MissionBody` 에는 주기 자리가 없다."""
    dumped = wire(CoachRunResult.model_validate(load("coach_run_result.json")))
    assert {"end_sec", "week", "cadence"} & keys_everywhere(dumped) == set()


def test_세션_키_집합이_와이어와_정확히_같다() -> None:
    session = wire(CoachRunResult.model_validate(load("coach_run_result.json")))
    mission = session["proposal"]["missions"][0]
    assert set(mission) == {"title", "period", "participants", "sessions", "copy"}
    assert set(mission["period"]) == {"start_date", "end_date"}
    assert set(mission["sessions"][0]) == {
        "day_offset",
        "exercise_name",
        "fitness_factor",
        "duration_min",
        "video",
        "evidence",
    }
    assert set(mission["sessions"][0]["video"]) == {"video_id", "start_sec"}


# ── 7. ref 는 바이트 그대로 ───────────────────────────────────────────


def test_profile_ref_를_바이트까지_그대로_보관한다() -> None:
    """백엔드 `refIndex` 에 없는 ref 는 조용히 버려지고 미션이 사라진다.

    공백 하나를 우리가 고치면 그 참여자가 없는 것이 된다.
    """
    dirty = "  P_c7A91f\t"
    body = {**REQUEST, "profile_refs": [profile(ref=dirty)]}
    parsed = CoachRunRequest.model_validate(body)
    assert parsed.profile_refs[0].ref == dirty
    assert parsed.profile_refs[0].ref.encode() == dirty.encode()
    assert wire(parsed)["profile_refs"][0]["ref"] == dirty


# ── 8. 길이 상한 ──────────────────────────────────────────────────────


def test_문구는_우리가_자른다() -> None:
    """title·copy·runs 거부 사유는 **자른다** — 잘려도 뜻이 남는다."""
    body = load("coach_run_result.json")
    mission = body["proposal"]["missions"][0]
    mission["title"] = "가" * (MAX_TITLE + 10)
    mission["copy"]["child"] = "나" * (MAX_COPY + 10)
    mission["copy"]["parent"] = "다" * (MAX_COPY + 10)
    body["refusal_reason"] = "라" * (MAX_RUN_REFUSAL_REASON + 10)

    parsed = CoachRunResult.model_validate(body)
    assert parsed.proposal is not None
    first = parsed.proposal.missions[0]
    assert len(first.title) == MAX_TITLE
    assert len(first.text.child) == MAX_COPY
    assert len(first.text.parent) == MAX_COPY
    assert parsed.refusal_reason is not None
    assert len(parsed.refusal_reason) == MAX_RUN_REFUSAL_REASON


def test_run_id_는_자르지_않고_거부한다() -> None:
    """잘린 id 로는 폴링이 되지 않는다. 우리가 내는 id 는 `cr_`+16자 = 19자다."""
    body = load("coach_run_refused.json")
    body["run_id"] = "cr_" + "0" * MAX_RUN_ID
    with pytest.raises(ValidationError):
        CoachRunResult.model_validate(body)

    ours = load("coach_run_refused.json")["run_id"]
    assert ours.startswith("cr_")
    assert len(ours) == 19 <= MAX_RUN_ID


def test_메시지_거부_사유는_네_값이고_모두_40자_안이다() -> None:
    """`coach_messages.refusal_reason varchar(40)` 에 잘리지 않고 들어간다."""
    reasons = get_args(MessageRefusalReason)
    assert set(reasons) == {
        "no_relevant_source",
        "age_filter_empty",
        "medical_query",
        "no_citation_generated",
    }
    assert max(len(r) for r in reasons) <= MAX_MESSAGE_REFUSAL_REASON

    body = load("coach_message.json")
    body["refusal_reason"] = "made_up_reason"
    with pytest.raises(ValidationError):
        CoachMessageResponse.model_validate(body)


# ── 9·10. 일일/주간은 기간으로 가른다 ─────────────────────────────────


def span_days(mission: Mission) -> int:
    return (mission.period.end_date - mission.period.start_date).days


def golden_proposal() -> Proposal:
    parsed = CoachRunResult.model_validate(load("coach_run_result.json"))
    assert parsed.proposal is not None
    return parsed.proposal


def test_일일은_하루_주간은_엿새_차다() -> None:
    """미션 하나의 `period` 가 1일이면 일일, 7일이면 주간이다. 계약은 안 바뀐다."""
    missions = golden_proposal().missions
    daily = [m for m in missions if span_days(m) == 0]
    weekly = [m for m in missions if span_days(m) == 6]

    assert len(daily) == 3
    assert len(weekly) == 1
    assert len(daily) + len(weekly) == len(missions)

    for mission in daily:
        assert mission.period.start_date == mission.period.end_date
        assert len(mission.sessions) == 1
        assert mission.sessions[0].day_offset == 0
    assert len(weekly[0].sessions) == 3


def test_세션_날짜가_기간_안에_있다() -> None:
    for mission in golden_proposal().missions:
        for session in mission.sessions:
            day = mission.period.start_date + dt.timedelta(days=session.day_offset)
            assert mission.period.start_date <= day <= mission.period.end_date


def overlapping_participants(proposal: Proposal) -> list[tuple[str, str, str]]:
    """기간이 겹치는 두 미션에 동시에 든 참여자를 찾는다.

    백엔드 `MissionCompletionPolicy` 의 TIMER_MINUTES 는 기간 안 활동 **합계**다
    (`activityQuery.totals(profileId, startsOn, endsOn)`). 같은 참여자가 기간이
    겹치는 두 미션을 동시에 가지면 같은 15분이 두 번 센다.
    """
    clashes = []
    missions = proposal.missions
    for i, left in enumerate(missions):
        for right in missions[i + 1 :]:
            if left.period.start_date > right.period.end_date:
                continue
            if right.period.start_date > left.period.end_date:
                continue
            shared = {p.ref for p in left.participants} & {p.ref for p in right.participants}
            clashes += [(ref, left.title, right.title) for ref in sorted(shared)]
    return clashes


def test_기간이_겹치는_두_미션에_같은_참여자가_없다() -> None:
    assert overlapping_participants(golden_proposal()) == []


def test_겹침_검사가_실제로_겹침을_잡는다() -> None:
    """단정문이 통과하는 이유가 검사가 헐거워서가 아님을 보인다."""
    body = load("coach_run_result.json")
    body["proposal"]["missions"][3]["participants"] = [{"ref": "p_c7a91f", "role": "주행자"}]
    parsed = CoachRunResult.model_validate(body)
    assert parsed.proposal is not None
    assert overlapping_participants(parsed.proposal) != []


# ── 거부 응답의 모양 ──────────────────────────────────────────────────


def test_거부는_proposal_이_null_이고_refused_가_true_다() -> None:
    parsed = CoachRunResult.model_validate(load("coach_run_refused.json"))
    assert parsed.status == "refused"
    assert parsed.proposal is None
    assert parsed.refused is True
    assert parsed.refusal_reason == "age_filter_empty"


def test_메시지_본문에_인용_번호가_박혀_있다() -> None:
    parsed = CoachMessageResponse.model_validate(load("coach_message.json"))
    assert "[1]" in parsed.answer
    assert [c.index for c in parsed.citations] == [1]


def test_세션의_근거가_인용_번호_안에_있다() -> None:
    proposal = golden_proposal()
    known = {c.index for c in proposal.citations}
    for mission in proposal.missions:
        for session in mission.sessions:
            assert session.evidence
            assert set(session.evidence) <= known


def test_영상은_실물과_null_이_섞여_있고_체력요인은_빈_것이_있다() -> None:
    sessions = [s for m in golden_proposal().missions for s in m.sessions]
    assert any(s.video is not None for s in sessions)
    assert any(s.video is None for s in sessions)
    assert any(s.fitness_factor == "" for s in sessions)
    assert any(s.fitness_factor != "" for s in sessions)
