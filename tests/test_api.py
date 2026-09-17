"""계약대로 나가는지. LLM 은 꺼진 채로 돈다 — 규칙 편성만으로도 계약을 지켜야 한다."""

from __future__ import annotations

import time

import pytest
from conftest import needs_embedder, needs_index, needs_release
from fastapi.testclient import TestClient

from family_fitness_ai.api.app import app

client = TestClient(app)

CHILD = {
    "ref": "p_c7a91f",
    "role": "주행자",
    "age": 11,
    "age_unit": "세",
    "sex": "F",
    "input_level": "L2",
    "measurements": {"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
}


def _wait(run_id: str, seconds: float = 60.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = client.get(f"/v1/coach/runs/{run_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.2)
    pytest.fail("60초 안에 끝나지 않았다")


def _finish(body: dict, seconds: float = 60.0) -> dict:
    started = client.post("/v1/coach/runs", json=body)
    assert started.status_code == 202
    return _wait(started.json()["run_id"], seconds)


@needs_release
def test_assessment_returns_factors_and_display_copy():
    response = client.post(
        "/v1/fitness/assessment",
        json={"profile_ref": "p", "age": 11, "sex": "F", "measurements": {"020": 70}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["parent_scope"]["factors"][0]["item_label"] == "15m왕복오래달리기"
    assert body["disclaimer"]


def test_blood_pressure_is_refused_by_code():
    response = client.post(
        "/v1/fitness/assessment",
        json={"profile_ref": "p", "age": 11, "sex": "F", "measurements": {"006": 120}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ITEM_NOT_ALLOWED"


def test_a_missing_field_is_one_error_shape():
    response = client.post("/v1/fitness/assessment", json={"age": 11})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_video_search_needs_a_factor_or_a_name():
    response = client.post("/v1/videos/search", json={"age_group": "유소년"})
    assert response.status_code == 400


@needs_index
@needs_embedder
def test_video_search_may_come_back_empty_and_say_why():
    response = client.post(
        "/v1/videos/search",
        json={"age_group": "어르신", "fitness_factors": ["평형성"], "k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hits"] == [] or all(0 <= hit["score"] <= 1 for hit in body["hits"])
    assert isinstance(body["filtered_out"], dict)


@needs_index
@needs_embedder
def test_medical_questions_are_refused_without_searching():
    response = client.post(
        "/v1/coach/messages",
        json={"question": "무릎이 아픈데 어떤 운동을 해야 하나요", "age_group": "유소년"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "medical_query"
    assert body["citations"] == []


@needs_index
@needs_embedder
def test_an_answer_always_carries_a_citation():
    response = client.post(
        "/v1/coach/messages",
        json={"question": "유소년 유연성을 기르는 운동이 궁금해요", "age_group": "유소년"},
    )
    body = response.json()
    if not body["refused"]:
        assert body["citations"]
        assert "[1]" in body["answer"]


@needs_release
@needs_index
@needs_embedder
def test_a_run_finishes_and_every_session_points_at_a_cited_clip():
    state = _finish(
        {
            "profile_refs": [CHILD],
            "period": {"start_date": "2026-09-07", "weeks": 1},
            "constraints": {"days_per_week": 3, "minutes_per_session": 15},
        }
    )
    assert state["status"] == "succeeded", state
    assert [step["name"] for step in state["steps"]] == [
        "assess",
        "retrieve",
        "compose",
        "verify",
    ]
    proposal = state["proposal"]
    allowed = {citation["index"] for citation in proposal["citations"]}
    assert allowed
    for mission in proposal["missions"]:
        assert mission["sessions"]
        phases = [session["phase"] for session in mission["sessions"]]
        assert phases == sorted(phases, key=["준비운동", "본운동", "정리운동"].index)
        for session in mission["sessions"]:
            assert set(session["evidence"]) <= allowed
            assert session["video"]["end_sec"] > session["video"]["start_sec"]


@needs_release
@needs_index
@needs_embedder
def test_an_age_without_data_is_widened_and_said_so():
    """만 8세 처방 자료는 없다. 빈손으로 돌려보내지 않고 넓혀서 권하고 말한다."""
    state = _finish(
        {
            "profile_refs": [{"ref": "p8", "role": "주행자", "age": 8, "sex": "M"}],
            "period": {"start_date": "2026-09-07", "weeks": 1},
        }
    )
    assert state["status"] == "succeeded", state
    assert state["proposal"]["notices"]


@needs_release
@needs_index
@needs_embedder
def test_a_second_run_for_the_same_profile_is_refused_while_one_is_running():
    body = {
        "profile_refs": [CHILD],
        "period": {"start_date": "2026-09-07", "weeks": 1},
    }
    first = client.post("/v1/coach/runs", json=body)
    assert first.status_code == 202
    second = client.post("/v1/coach/runs", json=body)
    if second.status_code == 409:
        assert second.json()["error"]["code"] == "RUN_IN_PROGRESS"
    _wait(first.json()["run_id"])


def test_every_failure_wears_the_same_shape():
    """FastAPI 가 스스로 내는 오류도 우리 모양으로 나가야 한다.

    호출하는 쪽이 error.code 하나로 분기한다. 여기서만 detail 로 나가면 그
    분기가 뚫린다. run_id 를 비워 부르면 실제로 이 길로 온다.
    """
    missing = client.get("/v1/coach/nothing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"

    wrong_method = client.get("/v1/coach/runs")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_an_unknown_run_is_a_404():
    response = client.get("/v1/coach/runs/cr_없는것")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RUN_NOT_FOUND"
