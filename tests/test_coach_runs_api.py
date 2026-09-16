"""`POST /v1/coach/runs` · `GET /v1/coach/runs/{run_id}` (docs/03 §5).

**백엔드가 실제로 보내는 본문**으로 시험한다 — `measurements: null` 이 섞이고 가족
전원이 오며 `weeks` 는 1이다 (`AiWire.CoachRunRequestBody`).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from family_fitness_ai.api.app import app
from family_fitness_ai.api.coach_schemas import MAX_RUN_ID
from family_fitness_ai.coach.store import RUN_PREFIX, RunStore, new_run_id


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _profile(ref: str, role: str, age: int, unit: str = "세", sex: str = "F", **kw: Any):
    body = {
        "ref": ref,
        "role": role,
        "age": age,
        "age_unit": unit,
        "sex": sex,
        "input_level": "L0",
        "height_cm": None,
        "weight_kg": None,
        "measurements": None,
    }
    body.update(kw)
    return body


def _request(*profiles: Any, days: int = 3, minutes: int = 15, start: str = "2026-09-21"):
    return {
        "profile_refs": list(profiles),
        "period": {"start_date": start, "weeks": 1},
        "constraints": {"days_per_week": days, "minutes_per_session": minutes},
    }


FAMILY = _request(
    _profile("p_c7a91f", "주행자", 11, sex="F", input_level="L2", measurements={"028": 41.3}),
    _profile("p_3d0b25", "동반자", 41, sex="F"),
    _profile("p_9ab0c2", "응원", 68, sex="M"),
)


def _run(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    accepted = client.post("/v1/coach/runs", json=body)
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run_id"]
    got = client.get(f"/v1/coach/runs/{run_id}")
    assert got.status_code == 200, got.text
    return got.json()


# ── 접수 ────────────────────────────────────────────────────────────


def test_post_returns_202_with_run_id(client: TestClient) -> None:
    response = client.post("/v1/coach/runs", json=FAMILY)
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"].startswith(RUN_PREFIX)
    assert len(body["run_id"]) <= MAX_RUN_ID
    assert body["status"] == "running"
    assert body["poll_after_ms"] > 0


def test_first_poll_does_not_wait(client: TestClient) -> None:
    """편성은 `POST` 안에서 끝난다 — 첫 폴링이 결과를 낸다."""
    assert _run(client, FAMILY)["status"] == "succeeded"


def test_never_returns_409(client: TestClient) -> None:
    """백엔드는 `POST` 의 409 를 재시도도 503 취급도 하지 않아 실행이 그냥 FAILED 가 된다."""
    for _ in range(3):
        assert client.post("/v1/coach/runs", json=FAMILY).status_code == 202


def test_unknown_run_id_is_404(client: TestClient) -> None:
    response = client.get("/v1/coach/runs/cr_does_not_exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RUN_NOT_FOUND"


# ── 응답 모양 ───────────────────────────────────────────────────────


def test_steps_are_the_four_nodes_in_order(client: TestClient) -> None:
    steps = _run(client, FAMILY)["steps"]
    assert [s["name"] for s in steps] == ["assess", "retrieve", "compose", "verify"]
    assert [s["seq"] for s in steps] == [1, 2, 3, 4]
    assert all(s["summary"] for s in steps)


def test_wire_keys_match_the_contract(client: TestClient) -> None:
    """와이어에 없는 이름을 내보내지 않는다 — `end_sec`·`week`·`cadence` 는 자리가 없다."""
    body = _run(client, FAMILY)
    assert set(body) == {"run_id", "status", "steps", "proposal", "refused", "refusal_reason"}
    mission = body["proposal"]["missions"][0]
    assert set(mission) == {"title", "period", "participants", "sessions", "copy"}
    assert set(mission["period"]) == {"start_date", "end_date"}
    assert set(mission["copy"]) == {"child", "parent"}
    session = mission["sessions"][0]
    assert set(session) == {
        "day_offset",
        "exercise_name",
        "fitness_factor",
        "duration_min",
        "video",
        "evidence",
    }
    for video in (s["video"] for m in body["proposal"]["missions"] for s in m["sessions"]):
        if video is not None:
            assert set(video) == {"video_id", "start_sec"}


def test_citations_are_never_empty_and_label_is_filled(client: TestClient) -> None:
    citations = _run(client, FAMILY)["proposal"]["citations"]
    assert citations
    for citation in citations:
        assert citation["label"].strip(), "label 이 비면 백엔드 역직렬화가 터진다"
        assert citation["chunk_id"].strip()
        assert citation["index"] >= 1


def test_daily_and_weekly_are_told_apart_by_period_only(client: TestClient) -> None:
    missions = _run(client, FAMILY)["proposal"]["missions"]
    daily = [m for m in missions if m["period"]["start_date"] == m["period"]["end_date"]]
    weekly = [m for m in missions if m["period"]["start_date"] != m["period"]["end_date"]]
    assert len(daily) == 3
    assert len(weekly) == 2  # 아이 몫 하나 · 동반자 몫 하나
    # 주기를 담는 값이 어디에도 없다
    assert "week" not in str(missions)
    assert "cadence" not in str(missions)


def test_refs_come_back_byte_for_byte(client: TestClient) -> None:
    """`refIndex` 에 없는 ref 는 조용히 버려지고 참여자 0명 항목은 미션이 안 생긴다."""
    odd = "  p_MiXeD_Case  "
    body = _run(client, _request(_profile(odd, "주행자", 11)))
    assert {p["ref"] for m in body["proposal"]["missions"] for p in m["participants"]} == {odd}


# ── 계약보다 넓게 받는다 ────────────────────────────────────────────


def test_accepts_six_profiles(client: TestClient) -> None:
    """계약은 1~4명이라 적었지만 백엔드는 가족 전원을 보낸다."""
    family = _request(
        _profile("p1", "주행자", 11),
        *[_profile(f"p{i}", "동반자", 30 + i) for i in range(2, 7)],
    )
    assert client.post("/v1/coach/runs", json=family).status_code == 202


def test_accepts_unknown_role_and_input_level(client: TestClient) -> None:
    """호출자가 못 고치는 400 을 만들지 않는다."""
    body = _request(_profile("p1", "주행자", 11, input_level="L9"), _profile("p2", "관찰자", 40))
    assert client.post("/v1/coach/runs", json=body).status_code == 202


@pytest.mark.parametrize("days", [1, 3, 7])
def test_days_per_week_range(client: TestClient, days: int) -> None:
    body = _run(client, _request(_profile("p1", "주행자", 11), days=days))
    daily = [
        m
        for m in body["proposal"]["missions"]
        if m["period"]["start_date"] == m["period"]["end_date"]
    ]
    assert len(daily) == days


def test_missing_required_field_is_400(client: TestClient) -> None:
    broken = {"period": {"start_date": "2026-09-21", "weeks": 1}}
    response = client.post("/v1/coach/runs", json=broken)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_measurements_are_not_echoed_in_errors(client: TestClient) -> None:
    """오류 응답에 측정값이 되돌아가지 않는다 (docs/01 §5)."""
    broken = _request(_profile("p1", "주행자", 11, measurements={"028": 41.3}))
    del broken["constraints"]
    response = client.post("/v1/coach/runs", json=broken)
    assert response.status_code == 400
    assert "41.3" not in response.text


# ── 거부 ────────────────────────────────────────────────────────────


def test_refuses_when_no_cell_matches(client: TestClient) -> None:
    """만 3세는 처방 칸이 없다. 인용이 0이면 거부다 (docs/03 §5.4)."""
    body = _run(client, _request(_profile("p1", "주행자", 3)))
    assert body["status"] == "refused"
    assert body["refused"] is True
    assert body["refusal_reason"]
    assert body["proposal"] is None


def test_toddler_in_years_is_refused_not_crashed(client: TestClient) -> None:
    """만 4~6세를 `세` 로 주면 유아기 칸(개월)에 닿지 못한다 — 거부로 드러낸다."""
    body = _run(client, _request(_profile("p1", "주행자", 5, unit="세")))
    assert body["status"] == "refused"


def test_toddler_in_months_works(client: TestClient) -> None:
    body = _run(client, _request(_profile("p1", "주행자", 60, unit="개월")))
    assert body["status"] == "succeeded"


# ── 저장소 ──────────────────────────────────────────────────────────


def test_store_evicts_by_ttl() -> None:
    store = RunStore(ttl_sec=10.0)
    from family_fitness_ai.api.coach_schemas import CoachRunResult

    result = CoachRunResult(run_id=new_run_id(), status="succeeded")
    store.put(result, now=0.0)
    assert store.get(result.run_id, now=5.0) is not None
    assert store.get(result.run_id, now=11.0) is None


def test_store_drops_the_oldest_when_full() -> None:
    from family_fitness_ai.api.coach_schemas import CoachRunResult

    store = RunStore(max_runs=2)
    ids = []
    for i in range(3):
        result = CoachRunResult(run_id=f"cr_{i}", status="succeeded")
        store.put(result, now=float(i))
        ids.append(result.run_id)
    assert len(store) == 2
    assert store.get(ids[0], now=3.0) is None
    assert store.get(ids[2], now=3.0) is not None


def test_run_id_is_unique() -> None:
    assert len({new_run_id() for _ in range(500)}) == 500
