"""돌고 있는 서비스에 한 바퀴 요청을 보내 눈으로 확인한다.

    make serve          # 먼저 띄운다
    python scripts/probe.py

백엔드 없이 우리 응답만 본다. 통과·실패를 가리는 시험이 아니라, 무엇이 어떻게
나오는지 보는 자리다 — 시험은 tests/ 에 있다.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

CHILD = {
    "ref": "p_c7a91f",
    "role": "주행자",
    "age": 11,
    "age_unit": "세",
    "sex": "F",
    "input_level": "L2",
    "measurements": {"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
}
PARENT = {
    "ref": "p_3d0b25",
    "role": "동반자",
    "age": 41,
    "age_unit": "세",
    "sex": "F",
    "input_level": "L1",
    "height_cm": 162.0,
    "weight_kg": 57.0,
}


def show(title: str, payload: object, limit: int = 900) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    print(text[:limit] + ("…" if len(text) > limit else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--question", default="유소년 유연성을 기르는 운동이 궁금해요")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=30.0)

    show(
        "체력 평가",
        client.post(
            "/fitness/assessment",
            json={k: v for k, v in CHILD.items() if k not in ("ref", "role", "input_level")}
            | {"profile_ref": CHILD["ref"]},
        ).json(),
    )
    show(
        "또래 분포",
        client.post(
            "/fitness/trajectory",
            json={
                "profile_ref": "p",
                "age": 11,
                "sex": "F",
                "item_code": "028",
                "horizon_years": 3,
            },
        ).json(),
        400,
    )
    show(
        "영상 찾기",
        client.post(
            "/videos/search",
            json={"age_group": "유소년", "fitness_factors": ["유연성"], "k": 3},
        ).json(),
        600,
    )
    show(
        "질문 · 의료",
        client.post("/coach/messages", json={"question": "무릎이 아픈데 뭘 해야 하나요"}).json(),
    )
    show(
        "질문",
        client.post(
            "/coach/messages", json={"question": args.question, "age_group": "유소년"}
        ).json(),
    )

    started = client.post(
        "/coach/runs",
        json={
            "profile_refs": [CHILD, PARENT],
            "period": {"start_date": "2026-09-07", "weeks": 1},
            "constraints": {"days_per_week": 3, "minutes_per_session": 15, "quiet": True},
        },
    ).json()
    print(f"\n── 편성 시작 {started}")
    run_id = started["run_id"]
    began = time.time()
    while time.time() - began < 70:
        time.sleep(1.5)
        state = client.get(f"/coach/runs/{run_id}").json()
        if state["status"] != "running":
            break
    print(f"   {time.time() - began:.1f}초 · {state['status']}")
    for step in state["steps"]:
        print(f"   {step['seq']} {step['name']:9s} {step['status']:8s} {step['summary']}")
    proposal = state.get("proposal")
    if not proposal:
        print(f"   거부 사유: {state['refusal_reason']}")
        return
    for notice in proposal.get("notices") or []:
        print(f"   알림: {notice}")
    for mission in proposal["missions"][:2]:
        print(f"\n   · {mission['title']} — {mission['duration_min']}분")
        print(f"     아이: {mission['copy']['child']}")
        print(f"     부모: {mission['copy']['parent']}")
        if mission.get("reason"):
            print(f"     근거: {mission['reason']}")
        for session in mission["sessions"]:
            video = session["video"]
            print(
                f"       [{session['phase']}] {session['exercise_name']} "
                f"{session['duration_sec']}초 · {video['video_id']}"
                f"@{video['start_sec']}-{video['end_sec']} 근거{session['evidence']}"
            )
    print(f"\n   인용 {len(proposal['citations'])}건")
    for citation in proposal["citations"][:4]:
        print(f"     [{citation['index']}] {citation['label']}")


if __name__ == "__main__":
    main()
