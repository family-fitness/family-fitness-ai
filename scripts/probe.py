"""돌고 있는 서비스에 요청을 보내 눈으로 확인한다.

**시험이 아니다.** 사람이 직접 돌려 보는 입력 도구라 `scripts/` 에 둔다 —
`wire_check.py` 와 같은 자리다. 자동 검사는 `tests/test_*.py` 가 하고 `make verify`
가 돈다. 이 파일은 **돌고 있는 서비스가 있어야** 쓸 수 있어 거기에 넣지 않는다.

먼저 서비스를 띄운다 (다른 창에서):

    source .venv/bin/activate
    make serve                      # uvicorn · 8000
    # 대화까지 보려면 임베딩 서버도
    llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082

쓰는 법:

    python scripts/probe.py                       # 두 흐름을 미리 정한 값으로 훑는다
    python scripts/probe.py --chat                # 질문을 직접 입력한다 (빈 줄이면 끝)
    python scripts/probe.py --age 8 --sex M       # 미션만 · 만 8세 남아
    python scripts/probe.py --age 60 --unit 개월   # 유아기는 개월로 준다
    python scripts/probe.py --ask "유아기 정리운동 알려줘" --age-group 유아기
    python scripts/probe.py --url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"
TIMEOUT_SEC = 30.0

# 미리 정한 질문. 마지막 둘은 **거부가 나와야 맞는 것**이다.
SAMPLE_QUESTIONS = [
    ("유아기 아이에게 어떤 준비운동이 많이 처방되나요?", "유아기"),
    ("유소년 유연성 운동 영상 있어?", "유소년"),
    ("성인 유산소 운동 뭐가 좋아요", "성인"),
    ("무릎이 아픈데 어떤 운동을 해야 해요?", "유소년"),
    ("주식 투자 어떻게 시작해요", "성인"),
]

# JSON 은 모양이 정해지지 않은 것을 그대로 찍는 자리라 `Any` 로 둔다. 계약의 모양을
# 잠그는 것은 `api/coach_schemas.py` 와 `tests/test_coach_wire.py` 가 한다.
Json = dict[str, Any]


def call(url: str, path: str, body: Json | None = None) -> tuple[int, Json]:
    """(상태, 본문). **오류 응답도 본문을 돌려준다** — 계약의 오류 봉투를 봐야 한다."""
    request = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except Exception:
            return error.code, {"error": {"message": error.reason}}
    except urllib.error.URLError as error:
        print(f"[중단] 서비스에 닿지 못했다 ({url}) — {error.reason}")
        print("       다른 창에서 `make serve` 를 돌렸는지 본다")
        raise SystemExit(1) from None


def show_health(url: str) -> None:
    _, health = call(url, "/healthz")
    status, ready = call(url, "/readyz")
    print(f"/healthz  {health}")
    print(f"/readyz   HTTP {status} · ready={ready.get('ready')}")
    checks = ready.get("checks") or {}
    if isinstance(checks, dict):
        print("          " + " · ".join(f"{k}={'O' if v else 'X'}" for k, v in checks.items()))
    if degraded := ready.get("degraded"):
        print(f"          못 쓰는 기능: {degraded}")
        reasons = ready.get("reasons") or {}
        if isinstance(reasons, dict):
            for name, why in reasons.items():
                print(f"            - {name}: {why}")


def profile(ref: str, role: str, age: int, unit: str, sex: str) -> Json:
    """백엔드가 실제로 보내는 모양이다 — `measurements` 는 빈 객체가 아니라 `null` 이다."""
    return {
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


def run_missions(url: str, members: list[Json], start: str, days: int, minutes: int) -> None:
    """흐름 1 — 프론트[미션 추천 버튼] → 서버 → 여기."""
    print("\n## 흐름 1 — 미션 편성  (POST /v1/coach/runs → GET)")
    status, accepted = call(
        url,
        "/v1/coach/runs",
        {
            "profile_refs": members,
            "period": {"start_date": start, "weeks": 1},
            "constraints": {"days_per_week": days, "minutes_per_session": minutes},
        },
    )
    if status != 202:
        print(f"   HTTP {status} · {json.dumps(accepted, ensure_ascii=False)}")
        return
    run_id = str(accepted.get("run_id"))
    print(f"   202 접수 · run_id={run_id} · poll_after_ms={accepted.get('poll_after_ms')}")

    status, result = call(url, f"/v1/coach/runs/{run_id}")
    if status != 200:
        print(f"   HTTP {status} · {json.dumps(result, ensure_ascii=False)}")
        return

    print(f"   200 · status={result.get('status')} · refused={result.get('refused')}")
    for step in result.get("steps") or []:
        print(f"     {step['seq']} {step['name']:9s} {step['status']:8s} {step['summary']}")
    if result.get("refusal_reason"):
        print(f"   거부 사유: {result['refusal_reason']}")
    proposal = result.get("proposal")
    if not isinstance(proposal, dict):
        return

    missions = proposal.get("missions") or []
    citations = proposal.get("citations") or []
    print(f"\n   미션 {len(missions)}건 · 인용 {len(citations)}건")
    for mission in missions:
        period = mission["period"]
        # 일일·주간은 **기간으로만** 가른다 (AI-11 §5.1) — 프론트도 이 규칙을 쓴다
        kind = "일일" if period["start_date"] == period["end_date"] else "주간"
        who = ",".join(p["ref"] for p in mission["participants"])
        print(f"\n   [{kind}] {period['start_date']} ~ {period['end_date']}  ({who})")
        print(f"        제목: {mission['title']}")
        for session in mission["sessions"]:
            video = session["video"]
            where = f"{video['video_id']}@{video['start_sec']}s" if video else "영상 없음"
            print(
                f"        D+{session['day_offset']} {session['exercise_name']} "
                f"{session['duration_min']}분 · {where} · 근거{session['evidence']}"
            )
        print(f"        아이: {mission['copy']['child'] or '(없음)'}")
        print(f"        부모: {mission['copy']['parent']}")
    print()
    for citation in citations:
        print(f"   [{citation['index']}] {citation['label']}")
        print(f"       {citation['chunk_id']}  {citation.get('url') or ''}")


def ask(url: str, question: str, age_group: str) -> None:
    """흐름 2 — 프론트[코치 대화] → 서버 → 여기."""
    status, body = call(
        url,
        "/v1/coach/messages",
        {"profile_ref": "p_probe", "age_group": age_group, "question": question},
    )
    print(f"\n   [{age_group}] {question}")
    if status != 200:
        print(f"   HTTP {status} · {json.dumps(body, ensure_ascii=False)}")
        return
    if body.get("refused"):
        print(f"   거부 · {body.get('refusal_reason')} — {body.get('answer')}")
        return
    print(f"   {body.get('answer')}")
    for citation in body.get("citations") or []:
        print(f"     [{citation['index']}] {citation['label']}")


def chat(url: str, age_group: str) -> None:
    """질문을 직접 입력한다. 빈 줄이면 끝난다."""
    print("\n## 대화 — 질문을 입력한다 (빈 줄이면 끝 · 연령대를 바꾸려면 `유아기: 질문`)")
    while True:
        try:
            line = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        group, _, rest = line.partition(":")
        if rest.strip() and group.strip() in ("유아기", "유소년", "청소년", "성인", "어르신"):
            ask(url, rest.strip(), group.strip())
        else:
            ask(url, line, age_group)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="돌고 있는 서비스에 요청을 보내 눈으로 확인한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--url", default=DEFAULT_URL, help=f"서비스 주소 (기본 {DEFAULT_URL})")
    ap.add_argument("--age", type=int, default=11, help="주행자 나이 (기본 11)")
    ap.add_argument("--unit", default="세", choices=["세", "개월"], help="유아기는 개월이다")
    ap.add_argument("--sex", default="F", choices=["M", "F"])
    ap.add_argument("--companion", type=int, help="동반자 나이. 없으면 동반자를 넣지 않는다")
    ap.add_argument("--days", type=int, default=3, help="주 몇 회 (기본 3)")
    ap.add_argument("--minutes", type=int, default=15, help="회당 몇 분 (기본 15)")
    ap.add_argument("--start", default="2026-09-21", help="시작일 (기본 2026-09-21 · 월요일)")
    ap.add_argument("--ask", help="이 질문 하나만 보낸다")
    ap.add_argument("--age-group", default="유아기", help="대화에 쓸 연령대")
    ap.add_argument("--chat", action="store_true", help="질문을 직접 입력한다")
    ap.add_argument("--missions-only", action="store_true", help="미션만 본다")
    args = ap.parse_args(argv)

    show_health(args.url)

    if args.ask:
        ask(args.url, args.ask, args.age_group)
        return 0
    if args.chat:
        chat(args.url, args.age_group)
        return 0

    members = [profile("p_child", "주행자", args.age, args.unit, args.sex)]
    if args.companion:
        members.append(profile("p_parent", "동반자", args.companion, "세", "F"))
    run_missions(args.url, members, args.start, args.days, args.minutes)

    if not args.missions_only:
        print("\n## 흐름 2 — 코치 대화  (POST /v1/coach/messages)")
        print("   마지막 둘은 거부가 나와야 맞는 것이다")
        for question, group in SAMPLE_QUESTIONS:
            ask(args.url, question, group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
