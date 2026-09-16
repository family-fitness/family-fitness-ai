"""`POST /v1/coach/runs` · `GET /v1/coach/runs/{run_id}` (docs/03).

**편성은 `POST` 안에서 끝낸다.** 계약은 비동기로 열어 두었지만 (15~40초를 예상한
것은 LLM 을 태울 때다) 지금 편성은 커밋된 CSV 조회뿐이라 밀리초다. 태스크 큐를
붙이면 없는 지연을 만들고 실패할 자리를 늘린다 (docs/01 §3.1).

그래서 `POST` 는 계약대로 `202` 와 `run_id` 를 내고, 결과는 이미 저장돼 있어
**첫 폴링이 기다리지 않는다.**

**409 를 어떤 경로로도 내지 않는다.** 백엔드는 `POST` 의 409 를 재시도도 503 취급도
하지 않아 그 실행이 그냥 `FAILED` 가 된다 — 중복 막기는 백엔드가 가족·주차로 이미 한다.
"""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import APIRouter, Response

from ..coach import answer as A
from ..coach import verify as V
from ..coach.llm import AnswerWriter, writer_from_settings
from ..coach.store import RunStore, new_run_id
from ..common import logging as reqlog
from ..common.errors import ApiError, ErrorCode
from ..common.settings import RELEASE_DIR, get_settings
from ..common.types import resolve_age_group
from ..mission import plan as PL
from ..mission.build import Mission as MissionRow
from ..mission.build import read_missions
from ..mission.cells import Cell, load_cells
from ..mission.segments import SEGMENTS_FILE, VideoSegment, read_segments
from ..rag.embed import Embedder, embedder_from_settings
from ..rag.search import Corpus
from .coach_schemas import (
    CoachMessageRequest,
    CoachMessageResponse,
    CoachRunAccepted,
    CoachRunRequest,
    CoachRunResult,
    Step,
)

router = APIRouter(prefix="/v1/coach", tags=["coach"])

POLL_AFTER_MS = 1500


@lru_cache(maxsize=1)
def _cells() -> list[Cell]:
    """산출물은 기동 시 한 번 올린다. 요청마다 CSV를 읽지 않는다."""
    return load_cells()


@lru_cache(maxsize=1)
def _segments() -> list[VideoSegment]:
    """**없으면 빈 목록이다** — 미션은 영상이 없어도 성립한다 (docs/03 §5.7).

    지금은 `verify` 가 영상의 연령대를 대조하는 데만 쓴다. 편성이 보는 것은
    미션 집합(`_mission_set`)이다.
    """
    return read_segments(RELEASE_DIR / SEGMENTS_FILE)


@lru_cache(maxsize=1)
def _mission_set() -> list[MissionRow]:
    """**추천의 검색 대상** (`missions.csv`). 기동 시 한 번 올린다."""
    return read_missions()


@lru_cache(maxsize=1)
def _store() -> RunStore:
    return RunStore()


def _video_groups(segments: list[VideoSegment]) -> dict[str, str]:
    return {s.video_id: s.age_group for s in segments}


def _refused(run_id: str, steps: list[Step], reason: str) -> CoachRunResult:
    return CoachRunResult(
        run_id=run_id,
        status="refused",
        steps=steps,
        proposal=None,
        refused=True,
        refusal_reason=reason,
    )


def _assess_summary(request: CoachRunRequest, planned: list[PL.MemberPlan]) -> str:
    cheer = sum(1 for p in request.profile_refs if p.role == PL.CHEER)
    roles = "·".join(f"{m.role} {m.age_group}" for m in planned) or "없음"
    return (
        f"구성원 {len(request.profile_refs)}명 · 편성 대상 {len(planned)}명({roles}) · 응원 {cheer}"
    )


def _retrieve_summary(planned: list[PL.MemberPlan]) -> str:
    parts = []
    for m in planned:
        pulled = f" 나이 {m.pulled_note}" if m.pulled_note else ""
        videos = sum(1 for c in m.picks if c.has_video)
        parts.append(
            f"{m.match.age_group} {m.match.age}{m.match.cells[0].age_unit}"
            f"{pulled} 후보 {len(m.pool)} · 영상 {videos}/{len(m.picks)}"
        )
    return " · ".join(parts)


@router.post("/runs", status_code=202, response_model=CoachRunAccepted)
def start_run(request: CoachRunRequest, response: Response) -> CoachRunAccepted:
    run_id = new_run_id()
    cells, segments, mission_set = _cells(), _segments(), _mission_set()

    profiles = [(p.ref, p.role, p.age, p.age_unit, p.sex) for p in request.profile_refs]
    start = request.period.start_date

    try:
        proposal, planned = PL.build(
            profiles,
            cells,
            mission_set,
            start_date=start,
            days_per_week=request.constraints.days_per_week,
            minutes_per_session=request.constraints.minutes_per_session,
            rotate=PL.iso_week(start),
            # **LLM 은 카드 문구만 쓴다.** 무엇을 추천할지는 미션 집합이 정했다.
            # 꺼져 있거나 실패하면 규칙 문구로 강등한다 (docs/01 §3.1)
            writer=_writer(),
        )
    except PL.Refusal as refusal:
        steps = [
            Step(seq=1, name="assess", status="ok", summary=_assess_summary(request, [])),
            Step(seq=2, name="retrieve", status="failed", summary=refusal.detail),
        ]
        _store().put(_refused(run_id, steps, refusal.reason), time.monotonic())
        reqlog.add_fields(run_id=run_id, refusal_reason=refusal.reason, citations_count=0)
        response.headers["Location"] = f"/v1/coach/runs/{run_id}"
        return CoachRunAccepted(run_id=run_id, status="running", poll_after_ms=POLL_AFTER_MS)

    daily, weekly = PL.count_kinds(proposal.missions)
    writer = _writer()
    steps = [
        Step(seq=1, name="assess", status="ok", summary=_assess_summary(request, planned)),
        Step(seq=2, name="retrieve", status="ok", summary=_retrieve_summary(planned)),
        Step(
            seq=3,
            name="compose",
            status="ok",
            summary=(
                f"미션 {len(proposal.missions)}건(일일 {daily} · 주간 {weekly})"
                f" · 회당 {request.constraints.minutes_per_session}분"
                f" · 문구 {writer.version() if writer else '규칙'}"
            ),
        ),
    ]

    known = {
        p.ref: (resolve_age_group(p.age, p.age_unit) or "", p.age, p.age_unit)  # type: ignore[arg-type]
        for p in request.profile_refs
    }
    failures = V.check_proposal(proposal, known, _video_groups(segments))
    if failures:
        steps.append(
            Step(
                seq=4,
                name="verify",
                status="failed",
                summary=" · ".join(f"[{f.check}] {f.detail}" for f in failures)[:400],
            )
        )
        result = _refused(run_id, steps, "verify_failed")
        reqlog.add_fields(
            run_id=run_id, refusal_reason="verify_failed", citations_count=0, age_match=False
        )
    else:
        videos = sum(1 for m in proposal.missions for s in m.sessions if s.video)
        steps.append(
            Step(
                seq=4,
                name="verify",
                status="ok",
                summary=(
                    f"인용 {len(proposal.citations)}건 · 근거 범위 안 · 금지 어휘 0"
                    f" · 연령대 일치 · 영상 붙은 세션 {videos}"
                ),
            )
        )
        result = CoachRunResult(run_id=run_id, status="succeeded", steps=steps, proposal=proposal)
        reqlog.add_fields(run_id=run_id, citations_count=len(proposal.citations), age_match=True)

    _store().put(result, time.monotonic())
    response.headers["Location"] = f"/v1/coach/runs/{run_id}"
    return CoachRunAccepted(run_id=run_id, status="running", poll_after_ms=POLL_AFTER_MS)


@router.get("/runs/{run_id}", response_model=CoachRunResult)
def get_run(run_id: str) -> CoachRunResult:
    result = _store().get(run_id, time.monotonic())
    if result is None:
        raise ApiError(ErrorCode.RUN_NOT_FOUND, f"모르는 run_id: {run_id}")
    reqlog.add_fields(run_id=run_id, run_status=result.status)
    return result


# ── coach/messages ──────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _corpus() -> Corpus:
    """색인은 기동 시 한 번 올린다. **없으면 예외다** — 라우터가 503 으로 바꾼다."""
    corpus = Corpus.load()
    corpus.expects(embedder_from_settings().version())
    return corpus


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return embedder_from_settings()


@lru_cache(maxsize=1)
def _writer() -> AnswerWriter | None:
    """`COACH_LLM` 이 꺼져 있으면 `None` — 발췌로 답한다."""
    return writer_from_settings()


@router.post("/messages", response_model=CoachMessageResponse)
def messages(request: CoachMessageRequest) -> CoachMessageResponse:
    """질문 하나에 답한다. **거부도 200 이다** (docs/04 §4.3).

    임계값이 설정되지 않았거나 색인·임베딩 서버가 없으면 503 이다 — **빈 인용으로
    답하지 않는다.** 근거 없는 답을 내는 것보다 못 한다고 말하는 쪽이 낫다.
    """
    threshold = get_settings().sim_threshold
    if threshold is None:
        raise ApiError(
            ErrorCode.TEMPORARILY_UNAVAILABLE,
            "SIM_THRESHOLD 가 설정되지 않았다 — 임계값 없이 검색하지 않는다",
        )

    try:
        corpus, embedder, writer = _corpus(), _embedder(), _writer()
    except Exception as failure:
        # 설정 오류(자격증명 없음)도 여기로 온다 — 조용히 발췌로 가지 않는다
        raise ApiError(
            ErrorCode.TEMPORARILY_UNAVAILABLE, f"코치 답변을 쓸 수 없다: {failure}"
        ) from failure

    def embed_query(text: str) -> object:
        try:
            return embedder.embed([text])[0]
        except Exception as failure:
            raise ApiError(
                ErrorCode.TEMPORARILY_UNAVAILABLE, f"임베딩 실패: {failure}"
            ) from failure

    result = A.answer(
        request.question,
        request.age_group,
        corpus,
        embed_query,
        threshold,
        writer=writer,
    )
    reqlog.add_fields(
        age_group=request.age_group,
        citations_count=len(result.response.citations),
        refusal_reason=result.response.refusal_reason or "",
        hit_count=result.hit_count,
        copy_source=result.source,
        medical_hit=bool(result.medical_terms),
    )
    return result.response


def reset_caches() -> None:
    """시험이 산출물을 갈아 끼울 때. 서비스 경로에서는 부르지 않는다."""
    _cells.cache_clear()
    _segments.cache_clear()
    _mission_set.cache_clear()
    _store.cache_clear()
    _corpus.cache_clear()
    _embedder.cache_clear()
    _writer.cache_clear()
