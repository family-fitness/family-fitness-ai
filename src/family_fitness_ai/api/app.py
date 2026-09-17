"""AI 서비스.

계약은 docs/인터페이스-명세.md 다. 기준 경로 /v1, JSON(UTF-8), 인증 없음.
성공은 payload 를 그대로 내고, 실패는 {"error": {...}} 한 모양이다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from family_fitness_ai.api.schemas import (
    MessageIn,
    ProfileIn,
    RunIn,
    TrajectoryIn,
    VideoSearchIn,
)
from family_fitness_ai.coach import answer as coach_answer
from family_fitness_ai.coach.compose import Constraints, RunProfile
from family_fitness_ai.coach.runs import store
from family_fitness_ai.common.errors import ApiError, temporarily_unavailable
from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.index import missing_files
from family_fitness_ai.stats.assess import Profile, assessment, trajectory
from family_fitness_ai.video.videos import search_videos

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """인덱스가 없으면 뜨지 않는다.

    없어도 서버는 떠서 평가·궤적만 답하고 코치 쪽은 요청이 올 때마다 깨진다.
    그러면 배포가 성공한 것처럼 보이고, 아무도 안 보는 새 절반이 죽어 있다.
    뜨지 않으면 컨테이너가 바로 알려 준다.
    """
    absent = missing_files()
    if absent:
        raise RuntimeError(
            f"코퍼스 인덱스가 없다: {settings().index_dir} ({', '.join(absent)} 없음). "
            "data/index 를 배포에 실었는지 보라"
        )
    yield


app = FastAPI(title="우리가족 체력키움 · AI", version="0.1.0", lifespan=lifespan)
v1 = APIRouter(prefix="/v1")


@app.exception_handler(ApiError)
async def _api_error(_: Request, error: ApiError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content=error.body())


#: FastAPI 가 스스로 내는 오류의 코드 이름. 우리가 던진 것이 아니어도 나가는
#: 모양은 같아야 한다 — 호출하는 쪽이 error.code 하나로 분기할 수 있게.
_STATUS_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
    code = _STATUS_CODES.get(
        error.status_code, "BAD_REQUEST" if error.status_code < 500 else "TEMPORARILY_UNAVAILABLE"
    )
    if error.status_code == 405:
        message = f"{request.method} {request.url.path} 는 받지 않습니다"
    elif error.status_code == 404:
        message = f"그런 경로가 없습니다: {request.url.path}"
    else:
        message = str(error.detail)
    return JSONResponse(
        status_code=error.status_code, content={"error": {"code": code, "message": message}}
    )


@app.exception_handler(Exception)
async def _unexpected(_: Request, error: Exception) -> JSONResponse:
    """예상 못 한 예외도 계약 모양으로 낸다.

    자세한 것은 로그에만 남긴다. 서버 안 경로나 스택이 응답에 실리면 배포한
    디렉터리 구조가 밖으로 나간다.
    """
    log.exception("처리 중 예외", exc_info=error)
    return JSONResponse(status_code=503, content=temporarily_unavailable().body())


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, error: RequestValidationError) -> JSONResponse:
    # 검증 중에 던진 우리 오류(혈압 항목 등)는 그 모양을 지킨다.
    for problem in error.errors():
        cause = problem.get("ctx", {}).get("error")
        if isinstance(cause, ApiError):
            return JSONResponse(status_code=cause.status, content=cause.body())
    first = error.errors()[0] if error.errors() else {}
    where = " → ".join(str(part) for part in first.get("loc", ())[1:])
    message = f"{where}: {first.get('msg', '요청을 읽을 수 없습니다')}"
    return JSONResponse(
        status_code=400, content={"error": {"code": "BAD_REQUEST", "message": message}}
    )


def _profile(body: ProfileIn) -> Profile:
    return Profile(
        profile_ref=body.profile_ref,
        age=body.age,
        age_unit=body.age_unit,
        sex=body.sex,
        height_cm=body.height_cm,
        weight_kg=body.weight_kg,
        measurements=body.measurements,
    )


@v1.post("/fitness/assessment")
def post_assessment(body: ProfileIn) -> dict[str, object]:
    return assessment(_profile(body))


@v1.post("/fitness/trajectory")
def post_trajectory(body: TrajectoryIn) -> dict[str, object]:
    return trajectory(_profile(body), body.item_code, body.horizon_years)


@v1.post("/videos/search")
def post_video_search(body: VideoSearchIn) -> dict[str, object]:
    if not body.fitness_factors and not body.exercise_names:
        raise ApiError(
            400,
            "BAD_REQUEST",
            "fitness_factors 와 exercise_names 중 최소 하나는 있어야 합니다",
        )
    return search_videos(
        body.age_group,
        tuple(body.fitness_factors),
        tuple(body.exercise_names),
        body.k,
    )


@v1.post("/coach/runs", status_code=202)
def post_run(body: RunIn) -> dict[str, object]:
    profiles = [
        RunProfile(
            ref=row.ref,
            role=row.role,
            age=row.age,
            age_unit=row.age_unit,
            sex=row.sex,
            input_level=row.input_level,
            height_cm=row.height_cm,
            weight_kg=row.weight_kg,
            measurements=row.measurements,
        )
        for row in body.profile_refs
    ]
    constraints = Constraints(
        days_per_week=body.constraints.days_per_week,
        minutes_per_session=body.constraints.minutes_per_session,
        quiet=body.constraints.quiet,
        small_space=body.constraints.small_space,
        no_props=body.constraints.no_props,
    )
    run = store.start(profiles, body.period.start_date, body.period.weeks, constraints)
    return {"run_id": run.run_id, "status": run.status, "poll_after_ms": 1500}


@v1.get("/coach/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    return store.get(run_id).dict()


@v1.post("/coach/messages")
def post_message(body: MessageIn) -> dict[str, object]:
    return coach_answer.answer(body.question, body.age_group or "", body.profile_ref)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(v1)
